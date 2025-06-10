import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_directories():
    """Create organized directory structure for results"""
    base_dir = Path("results")
    subdirs = [
        "one_shot_analysis",
        "one_shot_analysis/success_rates",
        "one_shot_analysis/runtime",
        "one_shot_analysis/accuracy",
        "one_shot_analysis/iterations",
    ]

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir


def load_and_prepare_data():
    """Load and prepare the experimental data"""
    df = pd.read_csv("results/full_experiment.csv")

    # Add derived columns
    df["accuracy_change"] = (
        df["repaired_accuracy_test_set"] - df["baseline_accuracy_test_set"]
    )
    df["edit_set_accuracy_change"] = (
        df["repaired_accuracy_edit_set"] - df["baseline_accuracy_edit_set"]
    )
    df["success_rate"] = df["repair_successful"].astype(int)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def plot_success_rate_analysis(df, base_dir):
    """Analyze repair success rates - individual plots"""
    save_dir = base_dir / "one_shot_analysis" / "success_rates"

    # 1. Success rate by repair set size
    fig, ax = plt.subplots(figsize=(10, 6))
    success_by_size = (
        df.groupby("requested_repair_set_size")["success_rate"]
        .agg(["mean", "count"])
        .reset_index()
    )
    bars = ax.bar(success_by_size["requested_repair_set_size"], success_by_size["mean"])
    ax.set_title("Success Rate by Repair Set Size", fontsize=14, fontweight="bold")
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)

    # Add count labels on bars
    for i, (size, rate, count) in enumerate(
        zip(
            success_by_size["requested_repair_set_size"],
            success_by_size["mean"],
            success_by_size["count"],
        )
    ):
        ax.text(size, rate + 0.02, f"n={count}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_dir / "success_rate_by_size.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Success rate by repair set type
    fig, ax = plt.subplots(figsize=(10, 6))
    success_by_type = (
        df.groupby("repair_set_type")["success_rate"]
        .agg(["mean", "count"])
        .reset_index()
    )
    ax.bar(success_by_type["repair_set_type"], success_by_type["mean"])
    ax.set_title("Success Rate by Repair Set Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Repair Set Type")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "success_rate_by_type.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Success rate by target class (for class_homogeneous_incorrect only)
    class_data = df[df["repair_set_type"] == "class_homogeneous_incorrect"].copy()
    if not class_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        success_by_class = (
            class_data.groupby("target_class")["success_rate"]
            .agg(["mean", "count"])
            .reset_index()
        )
        ax.bar(success_by_class["target_class"], success_by_class["mean"])
        ax.set_title("Success Rate by Target Class", fontsize=14, fontweight="bold")
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            save_dir / "success_rate_by_class.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 4. Success rate heatmap by size and type
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data = df.pivot_table(
        values="success_rate",
        index="repair_set_type",
        columns="requested_repair_set_size",
        aggfunc="mean",
    )
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Success Rate Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "success_rate_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_runtime_analysis(df, base_dir):
    """Analyze repair runtime characteristics - individual plots"""
    save_dir = base_dir / "one_shot_analysis" / "runtime"
    successful_df = df[df["repair_successful"] == True]

    # 1. Runtime vs repair set size scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        successful_df["requested_repair_set_size"],
        successful_df["repair_runtime_seconds"],
        alpha=0.6,
    )
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs Repair Set Size", fontsize=14, fontweight="bold")

    # Add trend line
    z = np.polyfit(
        successful_df["requested_repair_set_size"],
        successful_df["repair_runtime_seconds"],
        1,
    )
    p = np.poly1d(z)
    ax.plot(
        successful_df["requested_repair_set_size"],
        p(successful_df["requested_repair_set_size"]),
        "r--",
        alpha=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_dir / "runtime_vs_size_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Runtime by repair set type boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=successful_df,
        x="repair_set_type",
        y="repair_runtime_seconds",
        ax=ax,
    )
    ax.set_title(
        "Runtime Distribution by Repair Set Type", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "runtime_by_type_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Gurobi iterations vs runtime
    iteration_data = successful_df.dropna(subset=["gurobi_barrier_iterations"])
    if not iteration_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            iteration_data["gurobi_barrier_iterations"],
            iteration_data["repair_runtime_seconds"],
            alpha=0.6,
        )
        ax.set_xlabel("Gurobi Barrier Iterations")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime vs Gurobi Iterations", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            save_dir / "runtime_vs_gurobi_iterations.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 4. Runtime statistics by size
    fig, ax = plt.subplots(figsize=(10, 6))
    runtime_stats = (
        successful_df.groupby("requested_repair_set_size")["repair_runtime_seconds"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    x = runtime_stats["requested_repair_set_size"]
    ax.errorbar(
        x,
        runtime_stats["mean"],
        yerr=runtime_stats["std"],
        marker="o",
        capsize=5,
        label="Mean ± Std",
    )
    ax.plot(x, runtime_stats["median"], marker="s", label="Median")
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Statistics by Size", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        save_dir / "runtime_statistics_by_size.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_accuracy_analysis(df, base_dir):
    """Analyze accuracy changes - individual plots"""
    save_dir = base_dir / "one_shot_analysis" / "accuracy"
    successful_df = df[df["repair_successful"] == True]

    # 1. Test set accuracy change by repair set size
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_by_size = (
        successful_df.groupby("requested_repair_set_size")["accuracy_change"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    x = accuracy_by_size["requested_repair_set_size"]
    ax.errorbar(
        x, accuracy_by_size["mean"], yerr=accuracy_by_size["std"], marker="o", capsize=5
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Test Set Accuracy Change")
    ax.set_title("Test Set Accuracy Change by Size", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_change_by_size.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Accuracy change by repair set type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=successful_df, x="repair_set_type", y="accuracy_change", ax=ax)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Test Set Accuracy Change by Type", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_change_by_type.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Edit set accuracy (should be 1.0 for successful repairs)
    fig, ax = plt.subplots(figsize=(10, 6))
    edit_acc_by_size = (
        successful_df.groupby("requested_repair_set_size")["repaired_accuracy_edit_set"]
        .agg(["mean", "std"])
        .reset_index()
    )

    ax.bar(edit_acc_by_size["requested_repair_set_size"], edit_acc_by_size["mean"])
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Edit Set Accuracy (Repaired)")
    ax.set_title("Edit Set Accuracy After Repair", fontsize=14, fontweight="bold")
    ax.set_ylim(0.9, 1.01)

    plt.tight_layout()
    plt.savefig(
        save_dir / "edit_set_accuracy_after_repair.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Baseline vs repaired accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        successful_df["baseline_accuracy_test_set"],
        successful_df["repaired_accuracy_test_set"],
        alpha=0.6,
    )
    ax.plot([0, 1], [0, 1], "r--", alpha=0.7)  # Perfect correlation line
    ax.set_xlabel("Baseline Test Set Accuracy")
    ax.set_ylabel("Repaired Test Set Accuracy")
    ax.set_title("Baseline vs Repaired Accuracy", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        save_dir / "baseline_vs_repaired_accuracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Accuracy change by target class
    class_data = successful_df[
        successful_df["repair_set_type"] == "class_homogeneous_incorrect"
    ]
    if not class_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        class_accuracy = (
            class_data.groupby("target_class")["accuracy_change"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.bar(class_accuracy["target_class"], class_accuracy["mean"])
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Test Set Accuracy Change")
        ax.set_title("Accuracy Change by Target Class", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            save_dir / "accuracy_change_by_class.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 6. Accuracy vs runtime
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        successful_df["repair_runtime_seconds"],
        successful_df["accuracy_change"],
        alpha=0.6,
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Test Set Accuracy Change")
    ax.set_title("Accuracy Change vs Runtime", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        save_dir / "accuracy_change_vs_runtime.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_iterations_analysis(df, base_dir):
    """Analyze Gurobi iterations - individual plots"""
    save_dir = base_dir / "one_shot_analysis" / "iterations"

    # Filter for successful repairs with iteration data
    iteration_data = df[
        (df["repair_successful"] == True) & (df["gurobi_barrier_iterations"].notna())
    ]

    if iteration_data.empty:
        print("No iteration data available for analysis")
        return

    # 1. Iterations vs repair set size
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        iteration_data["requested_repair_set_size"],
        iteration_data["gurobi_barrier_iterations"],
        alpha=0.6,
    )
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Gurobi Barrier Iterations")
    ax.set_title("Iterations vs Repair Set Size", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "iterations_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Iterations by repair set type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=iteration_data,
        x="repair_set_type",
        y="gurobi_barrier_iterations",
        ax=ax,
    )
    ax.set_title("Iterations by Repair Set Type", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "iterations_by_type.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Iterations vs accuracy change
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        iteration_data["gurobi_barrier_iterations"],
        iteration_data["accuracy_change"],
        alpha=0.6,
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Gurobi Barrier Iterations")
    ax.set_ylabel("Test Set Accuracy Change")
    ax.set_title("Accuracy Change vs Iterations", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        save_dir / "accuracy_change_vs_iterations.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Iterations statistics by size
    fig, ax = plt.subplots(figsize=(10, 6))
    iter_stats = (
        iteration_data.groupby("requested_repair_set_size")["gurobi_barrier_iterations"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    x = iter_stats["requested_repair_set_size"]
    ax.errorbar(
        x,
        iter_stats["mean"],
        yerr=iter_stats["std"],
        marker="o",
        capsize=5,
        label="Mean ± Std",
    )
    ax.plot(x, iter_stats["median"], marker="s", label="Median")
    ax.set_xlabel("Repair Set Size")
    ax.set_ylabel("Gurobi Barrier Iterations")
    ax.set_title("Iteration Statistics by Size", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        save_dir / "iteration_statistics_by_size.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    print("=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)

    total_experiments = len(df)
    successful_experiments = df["repair_successful"].sum()

    print(f"\nOverall Statistics:")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful repairs: {successful_experiments}")
    print(f"Overall success rate: {successful_experiments / total_experiments:.1%}")

    print(f"\nSuccess Rate by Repair Set Size:")
    success_by_size = df.groupby("requested_repair_set_size")["repair_successful"].agg(
        ["sum", "count", "mean"]
    )
    for size, stats in success_by_size.iterrows():
        print(
            f"  Size {size:3d}: {int(stats['sum']):2d}/{int(stats['count']):2d} ({stats['mean']:.1%})"
        )

    print(f"\nSuccess Rate by Repair Set Type:")
    success_by_type = df.groupby("repair_set_type")["repair_successful"].agg(
        ["sum", "count", "mean"]
    )
    for repair_type, stats in success_by_type.iterrows():
        print(
            f"  {repair_type}: {int(stats['sum']):2d}/{int(stats['count']):2d} ({stats['mean']:.1%})"
        )

    successful_df = df[df["repair_successful"] == True]
    if not successful_df.empty:
        print(f"\nRuntime Statistics (successful repairs only):")
        runtime_stats = successful_df["repair_runtime_seconds"].describe()
        print(f"  Mean: {runtime_stats['mean']:.1f}s")
        print(f"  Median: {runtime_stats['50%']:.1f}s")
        print(f"  Min: {runtime_stats['min']:.1f}s")
        print(f"  Max: {runtime_stats['max']:.1f}s")

        print(f"\nAccuracy Change Statistics (successful repairs only):")
        acc_stats = successful_df["accuracy_change"].describe()
        print(f"  Mean change: {acc_stats['mean']:.4f}")
        print(f"  Median change: {acc_stats['50%']:.4f}")
        print(f"  Min change: {acc_stats['min']:.4f}")
        print(f"  Max change: {acc_stats['max']:.4f}")

        iteration_data = successful_df.dropna(subset=["gurobi_barrier_iterations"])
        if not iteration_data.empty:
            print(f"\nGurobi Iterations Statistics:")
            iter_stats = iteration_data["gurobi_barrier_iterations"].describe()
            print(f"  Mean: {iter_stats['mean']:.1f}")
            print(f"  Median: {iter_stats['50%']:.1f}")
            print(f"  Min: {iter_stats['min']:.0f}")
            print(f"  Max: {iter_stats['max']:.0f}")


def main():
    """Main analysis function"""
    # Create organized directory structure
    base_dir = create_directories()

    # Load data
    print("Loading experimental data...")
    df = load_and_prepare_data()

    # Generate summary statistics
    generate_summary_statistics(df)

    # Create visualizations
    print("\nGenerating visualizations...")

    print("  - Success rate analysis...")
    plot_success_rate_analysis(df, base_dir)

    print("  - Runtime analysis...")
    plot_runtime_analysis(df, base_dir)

    print("  - Accuracy analysis...")
    plot_accuracy_analysis(df, base_dir)

    print("  - Iterations analysis...")
    plot_iterations_analysis(df, base_dir)

    print(
        f"\nAnalysis complete! Individual plots saved to {base_dir}/one_shot_analysis/ directories."
    )


if __name__ == "__main__":
    main()
