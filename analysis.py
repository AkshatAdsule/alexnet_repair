import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


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


def plot_success_rate_analysis(df):
    """Analyze repair success rates"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Repair Success Rate Analysis", fontsize=16, fontweight="bold")

    # Success rate by repair set size
    success_by_size = (
        df.groupby("requested_repair_set_size")["success_rate"]
        .agg(["mean", "count"])
        .reset_index()
    )
    axes[0, 0].bar(
        success_by_size["requested_repair_set_size"], success_by_size["mean"]
    )
    axes[0, 0].set_title("Success Rate by Repair Set Size")
    axes[0, 0].set_xlabel("Repair Set Size")
    axes[0, 0].set_ylabel("Success Rate")
    axes[0, 0].set_ylim(0, 1)

    # Add count labels on bars
    for i, (size, rate, count) in enumerate(
        zip(
            success_by_size["requested_repair_set_size"],
            success_by_size["mean"],
            success_by_size["count"],
        )
    ):
        axes[0, 0].text(size, rate + 0.02, f"n={count}", ha="center", va="bottom")

    # Success rate by repair set type
    success_by_type = (
        df.groupby("repair_set_type")["success_rate"]
        .agg(["mean", "count"])
        .reset_index()
    )
    axes[0, 1].bar(success_by_type["repair_set_type"], success_by_type["mean"])
    axes[0, 1].set_title("Success Rate by Repair Set Type")
    axes[0, 1].set_xlabel("Repair Set Type")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Success rate by target class (for class_homogeneous_incorrect only)
    class_data = df[df["repair_set_type"] == "class_homogeneous_incorrect"].copy()
    if not class_data.empty:
        success_by_class = (
            class_data.groupby("target_class")["success_rate"]
            .agg(["mean", "count"])
            .reset_index()
        )
        axes[1, 0].bar(success_by_class["target_class"], success_by_class["mean"])
        axes[1, 0].set_title("Success Rate by Target Class")
        axes[1, 0].set_xlabel("Target Class")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].set_ylim(0, 1)

    # Success rate heatmap by size and type
    pivot_data = df.pivot_table(
        values="success_rate",
        index="repair_set_type",
        columns="requested_repair_set_size",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[1, 1]
    )
    axes[1, 1].set_title("Success Rate Heatmap")

    plt.tight_layout()
    plt.savefig("results/success_rate_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_runtime_analysis(df):
    """Analyze repair runtime characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Runtime Analysis", fontsize=16, fontweight="bold")

    # Runtime vs repair set size
    successful_df = df[df["repair_successful"] == True]

    axes[0, 0].scatter(
        successful_df["requested_repair_set_size"],
        successful_df["repair_runtime_seconds"],
        alpha=0.6,
    )
    axes[0, 0].set_xlabel("Repair Set Size")
    axes[0, 0].set_ylabel("Runtime (seconds)")
    axes[0, 0].set_title("Runtime vs Repair Set Size")

    # Add trend line
    z = np.polyfit(
        successful_df["requested_repair_set_size"],
        successful_df["repair_runtime_seconds"],
        1,
    )
    p = np.poly1d(z)
    axes[0, 0].plot(
        successful_df["requested_repair_set_size"],
        p(successful_df["requested_repair_set_size"]),
        "r--",
        alpha=0.8,
    )

    # Runtime by repair set type
    sns.boxplot(
        data=successful_df,
        x="repair_set_type",
        y="repair_runtime_seconds",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Runtime Distribution by Repair Set Type")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Gurobi iterations vs runtime
    iteration_data = successful_df.dropna(subset=["gurobi_barrier_iterations"])
    axes[1, 0].scatter(
        iteration_data["gurobi_barrier_iterations"],
        iteration_data["repair_runtime_seconds"],
        alpha=0.6,
    )
    axes[1, 0].set_xlabel("Gurobi Barrier Iterations")
    axes[1, 0].set_ylabel("Runtime (seconds)")
    axes[1, 0].set_title("Runtime vs Gurobi Iterations")

    # Runtime statistics by size
    runtime_stats = (
        successful_df.groupby("requested_repair_set_size")["repair_runtime_seconds"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    x = runtime_stats["requested_repair_set_size"]
    axes[1, 1].errorbar(
        x,
        runtime_stats["mean"],
        yerr=runtime_stats["std"],
        marker="o",
        capsize=5,
        label="Mean ± Std",
    )
    axes[1, 1].plot(x, runtime_stats["median"], marker="s", label="Median")
    axes[1, 1].set_xlabel("Repair Set Size")
    axes[1, 1].set_ylabel("Runtime (seconds)")
    axes[1, 1].set_title("Runtime Statistics by Size")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results/runtime_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy_analysis(df):
    """Analyze accuracy changes"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Accuracy Analysis", fontsize=16, fontweight="bold")

    successful_df = df[df["repair_successful"] == True]

    # Test set accuracy change by repair set size
    accuracy_by_size = (
        successful_df.groupby("requested_repair_set_size")["accuracy_change"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    x = accuracy_by_size["requested_repair_set_size"]
    axes[0, 0].errorbar(
        x, accuracy_by_size["mean"], yerr=accuracy_by_size["std"], marker="o", capsize=5
    )
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_xlabel("Repair Set Size")
    axes[0, 0].set_ylabel("Test Set Accuracy Change")
    axes[0, 0].set_title("Test Set Accuracy Change by Size")

    # Accuracy change by repair set type
    sns.boxplot(
        data=successful_df, x="repair_set_type", y="accuracy_change", ax=axes[0, 1]
    )
    axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Test Set Accuracy Change by Type")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Edit set accuracy (should be 1.0 for successful repairs)
    edit_acc_by_size = (
        successful_df.groupby("requested_repair_set_size")["repaired_accuracy_edit_set"]
        .agg(["mean", "std"])
        .reset_index()
    )

    axes[0, 2].bar(
        edit_acc_by_size["requested_repair_set_size"], edit_acc_by_size["mean"]
    )
    axes[0, 2].set_xlabel("Repair Set Size")
    axes[0, 2].set_ylabel("Edit Set Accuracy (Repaired)")
    axes[0, 2].set_title("Edit Set Accuracy After Repair")
    axes[0, 2].set_ylim(0.9, 1.01)

    # Baseline vs repaired accuracy scatter
    axes[1, 0].scatter(
        successful_df["baseline_accuracy_test_set"],
        successful_df["repaired_accuracy_test_set"],
        alpha=0.6,
    )
    axes[1, 0].plot([0, 1], [0, 1], "r--", alpha=0.7)  # Perfect correlation line
    axes[1, 0].set_xlabel("Baseline Test Set Accuracy")
    axes[1, 0].set_ylabel("Repaired Test Set Accuracy")
    axes[1, 0].set_title("Baseline vs Repaired Accuracy")

    # Accuracy change by target class
    class_data = successful_df[
        successful_df["repair_set_type"] == "class_homogeneous_incorrect"
    ]
    if not class_data.empty:
        class_accuracy = (
            class_data.groupby("target_class")["accuracy_change"]
            .agg(["mean", "std"])
            .reset_index()
        )

        axes[1, 1].bar(class_accuracy["target_class"], class_accuracy["mean"])
        axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
        axes[1, 1].set_xlabel("Target Class")
        axes[1, 1].set_ylabel("Test Set Accuracy Change")
        axes[1, 1].set_title("Accuracy Change by Target Class")

    # Accuracy vs runtime
    axes[1, 2].scatter(
        successful_df["repair_runtime_seconds"],
        successful_df["accuracy_change"],
        alpha=0.6,
    )
    axes[1, 2].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 2].set_xlabel("Runtime (seconds)")
    axes[1, 2].set_ylabel("Test Set Accuracy Change")
    axes[1, 2].set_title("Accuracy Change vs Runtime")

    plt.tight_layout()
    plt.savefig("results/accuracy_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_iterations_analysis(df):
    """Analyze Gurobi iterations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Gurobi Iterations Analysis", fontsize=16, fontweight="bold")

    # Filter for successful repairs with iteration data
    iteration_data = df[
        (df["repair_successful"] == True) & (df["gurobi_barrier_iterations"].notna())
    ]

    # Iterations vs repair set size
    axes[0, 0].scatter(
        iteration_data["requested_repair_set_size"],
        iteration_data["gurobi_barrier_iterations"],
        alpha=0.6,
    )
    axes[0, 0].set_xlabel("Repair Set Size")
    axes[0, 0].set_ylabel("Gurobi Barrier Iterations")
    axes[0, 0].set_title("Iterations vs Repair Set Size")

    # Iterations by repair set type
    sns.boxplot(
        data=iteration_data,
        x="repair_set_type",
        y="gurobi_barrier_iterations",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Iterations by Repair Set Type")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Iterations vs accuracy change
    axes[1, 0].scatter(
        iteration_data["gurobi_barrier_iterations"],
        iteration_data["accuracy_change"],
        alpha=0.6,
    )
    axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].set_xlabel("Gurobi Barrier Iterations")
    axes[1, 0].set_ylabel("Test Set Accuracy Change")
    axes[1, 0].set_title("Accuracy Change vs Iterations")

    # Iterations statistics by size
    iter_stats = (
        iteration_data.groupby("requested_repair_set_size")["gurobi_barrier_iterations"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    x = iter_stats["requested_repair_set_size"]
    axes[1, 1].errorbar(
        x,
        iter_stats["mean"],
        yerr=iter_stats["std"],
        marker="o",
        capsize=5,
        label="Mean ± Std",
    )
    axes[1, 1].plot(x, iter_stats["median"], marker="s", label="Median")
    axes[1, 1].set_xlabel("Repair Set Size")
    axes[1, 1].set_ylabel("Gurobi Barrier Iterations")
    axes[1, 1].set_title("Iteration Statistics by Size")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results/iterations_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


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
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Load data
    print("Loading experimental data...")
    df = load_and_prepare_data()

    # Generate summary statistics
    generate_summary_statistics(df)

    # Create visualizations
    print("\nGenerating visualizations...")

    print("  - Success rate analysis...")
    plot_success_rate_analysis(df)

    print("  - Runtime analysis...")
    plot_runtime_analysis(df)

    print("  - Accuracy analysis...")
    plot_accuracy_analysis(df)

    print("  - Iterations analysis...")
    plot_iterations_analysis(df)

    print("\nAnalysis complete! Plots saved to results/ directory.")


if __name__ == "__main__":
    main()
