#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_directories():
    """Create organized directory structure for stochastic results"""
    base_dir = Path("results")
    subdirs = [
        "stochastic_analysis",
        "stochastic_analysis/batch_iterations",
        "stochastic_analysis/comparisons",
        "stochastic_analysis/runtime",
    ]

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir


def load_and_preprocess_data(csv_path):
    """Load and preprocess the experimental data."""
    df = pd.read_csv(csv_path)

    # Filter out rows with errors
    df = df[df["error_message"].isna() | (df["error_message"] == "")]

    # Extract class number for class_homogeneous experiments
    df["class_number"] = df["experiment_id"].str.extract(r"_cls(\d+)$").astype("Int64")

    # Create a more readable experiment type
    df["exp_type"] = df["homogeneity"].apply(
        lambda x: "Misclassified" if x == "misclassified" else "Class Homogeneous"
    )

    return df


def plot_accuracy_over_iterations(df, base_dir):
    """Create individual plots showing accuracy over iterations for each batch size."""
    save_dir = base_dir / "stochastic_analysis" / "batch_iterations"

    # Get unique batch sizes
    batch_sizes = sorted(df["batch_size"].unique())

    for batch_size in batch_sizes:
        batch_df = df[df["batch_size"] == batch_size]

        # Plot 1: Test Accuracy for this batch size
        fig, ax = plt.subplots(figsize=(12, 8))

        # Misclassified experiments
        misc_df = batch_df[batch_df["homogeneity"] == "misclassified"]
        if not misc_df.empty:
            ax.plot(
                misc_df["iteration"],
                misc_df["test_accuracy"],
                "o-",
                linewidth=3,
                label="Misclassified",
                color="red",
            )

        # Individual class experiments (lighter opacity)
        class_df = batch_df[batch_df["homogeneity"] == "class_homogeneous_correct"]
        if not class_df.empty:
            for class_num in sorted(class_df["class_number"].unique()):
                class_data = class_df[class_df["class_number"] == class_num]
                ax.plot(
                    class_data["iteration"],
                    class_data["test_accuracy"],
                    "o-",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Class {class_num}",
                )

            # Average across classes
            class_avg = class_df.groupby("iteration")["test_accuracy"].mean()
            ax.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                linewidth=3,
                label="Class Average",
                color="blue",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(
            f"Test Accuracy Over Iterations (Batch Size {batch_size})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / f"test_accuracy_batch_{batch_size}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 2: Full Repair Set Accuracy for this batch size
        fig, ax = plt.subplots(figsize=(12, 8))

        # Misclassified experiments
        if not misc_df.empty:
            ax.plot(
                misc_df["iteration"],
                misc_df["full_repair_accuracy"],
                "o-",
                linewidth=3,
                label="Misclassified",
                color="red",
            )

        # Individual class experiments
        if not class_df.empty:
            for class_num in sorted(class_df["class_number"].unique()):
                class_data = class_df[class_df["class_number"] == class_num]
                ax.plot(
                    class_data["iteration"],
                    class_data["full_repair_accuracy"],
                    "o-",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Class {class_num}",
                )

            # Average across classes
            class_avg = class_df.groupby("iteration")["full_repair_accuracy"].mean()
            ax.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                linewidth=3,
                label="Class Average",
                color="blue",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Full Repair Set Accuracy")
        ax.set_title(
            f"Full Repair Set Accuracy Over Iterations (Batch Size {batch_size})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / f"repair_accuracy_batch_{batch_size}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_batch_size_comparisons(df, base_dir):
    """Create individual comparison plots for different aspects of batch size analysis."""
    save_dir = base_dir / "stochastic_analysis" / "comparisons"

    batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))

    # Plot 1: Misclassified - Test Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    misc_df = df[df["homogeneity"] == "misclassified"]
    for i, batch_size in enumerate(batch_sizes):
        batch_data = misc_df[misc_df["batch_size"] == batch_size]
        if not batch_data.empty:
            ax.plot(
                batch_data["iteration"],
                batch_data["test_accuracy"],
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(
        "Misclassified: Test Accuracy by Batch Size", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "misclassified_test_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Misclassified - Full Repair Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, batch_size in enumerate(batch_sizes):
        batch_data = misc_df[misc_df["batch_size"] == batch_size]
        if not batch_data.empty:
            ax.plot(
                batch_data["iteration"],
                batch_data["full_repair_accuracy"],
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Full Repair Set Accuracy")
    ax.set_title(
        "Misclassified: Full Repair Set Accuracy by Batch Size",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "misclassified_repair_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Class Average - Test Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    class_df = df[df["homogeneity"] == "class_homogeneous_correct"]
    for i, batch_size in enumerate(batch_sizes):
        batch_data = class_df[class_df["batch_size"] == batch_size]
        if not batch_data.empty:
            class_avg = batch_data.groupby("iteration")["test_accuracy"].mean()
            ax.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(
        "Class Average: Test Accuracy by Batch Size", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "class_average_test_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 4: Class Average - Full Repair Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, batch_size in enumerate(batch_sizes):
        batch_data = class_df[class_df["batch_size"] == batch_size]
        if not batch_data.empty:
            class_avg = batch_data.groupby("iteration")["full_repair_accuracy"].mean()
            ax.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Full Repair Set Accuracy")
    ax.set_title(
        "Class Average: Full Repair Set Accuracy by Batch Size",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "class_average_repair_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_repair_time_analysis(df, base_dir):
    """Create individual plots analyzing repair times."""
    save_dir = base_dir / "stochastic_analysis" / "runtime"

    # Plot 1: Repair time by batch size
    fig, ax = plt.subplots(figsize=(12, 8))
    batch_sizes = sorted(df["batch_size"].unique())
    repair_times_by_batch = []

    for batch_size in batch_sizes:
        batch_data = df[df["batch_size"] == batch_size]
        repair_times_by_batch.append(batch_data["batch_repair_time"].values)

    ax.boxplot(repair_times_by_batch, labels=[f"Batch {bs}" for bs in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Repair Time (seconds)")
    ax.set_title(
        "Repair Time Distribution by Batch Size", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "repair_time_distribution_by_batch.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Repair time over iterations
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))

    for i, batch_size in enumerate(batch_sizes):
        batch_data = df[df["batch_size"] == batch_size]
        misc_data = batch_data[batch_data["homogeneity"] == "misclassified"]
        if not misc_data.empty:
            ax.plot(
                misc_data["iteration"],
                misc_data["batch_repair_time"],
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
                alpha=0.7,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Repair Time (seconds)")
    ax.set_title(
        "Repair Time Over Iterations (Misclassified)", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "repair_time_over_iterations.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Repair time statistics by batch size
    fig, ax = plt.subplots(figsize=(12, 8))

    time_stats = []
    labels = []

    for batch_size in batch_sizes:
        batch_data = df[df["batch_size"] == batch_size]
        mean_time = batch_data["batch_repair_time"].mean()
        std_time = batch_data["batch_repair_time"].std()
        time_stats.append((mean_time, std_time))
        labels.append(f"Batch {batch_size}")

    means = [stats[0] for stats in time_stats]
    stds = [stats[1] for stats in time_stats]

    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Repair Time (seconds)")
    ax.set_title("Average Repair Time by Batch Size", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "average_repair_time_by_batch.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_summary_stats(df):
    """Generate summary statistics."""
    print("=" * 60)
    print("STOCHASTIC REPAIR EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"Total experiments: {df['experiment_id'].nunique()}")
    print(f"Batch sizes tested: {sorted(df['batch_size'].unique())}")
    print(f"Max iterations per experiment: {df['iteration'].max()}")
    print(f"Homogeneity types: {df['homogeneity'].unique().tolist()}")

    print("\n" + "=" * 40)
    print("FINAL ACCURACIES (Last Iteration)")
    print("=" * 40)

    # Get final iteration for each experiment
    final_iter_df = df.loc[df.groupby("experiment_id")["iteration"].idxmax()]

    # Misclassified results
    misc_final = final_iter_df[final_iter_df["homogeneity"] == "misclassified"]
    if not misc_final.empty:
        print("Misclassified Experiments:")
        for _, row in misc_final.iterrows():
            print(
                f"  Batch {row['batch_size']}: Test={row['test_accuracy']:.3f}, Repair={row['full_repair_accuracy']:.3f}"
            )

    # Class homogeneous averages
    class_final = final_iter_df[
        final_iter_df["homogeneity"] == "class_homogeneous_correct"
    ]
    if not class_final.empty:
        print("\nClass Homogeneous Experiments (Averages):")
        for batch_size in sorted(class_final["batch_size"].unique()):
            batch_data = class_final[class_final["batch_size"] == batch_size]
            avg_test = batch_data["test_accuracy"].mean()
            avg_repair = batch_data["full_repair_accuracy"].mean()
            print(f"  Batch {batch_size}: Test={avg_test:.3f}, Repair={avg_repair:.3f}")

    print("\n" + "=" * 40)
    print("REPAIR TIME STATISTICS")
    print("=" * 40)

    for batch_size in sorted(df["batch_size"].unique()):
        batch_data = df[df["batch_size"] == batch_size]
        mean_time = batch_data["batch_repair_time"].mean()
        std_time = batch_data["batch_repair_time"].std()
        print(f"Batch {batch_size}: {mean_time:.2f} ± {std_time:.2f} seconds")


def main():
    # Create organized directory structure
    base_dir = create_directories()

    # Load data
    df = load_and_preprocess_data("results/stochastic_experiment.csv")

    # Generate summary statistics
    generate_summary_stats(df)

    # Create plots
    print("\nGenerating individual plots...")

    print("  - Accuracy over iterations by batch size...")
    plot_accuracy_over_iterations(df, base_dir)

    print("  - Batch size comparisons...")
    plot_batch_size_comparisons(df, base_dir)

    print("  - Repair time analysis...")
    plot_repair_time_analysis(df, base_dir)

    print(
        f"\nAnalysis complete! Individual plots saved to {base_dir}/stochastic_analysis/ directories."
    )


if __name__ == "__main__":
    main()
