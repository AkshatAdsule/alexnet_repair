#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


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


def plot_accuracy_over_iterations(df, save_dir="plots"):
    """Create plots showing accuracy over iterations."""
    Path(save_dir).mkdir(exist_ok=True)

    # Get unique batch sizes
    batch_sizes = sorted(df["batch_size"].unique())

    for batch_size in batch_sizes:
        batch_df = df[df["batch_size"] == batch_size]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Test Accuracy
        # Misclassified experiments
        misc_df = batch_df[batch_df["homogeneity"] == "misclassified"]
        if not misc_df.empty:
            ax1.plot(
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
                ax1.plot(
                    class_data["iteration"],
                    class_data["test_accuracy"],
                    "o-",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Class {class_num}",
                )

            # Average across classes
            class_avg = class_df.groupby("iteration")["test_accuracy"].mean()
            ax1.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                linewidth=3,
                label="Class Average",
                color="blue",
            )

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Test Accuracy")
        ax1.set_title(f"Test Accuracy Over Iterations (Batch Size {batch_size})")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Full Repair Set Accuracy
        # Misclassified experiments
        if not misc_df.empty:
            ax2.plot(
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
                ax2.plot(
                    class_data["iteration"],
                    class_data["full_repair_accuracy"],
                    "o-",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Class {class_num}",
                )

            # Average across classes
            class_avg = class_df.groupby("iteration")["full_repair_accuracy"].mean()
            ax2.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                linewidth=3,
                label="Class Average",
                color="blue",
            )

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Full Repair Set Accuracy")
        ax2.set_title(
            f"Full Repair Set Accuracy Over Iterations (Batch Size {batch_size})"
        )
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/accuracy_batch_{batch_size}.png", dpi=300, bbox_inches="tight"
        )
        plt.show()


def plot_batch_size_comparison(df, save_dir="plots"):
    """Create plots comparing different batch sizes."""
    Path(save_dir).mkdir(exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))

    # Plot 1: Misclassified - Test Accuracy
    misc_df = df[df["homogeneity"] == "misclassified"]
    for i, batch_size in enumerate(batch_sizes):
        batch_data = misc_df[misc_df["batch_size"] == batch_size]
        if not batch_data.empty:
            ax1.plot(
                batch_data["iteration"],
                batch_data["test_accuracy"],
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Misclassified: Test Accuracy by Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Misclassified - Full Repair Accuracy
    for i, batch_size in enumerate(batch_sizes):
        batch_data = misc_df[misc_df["batch_size"] == batch_size]
        if not batch_data.empty:
            ax2.plot(
                batch_data["iteration"],
                batch_data["full_repair_accuracy"],
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Full Repair Set Accuracy")
    ax2.set_title("Misclassified: Full Repair Set Accuracy by Batch Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Class Average - Test Accuracy
    class_df = df[df["homogeneity"] == "class_homogeneous_correct"]
    for i, batch_size in enumerate(batch_sizes):
        batch_data = class_df[class_df["batch_size"] == batch_size]
        if not batch_data.empty:
            class_avg = batch_data.groupby("iteration")["test_accuracy"].mean()
            ax3.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Test Accuracy")
    ax3.set_title("Class Average: Test Accuracy by Batch Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Class Average - Full Repair Accuracy
    for i, batch_size in enumerate(batch_sizes):
        batch_data = class_df[class_df["batch_size"] == batch_size]
        if not batch_data.empty:
            class_avg = batch_data.groupby("iteration")["full_repair_accuracy"].mean()
            ax4.plot(
                class_avg.index,
                class_avg.values,
                "o-",
                color=colors[i],
                label=f"Batch Size {batch_size}",
            )

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Full Repair Set Accuracy")
    ax4.set_title("Class Average: Full Repair Set Accuracy by Batch Size")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/batch_size_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_repair_time_analysis(df, save_dir="plots"):
    """Create plots analyzing repair times."""
    Path(save_dir).mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Repair time by batch size
    batch_sizes = sorted(df["batch_size"].unique())
    repair_times_by_batch = []

    for batch_size in batch_sizes:
        batch_data = df[df["batch_size"] == batch_size]
        repair_times_by_batch.append(batch_data["batch_repair_time"].values)

    ax1.boxplot(repair_times_by_batch, labels=[f"Batch {bs}" for bs in batch_sizes])
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Repair Time (seconds)")
    ax1.set_title("Repair Time Distribution by Batch Size")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Repair time over iterations
    for batch_size in batch_sizes:
        batch_data = df[df["batch_size"] == batch_size]
        misc_data = batch_data[batch_data["homogeneity"] == "misclassified"]
        if not misc_data.empty:
            ax2.plot(
                misc_data["iteration"],
                misc_data["batch_repair_time"],
                "o-",
                label=f"Batch Size {batch_size}",
                alpha=0.7,
            )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Repair Time (seconds)")
    ax2.set_title("Repair Time Over Iterations (Misclassified)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/repair_time_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


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
    # Load data
    df = load_and_preprocess_data("results/stochastic_experiment.csv")

    # Generate summary statistics
    generate_summary_stats(df)

    # Create plots
    print("\nGenerating plots...")
    plot_accuracy_over_iterations(df)
    plot_batch_size_comparison(df)
    plot_repair_time_analysis(df)

    print("Analysis complete! Check the 'plots' directory for generated figures.")


if __name__ == "__main__":
    main()
