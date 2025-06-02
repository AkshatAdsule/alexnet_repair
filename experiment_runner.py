#!/usr/bin/env python3
"""
Experiment Runner - Automated Model Repair Experiments

This script automates the execution of multiple model repair experiments with varying
configurations to study the impact of repair-set size and type on the repair process.
"""

import argparse
import time
import os
import csv
import itertools
import subprocess
import re
import sys
import json
import uuid
from datetime import datetime
import torch
import torch.nn.functional as F

# Import existing evaluation functions
from eval import eval_accuracy, load_alexnet_with_weights
from helpers.cifar import get_cifar10_dataloader

# CIFAR-10 classes for reference
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# All metrics fields for CSV output
FIELDNAMES = [
    # Experiment metadata
    "experiment_id",
    "timestamp",
    # Configuration parameters
    "repair_set_type",
    "repair_set_focus",
    "requested_repair_set_size",
    "edit_set_sizing_strategy",
    "actual_repair_set_size",
    "edit_set_path",
    "target_class",
    "param_bound",
    "margin",
    # Baseline model performance
    "baseline_model_path",
    "baseline_accuracy_test_set",
    "baseline_accuracy_edit_set",
    # Repair process
    "repair_runtime_seconds",
    "gurobi_barrier_iterations",
    "repair_successful",
    # Repaired model performance (if successful)
    "repaired_model_path",
    "repaired_accuracy_test_set",
    "repaired_accuracy_edit_set",
    "repaired_accuracy_drawdown_set",
    # Error handling
    "error_message",
]


def parse_gurobi_log(log_output):
    """
    Parse Gurobi solver log to extract barrier iterations and other stats.

    Args:
        log_output (str): Complete stdout/stderr from repair process

    Returns:
        dict: Parsed metrics including gurobi_barrier_iterations
    """
    iterations = None

    # Search for barrier iterations
    # Example: "Barrier solved model in 24 iterations and 1.55 seconds"
    match = re.search(r"Barrier solved model in (\d+) iterations", log_output)
    if match:
        iterations = int(match.group(1))

    return {"gurobi_barrier_iterations": iterations}


def generate_edit_set(config, base_model_path):
    """
    Generate edit set using editset_generator.py based on configuration.

    Args:
        config (dict): Experiment configuration
        base_model_path (str): Path to base model

    Returns:
        tuple: (edit_set_path, actual_size) or (None, None) if failed
    """
    try:
        repair_set_type = config["repair_set_type"]
        target_size = config["requested_repair_set_size"]
        sizing_strategy = config["edit_set_sizing_strategy"]

        # For now, we'll use the existing editset_generator.py functionality
        # and extend it later to support size limits and strategies

        if repair_set_type == "misclassified":
            # Use existing misclassified generator
            cmd = [
                sys.executable,
                "editset_generator.py",
                "misclassified",
                "--max-samples",
                str(target_size) if target_size > 0 else str(-1),
                "--force",
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd()
            )

            if result.returncode == 0:
                edit_set_path = "data/edit_sets/misclassified_edit_dataset.pt"
                # Get actual size by loading the dataset
                data = torch.load(edit_set_path, map_location="cpu")
                actual_size = len(data["images"])
                return edit_set_path, actual_size
            else:
                print(f"Error generating misclassified edit set: {result.stderr}")
                return None, None

        elif repair_set_type.startswith("class_homogeneous"):
            # Extract target class from config
            target_class = config.get("target_class")
            if target_class is None:
                print("Error: target_class required for class_homogeneous types")
                return None, None

            cmd = [
                sys.executable,
                "editset_generator.py",
                "by-class",
                "--target-class",
                str(target_class),
                "--max-samples",
                str(target_size) if target_size > 0 else str(-1),
                "--force",
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd()
            )

            if result.returncode == 0:
                class_name = CIFAR10_CLASSES[target_class]
                edit_set_path = (
                    f"data/edit_sets/class_{target_class}_{class_name}_edit_dataset.pt"
                )
                # Get actual size by loading the dataset
                data = torch.load(edit_set_path, map_location="cpu")
                actual_size = len(data["images"])
                return edit_set_path, actual_size
            else:
                print(f"Error generating by-class edit set: {result.stderr}")
                return None, None
        else:
            print(f"Unsupported repair_set_type: {repair_set_type}")
            return None, None

    except Exception as e:
        print(f"Exception in generate_edit_set: {e}")
        return None, None


def run_repair_process(edit_set_path, param_bound, margin, output_model_path):
    """
    Run the repair process using run_repair.py.

    Args:
        edit_set_path (str): Path to edit set
        param_bound (float): Parameter bound for repair
        margin (float): Classification margin
        output_model_path (str): Output path for repaired model

    Returns:
        tuple: (success_status, repair_time, gurobi_stats)
    """
    try:
        # Determine the repair command based on edit set path
        if "misclassified" in edit_set_path:
            cmd = [
                sys.executable,
                "run_repair.py",
                "misclassified",
                "--param-bound",
                str(param_bound),
                "--margin",
                str(margin),
                "--output",
                output_model_path,
            ]
        elif "class_" in edit_set_path:
            # Extract class number from path
            import re

            match = re.search(r"class_(\d+)_", edit_set_path)
            if match:
                target_class = int(match.group(1))
                cmd = [
                    sys.executable,
                    "run_repair.py",
                    "by-class",
                    "--target-class",
                    str(target_class),
                    "--param-bound",
                    str(param_bound),
                    "--margin",
                    str(margin),
                    "--output",
                    output_model_path,
                ]
            else:
                print(f"Could not extract class from path: {edit_set_path}")
                return False, 0, {}
        else:
            print(f"Unknown edit set type for path: {edit_set_path}")
            return False, 0, {}

        print(f"Running repair: {' '.join(cmd)}")
        start_time = time.time()

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        end_time = time.time()
        repair_time = end_time - start_time

        # Parse Gurobi log from output
        combined_output = result.stdout + result.stderr
        gurobi_stats = parse_gurobi_log(combined_output)

        success = result.returncode == 0

        if not success:
            print(f"Repair failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")

        return success, repair_time, gurobi_stats

    except Exception as e:
        print(f"Exception in run_repair_process: {e}")
        return False, 0, {}


def create_drawdown_dataloader(edit_set_path, batch_size=64, num_workers=2):
    """
    Create a dataloader for the "drawdown set" (test set excluding edit set samples).

    Args:
        edit_set_path (str): Path to edit set
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of workers

    Returns:
        torch.utils.data.DataLoader: Drawdown dataloader
    """
    try:
        # Load edit set to get image indices
        edit_data = torch.load(edit_set_path, map_location="cpu")
        edit_metadata = edit_data.get("metadata", [])

        # Extract image indices from metadata
        edit_indices = set()
        for meta in edit_metadata:
            if "image_idx" in meta:
                edit_indices.add(meta["image_idx"])

        # Get full test dataloader
        full_dataloader = get_cifar10_dataloader(
            batch_size=batch_size, train=False, num_workers=num_workers
        )

        # For simplicity, we'll create a filtered dataset
        # This is a basic implementation - in practice, you might want a more efficient approach
        filtered_images = []
        filtered_labels = []

        current_idx = 0
        for images, labels in full_dataloader:
            for i in range(images.size(0)):
                if current_idx not in edit_indices:
                    filtered_images.append(images[i])
                    filtered_labels.append(labels[i])
                current_idx += 1

        if not filtered_images:
            print("Warning: No images left for drawdown set")
            return None

        # Create new dataset and dataloader
        filtered_images = torch.stack(filtered_images)
        filtered_labels = torch.stack(filtered_labels)

        from torch.utils.data import TensorDataset, DataLoader

        drawdown_dataset = TensorDataset(filtered_images, filtered_labels)
        drawdown_dataloader = DataLoader(
            drawdown_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return drawdown_dataloader

    except Exception as e:
        print(f"Error creating drawdown dataloader: {e}")
        return None


def evaluate_model_on_edit_set(model, edit_set_path, device):
    """
    Evaluate model accuracy on a specific edit set.

    Args:
        model: PyTorch model
        edit_set_path (str): Path to edit set
        device: PyTorch device

    Returns:
        float: Accuracy on edit set
    """
    try:
        edit_data = torch.load(edit_set_path, map_location=device)
        edit_images = edit_data["images"].to(device)
        edit_labels = edit_data["labels"].to(device)

        model.eval()
        correct = 0
        total = len(edit_labels)

        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, total, batch_size):
                batch_images = edit_images[i : i + batch_size]
                batch_labels = edit_labels[i : i + batch_size]

                outputs = model(batch_images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    except Exception as e:
        print(f"Error evaluating model on edit set: {e}")
        return 0.0


def run_single_experiment(exp_config, results_writer):
    """
    Run a single experiment with the given configuration.

    Args:
        exp_config (dict): Experiment configuration
        results_writer: CSV writer for results
    """
    print(f"INFO: Starting experiment ID: {exp_config['experiment_id']}")
    print(f"      Config: {exp_config}")

    # Initialize result data with config
    current_run_data = {**exp_config}
    current_run_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_run_data["baseline_model_path"] = "artifacts/alexnet_base.pth"
    current_run_data["error_message"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 1. Setup: Generate edit set
        print("  Step 1: Generating edit set...")
        edit_set_path, actual_size = generate_edit_set(
            exp_config, current_run_data["baseline_model_path"]
        )

        if edit_set_path is None:
            current_run_data["error_message"] = "Failed to generate edit set"
            current_run_data["repair_successful"] = False
            results_writer.writerow(
                {k: current_run_data.get(k, "") for k in FIELDNAMES}
            )
            return

        current_run_data["edit_set_path"] = edit_set_path
        current_run_data["actual_repair_set_size"] = actual_size

        # 2. Pre-Repair Evaluation
        print("  Step 2: Pre-repair evaluation...")
        try:
            baseline_model = load_alexnet_with_weights(
                current_run_data["baseline_model_path"]
            )
            baseline_model.to(device)

            # Evaluate on full test set
            test_dataloader = get_cifar10_dataloader(
                batch_size=64, train=False, num_workers=2
            )
            baseline_accuracy_test = eval_accuracy(
                baseline_model, test_dataloader, device
            )
            current_run_data["baseline_accuracy_test_set"] = baseline_accuracy_test

            # Evaluate on edit set
            baseline_accuracy_edit = evaluate_model_on_edit_set(
                baseline_model, edit_set_path, device
            )
            current_run_data["baseline_accuracy_edit_set"] = baseline_accuracy_edit

            print(f"    Baseline test accuracy: {baseline_accuracy_test:.4f}")
            print(f"    Baseline edit set accuracy: {baseline_accuracy_edit:.4f}")

        except Exception as e:
            current_run_data["error_message"] = (
                f"Pre-repair evaluation failed: {str(e)}"
            )
            current_run_data["repair_successful"] = False
            results_writer.writerow(
                {k: current_run_data.get(k, "") for k in FIELDNAMES}
            )
            return

        # 3. Repair Process
        print("  Step 3: Running repair...")

        # Create unique output path for this experiment
        os.makedirs("artifacts/experiments", exist_ok=True)
        output_model_path = (
            f"artifacts/experiments/{exp_config['experiment_id']}_repaired.pth"
        )
        current_run_data["repaired_model_path"] = output_model_path

        success, repair_time, gurobi_stats = run_repair_process(
            edit_set_path,
            exp_config["param_bound"],
            exp_config["margin"],
            output_model_path,
        )

        current_run_data["repair_successful"] = success
        current_run_data["repair_runtime_seconds"] = repair_time
        current_run_data["gurobi_barrier_iterations"] = gurobi_stats.get(
            "gurobi_barrier_iterations"
        )

        print(f"    Repair successful: {success}")
        print(f"    Repair time: {repair_time:.2f} seconds")
        print(
            f"    Gurobi iterations: {gurobi_stats.get('gurobi_barrier_iterations', 'N/A')}"
        )

        # 4. Post-Repair Evaluation (if repair was successful)
        if success and os.path.exists(output_model_path):
            print("  Step 4: Post-repair evaluation...")
            try:
                repaired_model = load_alexnet_with_weights(output_model_path)
                repaired_model.to(device)

                # Evaluate on full test set
                repaired_accuracy_test = eval_accuracy(
                    repaired_model, test_dataloader, device
                )
                current_run_data["repaired_accuracy_test_set"] = repaired_accuracy_test

                # Evaluate on edit set
                repaired_accuracy_edit = evaluate_model_on_edit_set(
                    repaired_model, edit_set_path, device
                )
                current_run_data["repaired_accuracy_edit_set"] = repaired_accuracy_edit

                # Evaluate on drawdown set
                drawdown_dataloader = create_drawdown_dataloader(edit_set_path)
                if drawdown_dataloader is not None:
                    repaired_accuracy_drawdown = eval_accuracy(
                        repaired_model, drawdown_dataloader, device
                    )
                    current_run_data["repaired_accuracy_drawdown_set"] = (
                        repaired_accuracy_drawdown
                    )
                    print(
                        f"    Repaired drawdown accuracy: {repaired_accuracy_drawdown:.4f}"
                    )

                print(f"    Repaired test accuracy: {repaired_accuracy_test:.4f}")
                print(f"    Repaired edit set accuracy: {repaired_accuracy_edit:.4f}")

            except Exception as e:
                current_run_data["error_message"] = (
                    f"Post-repair evaluation failed: {str(e)}"
                )
                print(f"    Post-repair evaluation failed: {e}")

    except Exception as e:
        current_run_data["error_message"] = f"Experiment failed: {str(e)}"
        current_run_data["repair_successful"] = False
        print(f"    Experiment failed: {e}")

    # 5. Log results
    results_writer.writerow({k: current_run_data.get(k, "") for k in FIELDNAMES})

    success_status = current_run_data.get("repair_successful", False)
    print(
        f"INFO: Finished experiment ID: {exp_config['experiment_id']}. Success: {success_status}"
    )
    print()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run model repair experiments with varying configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic experiment with misclassified edit sets
  uv run python experiment_runner.py \\
    --repair-set-sizes 10 50 \\
    --repair-set-types misclassified \\
    --output-file results/test_experiment.csv

  # Full experiment with class-specific sets
  uv run python experiment_runner.py \\
    --repair-set-sizes 25 50 100 \\
    --repair-set-types misclassified class_homogeneous_incorrect \\
    --target-classes 0 3 5 \\
    --output-file results/full_experiment.csv
        """,
    )

    parser.add_argument(
        "--repair-set-sizes",
        nargs="+",
        type=int,
        required=True,
        help="List of repair set sizes to test (e.g., 10 50 100)",
    )

    parser.add_argument(
        "--repair-set-types",
        nargs="+",
        type=str,
        required=True,
        choices=[
            "misclassified",
            "class_homogeneous_correct",
            "class_homogeneous_incorrect",
        ],
        help="Types of repair sets to generate",
    )

    parser.add_argument(
        "--target-classes",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Target classes for class_homogeneous types (default: all classes 0-9)",
    )

    parser.add_argument(
        "--classification-focus",
        nargs="*",
        type=str,
        default=["incorrect_only"],
        choices=["incorrect_only", "correct_and_incorrect"],
        help="Classification focus for repair set generation (not fully implemented yet)",
    )

    parser.add_argument(
        "--sizing-strategy",
        nargs="*",
        type=str,
        default=["flexible"],
        choices=["strict", "flexible"],
        help="Edit set sizing strategy (not fully implemented yet)",
    )

    parser.add_argument(
        "--param-bound",
        type=float,
        default=5.0,
        help="Parameter change bound for repair (default: 5.0)",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Classification margin for repair (default: 2.0)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="results/experiment_data.csv",
        help="Output CSV file for results (default: results/experiment_data.csv)",
    )

    args = parser.parse_args()

    # Validate target classes
    for cls in args.target_classes:
        if not (0 <= cls <= 9):
            print(f"Error: Invalid target class {cls}. Must be 0-9 for CIFAR-10.")
            sys.exit(1)

    # Generate experiment configurations
    print("Generating experiment configurations...")

    experiment_configs = []
    experiment_counter = 0

    for size in args.repair_set_sizes:
        for repair_type in args.repair_set_types:
            for focus in args.classification_focus:
                for strategy in args.sizing_strategy:
                    if repair_type.startswith("class_homogeneous"):
                        # Create separate experiments for each target class
                        for target_class in args.target_classes:
                            experiment_counter += 1
                            config = {
                                "experiment_id": f"exp_{experiment_counter:04d}",
                                "repair_set_type": repair_type,
                                "repair_set_focus": focus,
                                "requested_repair_set_size": size,
                                "edit_set_sizing_strategy": strategy,
                                "target_class": target_class,
                                "param_bound": args.param_bound,
                                "margin": args.margin,
                            }
                            experiment_configs.append(config)
                    else:
                        # No target class needed for misclassified
                        experiment_counter += 1
                        config = {
                            "experiment_id": f"exp_{experiment_counter:04d}",
                            "repair_set_type": repair_type,
                            "repair_set_focus": focus,
                            "requested_repair_set_size": size,
                            "edit_set_sizing_strategy": strategy,
                            "target_class": None,
                            "param_bound": args.param_bound,
                            "margin": args.margin,
                        }
                        experiment_configs.append(config)

    print(f"Generated {len(experiment_configs)} experiment configurations.")
    print(f"Results will be saved to: {args.output_file}")
    print()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Run experiments and log results
    with open(args.output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, config in enumerate(experiment_configs, 1):
            print(f"Progress: {i}/{len(experiment_configs)}")
            try:
                run_single_experiment(config, writer)
                csvfile.flush()  # Ensure data is written immediately
            except KeyboardInterrupt:
                print("\nExperiment interrupted by user.")
                break
            except Exception as e:
                print(
                    f"FATAL ERROR in experiment {config.get('experiment_id', 'UNKNOWN')}: {e}"
                )
                # Log error to CSV
                error_data = {
                    **config,
                    "error_message": str(e),
                    "repair_successful": False,
                }
                writer.writerow({k: error_data.get(k, "") for k in FIELDNAMES})

    print(f"\nExperiment runner completed. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
