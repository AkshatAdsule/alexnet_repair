#!/usr/bin/env python3
import sys, os, time, csv, random

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from edits.repair import repair_model
from experiment_runner import (
    generate_edit_set,
    evaluate_model_on_edit_set,
    load_alexnet_with_weights,
)
from eval import eval_accuracy
from helpers.cifar import get_cifar10_dataloader
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Stochastic Repair Experiments")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        required=True,
        help="List of batch sizes to sample each iteration",
    )
    parser.add_argument(
        "--repair-set-homogeneities",
        nargs="+",
        type=str,
        required=True,
        choices=[
            "misclassified",
            "class_homogeneous_correct",
            "class_homogeneous_incorrect",
        ],
        help="Homogeneity types for the full repair set",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        required=True,
        help="Number of sequential repair passes per experiment",
    )
    parser.add_argument(
        "--param-bound",
        type=float,
        default=5.0,
        help="Parameter change bound for repair",
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="Classification margin for repair"
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        type=int,
        default=list(range(10)),
        help="Target classes for class_homogeneous types",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/stochastic_experiment.csv",
        help="CSV file to write results to",
    )
    args = parser.parse_args()

    # Prepare output directories
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs("artifacts/experiments", exist_ok=True)
    # Load resume info if CSV exists
    resume_iters = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                exp = row.get("experiment_id")
                try:
                    it = int(row.get("iteration", 0))
                except ValueError:
                    continue
                resume_iters[exp] = max(resume_iters.get(exp, 0), it)
        file_mode = "a"
        write_header = False
    else:
        file_mode = "w"
        write_header = True
    # CSV fieldnames
    fieldnames = [
        "experiment_id",
        "timestamp",
        "batch_size",
        "homogeneity",
        "iteration",
        "test_accuracy",
        "full_repair_accuracy",
        "batch_repair_time",
        "error_message",
    ]
    # Open CSV in resume or write mode
    with open(args.output_file, file_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # Loop over configs
        for batch_size in args.batch_sizes:
            for hom in args.repair_set_homogeneities:
                # Determine target classes list
                target_list = [None]
                if hom.startswith("class_homogeneous"):
                    target_list = args.target_classes

                for target_class in target_list:
                    # Build a base experiment ID
                    tag = f"{hom}" + (
                        f"_cls{target_class}" if target_class is not None else ""
                    )
                    base_exp_id = f"stoch_bs{batch_size}_{tag}"
                    # Skip fully completed experiments without regenerating the edit set
                    if resume_iters.get(base_exp_id, 0) >= args.num_iterations:
                        print(
                            f"Skipping {base_exp_id}: already completed {resume_iters.get(base_exp_id)} iterations"
                        )
                        continue

                    # Generate full edit set (size=-1 means all)
                    config = {
                        "repair_set_type": hom,
                        "requested_repair_set_size": -1,
                        "edit_set_sizing_strategy": "flexible",
                        "target_class": target_class,
                    }
                    edit_set_path, full_size = generate_edit_set(
                        config, "artifacts/alexnet_base.pth"
                    )
                    if edit_set_path is None:
                        continue

                    # Load full repair set
                    data = torch.load(edit_set_path, map_location="cpu")
                    full_images = data["images"]
                    full_labels = data["labels"]
                    full_size = len(full_labels)

                    # Prepare sampling indices
                    indices = list(range(full_size))
                    random.shuffle(indices)
                    pos = 0

                    # Determine resume iteration for this experiment
                    resume_start = resume_iters.get(base_exp_id, 0) + 1
                    if resume_start > args.num_iterations:
                        print(
                            f"Skipping {base_exp_id}, already completed {resume_iters.get(base_exp_id)} iterations"
                        )
                        continue
                    # Initialize current model path
                    if resume_start > 1:
                        current_model = f"artifacts/experiments/{base_exp_id}_iter{resume_start - 1}_model.pth"
                    else:
                        current_model = "artifacts/alexnet_base.pth"

                    # Sequential repair passes
                    for it in range(resume_start, args.num_iterations + 1):
                        if pos + batch_size > full_size:
                            random.shuffle(indices)
                            pos = 0
                        batch_idx = indices[pos : pos + batch_size]
                        pos += batch_size

                        # Extract and save batch edit set
                        batch_images = full_images[batch_idx]
                        batch_labels = full_labels[batch_idx]
                        batch_set_path = (
                            f"artifacts/experiments/{base_exp_id}_iter{it}_batch.pt"
                        )
                        torch.save(
                            {"images": batch_images, "labels": batch_labels},
                            batch_set_path,
                        )

                        # Run repair on this batch
                        output_model_path = (
                            f"artifacts/experiments/{base_exp_id}_iter{it}_model.pth"
                        )
                        start = time.time()
                        try:
                            success, _ = repair_model(
                                edit_set_path=batch_set_path,
                                param_bound=args.param_bound,
                                margin=args.margin,
                                output_path=output_model_path,
                                base_state_dict_path=current_model,
                            )
                        except Exception as e:
                            # Log error and break
                            writer.writerow(
                                {
                                    "experiment_id": base_exp_id,
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "batch_size": batch_size,
                                    "homogeneity": hom,
                                    "iteration": it,
                                    "test_accuracy": "",
                                    "full_repair_accuracy": "",
                                    "batch_repair_time": "",
                                    "error_message": str(e),
                                }
                            )
                            break
                        duration = time.time() - start

                        # Evaluate repaired model
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        model = load_alexnet_with_weights(output_model_path)
                        model.to(device)
                        test_dataloader = get_cifar10_dataloader(
                            batch_size=64, train=False, num_workers=2
                        )
                        test_acc = eval_accuracy(model, test_dataloader, device)
                        full_acc = evaluate_model_on_edit_set(
                            model, edit_set_path, device
                        )

                        # Log results
                        writer.writerow(
                            {
                                "experiment_id": base_exp_id,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "batch_size": batch_size,
                                "homogeneity": hom,
                                "iteration": it,
                                "test_accuracy": test_acc,
                                "full_repair_accuracy": full_acc,
                                "batch_repair_time": duration,
                                "error_message": "",
                            }
                        )
                        csvfile.flush()

                        # Update base for next iteration
                        current_model = output_model_path


if __name__ == "__main__":
    main()
