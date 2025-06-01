#!/usr/bin/env python3
"""
Generate By-Class Edit Set - Create a repair set containing all images of a specific class

This script generates a repair set containing all images from the test set that belong
to a specific true class, regardless of whether they were correctly classified or not.
"""

import os
import json
import torch
import torch.nn.functional as F
from helpers.cifar import get_cifar10_dataloader
from models.alexnet import alexnet


def create_by_class_editset(
    model,
    target_class,
    max_samples=-1,
    output_dir="data/edit_sets",
    output_prefix=None,
):
    """
    Create a repair set from all images of a specific class.

    Args:
        model: The model to evaluate
        target_class: The class index (0-9 for CIFAR-10) to collect images for
        max_samples: Maximum number of samples to collect (-1 for all)
        output_dir: Directory to save the repair set
        output_prefix: Prefix for output files (auto-generated if None)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get CIFAR-10 test dataset
    dataloader = get_cifar10_dataloader(batch_size=64, train=False, num_workers=2)

    # CIFAR-10 class names
    cifar10_classes = [
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

    # Validate target class
    if not (0 <= target_class <= 9):
        raise ValueError(f"Target class must be between 0 and 9, got {target_class}")

    target_class_name = cifar10_classes[target_class]

    # Auto-generate prefix if not provided
    if output_prefix is None:
        output_prefix = f"class_{target_class}_{target_class_name}"

    # Storage for edit dataset
    edit_images = []
    edit_labels = []
    edit_metadata = []

    samples_collected = 0
    total_processed = 0
    total_target_class = 0
    correct_predictions = 0
    incorrect_predictions = 0

    print(
        f"Creating by-class repair set for class {target_class} ({target_class_name})..."
    )
    print(f"Max samples: {max_samples if max_samples > 0 else 'unlimited'}")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move to same device as model
            device = next(model.parameters()).device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            predicted_probs, predicted = torch.max(probabilities, 1)

            # Handle all images of the target class
            for i in range(images.size(0)):
                total_processed += 1
                true_label_idx = labels[i].item()

                if true_label_idx == target_class:
                    total_target_class += 1

                    if max_samples == -1 or samples_collected < max_samples:
                        predicted_prob = predicted_probs[i].item()
                        true_prob = probabilities[i][true_label_idx].item()
                        pred_label_idx = predicted[i].item()
                        is_correct = pred_label_idx == true_label_idx

                        if is_correct:
                            correct_predictions += 1
                        else:
                            incorrect_predictions += 1

                        # Store the image tensor (normalized)
                        edit_images.append(images[i].cpu())
                        # Store the correct label
                        edit_labels.append(labels[i].cpu())

                        # Store metadata including probabilities and prediction status
                        batch_size = dataloader.batch_size or 1
                        metadata = {
                            "image_idx": (batch_idx * batch_size) + i,
                            "true_label": int(true_label_idx),
                            "predicted_label": int(pred_label_idx),
                            "true_probability": float(true_prob),
                            "predicted_probability": float(predicted_prob),
                            "margin": float(predicted_prob - true_prob),
                            "true_class": target_class_name,
                            "predicted_class": cifar10_classes[pred_label_idx],
                            "is_correct": bool(is_correct),
                            "type": "by_class",
                            "target_class": int(target_class),
                        }
                        edit_metadata.append(metadata)

                        samples_collected += 1

                        if samples_collected % 50 == 0:
                            print(
                                f"Collected {samples_collected} images of class '{target_class_name}'..."
                            )

                # Progress update
                if total_processed % 1000 == 0:
                    print(
                        f"Processed {total_processed} images, found {total_target_class} images of target class..."
                    )

            # Exit early if we've found enough samples
            if max_samples != -1 and samples_collected >= max_samples:
                break

    if not edit_images:
        print(f"No images found for class {target_class} ({target_class_name})")
        return None, None, None

    # Convert to tensors
    edit_images_tensor = torch.stack(edit_images)
    edit_labels_tensor = torch.stack(edit_labels)

    # Save the edit dataset
    dataset_path = os.path.join(output_dir, f"{output_prefix}_edit_dataset.pt")
    metadata_path = os.path.join(output_dir, f"{output_prefix}_edit_metadata.json")

    torch.save(
        {
            "images": edit_images_tensor,
            "labels": edit_labels_tensor,
            "metadata": edit_metadata,
        },
        dataset_path,
    )

    # Save metadata as JSON for easy inspection
    with open(metadata_path, "w") as f:
        json.dump(edit_metadata, f, indent=2)

    print("-" * 80)
    print(f"BY-CLASS REPAIR SET CREATED FOR {target_class_name.upper()}!")
    print("-" * 80)
    print(f"✓ Dataset saved to: {dataset_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ Total images processed: {total_processed}")
    print(f"✓ Total images of target class: {total_target_class}")
    print(f"✓ Images collected: {len(edit_images)}")
    print(f"✓ Correctly predicted: {correct_predictions}")
    print(f"✓ Incorrectly predicted: {incorrect_predictions}")
    print(
        f"✓ Accuracy for this class: {correct_predictions / len(edit_images) * 100:.1f}%"
    )
    print(f"✓ Image tensor shape: {edit_images_tensor.shape}")
    print(f"✓ Labels tensor shape: {edit_labels_tensor.shape}")

    # Calculate and display statistics
    true_probs = [meta["true_probability"] for meta in edit_metadata]
    margins = [meta["margin"] for meta in edit_metadata]
    if true_probs and margins:
        print(f"✓ Average true probability: {sum(true_probs) / len(true_probs):.3f}")
        print(f"✓ Average margin: {sum(margins) / len(margins):.3f}")
        print(f"✓ Min margin: {min(margins):.3f}")
        print(f"✓ Max margin: {max(margins):.3f}")

    return edit_images_tensor, edit_labels_tensor, edit_metadata
