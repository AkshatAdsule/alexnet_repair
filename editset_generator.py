#!/usr/bin/env python3
"""
Edit Set Generator - Frontend for creating different types of edit sets

This script provides a command-line interface for generating various types of edit sets
using the editset_helpers functions. It supports generating misclassified sets, by-class sets,
and other repair sets for model analysis.
"""

import sys
import os
import argparse
import torch
from models import alexnet
from editset_helpers.missclassified import create_misclassified_editset
from editset_helpers.by_class import create_by_class_editset


def load_model():
    """Load the AlexNet model."""
    print("Loading AlexNet model...")
    try:
        model = alexnet("artifacts/alexnet_base.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✓ Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease check that:")
        print("  - The AlexNet model file exists at artifacts/alexnet_base.pth")
        print("  - All dependencies are installed")
        return None


def generate_misclassified(args):
    """Generate misclassified edit set."""
    print("=" * 80)
    print("GENERATING MISCLASSIFIED EDIT SET")
    print("=" * 80)

    model = load_model()
    if model is None:
        return False

    # Check if repair set already exists
    dataset_path = "data/edit_sets/misclassified_edit_dataset.pt"
    metadata_path = "data/edit_sets/misclassified_edit_metadata.json"

    if (
        os.path.exists(dataset_path)
        and os.path.exists(metadata_path)
        and not args.force
    ):
        print(f"Misclassified edit set already exists at {dataset_path}")
        response = input("Do you want to regenerate? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Using existing edit set.")
            return True

    try:
        edit_images, edit_labels, edit_metadata = create_misclassified_editset(
            model=model,
            max_samples=args.max_samples,
            output_prefix="misclassified",
        )

        if edit_images is not None:
            print("\n" + "=" * 80)
            print("SUCCESS! Misclassified edit set created.")
            print("=" * 80)
            return True
        else:
            print("Failed to create misclassified edit set.")
            return False

    except Exception as e:
        print(f"\n✗ Error generating edit set: {e}")
        return False


def generate_by_class(args):
    """Generate by-class edit set."""
    print("=" * 80)
    print(f"GENERATING BY-CLASS EDIT SET FOR CLASS {args.target_class}")
    print("=" * 80)

    model = load_model()
    if model is None:
        return False

    # CIFAR-10 class names for reference
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

    if not (0 <= args.target_class <= 9):
        print(f"✗ Invalid target class: {args.target_class}")
        print("Target class must be between 0 and 9 for CIFAR-10:")
        for i, class_name in enumerate(cifar10_classes):
            print(f"  {i}: {class_name}")
        return False

    target_class_name = cifar10_classes[args.target_class]
    print(f"Target class: {args.target_class} ({target_class_name})")

    # Check if repair set already exists
    output_prefix = f"class_{args.target_class}_{target_class_name}"
    dataset_path = f"data/edit_sets/{output_prefix}_edit_dataset.pt"
    metadata_path = f"data/edit_sets/{output_prefix}_edit_metadata.json"

    if (
        os.path.exists(dataset_path)
        and os.path.exists(metadata_path)
        and not args.force
    ):
        print(
            f"By-class edit set for {target_class_name} already exists at {dataset_path}"
        )
        response = input("Do you want to regenerate? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Using existing edit set.")
            return True

    try:
        edit_images, edit_labels, edit_metadata = create_by_class_editset(
            model=model,
            target_class=args.target_class,
            max_samples=args.max_samples,
            output_prefix=output_prefix,
        )

        if edit_images is not None:
            print("\n" + "=" * 80)
            print(f"SUCCESS! By-class edit set for {target_class_name} created.")
            print("=" * 80)
            return True
        else:
            print(f"Failed to create by-class edit set for {target_class_name}.")
            return False

    except Exception as e:
        print(f"\n✗ Error generating edit set: {e}")
        return False


def list_existing_sets():
    """List all existing edit sets."""
    print("=" * 80)
    print("EXISTING EDIT SETS")
    print("=" * 80)

    edit_sets_dir = "data/edit_sets"
    if not os.path.exists(edit_sets_dir):
        print("No edit sets directory found.")
        return

    found_sets = []
    for filename in os.listdir(edit_sets_dir):
        if filename.endswith("_dataset.pt"):
            base_name = filename[:-11]  # Remove "_dataset.pt"
            metadata_file = f"{base_name}_metadata.json"

            if os.path.exists(os.path.join(edit_sets_dir, metadata_file)):
                dataset_path = os.path.join(edit_sets_dir, filename)
                metadata_path = os.path.join(edit_sets_dir, metadata_file)

                # Get file sizes
                dataset_size = os.path.getsize(dataset_path) / (1024 * 1024)  # MB

                # Try to get number of images from the dataset
                try:
                    data = torch.load(dataset_path, map_location="cpu")
                    num_images = len(data["images"])
                    print(f"✓ {base_name}")
                    print(f"    Images: {num_images}")
                    print(f"    Size: {dataset_size:.1f} MB")
                    print(f"    Dataset: {dataset_path}")
                    print(f"    Metadata: {metadata_path}")
                    print()
                    found_sets.append(base_name)
                except Exception as e:
                    print(f"✗ {base_name} (error reading: {e})")

    if not found_sets:
        print("No valid edit sets found.")
    else:
        print(f"Found {len(found_sets)} edit set(s).")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate edit sets for AlexNet model analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate misclassified edit set with all samples
  python editset_generator.py misclassified

  # Generate misclassified edit set with max 1000 samples  
  python editset_generator.py misclassified --max-samples 1000

  # Generate by-class edit set for cats (class 3)
  python editset_generator.py by-class --target-class 3

  # Generate by-class edit set for dogs with max 500 samples
  python editset_generator.py by-class --target-class 5 --max-samples 500

  # List all existing edit sets
  python editset_generator.py list

  # Force regeneration of existing sets
  python editset_generator.py misclassified --force

CIFAR-10 Classes:
  0: airplane    1: automobile  2: bird     3: cat      4: deer
  5: dog         6: frog        7: horse    8: ship     9: truck
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Misclassified command
    misc_parser = subparsers.add_parser(
        "misclassified", help="Generate misclassified edit set"
    )
    misc_parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to collect (-1 for all)",
    )
    misc_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if edit set exists",
    )

    # By-class command
    class_parser = subparsers.add_parser("by-class", help="Generate by-class edit set")
    class_parser.add_argument(
        "--target-class",
        type=int,
        required=True,
        help="Target class index (0-9 for CIFAR-10)",
    )
    class_parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to collect (-1 for all)",
    )
    class_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if edit set exists",
    )

    # List command
    subparsers.add_parser("list", help="List existing edit sets")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Create output directory if it doesn't exist
    os.makedirs("data/edit_sets", exist_ok=True)

    # Route to appropriate function
    if args.command == "misclassified":
        success = generate_misclassified(args)
    elif args.command == "by-class":
        success = generate_by_class(args)
    elif args.command == "list":
        list_existing_sets()
        success = True
    else:
        parser.print_help()
        success = False

    if success:
        print(
            "\nTip: Use 'python editset_generator.py list' to see all available edit sets"
        )
        print(
            "     Use 'python edit_set_visualizer.py' to explore edit sets in the web interface"
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
