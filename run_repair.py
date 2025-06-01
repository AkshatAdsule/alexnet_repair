#!/usr/bin/env python3
"""
Model Repair Runner - Frontend for running model repair operations

This script provides a command-line interface for repairing AlexNet models
using different edit sets and parameters.
"""

import os
import argparse
from edits.repair import repair_model


def repair_misclassified(args):
    """Repair model on misclassified edit set."""
    print("=" * 80)
    print("REPAIRING MODEL ON MISCLASSIFIED EDIT SET")
    print("=" * 80)

    edit_set_path = "data/edit_sets/misclassified_edit_dataset.pt"
    
    if not os.path.exists(edit_set_path):
        print(f"✗ Edit set not found: {edit_set_path}")
        print("Please generate the edit set first using:")
        print("  uv run python editset_generator.py misclassified")
        return False

    try:
        success, num_images = repair_model(
            edit_set_path=edit_set_path,
            param_bound=args.param_bound,
            margin=args.margin,
            output_path=args.output
        )
        
        if success:
            print("\n" + "=" * 80)
            print(f"SUCCESS! Model repaired on {num_images} images.")
            print(f"Repaired model saved to: {args.output}")
            print("=" * 80)
            return True
        else:
            print("\n✗ Repair failed - could not find solution.")
            return False

    except Exception as e:
        print(f"\n✗ Error during repair: {e}")
        return False


def repair_by_class(args):
    """Repair model on by-class edit set."""
    print("=" * 80)
    print(f"REPAIRING MODEL ON CLASS {args.target_class} EDIT SET")
    print("=" * 80)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    if not (0 <= args.target_class <= 9):
        print(f"✗ Invalid class: {args.target_class}. Must be 0-9.")
        return False

    target_class_name = cifar10_classes[args.target_class]
    output_prefix = f"class_{args.target_class}_{target_class_name}"
    edit_set_path = f"data/edit_sets/{output_prefix}_edit_dataset.pt"
    
    if not os.path.exists(edit_set_path):
        print(f"✗ Edit set not found: {edit_set_path}")
        print("Please generate the edit set first using:")
        print(f"  uv run python editset_generator.py by-class --target-class {args.target_class}")
        return False

    try:
        success, num_images = repair_model(
            edit_set_path=edit_set_path,
            param_bound=args.param_bound,
            margin=args.margin,
            output_path=args.output
        )
        
        if success:
            print("\n" + "=" * 80)
            print(f"SUCCESS! Model repaired on {num_images} images for class {target_class_name}.")
            print(f"Repaired model saved to: {args.output}")
            print("=" * 80)
            return True
        else:
            print("\n✗ Repair failed - could not find solution.")
            return False

    except Exception as e:
        print(f"\n✗ Error during repair: {e}")
        return False


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Repair AlexNet model using edit sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair on misclassified edit set
  uv run python run_repair.py misclassified

  # Repair on misclassified set with custom parameters
  uv run python run_repair.py misclassified --param-bound 3.0 --margin 1.5

  # Repair on class-specific edit set
  uv run python run_repair.py by-class --target-class 3

  # Repair with custom output path
  uv run python run_repair.py misclassified --output artifacts/custom_repaired.pth

CIFAR-10 Classes:
  0: airplane    1: automobile  2: bird     3: cat      4: deer
  5: dog         6: frog        7: horse    8: ship     9: truck
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Misclassified command
    misc_parser = subparsers.add_parser(
        "misclassified", help="Repair model on misclassified edit set"
    )
    misc_parser.add_argument(
        "--param-bound",
        type=float,
        default=5.0,
        help="Parameter change bound (default: 5.0)",
    )
    misc_parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Classification margin (default: 2.0)",
    )
    misc_parser.add_argument(
        "--output",
        type=str,
        default="artifacts/alexnet_repaired.pth",
        help="Output path for repaired model (default: artifacts/alexnet_repaired.pth)",
    )

    # By-class command
    class_parser = subparsers.add_parser("by-class", help="Repair model on by-class edit set")
    class_parser.add_argument(
        "--target-class",
        type=int,
        required=True,
        help="Target class index (0-9 for CIFAR-10)",
    )
    class_parser.add_argument(
        "--param-bound",
        type=float,
        default=5.0,
        help="Parameter change bound (default: 5.0)",
    )
    class_parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Classification margin (default: 2.0)",
    )
    class_parser.add_argument(
        "--output",
        type=str,
        default="artifacts/alexnet_repaired.pth",
        help="Output path for repaired model (default: artifacts/alexnet_repaired.pth)",
    )

    args = parser.parse_args()

    if args.command == "misclassified":
        success = repair_misclassified(args)
    elif args.command == "by-class":
        success = repair_by_class(args)
    else:
        parser.print_help()
        return

    if success:
        print("\nRepair completed successfully!")
    else:
        print("\nRepair failed!")
        exit(1)


if __name__ == "__main__":
    main()
