#!/usr/bin/env python3
"""
Convert repaired model weights from sytorch format to standard AlexNet format.

This utility fixes the naming issue where repaired models have keys like "0.0.weight" 
instead of "features.0.weight".
"""

import torch
import argparse
import os


def convert_repaired_state_dict(state_dict):
    """Convert repaired model state dict to standard AlexNet format."""
    mapping = {
        '0.0': 'features.0',
        '0.3': 'features.3', 
        '0.6': 'features.6',
        '0.8': 'features.8',
        '0.10': 'features.10',
        '3.1': 'classifier.1',
        '3.4': 'classifier.4', 
        '3.6': 'classifier.6'
    }
    
    converted_dict = {}
    for key, value in state_dict.items():
        # Extract the base key (without .weight or .bias)
        if '.' in key:
            base_key, suffix = key.rsplit('.', 1)
            if base_key in mapping:
                new_key = f"{mapping[base_key]}.{suffix}"
                converted_dict[new_key] = value
                print(f"  {key} -> {new_key}")
            else:
                converted_dict[key] = value
                print(f"  {key} (unchanged)")
        else:
            converted_dict[key] = value
            print(f"  {key} (unchanged)")
    
    return converted_dict


def main():
    parser = argparse.ArgumentParser(description="Convert repaired model weights to standard AlexNet format")
    parser.add_argument("input_path", help="Path to repaired model file")
    parser.add_argument("output_path", nargs='?', help="Output path (default: add _converted suffix)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_converted{ext}"
    
    # Check if output exists
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Output file {args.output_path} already exists. Use --overwrite to replace it.")
        return
    
    print(f"Loading model from: {args.input_path}")
    try:
        state_dict = torch.load(args.input_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if conversion is needed
    has_sytorch_keys = any(key.startswith(('0.', '3.')) for key in state_dict.keys())
    has_standard_keys = any(key.startswith(('features.', 'classifier.')) for key in state_dict.keys())
    
    if has_standard_keys and not has_sytorch_keys:
        print("Model already has standard AlexNet format. No conversion needed.")
        return
    elif not has_sytorch_keys:
        print("Model does not appear to be a repaired AlexNet model.")
        return
    
    print("Converting key names:")
    converted_dict = convert_repaired_state_dict(state_dict)
    
    print(f"Saving converted model to: {args.output_path}")
    torch.save(converted_dict, args.output_path)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
