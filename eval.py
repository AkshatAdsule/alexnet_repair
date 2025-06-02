import torch
import torch.nn.functional as F
import argparse

from helpers.cifar import get_cifar10_dataloader
from models import alexnet


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
            else:
                converted_dict[key] = value
        else:
            converted_dict[key] = value
    
    return converted_dict


def load_alexnet_with_weights(weights_path):
    """Load AlexNet model with automatic format detection."""
    try:
        # First try loading normally (for standard format)
        return alexnet(weights_path)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e):
            print("Detected repaired model format, converting...")
            # Load the state dict and convert it
            state_dict = torch.load(weights_path, map_location="cpu")
            converted_state_dict = convert_repaired_state_dict(state_dict)
            
            # Save the converted state dict temporarily and load it
            temp_path = weights_path + ".converted.tmp"
            torch.save(converted_state_dict, temp_path)
            
            try:
                model = alexnet(temp_path)
                # Clean up temporary file
                import os
                os.remove(temp_path)
                return model
            except Exception as inner_e:
                # Clean up temporary file even if loading fails
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise inner_e
        else:
            raise e


def eval(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return average_loss


def eval_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AlexNet model on CIFAR-10")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="artifacts/alexnet_base.pth",
        help="Path to AlexNet weights file (default: artifacts/alexnet_base.pth)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for data loading (default: 2)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.weights}")
    try:
        model = load_alexnet_with_weights(args.weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"This might happen if the weights file was saved in a different format.")
        print(f"Make sure the weights file is compatible with the AlexNet model structure.")
        exit(1)

    print(f"Loading CIFAR-10 test dataset (batch_size={args.batch_size})...")
    dataloader = get_cifar10_dataloader(
        batch_size=args.batch_size, 
        train=False, 
        num_workers=args.num_workers
    )
    
    print("Evaluating model...")
    acc = eval_accuracy(model, dataloader, device)
    loss = eval(model, dataloader, device)
    
    print(f"Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {loss:.4f}")
