import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import sytorch as st
from models import alexnet

# Setup
device = st.device("cpu")
dtype = st.float64


def verify_repaired_model():
    """
    Verify that the repaired model correctly classifies the target image
    and compare its performance with the original model.
    """
    
    print("Loading original and repaired models...")
    
    # Load original model
    original_model = alexnet(state_dict_path="artifacts/alexnet_base.pth")
    original_model = original_model.to(dtype=dtype, device=device)
    original_model.eval()
    
    # Load repaired model
    repaired_model = alexnet(state_dict_path="artifacts/alexnet_repaired.pth")
    repaired_model = repaired_model.to(dtype=dtype, device=device)
    repaired_model.eval()
    
    # Load the test image
    print("Loading test image...")
    edit_data = torch.load("data/edit_sets/misclassified_edit_dataset.pt", map_location=device)
    test_image = edit_data["images"][0].unsqueeze(0).to(dtype=dtype, device=device)
    true_label = edit_data["labels"][0].item()
    
    print(f"Test image shape: {test_image.shape}")
    print(f"True label: {true_label} (ship)")
    print()
    
    # Test original model
    print("Testing original model:")
    with torch.no_grad():
        orig_output = original_model(test_image)
        orig_probs = torch.nn.functional.softmax(orig_output, dim=1)
        _, orig_pred = torch.max(orig_output, 1)
        
        print(f"  Prediction: {orig_pred.item()}")
        print(f"  True class probability: {orig_probs[0, true_label].item():.4f}")
        print(f"  Predicted class probability: {orig_probs[0, orig_pred.item()].item():.4f}")
        print(f"  Correct: {'✅' if orig_pred.item() == true_label else '❌'}")
    
    print()
    
    # Test repaired model
    print("Testing repaired model:")
    with torch.no_grad():
        rep_output = repaired_model(test_image)
        rep_probs = torch.nn.functional.softmax(rep_output, dim=1)
        _, rep_pred = torch.max(rep_output, 1)
        
        print(f"  Prediction: {rep_pred.item()}")
        print(f"  True class probability: {rep_probs[0, true_label].item():.4f}")
        print(f"  Predicted class probability: {rep_probs[0, rep_pred.item()].item():.4f}")
        print(f"  Correct: {'✅' if rep_pred.item() == true_label else '❌'}")
    
    print()
    
    # Compare confidence improvements
    orig_confidence = orig_probs[0, true_label].item()
    rep_confidence = rep_probs[0, true_label].item()
    confidence_improvement = rep_confidence - orig_confidence
    
    print("Repair summary:")
    print(f"  Original confidence in true class: {orig_confidence:.4f}")
    print(f"  Repaired confidence in true class: {rep_confidence:.4f}")
    print(f"  Confidence improvement: {confidence_improvement:+.4f}")
    
    # Check weight differences
    orig_final_weights = original_model[3][6].weight.data
    rep_final_weights = repaired_model[3][6].weight.data
    weight_diff = torch.norm(rep_final_weights - orig_final_weights).item()
    print(f"  Final layer weight change magnitude: {weight_diff:.6f}")
    
    if rep_pred.item() == true_label and orig_pred.item() != true_label:
        print("\n🎉 Repair successful! The model now correctly classifies the target image.")
    elif rep_pred.item() == true_label and orig_pred.item() == true_label:
        print("\n✅ Both models classify correctly.")
    else:
        print("\n❌ Repair did not fix the misclassification.")


if __name__ == "__main__":
    verify_repaired_model()
