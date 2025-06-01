import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sytorch as st
from models import alexnet

# SyTorch setup
device = st.device("cpu")
dtype = st.float64


def repair_alexnet_on_misclassified_image():
    """
    Repair AlexNet classifier layers on a single misclassified image.
    Uses the misclassified edit dataset which contains exactly one image.
    """
    
    # Load and convert AlexNet model
    print("Loading and converting AlexNet model...")
    model = alexnet(state_dict_path="artifacts/alexnet_base.pth")
    model = model.to(dtype=dtype, device=device)
    model.eval()
    
    # Print model structure to understand architecture
    print("Model structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            print(f"  {name}: {module}")
    print()

    # Load the misclassified edit set
    print("Loading misclassified edit set...")
    edit_data = torch.load(
        "data/edit_sets/misclassified_edit_dataset.pt", map_location=device
    )
    edit_images = edit_data["images"].to(dtype=dtype, device=device)
    edit_labels = edit_data["labels"].to(device=device)

    # Get the single misclassified image and its true label
    image_to_repair = edit_images[0].unsqueeze(0)  # Shape: [1, 3, 32, 32]
    true_label = edit_labels[0].item()

    print(f"Image shape: {image_to_repair.shape}")
    print(f"True label: {true_label} (ship)")

    # Evaluate original model on the misclassified image
    print("\nEvaluating original model...")
    with st.no_grad():
        original_output = model(image_to_repair)
        original_probs = torch.nn.functional.softmax(original_output, dim=1)
        _, original_pred = torch.max(original_output, 1)
        print(f"Original prediction: {original_pred.item()} (airplane)")
        print(f"True class probability: {original_probs[0, true_label].item():.4f}")
        print(f"Predicted class probability: {original_probs[0, original_pred.item()].item():.4f}")

    # Setup repair - create a symbolic copy of the model
    print("\nSetting up repair...")
    solver = st.GurobiSolver()
    N = model.deepcopy().to(solver).repair()

    # Make the final classifier layer symbolic with bounded parameter changes
    param_change_bound = 5.0
    print(f"Making final classifier layer symbolic with bound ±{param_change_bound}")
    
    # The final layer is at position 3.6 (N[3][6] - last Linear layer)
    final_layer = N[3][6]
    final_layer.weight.requires_symbolic_(
        lb=-param_change_bound, ub=param_change_bound
    )
    if final_layer.bias is not None:
        final_layer.bias.requires_symbolic_(
            lb=-param_change_bound, ub=param_change_bound
        )

    # Define constraints to ensure correct classification
    print("Defining repair constraints...")
    
    # Get reference output (non-symbolic)
    with st.no_symbolic(), st.no_grad():
        reference_output = N(image_to_repair)

    # Get symbolic output
    symbolic_output = N(image_to_repair)
    true_class_logit = symbolic_output[0, true_label]
    
    # Define margin for classification constraint
    margin = 2.0
    print(f"Using margin of {margin} for classification constraints")

    # Create constraints: true class logit should be higher than all other class logits by margin
    constraints = []
    for i in range(10):  # CIFAR-10 has 10 classes
        if i != true_label:
            constraints.append(true_class_logit >= symbolic_output[0, i] + margin)

    print(f"Created {len(constraints)} classification constraints")

    # Define objective: minimize parameter changes and output changes
    print("Setting up optimization objective...")
    param_deltas = N.parameter_deltas(concat=True)
    output_deltas = (symbolic_output - reference_output).flatten().alias()
    
    # Combined objective: minimize both parameter changes and output changes
    objective = st.cat([output_deltas, param_deltas]).norm_ub("linf+l1_normalized")

    # Solve the repair problem
    print("\nSolving repair problem...")
    if solver.solve(*constraints, minimize=objective):
        print("✓ Repair successful!")
        
        # Update the model with the optimal solution
        N.update_()
        N.repair(False)
        N.eval()

        # Evaluate repaired model
        print("\nEvaluating repaired model...")
        with st.no_grad():
            repaired_output = N(image_to_repair)
            repaired_probs = torch.nn.functional.softmax(repaired_output, dim=1)
            _, repaired_pred = torch.max(repaired_output, 1)
            
            print(f"Repaired prediction: {repaired_pred.item()}")
            print(f"True class probability: {repaired_probs[0, true_label].item():.4f}")
            print(f"Predicted class probability: {repaired_probs[0, repaired_pred.item()].item():.4f}")
            
            if repaired_pred.item() == true_label:
                print("Successfully corrected the misclassification!")
            else:
                print("Model still misclassifies the image")
                
        # Calculate parameter changes
        with st.no_grad():
            original_params = model[3][6].weight.data.clone()
            repaired_params = N[3][6].weight.data.clone()
            param_change = torch.norm(repaired_params - original_params).item()
            print(f"Parameter change magnitude: {param_change:.6f}")
            
        # Save the repaired model weights
        print("\nSaving repaired model...")
        repaired_model_path = "artifacts/alexnet_repaired.pth"
        
        # Convert the repaired model back to a regular PyTorch model
        # First, get the original PyTorch model structure
        original_torch_model = alexnet()
        
        # Copy the repaired weights to the PyTorch model
        with torch.no_grad():
            # Copy all parameters from the repaired sytorch model to the PyTorch model
            for (name, param), (_, torch_param) in zip(N.named_parameters(), original_torch_model.named_parameters()):
                torch_param.data.copy_(param.data.to(torch_param.dtype))
        
        # Save the PyTorch model
        torch.save(original_torch_model.state_dict(), repaired_model_path)
        print(f"Repaired model saved to: {repaired_model_path}")
        
        # Verify by loading the saved model and testing
        print("\nVerifying saved model...")
        verification_model = alexnet()  # Load fresh model
        verification_model.load_state_dict(torch.load(repaired_model_path, map_location=device))
        verification_model.eval()
        
        # Test with the original image (convert to float32 for regular PyTorch)
        test_image = image_to_repair.to(torch.float32)
        
        with torch.no_grad():
            verify_output = verification_model(test_image)
            verify_probs = torch.nn.functional.softmax(verify_output, dim=1)
            _, verify_pred = torch.max(verify_output, 1)
            
            print(f"Verification - Prediction: {verify_pred.item()}")
            print(f"Verification - True class probability: {verify_probs[0, true_label].item():.4f}")
            print(f"Verification - Predicted class probability: {verify_probs[0, verify_pred.item()].item():.4f}")
            
            if verify_pred.item() == true_label:
                print("Verification successful - saved model correctly classifies the image!")
            else:
                print("Verification failed - saved model still misclassifies")
            
    else:
        print("Repair failed - could not find a solution that satisfies the constraints.")


if __name__ == "__main__":
    repair_alexnet_on_misclassified_image()