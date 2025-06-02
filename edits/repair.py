import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sytorch as st
from models import alexnet

device = st.device("cpu")
dtype = st.float32


def repair_model(edit_set_path="data/edit_sets/misclassified_edit_dataset.pt", 
                 param_bound=5.0, margin=2.0, output_path="artifacts/alexnet_repaired.pth"):
    """Repair AlexNet model on edit set images."""
    
    model = alexnet(state_dict_path="artifacts/alexnet_base.pth")
    model = model.to(dtype=dtype, device=device)
    model.eval()

    edit_data = torch.load(edit_set_path, map_location=device)
    edit_images = edit_data["images"].to(dtype=dtype, device=device)
    edit_labels = edit_data["labels"].to(device=device)

    images_to_repair = [edit_images[i].unsqueeze(0) for i in range(len(edit_images))]
    true_labels = [edit_labels[i].item() for i in range(len(edit_labels))]

    print(f"Loaded {len(images_to_repair)} images for repair")

    solver = st.GurobiSolver()
    N = model.deepcopy().to(solver).repair()

    final_layer = N[3][6]
    final_layer.weight.requires_symbolic_(lb=-param_bound, ub=param_bound)
    if final_layer.bias is not None:
        final_layer.bias.requires_symbolic_(lb=-param_bound, ub=param_bound)

    reference_outputs = []
    with st.no_symbolic(), st.no_grad():
        for image_to_repair in images_to_repair:
            reference_outputs.append(N(image_to_repair))

    symbolic_outputs = []
    for image_to_repair in images_to_repair:
        symbolic_outputs.append(N(image_to_repair))
    
    constraints = []
    for i, (symbolic_output, true_label) in enumerate(zip(symbolic_outputs, true_labels)):
        for c in range(10):
            if c != true_label:
                constraints.append(symbolic_outputs[i][0, true_label] >= symbolic_outputs[i][0, c] + margin)

    param_deltas = N.parameter_deltas(concat=True)
    all_output_deltas = []
    for symbolic_output, reference_output in zip(symbolic_outputs, reference_outputs):
        output_delta = (symbolic_output - reference_output).flatten().alias()
        all_output_deltas.append(output_delta)
    
    combined_output_deltas = st.cat(all_output_deltas)
    objective = st.cat([combined_output_deltas, param_deltas]).norm_ub("linf+l1_normalized")

    if solver.solve(*constraints, minimize=objective):
        N.update_()
        N.repair(False)
        N.eval()

        all_correct = True
        for i, (image_to_repair, true_label) in enumerate(zip(images_to_repair, true_labels)):
            with st.no_grad():
                repaired_output = N(image_to_repair)
                _, repaired_pred = torch.max(repaired_output, 1)
                if repaired_pred.item() != true_label:
                    all_correct = False

        # Create a fresh PyTorch AlexNet model (not sytorch)
        from model_classes.alexnet import AlexNet
        original_torch_model = AlexNet(10)
        
        # Load the base weights into the fresh model
        base_state_dict = torch.load("artifacts/alexnet_base.pth", map_location=device)
        original_torch_model.load_state_dict(base_state_dict)
        
        # Copy repaired parameters from sytorch model to PyTorch model
        with torch.no_grad():
            for (name, param), (_, torch_param) in zip(N.named_parameters(), original_torch_model.named_parameters()):
                torch_param.data.copy_(param.data.to(torch_param.dtype))
        
        torch.save(original_torch_model.state_dict(), output_path)
        
        return all_correct, len(images_to_repair)
    else:
        return False, 0