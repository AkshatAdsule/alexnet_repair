import torch
import torch.nn.functional as F

from helpers.cifar import get_cifar10_dataloader
from models import alexnet


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
    model = alexnet("artifacts/alexnet_base.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = get_cifar10_dataloader(batch_size=64, train=False, num_workers=2)
    acc = eval_accuracy(model, dataloader, device)
    loss = eval(model, dataloader, device)
    print(f"Accuracy: {acc:.4f}")
    print(f"Loss: {loss:.4f}")
