import torch
from torchvision import datasets, transforms

cifar_transforms = transforms.Compose(
    [
        transforms.Resize((70, 70)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
    ]
)


def get_cifar10_dataloader(batch_size=64, train=True, num_workers=2):
    dataset = datasets.CIFAR10(
        root="data", train=train, download=True, transform=cifar_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers
    )
    return dataloader
