import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(save_dir, dataset_name, img_size=32, split="train", batch_size=128, selected_classes=None, selected_attributes=None):
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Data transformations (adjust based on the dataset size, CelebA images are larger than CIFAR)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize all images to 32x32 for consistency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset loading logic
    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=(split=="train"), download=True, transform=transform)
        if selected_classes is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
            dataset = torch.utils.data.Subset(dataset, indices)

    elif dataset_name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=(split=="train"), download=True, transform=transform)
        if selected_classes is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
            dataset = torch.utils.data.Subset(dataset, indices)
    
    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = torchvision.datasets.MNIST(root=data_dir, train=(split=="train"), download=True, transform=transform)
        if selected_classes is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
            dataset = torch.utils.data.Subset(dataset, indices)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=(split=="train"), download=True, transform=transform)
        if selected_classes is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
            dataset = torch.utils.data.Subset(dataset, indices)

    elif dataset_name == "CelebA":
        # Define the attribute name to index mapping (you can check CelebA documentation for the correct mapping)
        attribute_name_to_idx = {
            "Blond_Hair": 9,  # Example index for Blond_Hair
            "Male": 20        # Example index for Male
        }
        dataset = torchvision.datasets.CelebA(root=data_dir, split=split, download=False, transform=transform)

        # Optionally filter CelebA based on specific attributes (e.g., hair color, gender, etc.)
        if selected_attributes is not None:
            # Ensure that the attributes are mapped to their corresponding indices in the tensor
            attr_indices = [
                i for i, (img, attr) in enumerate(dataset)
                if all(attr[attribute_name_to_idx[attr_name]] == val for attr_name, val in selected_attributes.items())
            ]
            dataset = torch.utils.data.Subset(dataset, attr_indices)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return dataloader
