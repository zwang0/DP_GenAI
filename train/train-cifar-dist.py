import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from modules import UNet  # Assuming you have a UNet model definition file
from rectified_flow import rectified_flow

# Distributed Training Setup
def setup(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

def plot_loss(losses, save_dir):
    plt.figure()
    plt.plot(range(len(losses)), losses, marker='o', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss-vs-epoch.png'))
    plt.close()

def save_losses(losses, save_dir):
    with open(os.path.join(save_dir, "loss.json"), 'w') as f:
        json.dump(losses, f)

def train(rank, world_size, dataset_name, save_dir, num_epochs=100, batch_size=128, lr=5e-4):
    setup(rank, world_size)
    
    # Directories
    model_dir = os.path.join(save_dir, "models")
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(model_dir, exist_ok=True)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR images are 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Dataset
    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Distributed sampler for DDP
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Model, optimizer, and loss function
    device = torch.device(f'cuda:{rank}')
    model = UNet(device=device).to(device)
    model = DDP(model, device_ids=[rank])
    rect_flow = rectified_flow(img_size=32, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Training Loop
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for _, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t = rect_flow.sample_timesteps(x.shape[0]).to(device)
            x_t, z = rect_flow.noise_images(x, t)
            predict_velocity = model(x_t, t)
            loss = mse(x-z, predict_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        # Save model every 10 epochs
        if (epoch+1)%5==0 or epoch==num_epochs-1:
            torch.save(model.state_dict(), os.path.join(model_dir, f'unet_epoch_{epoch+1}.pth'))
            save_losses(losses, save_dir)

        # Update loss plot
        plot_loss(losses, save_dir)

        if rank == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    cleanup()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train UNet on CIFAR-10 or CIFAR-100')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='Dataset to train on (CIFAR10 or CIFAR100)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    # parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and losses')
    parser.add_argument('--world_size', type=int, default=2, help='Total number of GPUs')
    
    args = parser.parse_args()

    world_size = args.world_size
    save_dir = "/datastor1/vansh/rectified-flow/saved/{}".format(args.dataset)

    torch.multiprocessing.spawn(
        train,
        args=(world_size, args.dataset, save_dir, args.epochs, args.batch_size),
        nprocs=world_size,
        join=True
    )
