import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from modules import UNet, UNet2
from rectified_flow import RectifiedFlow
from data import get_dataloader
from utils import create_save_dir

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

def train(device, save_dir, dataloader, img_size=32, c_in=3, num_epochs=100, lr=5e-4):
    
    # Directories
    model_dir = os.path.join(save_dir, "models-3")
    os.makedirs(model_dir, exist_ok=True)
    
    model = UNet2(device, c_in=c_in, c_out=c_in, img_size=img_size).to(device)
    rect_flow = RectifiedFlow(device=device, img_size=img_size, c_in=c_in)
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

            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        # Save model every 10 epochs
        if epoch<5 or (epoch+1)%5==0 or epoch==num_epochs-1:
            torch.save(model.state_dict(), os.path.join(model_dir, f'unet_epoch_{epoch+1}.pth'))
            save_losses(losses, save_dir)

        # Update loss plot
        plot_loss(losses, save_dir)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train UNet on CIFAR-10, CIFAR-100, or CelebA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'CelebA', "FashionMNIST", "MNIST"], help='Dataset to train on (CIFAR10, CIFAR100, or CelebA)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--device', type=int, required=True, default=1, help='GPU device')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes to include in training (e.g. any number upto 9 for CIFAR)')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='Classes to include in training (e.g. 0 1 2 for CIFAR)')
    parser.add_argument('--attributes', type=json.loads, default=None, help='Attributes to filter CelebA dataset (e.g., {"Blond_Hair": 1, "Male": 0})')

    args = parser.parse_args()
    print(args.attributes)
    NUM_CLASSES = [None]
    for args.num_classes in NUM_CLASSES:
        if args.dataset=="CelebA":
            img_size = 32
        else:
            img_size = 32

        if args.dataset=="MNIST" or args.dataset=="FashionMNIST":
            c_in = 1
        else:
            c_in = 3

        if args.num_classes is not None:
            selected_classes = np.arange(args.num_classes)
        else:
            selected_classes = args.classes
        
        # Base save directory
        base_save_dir = "/datastor1/vansh/rectified-flow/saved/{}".format(args.dataset)

        # Create save directory based on classes or attributes
        save_dir = create_save_dir(base_save_dir, selected_classes=selected_classes, selected_attributes=args.attributes)

        device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        print('Using {} device'.format(device))
        print("Selected classes: ", "all" if args.num_classes==None and args.classes==None else selected_classes)
        print("Selected attributes: ", "all" if args.attributes==None else args.attributes)
        dataloader = get_dataloader(base_save_dir, args.dataset, img_size, split="train", batch_size=args.batch_size, selected_classes=selected_classes, selected_attributes=args.attributes)
        train(device, save_dir, dataloader, img_size=img_size, c_in=c_in, num_epochs=args.epochs, lr=args.lr)
