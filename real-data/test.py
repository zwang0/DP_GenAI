import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from geomloss import SamplesLoss
from modules import UNet, UNet2
from data import get_dataloader
from rectified_flow import RectifiedFlow
from utils import create_save_dir, log_results, plot_generated_images, plot_wasserstein_gamma

def load_model(model_path, device, c_in, img_size):
    # Load the UNet model and its weights
    model = UNet(device=device, img_size=img_size, c_in=c_in, c_out=c_in).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

def calculate_gamma_for_disc_steps(rect_flow: RectifiedFlow, model, device, disc_steps_list, save_dir, epoch, time_steps=200):
    gamma_results = {}
    # Generate samples and the expected norm suqared of d model(x_t, t)/ dt on a fine grid of points
    expected_dmodel_dt_norm_sq_vec, generated_samples = rect_flow.d_model_dt_norm_sq(model, n_samples=1000, time_steps=time_steps)
    plot_generated_images(generated_samples, epoch, disc_steps=time_steps, save_dir=save_dir)

    for disc_steps in tqdm(disc_steps_list):
        # compute the gamma parameter
        gamma = rect_flow.gamma_st(expected_dmodel_dt_norm_sq_vec, time_steps=time_steps, disc_steps=disc_steps)
        gamma_results[disc_steps] = gamma
        print(f"Gamma for disc_steps={disc_steps}: {gamma}")
    
    # Save gamma results to the specified directory
    gamma_save_path = os.path.join(save_dir, f"gamma_results_epoch_{epoch}.json")
    log_results(gamma_results, gamma_save_path)
    print(f"Gamma results saved to {gamma_save_path}")
    
    return gamma_results

def calculate_wasserstein_distance(rect_flow: RectifiedFlow, model, test_loader, device, disc_steps_list, save_dir, epoch, seed, use_saved):
    wasserstein_results = {}
    criterion = SamplesLoss("sinkhorn", p=2, blur=0.01)

    wasserstein_save_path = os.path.join(save_dir, f"wasserstein_results_epoch_{epoch}.json")
    if use_saved and os.path.exists(wasserstein_save_path):
        print(f"loading file at {wasserstein_save_path}")
        with open(wasserstein_save_path, "r") as f:
            wasserstein_results = json.load(f)
        return wasserstein_results
    
    for disc_steps in tqdm(disc_steps_list):
        # Generate samples using the sample_and_compute_derivative function
        generated_samples = rect_flow.sample(model, n=2000, disc_steps=disc_steps, seed=seed)
        plot_generated_images(generated_samples, disc_steps, save_dir, epoch)
        # Flatten generated samples
        generated_samples = generated_samples.view(len(generated_samples), -1).to(device)

        test_images = []
        for images, _ in test_loader:
            test_images.append(images)
        
        # Concatenate test images into one tensor
        test_images = torch.cat(test_images).to(device)
        test_images = test_images.view(len(test_images), -1)  # Flatten test images

        # Compute Wasserstein distance between the test set and generated samples
        wasserstein_distance = criterion(generated_samples, test_images).item()
        wasserstein_results[disc_steps] = wasserstein_distance
        print(f"Wasserstein distance for disc_steps={disc_steps}: {wasserstein_distance}")

    # Save Wasserstein results to the specified directory
    log_results(wasserstein_results, wasserstein_save_path)
    print(f"Wasserstein results saved to {wasserstein_save_path}")
    
    return wasserstein_results

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Train UNet on CIFAR-10, CIFAR-100, or CelebA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'CelebA', "FashionMNIST", "MNIST"], help='Dataset to train on (CIFAR10, CIFAR100, or CelebA)')
    parser.add_argument('--device', type=int, required=True, default=1, help='GPU device')
    parser.add_argument('--num_classes', nargs='+', type=int, default=None, help='Number of classes to include in training (e.g. 0 to 9 for CIFAR10)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='Classes to include in training (e.g. 0 1 2 for CIFAR)')
    parser.add_argument('--attributes', type=json.loads, default=None, help='Attributes to filter CelebA dataset (e.g., {"Blond_Hair": 1, "Male": 0})')
    parser.add_argument('--seed', type=int, default=0, help='Number of classes to include in training (e.g. 0 to 9 for CIFAR10)')
    parser.add_argument('--epoch', type=int, default=30, help='upto which epoch should the models be tested?')
    parser.add_argument('--use_saved', type=int, default=0, help='use saved results?')
    
    args = parser.parse_args()
    seed = args.seed
    gamma_results = {}
    wasserstein_results = {}
    args.num_classes = [20, 50, 70, None]
    ATTRIBUTES = [None] # [{"Blond_Hair": 1,  "Male": 0},  {"Blond_Hair": 1}] 
    for args.attributes in ATTRIBUTES:
        for num_classes in args.num_classes:
        # Base save directory
            base_save_dir = "/datastor1/vansh/rectified-flow/saved/{}".format(args.dataset)

            if args.dataset=="CelebA":
                img_size = 32
            else:
                img_size = 32

            if args.dataset=="MNIST" or args.dataset=="FashionMNIST":
                c_in = 1
            else:
                c_in = 3
            
            if num_classes is not None:
                selected_classes = np.arange(num_classes)
            else:
                selected_classes = args.classes
                
            # Create save directory based on classes or attributes
            save_dir = create_save_dir(base_save_dir, selected_classes=selected_classes, selected_attributes=args.attributes)

            device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
            device = torch.device(device_name)
            print('Using {} device'.format(device))
            print("Selected classes: ", "all" if num_classes==None and args.classes==None else selected_classes)
            print("Selected Attributes: ", args.attributes)
            dataloader = get_dataloader(base_save_dir, args.dataset, img_size, split="test", batch_size=args.batch_size, selected_classes=selected_classes, selected_attributes=args.attributes)
            if num_classes==None and args.classes==None:
                num_classes = "all"
            wasserstein_results[num_classes] = {}
            # EPOCHS = np.arange(10, args.epochs+1, 5)
            EPOCHS = [90, 100]
            for epoch in EPOCHS:
                model_path = os.path.join(save_dir, "models-2", f"unet_epoch_{epoch}.pth")
                print(model_path)
                model = load_model(model_path, device, c_in, img_size).to(device)
                rect_flow = RectifiedFlow(device, img_size, c_in)
                # Define the range of disc_steps to test
                disc_steps_list = list(range(1, 31, 1))

                # # Calculate gamma for each disc_steps
                # print("Calculating gamma values...")
                # gamma_results = calculate_gamma_for_disc_steps(rect_flow, model, device=device, disc_steps_list=disc_steps_list, save_dir=save_dir)

                # Calculate Wasserstein distance for each disc_steps
                print("Calculating Wasserstein distances...")
                wasserstein_results[num_classes][epoch] = calculate_wasserstein_distance(rect_flow, 
                                                                                        model, 
                                                                                        dataloader, 
                                                                                        device=device, 
                                                                                        disc_steps_list=disc_steps_list, 
                                                                                        save_dir=save_dir, 
                                                                                        epoch=epoch, 
                                                                                        seed=seed, 
                                                                                        use_saved=bool(args.use_saved))

            # Plot and save the results
            plot_wasserstein_gamma(gamma_results, wasserstein_results, save_dir, num_classes, 0)

        for epoch in EPOCHS:
            plot_wasserstein_gamma(gamma_results, wasserstein_results, save_dir, 0, epoch)