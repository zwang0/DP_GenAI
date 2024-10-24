import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

def postprocess(x):
    x = (x + 1)/2
    x = (x * 255).type(torch.uint8)
    return x

def create_save_dir(save_dir, selected_classes=None, num_classes = None, selected_attributes=None, exist_ok=True):
    """
    Create a directory for saving models and results, with subdirectory names reflecting
    the specific classes or attributes being used for training.
    """
    # Append class information if training on CIFAR-10/100 with specific classes
    if selected_classes is not None:
        class_str = "_".join(map(str, selected_classes))  # Convert list of classes to a string
        save_dir = os.path.join(save_dir, f"classes_{class_str}")

    # Append attribute information if training on CelebA with specific attributes
    elif selected_attributes is not None:
        attr_str = "_".join([f"{k}{v}" for k, v in selected_attributes.items()])  # Convert dict to a string
        save_dir = os.path.join(save_dir, f"attributes_{attr_str}")

    else:
        save_dir = os.path.join(save_dir, "classes_all")

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=exist_ok)

    return save_dir

def log_results(losses, log_dir):
    with open(os.path.join(log_dir), 'w') as f:
        json.dump(losses, f)

def plot_generated_images(gen_images, disc_steps, save_dir, epoch, num_images=100):
    """
    Plot and save a grid of generated images for visualization.
    
    Args:
        images (Tensor): A batch of images.
        disc_steps (int): The current disc_steps value (used in the filename).
        save_dir (str): The directory to save the image grid.
        num_images (int): Number of images to plot in the grid.
    """
    
    # Select a subset of images to plot
    images = gen_images[:num_images]
    images = postprocess(images)
    # Create a grid of images
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))  # 10x10 grid
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert to HWC format for plotting
        ax.axis('off')
    
    plot_dir = os.path.join(save_dir, "gen_images", f"epoch_{epoch}")
    os.makedirs(plot_dir, exist_ok=True)
    # Save the plot
    plt.tight_layout()
    plot_save_path = os.path.join(plot_dir, f"disc_steps_{disc_steps}.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Generated images saved to {plot_save_path}")

def plot_wasserstein_gamma(gamma_results, wasserstein_results, save_dir, num_classes, epoch):
    wasserstein_plot_dir = os.path.join(save_dir, "plots_wasserstein")
    os.makedirs(wasserstein_plot_dir, exist_ok=True)
    # gamma_plot_dir = os.path.join(save_dir, "gamma")
    plt.figure()

    if num_classes==0 and epoch!=0:
        plot_results = {}
        for c in wasserstein_results.keys():
            
            plot_results = wasserstein_results[c][epoch]
            vals = np.array(list(plot_results.values()))
            keys = np.array([int(t) for t in plot_results.keys()])
            # vals = vals - min(vals)
            if c=="all": c=10
            plt.plot(keys, np.square(vals), marker='o', label=f'num_labels = {c}')
            wasserstein_plot_path = os.path.join(wasserstein_plot_dir, f"wasserstein_plot_epoch_{epoch}.png")
    elif num_classes!=0 and epoch==0:
        plot_results = {}
        for e in wasserstein_results[num_classes].keys():
            plot_results = wasserstein_results[num_classes][e]
            vals = np.array(list(plot_results.values()))
            keys = np.array([int(t) for t in plot_results.keys()])
            # vals = vals - min(vals)
            plt.plot(keys, np.square(vals), marker='o', label=f'at epoch = {e}')
            wasserstein_plot_path = os.path.join(wasserstein_plot_dir, f"wasserstein_plot_num_labels_{num_classes}.png")
    # if gamma_results is not None:
    #     # Plot gamma vs T (disc steps) in normal scale
    #     plt.figure()
    #     plt.plot(list(gamma_results.keys()), list(gamma_results.values()), marker='o', label='Gamma')
    #     plt.xlabel('Disc Steps (T)')
    #     plt.ylabel('Gamma')
    #     plt.title('Gamma vs Disc Steps (T)')
    #     plt.legend()
    #     gamma_plot_path = os.path.join(save_dir, "gamma_plot.png")
    #     plt.savefig(gamma_plot_path)
    #     print(f"Gamma plot saved to {gamma_plot_path}")
    #     plt.close()

    
    # Plot Wasserstein vs T (disc steps) in log-log scale
    plt.xlabel('T', fontsize=20)
    plt.ylabel(r'$W_2^2$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=12) 
    # plt.title('Wasserstein Distance vs Disc Steps (T) [Log-Log]')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(wasserstein_plot_path)
    print(f"Wasserstein plot saved to {wasserstein_plot_path}")
    plt.close()