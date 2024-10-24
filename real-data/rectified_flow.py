import torch
import torch.nn as nn
from tqdm import tqdm

from straightness import compute_norm_squared_full_derivative

class RectifiedFlow():
    def __init__(self, device, img_size=32, c_in=3):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.c_in = c_in

    def noise_images(self, x, t):
        z = torch.randn_like(x)
        tt1 = (1-t)[:, None, None, None]
        tt2 = t[:, None, None, None]
        return tt1 * z + tt2 * x, z
    
    def sample_timesteps(self, n):
        return torch.rand(n)
    
    def d_model_dt_norm_sq(self, model, n_samples=1024, time_steps=1000, batch_size=128):
        """
        Samples n new images using the diffusion model and, at each time step, estimates the expectation of
        the norm squared of the full derivative of model(x_t, t) with respect to t.

        Args:
            model: The trained diffusion model.
            n_samples: The number of samples to generate at each time step.
            time_steps: The number of fine-grained time steps for the sampling process.
            batch_size: Batch size for sampling.

        Returns:
            expected_norm_squared_vector: A vector of the expected norm squared of the full derivative at each time step (shape: [time_steps + 1]).
            x_new: The generated samples from the last batch.
        """
        model.eval()  # Ensure the model is in evaluation mode
        num_batches = n_samples // batch_size
        expected_norm_squared_vector = torch.zeros(time_steps)
        x_new_last_batch = None  # Placeholder to store the last batch of x_new

        for b in range(num_batches):
            n = batch_size
            # Initialize x with random noise
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)

            # Iterate over the time steps (starting from 1 since we already did t=0)
            for i in tqdm(range(0, time_steps)):
                # Compute the current time step value t
                t = (torch.ones(n) * i / time_steps).to(self.device)

                # 1. Compute the expectation of the norm squared of the full derivative with gradient
                norm_squared = compute_norm_squared_full_derivative(model, x, t)

                # Average the norm squared over the batch and store it
                expected_norm_squared = norm_squared.mean().item() / num_batches
                expected_norm_squared_vector[i] += expected_norm_squared

                # 2. Sample according to the model using torch.no_grad to prevent building computation graph
                x = x + model(x, t) / time_steps

                # If it's the last batch, store x_new
                if b == num_batches - 1 and i == time_steps:
                    x_new_last_batch = x.detach().cpu()

                # Clear unnecessary memory and cache
                del norm_squared, t  # Explicitly delete to free memory
                torch.cuda.empty_cache()

        return expected_norm_squared_vector.cpu().numpy(), x_new_last_batch

    
    def gamma_st(expected_norm_squared_vector, time_steps, disc_steps):
        # Now we compute gamma based on the coarser intervals of disc_steps (T steps)
        T = disc_steps
        gamma = 0

        # Divide the fine time steps into T discrete intervals
        fine_step_size = time_steps // T  # Number of fine steps per discrete step
        for j in range(T):
            # Define the start and end indices for this discrete interval
            start_idx = j * fine_step_size
            end_idx = (j + 1) * fine_step_size - 1

            # Approximate the integral over this interval using the average norm in the interval
            norm_squared_avg = sum(expected_norm_squared_vector[start_idx:end_idx]) / time_steps

            # Compute the integral approximation and update gamma
            integral_approx = norm_squared_avg / fine_step_size # This approximates ∫ norm^2 dt over [t_j, t_{j+1}]
            gamma = max(gamma, integral_approx)  # Maximize over j

        return gamma

    def sample(self, model, n, disc_steps=10, batch_size=128, seed=0):
        print(f"Sampling {n} new images in batches of {batch_size}")
        model.eval()
        samples = []

        # Set the seed if provided for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            for batch_start in tqdm(range(0, n, batch_size)):
                current_batch_size = min(batch_size, n - batch_start)

                # Change the seed for each batch to ensure different random starting points
                if seed is not None:
                    torch.manual_seed(seed + batch_start)  # Use batch_start to vary the seed

                # Sample the Gaussian noise
                x = torch.randn((current_batch_size, self.c_in, self.img_size, self.img_size)).to(self.device)

                # Iterate over the diffusion steps
                for i in range(disc_steps):
                    t = (torch.ones(current_batch_size) * i / disc_steps).to(self.device)
                    x += model(x, t) / disc_steps

                x.clamp(-1, 1)
                # Append the generated samples
                samples.append(x.cpu())

        return torch.cat(samples)








