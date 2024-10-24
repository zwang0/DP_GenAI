import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from geomloss import SamplesLoss

device_id = 1
device_name = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(device_name)
print('Using {} device'.format(DEVICE))

def generate_mixture_samples(means, covariances, pi, num_samples, seed=10):
    """
    Generate samples from a mixture of Gaussians.

    Parameters:
    - means: (K, d) array of means for K Gaussian components
    - covariances: (K, d, d) array of covariance matrices for K Gaussian components
    - weights: (K,) array of weights for K Gaussian components
    - num_samples: Total number of samples to generate

    Returns:
    - X_np: (num_samples, d) array of generated samples
    """
    K = len(pi)  # Number of Gaussian components
    d = means.shape[1]  # Dimensionality of the data
    np.random.seed(seed)
    # Determine the number of samples from each component
    num_samples_component = np.random.multinomial(num_samples, pi)

    # Generate the samples
    samples = []
    for k in range(K):
        component_samples = np.random.multivariate_normal(means[k], covariances[k], num_samples_component[k])
        samples.append(component_samples)
    # Combine samples and shuffle
    X_np = np.vstack(samples)
    np.random.shuffle(X_np)
    # print(X_np.shape)
    return X_np

def compute_weights(X, mu, sigma2, pi):
    # Calculate squared distances between points and means
    squared_distances = np.sum(-((X[:, np.newaxis, :] - mu[np.newaxis, :, :])**2)/sigma2[np.newaxis,:,:], axis=2)
    # Apply the log-sum-exp trick
    max_neg_distance = np.max(squared_distances, axis=1, keepdims=True)
    stabilized_exp_distances = np.exp(squared_distances - max_neg_distance)

    weighted_stabilized_exp = pi.T * stabilized_exp_distances
    # Compute the normalization term using the stabilized distances
    sum_weighted_stabilized_exp = np.sum(weighted_stabilized_exp, axis=1, keepdims=True)

    # Normalize to get weights
    weights = weighted_stabilized_exp / sum_weighted_stabilized_exp

    return weights

def compute_scores(X, t, mu_0, mu_1, sigma2_0, sigma2_1, pi):
    K_0, d = mu_0.shape[0], mu_0.shape[1]
    K_1 = mu_1.shape[0]
    K = K_0 * K_1
    assert mu_0.shape[1]==mu_1.shape[1]
    Mu_0 = np.repeat(mu_0, K_1, axis=0)
    Mu_1 = np.tile(mu_1, (K_0, 1))

    Sigma2_0 = np.repeat(sigma2_0, K_1, axis=0)
    Sigma2_1 = np.tile(sigma2_1, (K_0, d))

    mu_t = t*Mu_1 + (1-t)*Mu_0
    sigma2_t = ((t**2)*Sigma2_1 + ((1-t)**2)*Sigma2_0)
    W_t = compute_weights(X, mu_t, sigma2_t, pi)
    diff = (mu_t[np.newaxis, :, :] - X[:, np.newaxis, :]) / sigma2_t[np.newaxis, :, :]  # Shape (n, k, d)

    # Multiply by W_t and sum along axis 1 (k)
    score_t = np.sum(W_t[:, :, np.newaxis] * diff, axis=1)  # Shape (n, d)

    return score_t, W_t

def drift(X, t, mu_0, mu_1, sigma2_0, sigma2_1, pi):
    s_t, W_t = compute_scores(X, t, mu_0, mu_1, sigma2_0, sigma2_1, pi)

    if t==0.0:
        v_t = pi.T@mu_1 - X
    elif t==1.0:
        v_t = X - np.mean(mu_0, axis=0)
    else:
        v_t = X/t + ((1-t)/t) * sigma2_0 * s_t
    return v_t, W_t

def sample_rectified_flow(mu_0, mu_1, Sigma_0, Sigma_1, pi_0, pi_1, T, num_samples=None, Z_0=None, seed=100):
    # Generate initial samples
    K_0, d = mu_0.shape[0], mu_0.shape[1]
    cov_0 = np.repeat(Sigma_0*np.eye(d)[np.newaxis, :, :], K_0, axis=0)
    # print(cov_0.shape)
    if Z_0 is None:
        if num_samples is None:
            num_samples = 1000
        Z_0 = generate_mixture_samples(mu_0, cov_0, pi_0[:, 0], num_samples, seed)
    num_samples = Z_0.shape[0]
    K_1 = mu_1.shape[0]
    # Initialize trajectory and drift_save
    trajectory = np.zeros((T+1, num_samples, Z_0.shape[1]))
    drift_save = np.zeros((T+1, num_samples, Z_0.shape[1]))
    weight_save = np.zeros((T+1, num_samples, K_1))
    trajectory[0] = Z_0

    # Initial drift computation
    t_0 = 0
    drift_save[0], weight_save[0]  = drift(Z_0, t_0, mu_0, mu_1, Sigma_0, Sigma_1, pi_1)
    # print(trajectory[0].shape)
    for i in range(1, T+1):
        t_i = i/T
        trajectory[i] = trajectory[i-1] + drift_save[i-1]/T
        drift_save[i], weight_save[i] = drift(trajectory[i], t_i, mu_0, mu_1, Sigma_0, Sigma_1, pi_1)

    return trajectory, drift_save, weight_save

# def plot_trajectories(trajectory, plot_paths=True, plot_original=True, t=None):
#     trajectory_np = trajectory
#     (T, n, d) = trajectory.shape
#     if t is None: t = T-1
#     diff = trajectory[T-1, :, :] - trajectory[0, :, :]
#     green = 0
#     red = 0
#     plt.figure(1)
#     for i in range(n):
#         particle_trajectory = trajectory_np[:t, i, :]
#         if diff[i, 0]>0:
#             if diff[i, 1]>0:
#                 color = "red"
#                 red = red + 1
#             else: color = "yellow"
#         else:
#             if diff[i, 1]<0:
#                 color = "purple"
#             else:
#                 green = green +1
#                 color = "green"
#         plt.scatter(trajectory_np[0, i, 0], trajectory_np[0, i, 1], alpha=0.2, color=color)
#         if plot_paths: plt.plot(particle_trajectory[:, 0], particle_trajectory[:, 1], alpha=0.1, color=color)
#         # print("green, red: ",green/n, red/n)
# #    plt.scatter(trajectory_np[T-1, :, 0], trajectory_np[T-1, :, 1], label="End points", color="orange")
#     plt.scatter(trajectory_np[t, :, 0],   trajectory_np[t, :, 1], label="End points", color="orange")
#     if plot_original: plt.scatter(X_np[:, 0], X_np[:, 1], label="Original data points", color="blue", alpha=0.05)
#     # plt.xlim((-10, 10))
#     # plt.ylim((-10, 10))
#     plt.title('Trajectories of particles over time, T={}, time={}'.format(T-1, np.round(t/T, 2)))
#     plt.legend()
#     plt.show()
#     plt.close()
#     return

def plot_weights(weights, trajectory, j):
    indices = np.where(trajectory[0, :, 1]>0)[0]

    weights_np = weights[:, indices, :]
    (T, n, K) = weights_np.shape
    time = np.arange(T)/T
    plt.figure(1)
    for i in range(n):
        particle_weights = weights_np[:, i, j]
        plt.plot(time, particle_weights, alpha=0.1, color="red")
    # plt.xlim((-10, 10))
    # plt.ylim((-10, 10))
    plt.title('Weights w_{} of particles (y>0) over time, T={}'.format(j+1, T-1))
    plt.legend()
    plt.show()
    plt.close()
    return

criterion = SamplesLoss("sinkhorn", p=2, blur=0.00001)
wasserstein_distances = {}
pot_wasserstein_distances = {}
device_name = f'cuda:{1}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print('Using {} device'.format(device))

# Define the vectorized version of w_i(x_t, t) using log_sum_exp trick
def w_i_vectorized(x_t, t, mu_list):
    norm_term = (1 - t).unsqueeze(-1) ** 2 + t.unsqueeze(-1) ** 2  # Shape: [batch_size, 1]
    
    # Compute the norms of (x_t - t * mu_list) for all mu_i's at once
    diffs = x_t.unsqueeze(1) - t.unsqueeze(-1).unsqueeze(-1) * mu_list  # Shape: [batch_size, 2, 2]
    norms_sq = torch.norm(diffs, dim=-1) ** 2  # Shape: [batch_size, 2]
    
    # Compute the exponent terms (exponent before normalization)
    exps = -norms_sq / (2 * norm_term)  # Shape: [batch_size, 2]
    
    # Use log_sum_exp trick for the denominator
    max_exp = torch.max(exps, dim=1, keepdim=True)[0]  # Shape: [batch_size, 1]
    log_sum_exp_denominator = max_exp + torch.log(torch.sum(torch.exp(exps - max_exp), dim=1, keepdim=True))
    
    # Compute the softmax weights (w_i)
    w = torch.exp(exps - log_sum_exp_denominator)  # Shape: [batch_size, 2]
    
    return w  # Shape: [batch_size, 2]

# Define the main function v(x_t, t)
def v_vectorized(x_t, t, mu_list):
    norm_term = (1 - t).unsqueeze(-1) ** 2 + t.unsqueeze(-1) ** 2  # Shape: [batch_size, 1]
    
    # First term: (2t - 1) * x_t / norm_term
    term_1 = (2 * t - 1).unsqueeze(-1) * x_t / norm_term  # Shape: [batch_size, 2]
    
    # Second term: w_i * mu_i and sum over all i
    w = w_i_vectorized(x_t, t, mu_list)  # Weights for each mu_i, shape: [batch_size, 2]
    weighted_sum = torch.sum(w.unsqueeze(-1) * mu_list, dim=1)  # Shape: [batch_size, 2]
    
    term_2 = (1 - t).unsqueeze(-1) * weighted_sum / norm_term  # Shape: [batch_size, 2]
    
    # v(x_t, t) = term_1 + term_2
    return term_1 + term_2  # Shape: [batch_size, 2]

# Compute the Jacobian of v(x_t, t) with respect to x_t
def jacobian(y, x):
    """ Compute the Jacobian of y with respect to x """
    jac = []
    for i in range(y.shape[1]):  # Iterate over the output dimensions
        grad_y = torch.zeros_like(y)
        grad_y[:, i] = 1  # One-hot vector for the i-th output component
        jac_i = torch.autograd.grad(y, x, grad_outputs=grad_y, retain_graph=True, create_graph=True)[0]
        jac.append(jac_i)
    return torch.stack(jac, dim=-1)  # Stack along the last dimension

# Compute total derivative using autograd
def compute_total_derivative_vectorized(x_t, t, mu_list):
    # Enable gradients for t and x_t
    x_t = x_t.requires_grad_(True)
    t = t.requires_grad_(True)

    # Compute v(x_t, t)
    v_value = v_vectorized(x_t, t, mu_list)  # Shape: [batch_size, 2]
    
    # Compute partial derivative w.r.t. t (treating x_t as constant)
    v_t = []
    for i in range(v_value.shape[1]):  # Iterate over the output dimensions of v
        grad_outputs = torch.zeros_like(v_value)
        grad_outputs[:, i] = 1
        v_t_i = torch.autograd.grad(v_value[:, i], t, grad_outputs=grad_outputs[:, i], create_graph=True)[0]
        v_t.append(v_t_i)
    
    v_t = torch.stack(v_t, dim=1)  # Shape: [batch_size, 2]
    
    # Compute Jacobian w.r.t. x_t (should be of shape [batch_size, 2, 2])
    v_x = jacobian(v_value, x_t)  # Shape: [batch_size, 2, 2]

    # Compute total derivative: dv/dt = dv/dt + (dv/dx_t) * (dx_t/dt)
    # Multiply the Jacobian by v_value, keeping the dimensions aligned
    total_derivative = v_t + torch.einsum('bij,bi->bj', v_x, v_value)  # Shape: [batch_size, 2]

    return total_derivative


def d_model_dt_norm_sq(trajectory, mu_list):

    """
    Returns:
        expected_norm_squared_vector: A vector of the expected norm squared of the full derivative at each time step (shape: [time_steps]).
    """
    (time_steps, n, _) = trajectory.shape
    expected_norm_squared = torch.zeros(time_steps)
    for i in range(time_steps):
        x_t = trajectory[i, :, :]
        t = torch.ones(n)*i/time_steps
        # 1. Compute the expectation of the norm squared of the full derivative with gradient
        norm_squared = torch.sum(compute_total_derivative_vectorized(x_t, t, mu_list)**2, dim=1)

        # Average the norm squared over the batch and store it
        expected_norm_squared[i] = norm_squared.mean().item()

    return expected_norm_squared

def gamma_st(expected_norm_squared_vector, disc_steps):
    # times_steps = 5000, disc_steps = 10, fine_step_size = 500, t_0 = 0, t_1 = 500
    time_steps = expected_norm_squared_vector.size(0)
    T = disc_steps
    gamma = 0

    fine_step_size = time_steps // T
    # print(fine_step_size)Anju
    for j in range(T):
        start_idx = j * fine_step_size
        end_idx = (j + 1) * fine_step_size - 1

        norm_squared_avg = sum(expected_norm_squared_vector[start_idx:end_idx]) / time_steps

        # Compute the integral approximation and update gamma
        integral_approx = time_steps * norm_squared_avg / fine_step_size # This approximates ∫ norm^2 dt over [t_j, t_{j+1}]
        gamma = max(gamma, integral_approx)  # Maximize over j

    return gamma

from math import sqrt

mu_0 = np.array([[0, 0]])
K_0, d = mu_0.shape[0], mu_0.shape[1]
Sigma_0 = np.array([1.0]).reshape(K_0, 1)

pi_0 = np.ones((K_0, 1))/K_0
# plt.scatter(X_test[:, 0], X_test[:, 1])
# plt.title("Test dist")
# plt.show()
# plt.close()
disc_steps_list = np.arange(1, 2001)
mu_dict = {1: np.array([[7/sqrt(2), 5/sqrt(2)]]), 
           2: np.array([[2, 5/sqrt(2)], [-2, 5/sqrt(2)]]), 
           3: np.array([[7/sqrt(2), 5/sqrt(2)], [-9/sqrt(2), 5/sqrt(2)], [-5/sqrt(2), -2/sqrt(2)]]),
           4: np.array([[7/sqrt(2), 5/sqrt(2)], [-9/sqrt(2), 5/sqrt(2)], [-5/sqrt(2), -2/sqrt(2)], [5/sqrt(2), -8/sqrt(2)]])}
wasserstein_distances = {}
gamma = {}

exp_norm_dot_vt = {}
for K_1 in range(1, 5):
    wasserstein_distances[K_1] = {}
    gamma[K_1] = {}

    mu_1 = mu_dict[K_1]
    K_1, d = mu_1.shape[0], mu_1.shape[1]
    # print(mu_1, K_1)
    Sigma_1 = np.ones(K_1).reshape(K_1, 1)
    pi_1 = np.ones((K_1, 1))/K_1
    cov1 = Sigma_1[:, :, np.newaxis] * np.eye(d)
    X_test = generate_mixture_samples(mu_1, cov1, pi_1.flatten(), num_samples=5000)

    trajectory, _, _ = sample_rectified_flow(mu_0, mu_1, Sigma_0, Sigma_1, pi_0, pi_1, T=20000, seed=0, num_samples=5000)
    expected_norm_squared_vector = d_model_dt_norm_sq(torch.from_numpy(trajectory), torch.from_numpy(mu_1))
    exp_norm_dot_vt[K_1] = expected_norm_squared_vector
    # plt.scatter(X_test[:, 0], X_test[:, 1])
    # plt.title("Test dist")
    # plt.show()
    # plt.close()
    for disc_steps in disc_steps_list:

        if (disc_steps-1)%100==0:
            print(disc_steps, K_1)
        trajectory, _, _ = sample_rectified_flow(mu_0, mu_1, Sigma_0, Sigma_1, pi_0, pi_1, T=disc_steps, seed=0, num_samples=5000)
        # plt.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1])
        # plt.title(f"disc_steps: {disc_steps}")
        # plt.show()
        # plt.close()
        gamma[K_1][disc_steps] = gamma_st(expected_norm_squared_vector, disc_steps)
        wasserstein_distances[K_1][disc_steps] = criterion(torch.from_numpy(trajectory[-1, :, :]).to(device), torch.from_numpy(X_test).to(device)).cpu().numpy()
    
    path = "/datastor1/vansh/rectified-flow/saved/gaussian-mixture/"
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "wasserstein_distances.npz"), wasserstein_distances=wasserstein_distances)
    np.savez(os.path.join(path, "gamma.npz"), gamma=gamma)
    np.savez(os.path.join(path, "expected_norm_dot_vt.npz"), exp_norm_dot_vt=exp_norm_dot_vt)

    dist = np.array(list(wasserstein_distances[K_1].values()))

    # mins.append(disc_steps_list[np.argmin(dist)])
    # print(disc_steps_list[np.argmin(dist)])
    plt.plot(disc_steps_list, np.log(dist), marker="o", label=f"{K_1} mixture components")
    plt.legend()
    plt.savefig(os.path.join(path, f"log-wasserstein-vs-T-K_1-{K_1}.png"))
    plt.close()

    dist = np.array(list(gamma[K_1].values()))
    plt.plot(disc_steps_list, (dist), marker="o", label=f"{K_1} mixture components")
    plt.legend()
    plt.savefig(os.path.join(path, f"gamma-vs-T-K_1-{K_1}.png"))
    plt.close()

    plt.plot(disc_steps_list, (dist)/np.square(disc_steps_list), marker="o", label=f"{K_1} mixture components")
    plt.legend()
    plt.savefig(os.path.join(path, f"gamma_bt_T2-vs-T-K_1-{K_1}.png"))
    plt.close()

for K_1 in range(1, 5):
    dist = np.array(list(wasserstein_distances[K_1].values()))
    plt.plot(disc_steps_list, np.log(dist), marker="o", label=f"{K_1} mixture components")
    plt.legend()
plt.savefig(os.path.join(path, "log-wasserstein-vs-T.png"))
plt.close()

for K_1 in range(1, 5):
    dist = np.array(list(gamma[K_1].values()))
    plt.plot(disc_steps_list, (dist), marker="o", label=f"{K_1} mixture components")
    plt.legend()
plt.savefig(os.path.join(path, "gamma-vs-T.png"))
plt.close()

for K_1 in range(1, 5):
    dist = np.array(list(gamma[K_1].values()))
    plt.plot(disc_steps_list, (dist)/np.square(disc_steps_list), marker="o", label=f"{K_1} mixture components")
    plt.legend()
plt.savefig(os.path.join(path, "gamma_bt_T2-vs-T.png"))
plt.close()