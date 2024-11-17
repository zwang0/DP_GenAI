#############################################
## Rectified flow with DP gradient descent ##
#############################################

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.utils.data import DataLoader, TensorDataset, Sampler
import matplotlib.pyplot as plt
import copy, os
import scipy as sp
import ot
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from opacus.schedulers import _NoiseScheduler
from opacus.data_loader import DPDataLoader
from opacus import PrivacyEngine
from typing import Callable, Optional



fig_path = "trials/Nov_16_2024/test_fig/" # root path for saving figures
seed = 42 # set seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If multi-GPU.
np.random.seed(seed)

## generate distribution pi_0 and pi_1
D = 10.
M = D+5
VAR = 0.3

DOT_SIZE = 4
COMP_init = 1
COMP_target = 3
scaling_factors = [VAR, 3 * VAR,  VAR] # changes with COMP_Target
sample_size = 10000

pi_init = [1/COMP_init for i in range(COMP_init)]
pi_target = [1/COMP_target for i in range(COMP_target)]

initial_mix = Categorical(torch.tensor(pi_init))
initial_comp = MultivariateNormal(torch.tensor([0., 0.]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP_init)]))
initial_model = MixtureSameFamily(initial_mix, initial_comp)
samples_0 = initial_model.sample([sample_size])

target_mix = Categorical(torch.tensor(pi_target))
target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(5) / 2., - D / 2.], [-D * np.sqrt(7) / 2., - D / 2.], [0.0, D * np.sqrt(2) / 2.]]).float(),  torch.stack([factor * torch.eye(2) for factor in scaling_factors]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([sample_size])

# Variables for DP
DP_EPSILON = 3.0
DP_DELTA = 0.0
DP_MAX_GRAD_NORM = 1.0
DP_NOISE_MULTIPLIER = 1.1
PROB_ADD_NOISE = 0.3

# print('Shape of the samples:', samples_0.shape, samples_1.shape)
# plt.figure(figsize=(4,4))
# plt.xlim(-M,M)
# plt.ylim(-M,M)
# plt.title(r'Samples from $\pi_0$ and $\pi_1$')
# plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_0$')
# plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_1$')
# plt.legend()
# plt.savefig(fig_path + 'sample_from_pi_0_and_pi_1.png', dpi=500)

## Define the Flow Model
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
        # self.act = nn.ReLU()

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000, device=torch.device("cpu")):
        self.model = model
        self.N = num_steps
        self.device = device

    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1)).to(self.device)
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
          N = self.N
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = (torch.ones((batchsize,1)) * i / N).to(self.device)
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt

            traj.append(z.detach().clone())

        return traj

# class ProbNoiseScheduler(_NoiseScheduler):
#     def __init__(self, optimizer: DPOptimizer, *, prob_no_noise: float, last_epoch: int = -1):
#         """
#         DP noise scheduler that sets noise_multiplier to 0 with a given probability.
        
#         Args:
#             optimizer (Optimizer): Optimizer linked to the Privacy Engine.
#             noise_multiplier (float): Initial noise multiplier.
#             prob_no_noise (float): Probability of setting noise_multiplier to 0 (default: 0.3).
#         """
#         self.prob_no_noise = prob_no_noise
#         super().__init__(optimizer=optimizer, last_epoch=last_epoch)


#     def get_noise_multiplier(self):
#         uniform_output = np.random.uniform(0, 1)
#         print(f"Uniform Output: {uniform_output}")
#         if uniform_output < self.prob_no_noise:
#             return self.optimizer.noise_multiplier * 0.0  # No noise this step
#         else:
#             return self.optimizer.noise_multiplier


class ProbDPOptimizerFastGradientClipping(DPOptimizerFastGradientClipping):
    """
    :class:`opacus.optimizers.optimizer.DPOptimizer` compatible with
    add_noise in each epoch with prob
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        prob_add_noise: float,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        self.prob_add_noise = prob_add_noise
    
    def add_noise(self):
        # Noise only added when uniform output less than prob_add_noise
        uniform_output = np.random.uniform(0, 1)
        if uniform_output < self.prob_add_noise:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

class ProbDPOptimizer(DPOptimizer):
    """
    :class:`opacus.optimizers.optimizer.DPOptimizer` compatible with
    add_noise in each epoch with prob
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        use_original_optimizer: bool=True,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        self.use_original_optimizer = use_original_optimizer

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.use_original_optimizer:
            return self.original_optimizer.step()
        elif self.pre_step():
            return self.original_optimizer.step()
        else:
            return None

    # def add_noise(self):
    #     # Noise only added when uniform output less than prob_add_noise
    #     uniform_output = np.random.uniform(0, 1)
    #     if uniform_output < self.prob_add_noise:
    #         super().add_noise()
    #     else:
    #         for p in self.params:
    #             p.grad = p.summed_grad.view_as(p)


## Define training method  
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    for i in range(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize] # only one batch training per iter?
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)
        pred = rectified_flow.model(z_t, t)
        # loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        # loss = loss.mean()
        loss = nn.MSELoss(reduction='none')(pred, target).sum(dim=1).mean()
        loss.backward()

        optimizer.step()
        # loss_curve.append(np.log(loss.item())) ## to store the loss curve
        loss_curve.append(loss.item())

    return rectified_flow, loss_curve

def train_dp_rectified_flow(rectified_flow, optimizer, data_loader, batchsize, inner_iters):
    loss_curve = []
    ## UNUSED CODE
    # rectified_flow_MLP, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    #     module=rectified_flow.model,
    #     optimizer=optimizer,
    #     data_loader=data_loader,
    #     epcohs=inner_iters,
    #     target_epsilon=DP_EPSILON,
    #     target_delta=DP_DELTA,
    #     max_grad_norm=DP_MAX_GRAD_NORM,
    #     grad_sample_mode="ghost"
    # )

    # optimizer = ProbDPOptimizerFastGradientClipping( # dpoptimizer fast gradient clipping
    #     optimizer, # torch.optim.Optimizer
    #     noise_multiplier=DP_NOISE_MULTIPLIER,
    #     max_grad_norm=DP_MAX_GRAD_NORM,
    #     prob_add_noise=PROB_ADD_NOISE,
    #     # expected_batch_size=expected_batch_size
    #     expected_batch_size=batchsize
    # )
    ## UNUSED CODE END

    # if hasattr(rectified_flow.model._module, "autograd_grad_sample_hooks"):
    #     rectified_flow.model.remove_hooks() # remove hooks if add hooks
    rectified_flow.model.disable_hooks()
 
    # dp_prob_scheduler = ProbNoiseScheduler(optimizer=optimizer, prob_no_noise=PROB_NO_NOISE)

    for _ in range(inner_iters+1):
        optimizer.zero_grad()
        temp_z0, temp_z1 = next(iter(data_loader))
        indices = torch.randperm(len(temp_z0))[:batchsize]
        z0 = temp_z0[indices].detach().clone()
        z1 = temp_z1[indices].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

        if np.random.uniform(0, 1) < PROB_ADD_NOISE:
            # rectified_flow.model.add_hooks() # add hooks if add noise
            rectified_flow.model.enable_hooks()
            optimizer.use_original_optimizer = False

        pred = rectified_flow.model(z_t, t)
        # loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        # loss = loss.mean()
        loss = nn.MSELoss(reduction='none')(pred, target).sum(dim=1).mean()
        loss.backward()

        optimizer.step()

        # uniform_out = np.random.uniform(0, 1)
        # if uniform_out < PROB_NO_NOISE:
        #   optimizer.noise_multiplier = 0.0
        # else:
        #   optimizer.noise_multiplier = DP_NOISE_MULTIPLIER
        # dp_prob_scheduler.step()

        # loss_curve.append(np.log(loss.item())) ## to store the loss curve
        loss_curve.append(loss.item())

        # if hasattr(rectified_flow.model._module, "autograd_grad_sample_hooks"):
        #     rectified_flow.model.remove_hooks() # remove hooks if add hooks
        #     optimizer.use_original_optimizer = True
        rectified_flow.model.disable_hooks
        optimizer.use_original_optimizer = True
    
    return rectified_flow, loss_curve

## Misc. code for plotting figures
@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None, fig_path=fig_path):
    traj = rectified_flow.sample_ode(z0=z0, N=N)

    plt.figure(figsize=(4,4))
    plt.xlim(-M,M)
    plt.ylim(-M,M)

    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
    plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
    plt.legend()
    plt.title('Distribution')
    plt.savefig(fig_path + f' Distribution N={N}.png', dpi=500)

    traj_particles = torch.stack(traj)
    plt.figure(figsize=(4,4))
    plt.xlim(-M,M)
    plt.ylim(-M,M)
    plt.axis('equal')
    for i in range(50):
      plt.plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy())
    plt.title('Transport Trajectory')
    plt.savefig(fig_path + f' Transport Trajectory N={N}.png', dpi=500)


## One Rectified Flow
def train_rectified_flow_wrapper(z0, z1, prev_rectified_flow=None, prev_optimizer=None, reflow_idx=1, 
                                 dp=False, iterations=10000, batchsize=2048, input_dim=2, 
                                 fig_path=fig_path, device=torch.device("cpu")):
    z0 = z0.to(device)
    z1 = z1.to(device)
    z_pairs = torch.stack([z0, z1], dim=1).to(device)
    
    
    if prev_rectified_flow is not None: # not first rectified flow, using prev model
        rectified_flow = copy.deepcopy(prev_rectified_flow) # we fine-tune the model from previous Rectified Flow for faster training.
        if dp: 
            # # customized dpoptimizer
            optimizer = ProbDPOptimizer( 
                optimizer=prev_optimizer.original_optimizer,
                noise_multiplier=DP_NOISE_MULTIPLIER,
                max_grad_norm=DP_MAX_GRAD_NORM,
                expected_batch_size=batchsize,
                loss_reduction="mean",
                generator=None,
                secure_mode=False,
                use_original_optimizer=True
            )
            
            # dpdataloader
            tensor_dataset = TensorDataset(z_pairs[:, 0], z_pairs[:, 1])
            data_loader = DataLoader(tensor_dataset, batch_size=len(z_pairs), shuffle=True) # full batch
            data_loader = DPDataLoader.from_data_loader(
                data_loader, generator=None, distributed=False
            )
            rectified_flow, loss_curve = train_dp_rectified_flow(rectified_flow, optimizer, data_loader, batchsize, iterations)
        
        else:
            optimizer =  copy.deepcopy(prev_optimizer)
            rectified_flow, loss_curve = train_rectified_flow(rectified_flow, optimizer, z_pairs, batchsize, iterations)
    
    else: # first rectified flow
        rectified_flow = RectifiedFlow(model=MLP(input_dim, hidden_num=100).to(device), num_steps=100, device=device)
        optimizer = torch.optim.Adam(rectified_flow.model.parameters(), lr=5e-3)
        if dp:
            tensor_dataset = TensorDataset(z_pairs[:, 0], z_pairs[:, 1])
            data_loader = DataLoader(tensor_dataset, batch_size=len(z_pairs), shuffle=True) # full batch

            privacy_engine = PrivacyEngine()
            rectified_flow_MLP, _, data_loader = privacy_engine.make_private(
                module=rectified_flow.model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=DP_NOISE_MULTIPLIER,
                max_grad_norm=DP_MAX_GRAD_NORM,
                clipping="flat", # "flat" or "ghost"
                grad_sample_mode="hooks", # "hooks" , "ghost", "ew"
                batch_first=True,
                poisson_sampling=True,
            )
            optimizer = ProbDPOptimizer( # customized dpoptimizer
                optimizer=optimizer,
                noise_multiplier=DP_NOISE_MULTIPLIER,
                max_grad_norm=DP_MAX_GRAD_NORM,
                expected_batch_size=batchsize,
                loss_reduction="mean",
                generator=None,
                secure_mode=False,
                use_original_optimizer=True
            )
            rectified_flow.model = rectified_flow_MLP # GradSampleModule
            rectified_flow, loss_curve = train_dp_rectified_flow(rectified_flow, optimizer, data_loader, batchsize, iterations)
        
        else:
            rectified_flow, loss_curve = train_rectified_flow(rectified_flow, optimizer, z_pairs, batchsize, iterations)
    

    # plot training loss curve
    plt.figure()
    plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'{reflow_idx}-Rectified Flow Training Loss Curve')
    plt.savefig(fig_path+f'{reflow_idx}-Rectified Flow Training Loss Curve.png', dpi=500)
    # plot distribution and trajectory
    draw_plot(rectified_flow, z0=initial_model.sample([1000]).to(device), z1=samples_1.detach().clone(), N=100, fig_path=fig_path + f'{reflow_idx}-Rectified')
    draw_plot(rectified_flow, z0=initial_model.sample([1000]).to(device), z1=samples_1.detach().clone(), N=1, fig_path=fig_path + f'{reflow_idx}-Rectified')

    return rectified_flow, optimizer



num_rectified_flow = 2
iterations = 10000
batchsize = 32 # 2048
input_dim = 2
dp = True
verbose = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GW_dict = {"CG_GW": [], "PP_GW": [], "Entropic_GW": []}
GW_dict = {"CG_GW": []}

for i in range(1, num_rectified_flow+1):
    if dp: # create folder for each Rectified_Flow
        os.makedirs(fig_path+f'{i}-DP_Rectified_Flow', exist_ok=True)
        fig_path_reflow = fig_path+f'{i}-DP_Rectified_Flow/'
    else:
        os.makedirs(fig_path+f'{i}-Rectified_Flow', exist_ok=True)
        fig_path_reflow = fig_path+f'{i}-Rectified_Flow/'

    if i == 1: # init for the first rectified flow
        z_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
        z_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
        rectified_flow, optimizer = train_rectified_flow_wrapper(z_0, z_1, None, None, reflow_idx=i, iterations=iterations, dp=dp,
                                              batchsize=batchsize, input_dim=input_dim, fig_path=fig_path_reflow, device=device)
    else:
        rectified_flow, optimizer = train_rectified_flow_wrapper(z_0, z_1, rectified_flow, optimizer, reflow_idx=i, iterations=iterations, dp=dp,
                                              batchsize=batchsize, input_dim=input_dim, fig_path=fig_path_reflow, device=device)
    
    # generate z1 by simulating rectified flow
    z_0 = samples_0.detach().clone()
    traj = rectified_flow.sample_ode(z0=z_0.to(device), N=100)
    z_1 = traj[-1].detach().clone()
    
    # compute Gromov-Wasserstein distance of z_0 and z_1
    
    # scipy
    # w_dist_sp = sp.stats.wasserstein_distance_nd(z_0.cpu().numpy(), z_1.cpu().numpy())
    # print(f"Wasserstein-1 distance for {i}-Rectified_Flow Scipy: {w_dist_sp}" )

    # POT
    # cost_matrix = ot.dist(z_0.cpu().numpy(), z_1.cpu().numpy(), metric='euclidean')  # Compute cost matrix
    # w_distance = ot.emd2(ot.unif(len(z_0)), ot.unif(len(z_1)), cost_matrix)  # Exact Wasserstein distance
    # w_distance_sinkhorn = ot.sinkhorn2(ot.unif(len(z_0)), ot.unif(len(z_1)), cost_matrix, 1e-3)  # Regularized Wasserstein distance for large problems
    # print(f"Wasserstein-1 distance for {i}-Rectified_Flow: {w_distance}" )
    # print(f"Wasserstein-1 distance for {i}-Rectified_Flow sinkhorn: {w_distance_sinkhorn}" )

    # Gromov-Wasserstein https://pythonot.github.io/auto_examples/gromov/plot_gromov.html

    # since the size of samples is too large, pick 2000 datapoint randomly to compute the Wasserstein distance
    n_sub_samples = 2000
    subset_indices = torch.randperm(len(z_0))[:n_sub_samples]
    sub_samples_1 = samples_1.detach().clone()[subset_indices]
    sub_z_1 = z_1[subset_indices]
    C0 = sp.spatial.distance.cdist(sub_samples_1.cpu().numpy(), sub_samples_1.cpu().numpy())
    C1 = sp.spatial.distance.cdist(sub_z_1.cpu().numpy(), sub_z_1.cpu().numpy())
    C0 /= C0.max()
    C1 /= C1.max()
    p = ot.unif(n_sub_samples)
    q = ot.unif(n_sub_samples)
    # Conditional Gradient algorithm
    gw0, log0 = ot.gromov.gromov_wasserstein(
        C0, C1, p, q, 'square_loss', verbose=verbose, log=True)
    # # Proximal Point algorithm with Kullback-Leibler as proximal operator
    # gw, log = ot.gromov.entropic_gromov_wasserstein(
    #     C0, C1, p, q, 'square_loss', epsilon=0.1, solver="PPA", log=True, verbose=verbose, max_iter=20, tol=1e-07)
    # # Projected Gradient algorithm with entropic regularization
    # gwe, loge = ot.gromov.entropic_gromov_wasserstein(
    #     C0, C1, p, q, "square_loss", epsilon=0.1, solver="PGD", log=True, verbose=verbose, max_iter=20, tol=1e-07)

    GW_dict['CG_GW'].append(log0["gw_dist"])
    # GW_dict['PP_GW'].append(log["gw_dist"])
    # GW_dict['Entropic_GW'].append(loge["gw_dist"])

    print(f"{i}-Rectified Flow Complete")

    


# saveing GW dict
import pandas as pd
if dp:
    csv_filename = fig_path + "dp_gw_dist.csv"
    gw_fig_path = fig_path + "dp_gromov_wasserstein.png"
else:
    csv_filename = fig_path + "gw_dist.csv"
    gw_fig_path = fig_path + "gromov_wasserstein.png"

GW_df = pd.DataFrame(GW_dict)
GW_df.to_csv(csv_filename, index=False)

plt.figure(figsize=(8 , 6))
plt.plot(range(1, num_rectified_flow+1), GW_dict['CG_GW'], label='Conditional Gradient GW', marker='o', color='b')
# plt.plot(range(1, num_rectified_flow+1), GW_dict['PP_GW'], label='Proximal Pointt GW', marker='o', color='r')
# plt.plot(range(1, num_rectified_flow+1), GW_dict['Entropic_GW'], label='Projected Gradient GW', marker='o', color='g')
# plt.plot(GW_dict['CG_GW'], label='Conditional Gradient GW', marker='o', color='b')
# plt.plot(GW_dict['PP_GW'], label='Proximal Pointt GW', marker='o', color='r')
# plt.plot(GW_dict['Entropic_GW'], label='Projected Gradient GW', marker='o', color='g')
plt.legend()
plt.xlabel('Number of Rectified Flow')
plt.ylabel('Distance')
plt.title('Gromov-Wasserstein distance over Rectified Flow')
plt.savefig(gw_fig_path, dpi=500)
