from typing import Tuple

import numpy as np

import torch
from torch import Tensor, IntTensor
from torch_geometric.utils import subgraph
from datatypes.dense import DenseGraph

from . import TimeSampler, NoiseSchedule, NoiseProcess
from src.noise.graph_diffusion import cosine_beta_schedule_discrete


class CosineDiffusionSchedule(torch.nn.Module, NoiseSchedule):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, max_time: int):
        super(CosineDiffusionSchedule, self).__init__()

        self.max_time = max_time

        # compute betas (parameter next)
        betas = cosine_beta_schedule_discrete(max_time)
        # clamp values as in the original paper
        betas = torch.clamp(torch.from_numpy(betas), min=0, max=0.999)
        self.register_buffer('betas', betas.float())

        # compute alpha = 1 - beta
        alphas = 1 - self.betas

        # recompute alpha_bar (parameter time 0->t)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.register_buffer('alphas_bar', torch.exp(log_alpha_bar))


    def params_next(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.betas[t_int]

    def params_time_t(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.alphas_bar[t_int]

    def params_posterior(self, t, **kwargs):
        raise NotImplementedError

    def get_max_time(self, **kwargs):
        return self.max_time
    

    def forward(self, t: Tensor, **kwargs):
        return self.params_next(t)

################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################

class ContinuousGaussianDiffusionProcess(NoiseProcess):

    def __init__(
            self,
            seed_dim: int,
            schedule : NoiseSchedule
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(schedule=schedule)

        # set seed dimension
        self.seed_dim = seed_dim

    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: Tensor, t: IntTensor, **kwargs) -> Tensor:
        
        return torch.randn_like(current_datapoint)


    def apply_noise_next(self, current_datapoint: Tensor, noise: Tensor, t: IntTensor, **kwargs) -> Tensor:
        raise NotImplementedError

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(self, original_datapoint: Tensor, t: IntTensor, **kwargs):

        return torch.randn_like(original_datapoint)


    def apply_noise_from_original(
            self,
            original_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        alpha_bar_t = self.get_params_from_original(t, **kwargs)

        return torch.sqrt(alpha_bar_t) * original_datapoint + torch.sqrt(1 - alpha_bar_t) * noise
    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(self, original_datapoint: Tensor, current_datapoint: Tensor, t: Tensor, **kwargs) -> Tensor:

        return torch.randn_like(original_datapoint)
    
    
    def apply_noise_posterior(self, original_datapoint: Tensor, current_datapoint: Tensor, noise: Tensor, t: Tensor, **kwargs) -> Tensor:

        alpha_bar_t = self.get_params_from_original(t, **kwargs)

        return torch.sqrt(alpha_bar_t) * original_datapoint + torch.sqrt(1 - alpha_bar_t) * noise

        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_TIMESAMPLER_UNIFORM = 'uniform'

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_DISCRETE = 'discrete_uniform'

def resolve_diffusion_timesampler(name: str) -> type:
    if name == DIFFUSION_TIMESAMPLER_UNIFORM:
        return UniformTimeSampler
    else:
        raise ContinuousDiffusionProcessException(f'Could not resolve diffusion time sampler name: {name}')

def resolve_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise ContinuousDiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_DISCRETE:
        return DiscreteUniformDiffusionProcess
    else:
        raise ContinuousDiffusionProcessException(f'Could not resolve diffusion process name: {name}')