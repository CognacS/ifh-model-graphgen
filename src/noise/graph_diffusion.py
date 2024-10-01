from typing import Tuple, Dict

import numpy as np

import torch
from torch import Tensor, IntTensor, BoolTensor
from torch_geometric.utils.to_dense_batch import to_dense_batch

from src.models.denoising.graph_transformer import DIM_X, DIM_E

from src.datatypes.dense import (
    DenseGraph,
    DenseEdges,
    get_bipartite_edge_mask_dense,
    get_edge_mask_dense
)

from src.noise import NoiseSchedule, NoiseProcess

################################################################################
#                              UTILITY FUNCTIONS                               #
################################################################################

class DiffusionProcessException(Exception):
    pass


def cosine_beta_schedule_discrete(max_steps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = max_steps + 1
    x = np.linspace(0, max_steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / max_steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = np.concatenate([np.ones(1), alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas.squeeze()


def time_to_long(t: Tensor, timesteps: int):
    if t.dtype == torch.long:
        return t

    elif t.dtype == torch.int:
        return t.long()
    
    elif t.dtype == torch.float:
        t_int = torch.round(t * timesteps)
        return t_int.long()
    
    else:
        raise DiffusionProcessException(
            f'Given time tensor t has wrong dtype: {t.dtype}. Should be long, integer or float in [0,1]'
        )


################################################################################
#                         DIFFUSION PROCESS SCHEDULES                          #
################################################################################

class CosineDiffusionSchedule(NoiseSchedule):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, max_time: int):
        super().__init__()

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

if False:
    def apply_noise_graph(
            datapoint: DenseGraph,
            ext_edge_adjmat: Tensor,
            ext_node_mask: Tensor,
            noise_graph: DenseGraph
        ) -> Tuple[DenseGraph, Tensor]:
        """
        adapted from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py
        """
        # apply noise to own nodes, edges, and edges leading to external graph
        prob_x = datapoint.x @ noise_graph.x									# (bs, nq, dx_out)
        prob_e = datapoint.edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1)	# (bs, nq, nq, de_out)
        prob_ext_e = ext_edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1)		# (bs, nq, nk, de_out)

        # dimensions
        bs, nq, nk, _ = prob_ext_e.shape

        #############  APPLY NOISE TO X  #############
        # Noise X
        # The masked rows should define probability distributions as well
        prob_x[~datapoint.node_mask] = 1 / prob_x.shape[-1]

        # Flatten the probability tensor to sample with multinomial
        prob_x = prob_x.reshape(bs * nq, -1)		# (bs * nq, dx_out)

        # Sample X
        x = prob_x.multinomial(1)	# (bs * nq, 1)
        x = x.reshape(bs, nq)		# (bs, nq)

        #############  APPLY NOISE TO E  #############

        # Noise E
        # The masked rows should define probability distributions as well
        inverse_edge_mask = ~get_edge_mask_dense(datapoint.node_mask)
        diag_mask = torch.eye(nq).unsqueeze(0).expand(bs, -1, -1)

        prob_e[inverse_edge_mask] = 1 / prob_e.shape[-1] # fake nodes
        prob_e[diag_mask.bool()] = 1 / prob_e.shape[-1] # self loops

        prob_e = prob_e.reshape(bs * nq * nq, -1)    # (bs * nq * nq, de_out)

        # Sample E
        edge_adjmat = prob_e.multinomial(1).reshape(bs, nq, nq)   # (bs, nq, nq)
        edge_adjmat = torch.triu(edge_adjmat, diagonal=1)
        edge_adjmat = edge_adjmat + torch.transpose(edge_adjmat, 1, 2)

        #########  APPLY NOISE TO EXTERNAL E  ########

        # Noise E
        # The masked rows should define probability distributions as well
        inverse_edge_mask = ~get_bipartite_edge_mask_dense(datapoint.node_mask, ext_node_mask) # (bs, nq, nk)

        prob_ext_e[inverse_edge_mask] = 1 / prob_ext_e.shape[-1]

        prob_ext_e = prob_ext_e.reshape(bs * nq * nk, -1)    # (bs * nq * nk, de_out)

        # Sample E
        ext_edge_adjmat = prob_ext_e.multinomial(1).reshape(bs, nq, nk)   # (bs, qn, nk)

        # prepare new graph
        datapoint = DenseGraph(
            x =				x,
            edge_adjmat =	edge_adjmat,
            y =				datapoint.y,
            node_mask =		datapoint.node_mask
        )

        return datapoint, ext_edge_adjmat


# adapted from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py

def compute_matmul_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph: DenseGraph
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    graph, ext_edges = datapoint
    
    # apply noise to own nodes, edges, and edges leading to external graph
    prob_x = graph.x @ noise_graph.x									# (bs, nq, dx_out)
    prob_e = graph.edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1)	# (bs, nq, nq, de_out)
    prob_ext_e = ext_edges.edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1)		# (bs, nq, nk, de_out)

    # prepare probability graph
    prob_graph = DenseGraph(
        x =				prob_x,
        edge_adjmat =	prob_e,
        y =				graph.y,
        node_mask =		graph.node_mask,
        edge_mask =     graph.edge_mask
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   prob_ext_e,
        edge_mask =     ext_edges.edge_mask
    )

    return (prob_graph, prob_ext_edges)


def compute_elementwise_graph(
        first_datapoint: Tuple[DenseGraph, DenseEdges],
        second_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    first_graph, first_ext_edges = first_datapoint
    second_graph, second_ext_edges = second_datapoint
    
    # apply noise to own nodes, edges, and edges leading to external graph
    prob_x = first_graph.x * second_graph.x							# (bs, nq, dx_out)
    prob_e = first_graph.edge_adjmat * second_graph.edge_adjmat		# (bs, nq, nq, de_out)
    prob_ext_e = first_ext_edges * second_ext_edges		# (bs, nq, nk, de_out)

    # prepare probability graph
    prob_graph = DenseGraph(
        x =				prob_x,
        edge_adjmat =	prob_e,
        y =				first_graph.y,
        node_mask =		first_graph.node_mask,
        edge_mask =     first_graph.edge_mask
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   prob_ext_e,
        edge_mask =     first_ext_edges.edge_mask
    )

    return (prob_graph, prob_ext_edges)


def compute_prob_s_t_given_0(
        X_t: Tensor,
        Qt: Tensor,
        Qsb: Tensor,
        Qtb: Tensor
    ):
    """Borrowed from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py"""

    X_t = X_t.flatten(start_dim=1, end_dim=-2)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out



def normalize_probability(
        x: Tensor,
        norm_x: Tensor
    ) -> Tensor:

    denominator = norm_x.sum(-1, keepdim=True)
    denominator[denominator == 0] = 1

    return x / denominator

def normalize_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        norm_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # gather all relevant tensors
    tuple_numerators = (datapoint[0].x, datapoint[0].edge_adjmat, datapoint[1].edge_adjmat)
    tuple_denominators = (norm_datapoint[0].x, norm_datapoint[0].edge_adjmat, norm_datapoint[1].edge_adjmat)

    # apply normalization
    tuple_normalized = [
        normalize_probability(*tup) for tup in zip(tuple_numerators, tuple_denominators)
    ]

    prob_graph = DenseGraph(
        x =				tuple_normalized[0],
        edge_adjmat =	tuple_normalized[1],
        y =				datapoint[0].y,
        node_mask =		datapoint[0].node_mask,
        edge_mask =     datapoint[0].edge_mask
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   tuple_normalized[2],
        edge_mask =     datapoint[1].edge_mask
    )

    return prob_graph, prob_ext_edges


def fill_out_prob_graph(
        prob_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    prob_graph, prob_ext_edges = prob_datapoint

    # dimensions
    bs, nq, nk, _ = prob_ext_edges.edge_adjmat.shape

    num_cls_x = prob_graph.x.shape[-1]
    num_cls_e = prob_graph.edge_adjmat.shape[-1]

    #############  APPLY NOISE TO X  #############
    # Noise X
    # The masked rows should define probability distributions as well
    prob_graph.x[~prob_graph.node_mask] = 1 / num_cls_x

    #############  APPLY NOISE TO E  #############

    # Noise E
    # The masked rows should define probability distributions as well
    diag_mask = torch.eye(nq, dtype=torch.bool, device=prob_graph.device).unsqueeze(0).expand(bs, -1, -1)

    prob_graph.edge_adjmat[~prob_graph.edge_mask] = 1 / num_cls_e # fake nodes
    prob_graph.edge_adjmat[diag_mask] = 1 / num_cls_e # self loops

    #########  APPLY NOISE TO EXTERNAL E  ########

    # Noise E
    # The masked rows should define probability distributions as well
    prob_ext_edges.edge_adjmat[~prob_ext_edges.edge_mask] = 1 / num_cls_e

    return (prob_graph, prob_ext_edges)


def sample_from_probabilities(
        prob_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    prob_graph, prob_ext_edges = prob_datapoint

    # dimensions
    bs, nq, nk, _ = prob_ext_edges.edge_adjmat.shape

    if nq > 0:
        ##############  SAMPLE NODES X  ##############
        # Flatten the probability tensor to sample with multinomial
        prob_x = prob_graph.x.reshape(bs * nq, -1)		# (bs * nq, dx_out)

        # Sample X
        x = prob_x.multinomial(1).reshape(bs, nq)	# (bs, nq)

        #########  SAMPLE INTERNAL EDGES E  ##########
        # Flatten the probability tensor to sample with multinomial
        prob_e = prob_graph.edge_adjmat.reshape(bs * nq * nq, -1)	# (bs * nq * nq, de_out)

        # Sample E
        edge_adjmat = prob_e.multinomial(1).reshape(bs, nq, nq)	# (bs, nq, nq)
        edge_adjmat = torch.triu(edge_adjmat, diagonal=1)
        edge_adjmat = edge_adjmat + torch.transpose(edge_adjmat, 1, 2)

        #########  SAMPLE EXTERNAL EDGES E  ##########
        if nk == 0:
            ext_edge_adjmat = torch.zeros(bs, nq, nk, dtype=torch.long, device=prob_ext_edges.device)
        else:
            # Flatten the probability tensor to sample with multinomial
            prob_ext_e = prob_ext_edges.edge_adjmat.reshape(bs * nq * nk, -1)	# (bs * nq * nk, de_out)

            # Sample E external
            ext_edge_adjmat = prob_ext_e.multinomial(1).reshape(bs, nq, nk)	# (bs, nq, nk)
    else:
        device = prob_ext_edges.device
        x = torch.zeros(bs, nq, dtype=torch.long, device=device)
        edge_adjmat = torch.zeros(bs, nq, nq, dtype=torch.long, device=device)
        ext_edge_adjmat = torch.zeros(bs, nq, nk, dtype=torch.long, device=device)

    #############  FORMAT AND RETURN  ############
    # prepare sampled graph and mask
    sampled_graph = DenseGraph(
        x =				x,
        edge_adjmat =	edge_adjmat,
        y =				prob_graph.y,
        node_mask =		prob_graph.node_mask,
        edge_mask=      prob_graph.edge_mask
    ).apply_mask()

    # apply mask to external edges
    sampled_ext_edges = DenseEdges(
        edge_adjmat =   ext_edge_adjmat,
        edge_mask =     prob_ext_edges.edge_mask
    ).apply_mask()

    return sampled_graph, sampled_ext_edges


def apply_noise_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph: DenseGraph
    ) -> Tuple[DenseGraph, DenseEdges]:
    
    # get transition probabilities
    prob_datapoint = compute_matmul_graph(
        datapoint=datapoint,
        noise_graph=noise_graph
    )

    # fill out probability graph
    prob_datapoint = fill_out_prob_graph(
        prob_datapoint=prob_datapoint
    )

    # sample from transition probabilities
    sampled_datapoint = sample_from_probabilities(
        prob_datapoint=prob_datapoint
    )

    return sampled_datapoint

if False:
    def apply_posterior_noise_graph(
            original_datapoint: Tuple[DenseGraph, Tensor, Tensor],
            current_datapoint: Tuple[DenseGraph, Tensor, Tensor],
            noise_graph_bar_t: DenseGraph,
            noise_graph_bar_t_1: DenseGraph,
            noise_graph_t: DenseGraph
        ) -> Tuple[DenseGraph, Tensor, Tensor]:

        
        # transpose noise_graph_t
        noise_graph_t.x = noise_graph_t.x.transpose(-2, -1)
        noise_graph_t.edge_adjmat = noise_graph_t.edge_adjmat.transpose(-2, -1)

        # compute left term
        prob_current_datapoint = compute_matmul_graph(
            datapoint=current_datapoint,
            noise_graph=noise_graph_t
        )

        # compute right term
        prob_original_datapoint = compute_matmul_graph(
            datapoint=original_datapoint,
            noise_graph=noise_graph_bar_t_1
        )

        # compute numerator
        prob_numerator = compute_elementwise_graph(
            first_datapoint=prob_current_datapoint,
            second_datapoint=prob_original_datapoint
        )

        # compute denominator
        prob_denominator = compute_matmul_graph(
            datapoint=original_datapoint,
            noise_graph=noise_graph_bar_t
        )
        prob_denominator = compute_elementwise_graph(
            first_datapoint=prob_denominator,
            second_datapoint=current_datapoint
        )

        # compute posterior probabilities
        prob_datapoint = normalize_graph(
            datapoint=prob_numerator,
            norm_datapoint=prob_denominator
        )

        # fill out probability graph
        prob_datapoint = fill_out_prob_graph(
            prob_datapoint=prob_datapoint
        )

        print('Final probs:')
        print(prob_datapoint[0].x[0])
        print(prob_datapoint[0].edge_adjmat[0])

        # sample from transition probabilities
        sampled_datapoint = sample_from_probabilities(
            prob_datapoint=prob_datapoint
        )

        return sampled_datapoint

def weight_and_normalize_distribution(
        dist: Tensor,
        weights: Tensor
    ) -> Tensor:
    weighted_prob = dist.unsqueeze(-1) * weights        # bs, N, d0, d_t-1
    unnormalized_prob = weighted_prob.sum(dim=-2)       # bs, N, d_t-1
    unnormalized_prob[torch.sum(unnormalized_prob, dim=-1) == 0] = 1e-5
    return unnormalized_prob / torch.sum(unnormalized_prob, dim=-1, keepdim=True) # bs, n, d_t-1
    

def apply_posterior_noise_graph(
        original_datapoint: Tuple[DenseGraph, DenseEdges],
        current_datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph_bar_t: DenseGraph,
        noise_graph_bar_t_1: DenseGraph,
        noise_graph_t: DenseGraph
    ) -> Tuple[DenseGraph, DenseEdges]:

    bs, nq, nk, de = current_datapoint[1].edge_adjmat.shape

    # compute weights for the parameterization
    p_s_and_t_given_0_X = compute_prob_s_t_given_0(
        X_t =   current_datapoint[0].x,
        Qt =    noise_graph_t.x,
        Qsb =   noise_graph_bar_t_1.x,
        Qtb =   noise_graph_bar_t.x
    )

    p_s_and_t_given_0_E = compute_prob_s_t_given_0(
        X_t =   current_datapoint[0].edge_adjmat,
        Qt =    noise_graph_t.edge_adjmat,
        Qsb =   noise_graph_bar_t_1.edge_adjmat,
        Qtb =   noise_graph_bar_t.edge_adjmat
    )

    p_s_and_t_given_0_E_ext = compute_prob_s_t_given_0(
        X_t =   current_datapoint[1].edge_adjmat,
        Qt =    noise_graph_t.edge_adjmat,
        Qsb =   noise_graph_bar_t_1.edge_adjmat,
        Qtb =   noise_graph_bar_t.edge_adjmat
    )

    # weight the original datapoint probability distribution
    prob_x = weight_and_normalize_distribution(
        dist =      original_datapoint[0].x,
        weights =   p_s_and_t_given_0_X
    )

    prob_e = weight_and_normalize_distribution(
        dist =      original_datapoint[0].edge_adjmat.reshape(bs, -1, de),
        weights =   p_s_and_t_given_0_E
    )
    prob_e = prob_e.reshape(bs, nq, nq, de)

    prob_e_ext = weight_and_normalize_distribution(
        dist =      original_datapoint[1].edge_adjmat.reshape(bs, -1, de),
        weights =   p_s_and_t_given_0_E_ext
    )
    prob_e_ext = prob_e_ext.reshape(bs, nq, nk, de)

    # create probability graph
    prob_datapoint = (
        DenseGraph(
            x=prob_x,
            edge_adjmat=prob_e,
            y=current_datapoint[0].y,
            node_mask=current_datapoint[0].node_mask,
            edge_mask=current_datapoint[0].edge_mask
        ),
        DenseEdges(
            edge_adjmat=prob_e_ext,
            edge_mask=current_datapoint[1].edge_mask
        )
    )

    # fill out probability graph
    prob_datapoint = fill_out_prob_graph(
        prob_datapoint=prob_datapoint
    )

    #print('Final probs:')
    #print(prob_datapoint[0].x[0])
    #print(prob_datapoint[0].edge_adjmat[0])
    #print(prob_datapoint[1][0])

    # sample from transition probabilities
    sampled_datapoint = sample_from_probabilities(
        prob_datapoint=prob_datapoint
    )

    return sampled_datapoint



def get_num_classes(graph: DenseGraph) -> Tuple[int, int]:
    """
    Parameters
    ----------
    graph : DenseGraph
        the graph to get the number of classes for
    """
    # get number of classes
    x_classes = graph.x.shape[-1]
    e_classes = graph.edge_adjmat.shape[-1]

    return x_classes, e_classes

class DiscreteUniformDiffusionProcess(NoiseProcess):

    def __init__(
            self,
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

        """
        # setup number of classes
        self.x_classes = x_classes
        self.e_classes = e_classes

        # pre-build uniform transition probabilities
        self.u_x = torch.ones(1, self.x_classes, self.x_classes)
        if self.x_classes > 0:
            self.u_x = self.u_x / self.x_classes

        self.u_e = torch.ones(1, self.e_classes, self.e_classes)
        if self.e_classes > 0:
            self.u_e = self.u_e / self.e_classes
        """

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            num_new_nodes: IntTensor,
            ext_node_mask: BoolTensor,
            num_classes: Dict[str, int]
        ) -> Tuple[DenseGraph, DenseEdges]:
        # num new nodes has shape (bs,), and each element
        # is the number of nodes the graph should have
        bs = len(num_new_nodes)
        max_num_nodes = num_new_nodes.max().item()
        max_ext_nodes = ext_node_mask.shape[1]

        # get number of classes
        x_classes, e_classes = num_classes[DIM_X], num_classes[DIM_E]

        # ext_node_mask has shape (bs, nk), where nk is the
        # max number of external nodes there are. Each element
        # is a boolean indicating whether the node is fake or not

        # get current device
        device = num_new_nodes.device

        # generate uniform sparse nodes and batch index
        #tot_num_nodes = int(num_new_nodes.sum().item())

        x = torch.randint(low=0, high=x_classes, size=(bs, max_num_nodes), device=device)
        node_mask = torch.arange(max_num_nodes, device=device) < num_new_nodes.unsqueeze(-1)

        # generate uniform edge adjmat, without self loops
        edge_adjmat = torch.randint(low=0, high=e_classes, size=(bs, max_num_nodes, max_num_nodes), device=device)
        edge_adjmat = torch.triu(edge_adjmat, diagonal=1)
        edge_adjmat = edge_adjmat + edge_adjmat.transpose(1, 2)
        edge_mask = get_edge_mask_dense(node_mask)

        # compose graph
        graph = DenseGraph(
            x =				x,
            edge_adjmat =	edge_adjmat,
            y =				None,
            node_mask =		node_mask,
            edge_mask =     edge_mask
        ).apply_mask()


        # generate uniform external edge adjmat
        ext_edge_adjmat = torch.randint(low=0, high=e_classes, size=(bs, max_num_nodes, max_ext_nodes), device=device)
        # mask out fake nodes
        ext_edge_mask = get_bipartite_edge_mask_dense(node_mask, ext_node_mask)
        
        ext_edges = DenseEdges(
            edge_adjmat =   ext_edge_adjmat,
            edge_mask =     ext_edge_mask
        ).apply_mask()

        return graph, ext_edges


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: Tuple[DenseGraph, DenseEdges], t: IntTensor, **kwargs):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de).
        """

        # get diffusion parameter
        beta_t: Tensor = self.get_params_next(t, **kwargs).unsqueeze(-1).unsqueeze(-1)

        # get current device
        graph, ext_edges = current_datapoint
        device = graph.device

        # get number of classes
        x_classes, e_classes = get_num_classes(graph)

        #beta_t = beta_t.to(device)
        #u_x = self.u_x.to(device)
        #u_e = self.u_e.to(device)

        # exact definition from the discrete diffusion paper
        q_x = (1 - beta_t) * torch.eye(x_classes, device=device).unsqueeze(0) + beta_t / x_classes
        q_e = (1 - beta_t) * torch.eye(e_classes, device=device).unsqueeze(0) + beta_t / e_classes

        transition_graph = DenseGraph(
            x=q_x,
            edge_adjmat=q_e,
            y=None
        )

        return transition_graph


    def apply_noise_next(
            self,
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: DenseGraph,
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:

        graph, ext_edges = apply_noise_graph(
            datapoint =			current_datapoint,
            noise_graph =		noise
        )

        return graph, ext_edges

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(self, original_datapoint: Tuple[DenseGraph, DenseEdges], t: IntTensor, **kwargs):
        
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de).
        """

        # get diffusion parameter
        alpha_bar_t: Tensor = self.get_params_from_original(t, **kwargs).unsqueeze(-1).unsqueeze(-1)

        # get current device
        graph, ext_edges = original_datapoint
        device = graph.device

        # get number of classes
        x_classes, e_classes = get_num_classes(graph)
        
        #alpha_bar_t = alpha_bar_t.to(device).unsqueeze(-1)
        #self.u_x = self.u_x.to(device)
        #self.u_e = self.u_e.to(device)


        # exact definition from the discrete diffusion paper
        q_x = alpha_bar_t * torch.eye(x_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) / x_classes
        q_e = alpha_bar_t * torch.eye(e_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) / e_classes

        transition_graph = DenseGraph(
            x=q_x,
            edge_adjmat=q_e,
            y=None
        )

        return transition_graph


    def apply_noise_from_original(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: torch.BoolTensor,
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:

        #print('Check 5:', isinstance(original_datapoint[0], DenseGraph))

        graph, ext_edges = apply_noise_graph(
            datapoint =			original_datapoint,
            noise_graph =		noise
        )

        #print('Check 6:', isinstance(graph, DenseGraph))

        return graph, ext_edges
    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseGraph, DenseGraph]:

        trans_graph_bar_t = self.sample_noise_from_original(original_datapoint, t, **kwargs)
        trans_graph_bar_t_minus_one = self.sample_noise_from_original(original_datapoint, t-1, **kwargs)
        trans_graph_t = self.sample_noise_next(current_datapoint, t, **kwargs)

        return (
            trans_graph_bar_t,
            trans_graph_bar_t_minus_one,
            trans_graph_t
        )


    def apply_noise_posterior(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: Tuple[DenseGraph, DenseGraph, DenseGraph],
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:

        noise_graph_bar_t, noise_graph_bar_t_minus_one, noise_graph_t = noise

        graph, ext_edges = apply_posterior_noise_graph(
            original_datapoint =	original_datapoint,
            current_datapoint =		current_datapoint,
            noise_graph_bar_t =		noise_graph_bar_t,
            noise_graph_bar_t_1 =	noise_graph_bar_t_minus_one,
            noise_graph_t =			noise_graph_t
        )

        return graph, ext_edges

        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_DISCRETE = 'discrete_uniform'

def resolve_graph_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_graph_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_DISCRETE:
        return DiscreteUniformDiffusionProcess
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')