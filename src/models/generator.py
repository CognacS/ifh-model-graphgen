##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/8757353a61235fa499dea0cbcd4771eb79b22901/dgd/diffusion_model_discrete.py
#
##########################################################################################################

from typing import Dict, Tuple, Union, Optional, List, Callable, Any

import time
import os
from copy import deepcopy

from logging import Logger
import wandb

import numpy as np

# utils for debugging
import sys

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, BoolTensor, IntTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from torchmetrics import Metric

##############  DATATYPES IMPORTS  ###############
from src.datatypes import (
    dense,
    sparse,
    split
)
from src.datatypes.dense import DenseGraph, DenseEdges
from src.datatypes.sparse import SparseGraph, SparseEdges

###############  MODULES IMPORTS  ################
from src.models.denoising.graph_transformer import DIM_X, DIM_E, DIM_Y
from src.models.models import get_model_from_config, KEY_REINSERTION, KEY_HALTING, KEY_DENOISING

################  NOISE IMPORTS  #################
from src.noise.timesample import (
    resolve_timesampler
)
from src.noise.graph_diffusion import (
    resolve_graph_diffusion_process,
    resolve_graph_diffusion_schedule
)
from src.noise.removal import (
    resolve_removal_process,
    resolve_removal_schedule
)
from src.noise.config_support import build_noise_process
from src.noise.batch_transform.sequence_sampler import sample_sequences

###############  METRICS IMPORTS  ################
import src.metrics.metrics as m_list
from src.metrics.train_denoising import SimpleTrainLossDiscrete
from src.metrics.train_reinsertion import (
    RegressionReinsertionLoss,
    DistributionReinsertionLoss
)
from src.metrics.train_halting import HaltingLoss
from src.metrics.test import (
    SumExceptBatchMetric,
    regression_accuracy,
    classification_accuracy,
    binary_classification_accuracy,
    binary_classification_recall,
    halting_prior_emd
)
from src.metrics.sampling import (
    SamplingMetricsHandler
)


RUNTIME_METRICS_EVALUATION = {
    KEY_REINSERTION: [
        m_list.KEY_REINSERTION_LOSS,
        m_list.KEY_REINSERTION_ACC
    ],
    KEY_HALTING: [
        m_list.KEY_HALTING_LOSS_BCE,
        m_list.KEY_HALTING_ACC,
        m_list.KEY_HALTING_RECALL,
        m_list.KEY_HALTING_PRIOR_EMD
    ],
    KEY_DENOISING: [
        m_list.KEY_DENOISING_LOSS_X_CE,
        m_list.KEY_DENOISING_LOSS_E_CE,
        m_list.KEY_DENOISING_LOSS_EXT_E_CE,
        m_list.KEY_DENOISING_LOSS_TOTAL_CE,
        m_list.KEY_DENOISING_ACC_X,
        m_list.KEY_DENOISING_ACC_E,
        m_list.KEY_DENOISING_ACC_EXT_E
    ]
}

def specialize_metric(reins_loss: str='mse') -> Dict:
    copy_runtime_metrics = deepcopy(RUNTIME_METRICS_EVALUATION)
    copy_runtime_metrics[KEY_REINSERTION][0] += f'_{reins_loss}'
    return copy_runtime_metrics

class InsertDenoiseHaltModel(pl.LightningModule):

    def __init__(
            self,
            architecture_config: Dict,
            diffusion_config: Dict,
            removal_config: Dict,
            dataset_info: Dict,
            run_config: Dict,
            sampling_metrics,
            inference_samples_converter: Callable[[SparseGraph], Any],
            console_logger: Logger,
            conditional_generator: bool = True,
            enable_logging: bool = True
        ):
        
        super().__init__()

        ############################  CONFIGS SETUP  ###########################

        # setup console logger
        self.console_logger = console_logger
        # setup config on how to build the model and noise processes
        self.architecture_config = architecture_config
        # setup additional information from the dataset
        self.dataset_info = dataset_info
        self.run_config = run_config

        ######################  GLOBAL MODEL PARAMETERS  #######################
        # setup model config
        self.conditional_generator = conditional_generator
        self.model_dtype = torch.float32
        self.enable_logging = enable_logging

        #######################  GRAPHS DIMENSIONS SETUP  ######################
        # setup model input and output dimensions (based on the dataset)
        self.num_cls_nodes = dataset_info['num_cls_nodes']
        self.num_cls_edges = dataset_info['num_cls_edges']
        self.num_cls_edges_w_no_edge = self.num_cls_edges + 1
        #self.dim_targets = dataset_info['dim_targets'] if self.conditional_generator else 0
        self.dim_targets = 2 if self.conditional_generator else 0

        # models input and output dimensions
        self.input_dims = {
            DIM_X: self.num_cls_nodes,
            DIM_E: self.num_cls_edges,
            DIM_Y: self.dim_targets + 2 # account for removal and denoising times
        }
        self.output_dims = deepcopy(self.input_dims)

        self.exists_and_true = lambda key: key in self.architecture_config and self.architecture_config[key]

        if self.exists_and_true('use_indegree'):
            self.input_dims[DIM_X] += 1
        if self.exists_and_true('use_nodesnum'):
            self.input_dims[DIM_Y] += 1


        self.input_dims_w_no_edge = {
            DIM_X: self.num_cls_nodes,
            DIM_E: self.num_cls_edges_w_no_edge,
            DIM_Y: self.dim_targets + 2 # account for removal and denoising times
        }
        self.output_dims_w_no_edge = deepcopy(self.input_dims_w_no_edge)
        if self.exists_and_true('use_indegree'):
            self.input_dims_w_no_edge[DIM_X] += 1
        if self.exists_and_true('use_nodesnum'):
            self.input_dims_w_no_edge[DIM_Y] += 1


        ############################  BUILD MODELS  ############################
        self.has_reinsertion_model = KEY_REINSERTION in self.architecture_config
        self.has_halting_model = KEY_HALTING in self.architecture_config
        self.has_denoising_model = KEY_DENOISING in self.architecture_config

        ##########  REINSERTION MODEL  ###########
        if self.has_reinsertion_model:

            self.reinsertion_input = nn.Identity()
            self.reinsertion_model = get_model_from_config(
                config =                architecture_config[KEY_REINSERTION],
                which_type =            KEY_REINSERTION,
                dataset_info =          self.dataset_info,
                node_in_channels =      self.input_dims[DIM_X],
                edge_dim =              self.input_dims[DIM_E],
                globals_dim =           self.input_dims[DIM_Y] - 1, # no dependency on denoising time
            )

            ############  HALTING MODEL  #############
            if self.has_halting_model:

                self.halting_input = nn.Identity()
                self.halting_model = get_model_from_config(
                    config =                architecture_config[KEY_HALTING],
                    which_type =            KEY_HALTING, # same as reinsertion
                    dataset_info =          self.dataset_info,
                    node_in_channels =      self.input_dims[DIM_X],
                    edge_dim =              self.input_dims[DIM_E],
                    globals_dim =           self.input_dims[DIM_Y] - 1, # no dependency on denoising time
                )


        ###########  DENOISING MODEL  ############
        if self.has_denoising_model:

            self.denoising_input_before_noise = nn.Identity()
            self.denoising_input_after_noise = nn.Identity()
            self.denoising_model = get_model_from_config(
                config =                architecture_config[KEY_DENOISING],
                which_type =            KEY_DENOISING,
                dataset_info =          self.dataset_info,
                input_dims =            self.input_dims_w_no_edge,
                output_dims =           self.output_dims_w_no_edge
            )

        ########################  BUILD NOISE PROCESSES  #######################

        ###########  REMOVAL PROCESS  ############
        
        self.removal_config = removal_config

        if self.has_reinsertion_model:

            self.removal_process, self.removal_timesampler = build_noise_process(
                config =                removal_config,
                process_resolver =      resolve_removal_process,
                schedule_resolver =     resolve_removal_schedule,
                timesampler_resolver =  resolve_timesampler
            )

            self.console_logger.info(f"Removal process: {self.removal_process.__class__.__name__}")

            self.fixed_reinsertion_steps = hasattr(self.removal_process.schedule, 'max_time')

        ##########  DIFFUSION PROCESS  ###########

        self.diffusion_config = diffusion_config

        if self.has_denoising_model:

            self.diffusion_process, self.diffusion_timesampler = build_noise_process(
                config =                diffusion_config,
                process_resolver =      resolve_graph_diffusion_process,
                schedule_resolver =     resolve_graph_diffusion_schedule,
                timesampler_resolver =  resolve_timesampler
            )

            self.console_logger.info(f"Diffusion process: {self.diffusion_process.__class__.__name__}")

        ########################################################################
        #                                LOSSES                                #
        ########################################################################
            
        self.training_enabled = {
            KEY_REINSERTION: self.run_config['train_reinsertion'] and self.has_reinsertion_model,
            KEY_HALTING: self.run_config['train_halting'] and self.has_halting_model,
            KEY_DENOISING: self.run_config['train_denoising'] and self.has_denoising_model
        }

        self.evaluating_enabled = deepcopy(self.training_enabled)
        

        self.losses = nn.ModuleDict()

        ###########################  TRAINING LOSSES  ##########################
        # save training loss
        self.losses['_train'] = nn.ModuleDict()


        reins_loss_code = 'none'
        if self.has_reinsertion_model:
            #self.train_loss_reinsertion = ReinsertionLoss(self.run_config[KEY_REINSERTION]['lambda'])
            self.node_regressive = True
            #TODO: check node_regressive, because mse is being selected
            reins_cfg = architecture_config[KEY_REINSERTION]

            if 'params' in reins_cfg and 'out_properties' in reins_cfg['params']:
                out_props = architecture_config[KEY_REINSERTION]['params']['out_properties']
                self.node_regressive = isinstance(out_props, int) and out_props == 1

            if self.node_regressive:
                loss = RegressionReinsertionLoss()
                reins_loss_code = 'mse'
            else:
                loss = DistributionReinsertionLoss()
                reins_loss_code = 'kldiv'
            
            self.losses['_train'][KEY_REINSERTION] = loss

        if self.has_halting_model:
            args = {} if KEY_HALTING not in self.run_config else self.run_config[KEY_HALTING]
            self.losses['_train'][KEY_HALTING] = HaltingLoss(**args)

        if self.has_denoising_model:
            self.weighted_denoising_loss = 'weighted' in self.run_config[KEY_DENOISING] and self.run_config[KEY_DENOISING]['weighted']
            self.class_weighted_denoising_loss = 'class_weighted' in self.run_config[KEY_DENOISING] and self.run_config[KEY_DENOISING]['class_weighted']
            self.losses['_train'][KEY_DENOISING] = SimpleTrainLossDiscrete(
                **self.run_config[KEY_DENOISING]
            )

        ################  ADDING CUMULATIVE EVALUATION METRICS  ################

        # the following code adds a set of cumulative evaluation metrics
        # for each enabled modules. The metrics to be added
        # are defined in the RUNTIME_METRICS_EVALUATION dictionary
        # the final structure will be:
        # self.losses[<phase>][<module_name>][<metric_name>]

        ##########  VALIDATION METRICS  ##########
        # specialize generic names, e.g. reinsertion_loss -> reinsertion_loss_mse
        runtime_metrics_evaluation = specialize_metric(reins_loss_code)

        # create self.losses[<phase>][<module_name>][<metric_name>]
        for phase in ['_valid', '_test']:
            eval_metrics = nn.ModuleDict()

            for mod, enabled in self.training_enabled.items():
                if enabled:
                    eval_metrics[mod] = nn.ModuleDict({
                        metric: SumExceptBatchMetric()
                        for metric in runtime_metrics_evaluation[mod]
                    })

            self.losses[phase] = eval_metrics

        ############################  SAMPLING  ###############################
        
        if self.has_reinsertion_model and self.has_denoising_model:
            # save sampling metrics
            self.losses['sampling'] = sampling_metrics


        ############################  EXTRA SETUP  #############################

        self.inference_samples_converter = inference_samples_converter

        self._disable_generation = False
        self.total_elapsed_time = 0
        self.max_memory_reserved = torch.cuda.max_memory_reserved(0)
        self.start_time = None

        self.save_hyperparameters(ignore=[
            'sampling_metrics', 'inference_samples_converter', 'console_logger', 'enable_logging'
        ])


    def is_conditional(self):
        return self.conditional_generator
    def is_debug(self):
        return not self.enable_logging
    def is_generation_disabled(self):
        return self._disable_generation
    def disable_generation(self):
        self._disable_generation = True
    def enable_generation(self):
        self._disable_generation = False
    

    def get_module(self, module_name: str) -> nn.Module:
        if module_name == KEY_REINSERTION:
            return self.reinsertion_model
        elif module_name == KEY_HALTING:
            return self.halting_model
        elif module_name == KEY_DENOISING:
            return self.denoising_model
        else:
            raise ValueError(f'Invalid module name {module_name}')


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        checkpoint['total_elapsed_time'] = self.total_elapsed_time + overtime
        checkpoint['max_memory_reserved'] = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)
        checkpoint['training_enabled'] = self.training_enabled
        checkpoint['evaluating_enabled'] = self.evaluating_enabled

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.total_elapsed_time = checkpoint['total_elapsed_time']
        self.max_memory_reserved = checkpoint['max_memory_reserved']
        self.training_enabled = checkpoint['training_enabled']
        self.evaluating_enabled = checkpoint['evaluating_enabled']


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################


    def prepare_batch(self, batch: Union[Data, Dict[str, Data]]) -> Tuple[SparseGraph, SparseGraph, SparseGraph, SparseEdges]:

        seqs, max_seq_len = sample_sequences(
            batch = batch,
            removal_process = self.removal_process,
            need_preparation=False
        )

        batch = seqs['batch']
        surv_batch = seqs['surv_batch']
        remv_batch = seqs['remv_batch']
        remv_edges_ba = seqs['remv_edges_ba']

        if not self.conditional_generator:
            batch.y = None
            surv_batch.y = None
            remv_batch.y = None
        else:
            batch.y = batch.y[:, -2:].float()
            surv_batch.y = surv_batch.y[:, -2:].float()
            remv_batch.y = remv_batch.y[:, -2:].float()

        append_time_to_graph_globals(
            batch,
            time = batch.global_rev_t
        )
        append_time_to_graph_globals(
            surv_batch,
            time = surv_batch.global_rev_t
        )
        append_time_to_graph_globals(
            remv_batch,
            time = remv_batch.global_rev_t
        )

        ###########  FORMAT BEFORE BRANCHING INTO THE TWO TRAININGS  ###########
        self.add_additional_features(batch)
        self.add_additional_features(surv_batch)

        return batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len
    
    
    def compute_true_pred_reinsertion(
            self,
            batch: SparseGraph
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted properties of the reinsertion process:
        - true properties are generated from the removal process's posterior
        distribution.
        - predicted properties are generated by the reinsertion model from the
        survived batch graph.

        Parameters
        ----------
        batch : SparseGraph
            batch of graphs

        Returns
        -------
        true_props : List[Tensor]
            list of true properties
        pred_props : List[Tensor]
            list of predicted properties
        """
        ############################  CHECK INPUT  #############################
        # batch should be onehot
        assert batch.x.ndim == 2, f'Nodes are dimension {batch.x.ndim}, should be 2'
        assert batch.edge_attr.ndim == 2, f'Edges are dimension {batch.edge_attr.ndim}, should be 2'

        # generate the true probabilities
        # true_prob = self.removal_process.get_params_posterior(
        #     t = 		batch.global_t,
        #     max_time =	batch.global_n0
        # )

        # used for hooks, does nothing (identity)
        self.reinsertion_input(batch)

        # predict reverse process properties
        pred_props: Tensor = self.reinsertion_model(
            x =				batch.x,
            edge_index =	batch.edge_index,
            edge_attr =		batch.edge_attr,
            batch =			batch.batch,
            batch_size =	batch.num_graphs,
            y =				batch.y
        )

        loss_func = self.losses['_train'][KEY_REINSERTION]

        if isinstance(loss_func, RegressionReinsertionLoss) and self.node_regressive:

            # get missing nodes to reintegrate
            true_missing_nodes = (batch.global_n0 - batch.global_nt).float()

            # property to predict: number of missing nodes from true graph
            true_props = true_missing_nodes

        elif isinstance(loss_func, DistributionReinsertionLoss) and not self.node_regressive:

            true_dist = self.removal_process.schedule.get_posterior_distribution(
                n0 = batch.global_n0,
                nt = batch.global_nt,
                t = batch.global_t
            )

            # property to predict: distribution on the moves to make to insert nodes
            true_props = true_dist

        else:
            raise ValueError(
                f'Invalid loss function {type(loss_func)} ' +\
                f'for reinsertion process {type(self.removal_process)} with '+\
                f'scheduler {type(self.removal_process.schedule)}, and model '+\
                f'with {self.reinsertion_model.out_properties} properties'
            )

        return true_props, pred_props
    

    def compute_true_pred_halting(
            self,
            batch: SparseGraph
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted properties of the halting part of removal:
        - true properties are generated as the halt signal at time t=0.
        - predicted properties are generated by the halting model from the current
        batch graph

        Parameters
        ----------
        batch : SparseGraph
            batch of graphs

        Returns
        -------
        true_props : List[Tensor]
            list of true properties
        pred_props : List[Tensor]
            list of predicted properties
        """
        ############################  CHECK INPUT  #############################
        # batch should be onehot
        assert batch.x.ndim == 2, f'Nodes are dimension {batch.x.ndim}, should be 2'
        assert batch.edge_attr.ndim == 2, f'Edges are dimension {batch.edge_attr.ndim}, should be 2'

        # used for hooks, does nothing (identity)
        self.halting_input(batch)

        # predict reverse process properties
        pred_props: Tensor = self.halting_model(
            x =				batch.x,
            edge_index =	batch.edge_index,
            edge_attr =		batch.edge_attr,
            batch =			batch.batch,
            batch_size =	batch.num_graphs,
            y =				batch.y
        )

        # the halt signal indicates that the generator
        # should stop right at this point
        # during training: when time is t=0
        true_halt = (batch.global_t == 0).int()

        true_props = true_halt

        return true_props, pred_props
    

    
    def compute_true_pred_denoising(
            self,
            batch_to_generate: SparseGraph,
            batch_external: Optional[SparseGraph] = None,
            edges_external: Optional[SparseEdges] = None
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted nodes and egdes for the denoising
        process. The flow is as follows:
        1 - encode the batch_external to get encoded nodes
        2 - densify batch_to_generate as a DenseGraph, the encoded nodes,
            and the external edges, with onehot and masking
        3 - sample the diffusion process at uniformly random timesteps to
            make a noisy version of batch_to_generate (again requires onehot
            and masking)
        4 - try to denoise the above data which include the batch_to_generate
            and edges_external
        5 - flatten and pack the true and predicted nodes and edges
        The final order is: nodes, edges, external_edges.
        Predicted values are in expanded form, true values are collapsed. This is
        ideal for the cross-entropy loss function.

        Parameters
        ----------
        batch_to_generate : SparseGraph
            sparse graph with collapsed classes (i.e. class indices). This graph
            will be noised and denoised.
        batch_external : Optional[SparseGraph]
            sparse graph with onehot classes. The nodes of this graph will be
            encoded and used to denoise the batch_to_generate. Default is None,
            in which case only the batch_to_generate is noised and denoised.
        edges_external : Optional[Tuple[Tensor, Tensor]]
            external edges in edge_index and edge_attr form, to be noised and
            denoised. Default is None, in which case only the batch_to_generate
            is noised and denoised.

        Returns
        -------
        true_values : List[Tensor]
            list of true values of nodes and edges, in collapsed form.
        pred_values : List[Tensor]
            list of predicted values of nodes and edges, in expanded form.
        """

        if batch_external is None:
            raise NotImplementedError('None external graph is still to be implemented')

        ################  ENCODE EXTERNAL GRAPH INTO ITS NODES  ################
        batch_external.x = self.denoising_model.forward_encoding(batch_external)

        ####################  FORMAT INPUT FOR PREDICTION  #####################
        # 1 - densify
        # transform the removed nodes and edges to dense format
        # transform the survived nodes to dense format
        batch_to_generate_dense: DenseGraph
        ext_x: Tensor
        ext_node_mask: BoolTensor
        ext_adjmat: DenseEdges
        batch_to_generate_dense, ext_x, ext_node_mask, ext_adjmat = format_generation_task_data(
            surv_graph =		batch_external,
            remv_graph =		batch_to_generate,
            edges_remv_surv =	edges_external
        )

        # check
        # now remv_graph is dense, edges_remv_surv is dense, and surv_graph is sparse
        # from src.noise.batch_transform.sequence_sampler import check_connected_components
        # assert check_connected_components(
        #     graph_a = batch_external,
        #     graph_b = batch_to_generate_dense,
        #     edges_ba = ext_adjmat
        # ), f"External graph and batch to generate are not connected"

        # setup masks for edges
        node_mask = batch_to_generate_dense.node_mask
        triang_edge_mask = torch.tril(batch_to_generate_dense.edge_mask, diagonal=-1)
        ext_edge_mask = ext_adjmat.edge_mask

        # 2 - copy true masked data (to be returned later)
        true_x = batch_to_generate_dense.x.argmax(dim=-1)[node_mask]
        true_e = batch_to_generate_dense.edge_adjmat.argmax(dim=-1)[triang_edge_mask]
        true_ext_e = ext_adjmat.edge_adjmat.argmax(dim=-1)[ext_edge_mask]

        # used for hooks, does nothing (identity)
        self.denoising_input_before_noise({
            'batch_to_generate_dense': batch_to_generate_dense,
            'batch_to_generate': batch_to_generate,
            'batch_external': batch_external,
            'edges_external': ext_adjmat
        })

        #######################  APPLY GRAPH DIFFUSION  ########################
        # sample the timesteps for the diffusion process
        max_times = torch.full((batch_external.num_graphs,), self.diffusion_process.get_max_time()-1) # must be in cpu
        u: Tensor = self.diffusion_timesampler.sample_time(max_time=max_times).to(self.device) + 1 # do not sample u=0

        append_time_to_graph_globals(
            batch_to_generate_dense,
            time = self.diffusion_process.normalize_time(u)
        )

        # sample the noisy graph at timestep u

        # WARNING: here selfloops are not masked!!!
        noisy_data = self.diffusion_process.sample_from_original(
            original_datapoint=(batch_to_generate_dense, ext_adjmat),
            t=u
        )

        # onehot and mask the noisy data again (to remove the fake noisy components)
        onehot_data = to_onehot_all(
            *noisy_data,
            **self.output_dims_w_no_edge
        )

        # add features to the noisy data
        self.add_additional_features(onehot_data)

        masked_data = mask_all(
            *onehot_data
        )

        noisy_batch_to_generate_dense_onehot, noisy_ext_edges_onehot = masked_data

        #####################  PREDICT THE ORIGINAL GRAPH  #####################
        # used for hooks, does nothing (identity)
        self.denoising_input_after_noise({
            'batch_to_generate': noisy_batch_to_generate_dense_onehot,
            'batch_external': batch_external,
            'edges_external': noisy_ext_edges_onehot
        })
        gen_batch_dense: DenseGraph
        gen_ext_edges: DenseEdges
        gen_batch_dense, gen_ext_edges = self.denoising_model.forward_transformer(
            subgraph =				noisy_batch_to_generate_dense_onehot,
            ext_edges_new_to_ext =	noisy_ext_edges_onehot,
            ext_X =					ext_x,
            ext_node_mask =			ext_node_mask
        )

        pred_x = gen_batch_dense.x[node_mask]
        pred_e = gen_batch_dense.edge_adjmat[triang_edge_mask]
        if gen_ext_edges is not None:
            pred_ext_e = gen_ext_edges.edge_adjmat[ext_edge_mask]
        else:
            pred_ext_e = torch.empty((0, self.num_cls_edges_w_no_edge), dtype=torch.float, device=self.device)

        ###########################  FINAL PACKING  ############################
        true_values = [true_x, true_e, true_ext_e]
        if self.weighted_denoising_loss:
            pred_values = [pred_x, pred_e, pred_ext_e, node_mask, triang_edge_mask, ext_edge_mask]
        else:
            pred_values = [pred_x, pred_e, pred_ext_e]
        
        return true_values, pred_values
    

    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################

    def on_fit_start(self) -> None:
        self.console_logger.info(f"Size of input features X: {self.input_dims[DIM_X]}, E: {self.input_dims[DIM_E]}, y: {self.input_dims[DIM_Y]}")


    def on_train_epoch_start(self) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """"Recall that this method is called AFTER the validation epoch, if there is any!"""
        self.total_elapsed_time += time.time() - self.start_time
        self.max_memory_reserved = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)


    def training_step(self, batch: SparseGraph, batch_idx: int):

        ###########################  INITIAL SETUP  ############################
        true_bs = batch.num_graphs
        batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)

        train_loss = []
        logs = {}

        loss_functions = self.losses['_train']

        ######################  TRAIN REINSERTION MODEL  #######################
        if self.training_enabled[KEY_REINSERTION]:

            # FLOW DEFINITION
            # survived graph -> predict reverse process params -> match against true params

            # compute true and predicted params for the reinsertion process
            true_params, pred_params = self.compute_true_pred_reinsertion(
                batch = surv_batch
            )

            # compute reinsertion loss
            reintegr_loss, reintegr_logs = loss_functions[KEY_REINSERTION](
                pred_params,
                true_params,
                ret_log=True
            )

            # compute accuracy
            with torch.no_grad():
                if self.node_regressive:
                    reintegr_logs[
                        m_list.KEY_REINSERTION_ACC
                    ] = regression_accuracy(pred_params, true_params)

            # apply prefix to logs
            reintegr_logs = self.apply_prefix(
                metrics = reintegr_logs,
                prefix = f'train_{KEY_REINSERTION}'
            )

            logs.update(reintegr_logs)
            train_loss.append(reintegr_loss)


        #######################  TRAIN HALTING MODEL  ##########################
        if self.training_enabled[KEY_HALTING]:

            # FLOW DEFINITION
            # batch -> predict halting signal -> match against true halting signal (i.e. t=0)

            # use true and predicted halting signals from the batch
            true_halting, pred_halting = self.compute_true_pred_halting(
                batch = batch
            )

            # compute halting loss
            halting_loss, halting_logs = loss_functions[KEY_HALTING](
                pred_halting,
                true_halting,
                ret_log=True
            )

            # compute recall
            with torch.no_grad():
                halting_logs[
                    m_list.KEY_HALTING_ACC
                ] = binary_classification_accuracy(pred_halting, true_halting)
                halting_logs[
                    m_list.KEY_HALTING_RECALL
                ] = binary_classification_recall(pred_halting, true_halting)
                #halting_logs[
                #    m_list.KEY_HALTING_PRIOR_EMD
                #] = halting_prior_emd(
                #    pred_halting,
                #    true_halting,
                #    batch_idx   = batch.global_batch_idx,
                #    batch_size  = true_bs,
                #    max_seq_len = max_seq_len
                #)

            # apply prefix to logs
            halting_logs = self.apply_prefix(
                metrics = halting_logs,
                prefix = f'train_{KEY_HALTING}'
            )

            logs.update(halting_logs)
            train_loss.append(halting_loss)
                


        #######################  TRAIN DENOISING MODEL  ########################
        if self.training_enabled[KEY_DENOISING]:

            # FLOW DEFINITION
            # survived graph -> encoded survived graph
            # removed graph -> noisy graph -> denoised graph

            # compute true and predicted nodes and edges from the denoising process
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = remv_batch,
                batch_external =	surv_batch,
                edges_external =	remv_edges_ba
            )

            # compute denoising training loss
            denoise_loss, denoise_logs = loss_functions[KEY_DENOISING](
                pred_data,
                true_data,
                weighted=self.weighted_denoising_loss,
                class_weighted=self.class_weighted_denoising_loss,
                ret_log=True
            )

            # compute accuracy
            with torch.no_grad():
                denoise_logs[
                    m_list.KEY_DENOISING_ACC_X
                ] = classification_accuracy(pred_data[0], true_data[0])
                denoise_logs[
                    m_list.KEY_DENOISING_ACC_E
                ] = classification_accuracy(pred_data[1], true_data[1])
                denoise_logs[
                    m_list.KEY_DENOISING_ACC_EXT_E
                ] = classification_accuracy(pred_data[2], true_data[2])

            # apply prefix to logs
            denoise_logs = self.apply_prefix(
                metrics = denoise_logs,
                prefix = f'train_{KEY_DENOISING}'
            )

            logs.update(denoise_logs)
            train_loss.append(denoise_loss)


        # log current metrics
        current_training_step = self.trainer.num_training_batches * self.current_epoch + batch_idx
        self.log_metrics_custom(logs, step=current_training_step)

        return {'loss': sum(train_loss)}


    def configure_optimizers(self):

        # currently using the AdamW optimizer
        # NOTE: the original code used the option "amsgrad=True"
        return torch.optim.AdamW(
            self.parameters(), **self.run_config['optimizer']
        )
    
    ############################################################################
    #                         VALID/TEST PHASE SECTION                         #
    ############################################################################

    @torch.no_grad()
    def on_evaluation_epoch_start(self, which='_valid') -> None:

        # reset to zero all metrics to be
        # accumulated
        for metrics in self.losses[which].values():
            for metric in metrics.values():
                metric.reset()

        # part used for gathering conditioning
        # attributes from the validation or test set
        # to be used for generation
        self.conditioning_y = None
        if self.conditional_generator:
            self.conditioning_y = []
            self.num_cond_y = 0


    @torch.no_grad()
    def evaluation_step(self, batch: SparseGraph, batch_idx: int, which='_valid') -> None:

        #############  SAVE PROPERTIES FOR CONDITIONAL GENERATION  #############
        # save some target properties if needed for conditional generation
        if self.conditional_generator:

            # get how many will be sampled
            sampling_metrics = self.losses['sampling']
            if which in sampling_metrics:
                sampling_metrics = sampling_metrics[which]

            num_to_sample = sampling_metrics.generation_cfg['num_samples']

            # get the conditioning attributes from the batch
            if self.num_cond_y < num_to_sample:
                to_grab = min(num_to_sample - self.num_cond_y, batch.num_graphs)
                self.conditioning_y.append(batch.y[:to_grab, -2:].float())
                self.num_cond_y += to_grab

        ###########################  INITIAL SETUP  ############################
        true_bs = batch.num_graphs
        batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)

        eval_loss = []

        loss_functions = self.losses['_train']
        eval_metrics = self.losses[which]

        ######################  TRAIN REINSERTION MODEL  #######################
        if self.evaluating_enabled[KEY_REINSERTION]:

            # FLOW DEFINITION
            # survived graph -> predict reverse process params -> match against true params

            # compute true and predicted params for the reinsertion process
            true_params, pred_params = self.compute_true_pred_reinsertion(
                batch = surv_batch
            )

            # compute reinsertion loss
            reins_loss, reins_logs = loss_functions[KEY_REINSERTION](
                pred_params,
                true_params,
                reduce=False,
                ret_log=True
            )

            # compute accuracy
            if self.node_regressive:
                # compute accuracy on correctly predicting the number of nodes
                correct = regression_accuracy(pred_params, true_params, reduction='none')
            else:
                # compute accuracy on correctly predicting the halt signal
                correct = binary_classification_accuracy(pred_params[1], true_params[1], reduction='none')

            # compute metrics and add to logs
            reins_logs.update({
                m_list.KEY_REINSERTION_ACC: correct
            })

            # update metrics
            eval_loss.append(reins_loss.mean())
            reins_eval_metrics = eval_metrics[KEY_REINSERTION]
            for m, val in reins_logs.items():
                reins_eval_metrics[m](val)


        #######################  TRAIN HALTING MODEL  ##########################
        if self.evaluating_enabled[KEY_HALTING]:

            # FLOW DEFINITION
            # batch -> predict halting signal -> match against true halting signal (i.e. t=0)

            # use true and predicted halting signals from the batch
            true_halting, pred_halting = self.compute_true_pred_halting(
                batch = batch
            )

            # compute halting loss
            halting_loss, halting_logs = loss_functions[KEY_HALTING](
                pred_halting,
                true_halting,
                reduce=False,
                ret_log=True
            )

            # compute metrics and add to logs
            halting_logs.update({
                m_list.KEY_HALTING_ACC: binary_classification_accuracy(pred_halting, true_halting, reduction='none'),
                m_list.KEY_HALTING_RECALL: binary_classification_recall(pred_halting, true_halting, reduction='none'),
                m_list.KEY_HALTING_PRIOR_EMD: halting_prior_emd(
                    pred_halting,
                    true_halting,
                    batch_idx   = batch.global_batch_idx,
                    batch_size  = true_bs,
                    max_seq_len = max_seq_len,
                    reduction   = 'none'
                )
            })

            # update metrics
            eval_loss.append(halting_loss.mean())
            halt_eval_metrics = eval_metrics[KEY_HALTING]
            for m, val in halting_logs.items():
                halt_eval_metrics[m](val)


        #######################  TRAIN DENOISING MODEL  ########################
        if self.evaluating_enabled[KEY_DENOISING]:

            # FLOW DEFINITION
            # survived graph -> encoded survived graph
            # removed graph -> noisy graph -> denoised graph

            # compute true and predicted nodes and edges from the denoising process
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = remv_batch,
                batch_external =	surv_batch,
                edges_external =	remv_edges_ba
            )

            # compute denoising training loss
            denoise_loss, denoise_logs = loss_functions[KEY_DENOISING](
                pred_data,
                true_data,
                weighted=self.weighted_denoising_loss,
                class_weighted=self.class_weighted_denoising_loss,
                reduce=False,
                ret_log=True
            )

            # compute denoising training loss
            #nll = self.compute_val_loss(pred_data, noisy_data, true_data, node_mask, test=False)

            # compute accuracy and add to logs
            denoise_logs.update({
                m_list.KEY_DENOISING_ACC_X: classification_accuracy(pred_data[0], true_data[0], reduction='none'),
                m_list.KEY_DENOISING_ACC_E: classification_accuracy(pred_data[1], true_data[1], reduction='none'),
                m_list.KEY_DENOISING_ACC_EXT_E: classification_accuracy(pred_data[2], true_data[2], reduction='none')
            })

            # update metrics
            eval_loss.append(denoise_loss)
            denoise_eval_metrics = eval_metrics[KEY_DENOISING]
            for m, val in denoise_logs.items():
                denoise_eval_metrics[m](val)



        return {'loss': sum(eval_loss)}


    @torch.no_grad()
    def on_evaluation_epoch_end(self, which='_valid') -> None:
        metrics = {}

        # compute all metrics and log
        for mode, metrics_fun in self.losses[which].items():
            for name, metric_fun in metrics_fun.items():
                metrics[f'{which[1:]}_{mode}/{name}'] = metric_fun.compute()


        # compute sampling metrics
        if (
            self.has_reinsertion_model and
            self.has_denoising_model and
            which in self.losses['sampling']
            and not self._disable_generation
        ):

            # get generation parameters
            sampling_metrics = self.losses['sampling']
            if which in sampling_metrics:
                sampling_metrics = sampling_metrics[which]

            if isinstance(sampling_metrics, SamplingMetricsHandler):
                generation_params = sampling_metrics.generation_cfg
            else:
                generation_params = {}

            ######## compute the sampling metrics ########
            start = time.time()

            to_log, num_nodes_hist = self.compute_sampling_metrics(**generation_params, which=which)

            end = time.time()
            ##############################################

            # add prefix to logs
            to_log = self.apply_prefix(
                metrics = to_log,
                prefix = f'{which[1:]}_sampling'
            )

            metrics.update(to_log)

            if which == '_valid' and num_nodes_hist:
                # also add the histogram of number of nodes
                self.log_wandb_media(f'{which[1:]}_sampling/num_nodes_hist', wandb.Histogram(np_histogram=num_nodes_hist))

            self.console_logger.info(f'Done. Sampling took {end - start:.2f} seconds\n')
            #self.sampling_metrics.reset()


        overtime = 0 if self.start_time is None else time.time() - self.start_time
        metrics.update(
            self.apply_prefix(
                {
                    m_list.KEY_COMPUTATIONAL_TRAIN_TIME: self.total_elapsed_time + overtime,
                    m_list.KEY_COMPUTATIONAL_TRAIN_MEMORY:  max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)
                },
                prefix = f'{which[1:]}_computational'
            )
        )

        self.log_metrics_custom(metrics, step=None, train_log=False)


    @torch.no_grad()
    def compute_sampling_metrics(self, num_samples: int=128, batch_size: int=64, which='_valid'):

        if self._disable_generation:
            return {}, None
        
        samples_left_to_generate = num_samples

        samples = []

        self.console_logger.info('Sampling some graphs...')

        # initialize for process metrics
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats(0)

        if self.conditional_generator:
            conditioning_y = torch.cat(self.conditioning_y, dim=0)
            num_available_y = conditioning_y.shape[0]
            if num_available_y < num_samples:
                self.console_logger.warning(f'Only {num_available_y} conditioning attributes available, but {num_samples} requested')
                self.console_logger.info('Sampling with replacement...')
                conditioning_y_nolast = conditioning_y.repeat(num_samples // num_available_y, 1)
                if num_samples % num_available_y > 0:
                    conditioning_y = torch.cat([conditioning_y_nolast, conditioning_y[:num_samples % num_available_y]], dim=0)
                else:
                    conditioning_y = conditioning_y_nolast

            # split into the batches to generate
            conditioning_y = torch.split(conditioning_y, batch_size)
        else:
            conditioning_y = [None] * (num_samples // batch_size + 1)
        
        batch_idx = 0
        # sample required graphs
        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, batch_size)
            self.console_logger.info(f'Generating {to_generate} graphs...')

            samples.extend(
                self.sample_batch(
                    batch_size=to_generate,
                    conditioning_y=conditioning_y[batch_idx],
                    maximum_number_nodes=self.dataset_info['num_nodes_max']+4,
                )
            )

            samples_left_to_generate -= to_generate
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')

        # end for process metrics
        end_time = time.time()
        peak_memory_usage = float(torch.cuda.max_memory_allocated(0))

        # gather data for metrics computation
        data_for_metrics = {
            'generated_graphs': samples,
            'time': {'start': start_time, 'end': end_time},
            'memory': {'peak': peak_memory_usage}
        }

        # compute some statistics on the generated graphs
        num_nodes = [s.num_nodes for s in samples]
        num_edges = [s.num_edges for s in samples]
        self.console_logger.info(f'Number of nodes per graph: avg:{np.mean(num_nodes)}, min:{np.min(num_nodes)}, max:{np.max(num_nodes)}')
        self.console_logger.info(f'Number of edges per graph: avg:{np.mean(num_edges)}, min:{np.min(num_edges)}, max:{np.max(num_edges)}')

        # compute histogram of number of nodes
        num_nodes_hist = np.histogram(num_nodes, bins=np.arange(min(num_nodes)-0.5, max(num_nodes)+1.5), density=True)

        # get correct sampling metrics
        sampling_metrics = self.losses['sampling']
        if which in sampling_metrics:
            sampling_metrics = sampling_metrics[which]

        # compute metrics
        to_log = sampling_metrics(data_for_metrics)
        self.console_logger.info(str(to_log))

        return to_log, num_nodes_hist


    ############################################################################
    #           VALIDATION PHASE SECTION (executed during validation)          #
    ############################################################################

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start(which='_valid')

    def validation_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which='_valid')

    def on_validation_epoch_end(self):
        self.console_logger.info(self.get_training_status())
        return self.on_evaluation_epoch_end(which='_valid')

    ############################################################################
    #               TEST PHASE SECTION (executed during testing)               #
    ############################################################################

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start(which='_test')

    def test_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which='_test')

    def on_test_epoch_end(self):
        return self.on_evaluation_epoch_end(which='_test')

    ############################################################################
    #                           CHECKPOINT FUNCTIONS                           #
    ############################################################################

    def log_metrics_custom(self, metrics, step, train_log=True):
        if (not train_log or\
            (train_log and step % self.run_config['log_every_n_steps'] == 0)):
            # log to wandb if enabled
            if self.enable_logging:
                wandb.log(metrics, step=step)
            # log metrics to lightning anyway
            self.log_dict(metrics)

    
    def get_training_status(self) -> str:
        st = [
            f"{k}: {'[X]' if v else '[ ]'}"
            for k, v in self.training_enabled.items()
        ]
        return 'Modules enabled for training: ' + ', '.join(st)

    def apply_prefix(self, metrics, prefix):
        return {f'{prefix}/{k}': v for k, v in metrics.items()}


    def log_wandb_media(self, name, metric):
        if self.enable_logging:
            wandb.log({name: metric})


    ############################################################################
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################

    @torch.no_grad()
    def forward_reinsertion(
            self,
            graph: SparseGraph,
            reversed_reinsertion_time: IntTensor
        ) -> IntTensor:

        assert_is_onehot(graph)

        # used for hooks, does nothing (identity)
        self.reinsertion_input(graph)

        """Forward pass of the reinsertion model."""
        # predict final number of nodes from
        # current graph

        pred_props: Tensor = self.reinsertion_model(
            x =				graph.x,
            edge_index =	graph.edge_index,
            edge_attr =		graph.edge_attr,
            batch =			graph.batch if hasattr(graph, 'batch') else None,
            batch_size =	graph.num_graphs if hasattr(graph, 'num_graphs') else None,
            y =				graph.y
        )

        if self.node_regressive:
            sampled_num_new_nodes, new_time, reinsertion_time = self._sample_new_nodes_regressive(
                pred_props =                pred_props,
                graph =                     graph,
                reversed_reinsertion_time = reversed_reinsertion_time
            )

        else:
            sampled_num_new_nodes, new_time, reinsertion_time = self._sample_new_nodes_distribution(
                pred_props =                pred_props,
                graph =                     graph,
                reversed_reinsertion_time = reversed_reinsertion_time
            )


        return sampled_num_new_nodes, new_time, reinsertion_time
    

    def _sample_new_nodes_regressive(self, pred_props: Tensor, graph: SparseGraph, reversed_reinsertion_time: IntTensor):
        # transform the regression property to discrete prediction
        pred_num_remaining_nodes = torch.round(torch.relu(pred_props)).int()

        # compute number nodes currently (nt)
        num_nodes = graph.ptr[1:] - graph.ptr[:-1]
        pred_n0 = pred_num_remaining_nodes + num_nodes

        # compute the correct reinsertion time by reversing the reversed time
        reinsertion_time = self.removal_process.get_schedule().reverse_step(
            t =		reversed_reinsertion_time,
            n0 =	pred_n0
        )

        # sample number of nodes to add
        # using posterior of removal process
        sampled_num_new_nodes = self.removal_process.sample_noise_posterior(
            original_datapoint =    pred_n0,
            current_datapoint =     graph,
            t =                     reinsertion_time,
            return_quantity =       True
        )

        # get new time
        new_time = self.removal_process.normalize_time(
            t = reversed_reinsertion_time+1,
            n0 = pred_n0
        )

        return sampled_num_new_nodes, new_time, reinsertion_time
    

    def _sample_new_nodes_distribution(self, pred_props: Tensor, graph: SparseGraph, reversed_reinsertion_time: IntTensor):
        
        # transform the logits property to a distribution
        pred_nodes_logits = pred_props

        if pred_nodes_logits.ndim == 1:
            pred_nodes_logits = pred_nodes_logits.unsqueeze(-1)

        # these would be the probabilities from the logits
        #pred_nodes_probs = torch.softmax(pred_nodes_logits, dim=-1)

        # sample from the removal process
        # actually a categorical with a final mapping to the right
        # number of nodes
        sampled_num_new_nodes = self.removal_process.schedule.sample_nodes_from_dist(
            logits = pred_nodes_logits
        )

        #print('sampled_num_new_nodes', sampled_num_new_nodes)

        # get new time
        new_time = self.removal_process.normalize_time(
            t = reversed_reinsertion_time+1
        )

        return sampled_num_new_nodes, new_time, None
    

    def forward_halting(
            self,
            graph: SparseGraph
        ) -> IntTensor:

        assert_is_onehot(graph)

        # used for hooks, does nothing (identity)
        self.halting_input(graph)

        """Forward pass of the halting model."""
        # predict final number of nodes from
        # current graph
        #print(graph)
        #print(graph.y)
        pred_halt_logits: Tensor = self.halting_model(
            x =				graph.x,
            edge_index =	graph.edge_index,
            edge_attr =		graph.edge_attr,
            batch =			graph.batch if hasattr(graph, 'batch') else None,
            batch_size =	graph.num_graphs if hasattr(graph, 'num_graphs') else None,
            y =				graph.y
        )

        halt_signal = torch.distributions.Bernoulli(
            logits=pred_halt_logits
        ).sample()


        return halt_signal
    
    
    @torch.no_grad()
    def forward_denoising(
            self,
            graph_to_gen: DenseGraph,
            ext_edges_to_gen: DenseEdges,
            encoded_ext_x: Tensor,
            ext_node_mask: Tensor,
            denoising_time: IntTensor,
            return_onehot: bool=True,
            return_masked: bool=True,
            copy_globals_to_output: bool=True
        ) -> Tuple[DenseGraph, Tensor]:	

        #assert_is_onehot(graph_to_gen, ext_edges_to_gen)

        augmented_graph_to_gen = graph_to_gen.clone()
        self.add_additional_features((augmented_graph_to_gen, ext_edges_to_gen))

        # used for hooks, does nothing (identity)
        self.denoising_input_after_noise({
            'batch_to_generate': augmented_graph_to_gen,
            'batch_external': encoded_ext_x,
            'edges_external': ext_edges_to_gen
        })

        # predict final graph and edges
        final_graph: DenseGraph
        final_ext_edges: DenseEdges
        final_graph, final_ext_edges = self.denoising_model.forward_transformer(
            subgraph =				augmented_graph_to_gen,
            ext_edges_new_to_ext =	ext_edges_to_gen,
            ext_X =					encoded_ext_x,
            ext_node_mask =			ext_node_mask
        )

        has_ext_edges = final_ext_edges is not None

        #assert final_graph.edge_adjmat.shape[-1] == self.num_cls_edges_w_no_edge, \
        #    f"Expected {self.num_cls_edges_w_no_edge} edge classes, got {final_graph.shape[-1]}."
        #assert final_ext_adjmat.shape[-1] == self.num_cls_edges_w_no_edge, \
        #    f"Expected {self.num_cls_edges_w_no_edge} edge classes, got {final_ext_adjmat.shape[-1]}."
        
        
        # transform the logits to probabilities
        final_graph.x = torch.softmax(final_graph.x, dim=-1)
        final_graph.edge_adjmat = torch.softmax(final_graph.edge_adjmat, dim=-1)
        if has_ext_edges:
            final_ext_edges.edge_adjmat = torch.softmax(final_ext_edges.edge_adjmat, dim=-1)
        else:
            final_ext_edges = DenseEdges(
                #edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device),
                #edge_attr = torch.empty((0, self.num_cls_edges_w_no_edge), dtype=torch.float, device=self.device),
                edge_adjmat = torch.empty((*ext_edges_to_gen.edge_adjmat.shape[:-1], self.num_cls_edges_w_no_edge), dtype=torch.float, device=self.device),
                edge_mask = ext_edges_to_gen.edge_mask
            )
        

        # pack datapoints
        original_datapoint = (final_graph, final_ext_edges)
        current_datapoint = (graph_to_gen, ext_edges_to_gen)

        # sample graph at step t-1 from posterior
        generated_graph, generated_ext_edges = self.diffusion_process.sample_posterior(
            original_datapoint =	original_datapoint,
            current_datapoint =		current_datapoint,
            t =						denoising_time
        )

        if return_onehot:
            generated_graph, generated_ext_edges = to_onehot_all(
                generated_graph, generated_ext_edges,
                **self.output_dims_w_no_edge
            )

        if return_masked:
            generated_graph, generated_ext_adjmat = mask_all(
                generated_graph, generated_ext_edges
            )


        if copy_globals_to_output:
            generated_graph.y = graph_to_gen.y

        return generated_graph, generated_ext_adjmat
        
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        maximum_insertion_steps: int=1000,
        maximum_number_nodes: int=None,
        return_directed: bool=True,
        save_chains: int=0
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        if self._disable_generation:
            raise RuntimeError("Generation is disabled.")

        do_save_chains = save_chains > 0

        ########################################################################
        #                SAMPLE THE STARTING GRAPH TO GENERATE                 #
        ########################################################################

        ############  SAMPLE THE STARTING GRAPHS (AS EMPTY GRAPHS)  ############
        initialization = {}
        if conditioning_y is not None:
            initialization['y'] = conditioning_y
        else:
            initialization['y'] = torch.empty((batch_size, 0), dtype=torch.float, device=self.device)

        # generate the starting batch of graphs
        graph: SparseGraph
        graph = self.removal_process.sample_stationary(
            batch_size = batch_size,
            initialization = initialization,
            device = self.device
        ).to_onehot(
            num_classes_x =	self.output_dims[DIM_X],
            num_classes_e =	self.output_dims[DIM_E]
        )

        ############  INITIALIZE REINSERTION TIME AS 0 (REVERSED!)  ############
        # initialize the global time
        if not self.conditional_generator:
            graph.y = None

        append_time_to_graph_globals(
            graph = graph,
            time = torch.zeros(batch_size, device=self.device)
        )

        if do_save_chains:
            saved_chains = graph[:save_chains]
            saved_chains = [[g.collapse().cpu()] for g in saved_chains]


        ########################################################################
        #                            INSERTION LOOP                            #
        ########################################################################

        # generated is equal to the number of nodes predicted by the reinsertion
        # model
        #for t in range(self.removal_process.get_max_time().max().item()):
        t = 0
        remaining_graphs_idx = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # initialize batch of generated graphs
        output_batch = [None] * batch_size

        # initialize number of remaining graphs
        remaining_graphs_num = batch_size

        while t < maximum_insertion_steps and remaining_graphs_num > 0:

            ####################################################################
            #                SAMPLE THE NUMBER OF NODES TO ADD                 #
            ####################################################################

            curr_graph = graph.clone()
            self.add_additional_features(curr_graph)

            # sample the number of nodes to remove
            nodes_to_insert: IntTensor
            nodes_to_insert, new_time, true_reinsertion_time = self.forward_reinsertion(
                graph =						curr_graph,
                reversed_reinsertion_time =	torch.full((remaining_graphs_num,), t, dtype=torch.int, device=self.device)			
            )
                   

            if nodes_to_insert.sum() > 0:
                ####################################################################
                #             SAMPLE THE STARTING SUBGRAPH TO DENOISE              #
                ####################################################################

                ##########  PRE-ENCODE THE PREVIOUS GRAPH ONTO ITS NODES  ##########
                encoded_ext_x = self.denoising_model.forward_encoding(curr_graph)

                # extract the dense representation of the surviving nodes
                encoded_ext_x, ext_node_mask = to_dense_batch(
                    x =				encoded_ext_x,
                    batch =			graph.batch,
                    batch_size =	graph.num_graphs
                )

                ############  SAMPLE THE STARTING SUBGRAPHS (AS NOISE)  ############
                new_subgraph: DenseGraph
                new_ext_edges: DenseEdges
                new_subgraph, new_ext_edges = self.diffusion_process.sample_stationary(
                    num_new_nodes = nodes_to_insert,
                    ext_node_mask = ext_node_mask,
                    num_classes = self.output_dims_w_no_edge
                )

                # print('true num nodes', graph.num_nodes_per_sample)
                # print('mask num nodes', ext_node_mask.sum(dim=-1))
                # print('true new nodes', nodes_to_insert)
                # print('mask new nodes', new_subgraph.node_mask.sum(dim=-1))

                # convert the new subgraph to one-hot
                new_subgraph, new_ext_edges = to_onehot_all(
                    *(new_subgraph, new_ext_edges),
                    **self.output_dims_w_no_edge
                )

                # copy the global information to the new subgraph
                new_subgraph.y = graph.y.clone()
                
                #################  INITIALIZE DENOISING TIME AS U  #################
                diffusion_max_time = self.diffusion_process.get_max_time()

                diff_time = self.diffusion_process.normalize_time(
                    t = torch.full((remaining_graphs_num,), diffusion_max_time, dtype=torch.int, device=self.device)
                )

                append_time_to_graph_globals(
                    graph = new_subgraph,
                    time = diff_time,
                )

                ####################################################################
                #                          DENOISING LOOP                          #
                ####################################################################

                #print(f'################################### START NOW {t} ###################################')

                u_tensor = torch.empty(remaining_graphs_num, dtype=torch.int, device=self.device)

                for u in reversed(range(1,diffusion_max_time+1)):

                    #print(new_subgraph.y[0])

                    #print('Current subgraph:')
                    #print(new_subgraph.x[0])
                    #print(new_subgraph.edge_adjmat[0])
                    #print(new_subgraph.y[0])

                    if do_save_chains and u % 10 == 0:

                        chain_graphs = merge_sparse_dense_graphs_to_sparse(
                            sparse_subgraph = graph,
                            dense_subgraph = new_subgraph,
                            dense_ext_edges = new_ext_edges,
                            dense_nodes_num = nodes_to_insert
                        )

                        saved_mask = remaining_graphs_idx < save_chains
                        saved_idx = remaining_graphs_idx[saved_mask]

                        for i, g in zip(saved_idx, chain_graphs[saved_mask]):
                            g = g.collapse().cpu()
                            if return_directed:
                                g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
                            saved_chains[i].append(g)

                    u_tensor[:] = u
                    # sample graph at step u-1
                    new_subgraph, new_ext_edges = self.forward_denoising(
                        graph_to_gen =		new_subgraph,
                        ext_edges_to_gen =	new_ext_edges,
                        encoded_ext_x =		encoded_ext_x,
                        ext_node_mask =		ext_node_mask,
                        denoising_time = 	u_tensor,
                        return_onehot =		True
                    )

                    # print(f'##################### {u} ##################')
                    # print(new_subgraph.x[0])
                    # print(new_subgraph.edge_adjmat[0])
                    # print(new_adjmat[0])

                    # update denoising time (in-place), denoising go down!
                    new_subgraph.y[..., 0] = self.diffusion_process.normalize_time(
                        t = u-1
                    )

                #####################  END OF DENOISING LOOP  ######################

                ####################################################################
                #                  MERGE THE OLD AND NEW SUBGRAPHS                 #
                ####################################################################

                graph = merge_sparse_dense_graphs_to_sparse(
                    sparse_subgraph = graph,
                    dense_subgraph = new_subgraph,
                    dense_ext_edges = new_ext_edges,
                    dense_nodes_num = nodes_to_insert
                )

            if do_save_chains:
                saved_mask = remaining_graphs_idx < save_chains
                saved_idx = remaining_graphs_idx[saved_mask]

                for i, g in zip(saved_idx, graph[saved_mask]):
                    g = g.collapse().cpu()
                    if return_directed:
                        g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
                    saved_chains[i].append(g)


            ######  COMPUTE HALTING SIGNAL  ######

            # update insertion time (in-place), insertion go up!
            graph.y[..., 0] = new_time
            t += 1
            
            if true_reinsertion_time is not None:
                halt_signal = true_reinsertion_time <= 1
            elif self.has_halting_model:
                curr_graph = graph.clone()
                self.add_additional_features(curr_graph)

                halt_signal = self.forward_halting(
                    graph = curr_graph
                )
            else:
                halt_signal = torch.zeros_like(remaining_graphs_idx, dtype=torch.bool)


            ####################################################################
            #                      CHECK COMPLETED GRAPHS                      #
            ####################################################################
            # check if any of the graphs is completed
            if maximum_number_nodes is not None:
                halt_signal = torch.logical_or(
                    halt_signal,
                    graph.num_nodes_per_sample >= maximum_number_nodes
                )

            if t == maximum_insertion_steps:
                completed_graphs_mask = torch.ones_like(halt_signal, dtype=torch.bool)
            else:
                completed_graphs_mask = halt_signal.bool()
            

            ###########  IF SOME GRAPHS ARE COMPLETED, REMOVE THEM  ############
            # TODO: the following might be costly, check it!
            completed_graphs_num = completed_graphs_mask.sum().item()
            if completed_graphs_num > 0:

                graph_list = graph.to_data_list()

                # compute completed and remaining graphs indices
                remaining_graphs_mask = ~completed_graphs_mask
                completed_graphs_idx = remaining_graphs_idx[completed_graphs_mask]
                remaining_graphs_idx = remaining_graphs_idx[remaining_graphs_mask]
                remaining_graphs_num = remaining_graphs_idx.shape[0]
                new_time = new_time[remaining_graphs_mask]

                # get completed and remaining graphs
                remaining_graphs = [graph_list[i] for i in torch.nonzero(remaining_graphs_mask).squeeze(-1)]
                completed_graphs = [graph_list[i] for i in torch.nonzero(completed_graphs_mask).squeeze(-1)]

                # insert finished graphs into the output batch
                for i, g in zip(completed_graphs_idx, completed_graphs):
                    if return_directed:
                        g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)

                    if conditioning_y is not None:
                        g.y = conditioning_y[i]
                    output_batch[i] = g.collapse()

                if remaining_graphs_num == 0:
                    break
                # resume remaining batch
                graph = Batch.from_data_list(remaining_graphs)

                if graph.y.ndim == 1:
                    graph.y = graph.y.unsqueeze(-1)

        ########################  END OF INSERTION LOOP  #######################

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # insert remaining graphs into the output batch
        for i, g in zip(remaining_graphs_idx, remaining_graphs):
            if return_directed:
                g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
            output_batch[i] = g.collapse()

        # store the output batch to cpu
        for i, g in enumerate(output_batch):
            output_batch[i] = g.cpu()

        # replace globals with starting variables, removing time
        if conditioning_y is None:
            for i in range(batch_size):
                output_batch[i].y = None
        else:
            for i in range(batch_size):
                output_batch[i].y = conditioning_y[i]
        
        if do_save_chains:
            return output_batch, saved_chains
        else:
            return output_batch
    
    @torch.no_grad()
    def sample_inference(
            self,
            batch_size: int,
            conditioning_y: Optional[Tensor]=None,
            sample_directed: bool=True,
            also_return_raw_graphs: bool=False,
            save_chains: int=0
        ):

        # sample graphs from the model
        batch: List[SparseGraph] = self.sample_batch(
            batch_size,
            conditioning_y,
            return_directed=sample_directed,
            save_chains=save_chains
        )

        if save_chains > 0:
            batch, saved_chains = batch

        # transform graphs into the format required for inference
        out_batch = self.inference_samples_converter(batch)

        ret = [out_batch]

        if also_return_raw_graphs:
            ret.append(batch)
        if save_chains > 0:
            ret.append(saved_chains)
            
        return ret if len(ret) > 1 else ret[0]
    

    ############################################################################
    #                         UTILITY MODULE FUNCTIONS                         #
    ############################################################################


    def add_additional_features(self, graph: SparseGraph|DenseGraph|Tuple[DenseGraph, DenseEdges]) -> Tensor:
        if self.exists_and_true('use_indegree'):

            if isinstance(graph, (SparseGraph, DenseGraph)):
                indegree = graph.indegree
                graph.x = torch.cat([graph.x, indegree.unsqueeze(-1)], dim=-1)
            else:
                remv_graph, edges_remv_surv = graph
                # in undirected graph, remv -> surv is the same as surv -> remv
                # so outdegree of surv is the same as indegree of remv
                add_indegree = edges_remv_surv.outdegree
                indegree = remv_graph.indegree + add_indegree
                graph[0].x = torch.cat([graph[0].x, indegree.unsqueeze(-1)], dim=-1)

        if self.exists_and_true('use_nodesnum'):
            
            if isinstance(graph, (SparseGraph, DenseGraph)):
                nodes_num = graph.num_nodes_per_sample
                graph.y = torch.cat([graph.y, nodes_num.unsqueeze(-1)], dim=-1)
            else:
                nodes_num = graph[0].num_nodes_per_sample
                graph[0].y = torch.cat([graph[0].y, nodes_num.unsqueeze(-1)], dim=-1)
                

    


################################################################################
#                               UTILITY METHODS                                #
################################################################################

# the following methods are utility methods which could be an integral part of
# the main class, but have been put outside for readability

##############################  DATA FORMATTING  ###############################

def format_generation_task_data(
        surv_graph: SparseGraph,
        remv_graph: SparseGraph,
        edges_surv_remv: SparseEdges=None,
        edges_remv_surv: SparseEdges=None
    ) -> Tuple[DenseGraph, Tensor, BoolTensor, Optional[DenseEdges], Optional[DenseEdges]]:
    """transform the splitting of the two graphs into the format required by the
    model, that is:
    - extract a dense representation (and a node mask) of the Ns nodes from surv_graph
    - transform remv_graph into a DenseGraph (with Nr nodes, and adjmat of shape (*, Nr, Nr, *))
    - transform edges_surv_remv and edges_remv_surv into dense adjacency matrices
      each of shape (*, Ns, Nr, *) and (*, Nr, Ns, *) respectively.
      If one of the two is None, it is assumed that the graph is undirected and
      a single adjacency matrix ((*, Ns, Nr, *) or (*, Nr, Ns, *)) is returned.

    Notice that the possibly very big adjacency matrix of surv_graph (*, Ns, Ns, *)
    is never computed, so Ns >> Nr is allowed, avoiding a squared dependency on
    Ns.

    Parameters
    ----------
    surv_graph : SparseGraph
        graph of nodes surviving the removal process.
    remv_graph : SparseGraph
        graph of nodes removed by the removal process.
    edges_surv_remv : Tuple[Tensor, Tensor]
        edges going from the surviving nodes to the removed nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)
    edges_remv_surv : Tuple[Tensor, Tensor], optional
        edges going from the removed nodes to the surviving nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)

    Returns
    -------
    remv_graph_dense : DenseGraph
        graph of nodes removed by the removal process as a dense graph.
    surv_x_tensor : Tensor
        tensor of the surviving nodes, as a batched dense representation.
    surv_node_mask : BoolTensor
        mask of the true surviving nodes, as the process of densifying generates
        some dummy nodes.
    adjmat_surv_remv : Optional[Tensor]
        edges going from the surviving nodes to the removed nodes, as a dense
        adjacency matrix. If edges_surv_remv is None, this is not returned.
    adjmat_remv_surv : Optional[Tensor]
        edges going from the removed nodes to the surviving nodes, as a dense
        adjacency matrix. If edges_remv_surv is None, this is not returned.
    """

    batch_size = remv_graph.num_graphs

    # extract the dense representation of the surviving nodes
    surv_x_tensor, surv_node_mask = to_dense_batch(
        x =				surv_graph.x,
        batch =			surv_graph.batch,
        batch_size =	batch_size
    )

    # transform the removed graph into a dense representation
    remv_graph_dense = dense.sparse_graph_to_dense_graph(
        sparse_graph =		remv_graph,
        handle_one_hot =    True
    )

    adjmats = []

    if (edges_surv_remv is not None) or (edges_remv_surv is not None):
        edge_mask_surv_remv = dense.get_bipartite_edge_mask_dense(
            node_mask_a = surv_node_mask,
            node_mask_b = remv_graph_dense.node_mask
        )


    # transform the edges_surv_remv into a dense adjacency matrix
    if edges_surv_remv is not None:

        adjmat_surv_remv = dense.to_dense_adj_bipartite(
            edge_index =	edges_surv_remv.edge_index,
            edge_attr =		edges_surv_remv.edge_attr,
            batch_s =		surv_graph.batch,
            batch_t =		remv_graph.batch,
            batch_size =	batch_size,
            handle_one_hot =True,
            edge_mask =     edge_mask_surv_remv
        )

        edges_surv_remv = DenseEdges(
            edge_adjmat =   adjmat_surv_remv,
            edge_mask =     edge_mask_surv_remv
        )

        adjmats.append(edges_surv_remv)

    # transform the edges_remv_surv into a dense adjacency matrix
    if edges_remv_surv is not None:
        # transpose
        edge_mask_remv_surv = edge_mask_surv_remv.transpose(1, 2)

        adjmat_remv_surv = dense.to_dense_adj_bipartite(
            edge_index =	edges_remv_surv.edge_index,
            edge_attr =		edges_remv_surv.edge_attr,
            batch_s =		remv_graph.batch,
            batch_t =		surv_graph.batch,
            batch_size =	batch_size,
            handle_one_hot =True,
            edge_mask =     edge_mask_remv_surv
        )

        edges_remv_surv = DenseEdges(
            edge_adjmat =   adjmat_remv_surv,
            edge_mask =     edge_mask_remv_surv
        )

        adjmats.append(edges_remv_surv)

    return remv_graph_dense, surv_x_tensor, surv_node_mask, *adjmats



def sparsify_data(
        subgraph: DenseGraph,
        ext_edges: DenseEdges,
        subgraph_nodes_num: IntTensor,
        ext_ptr: Tensor,
    ) -> Tuple[SparseGraph, SparseEdges]:

    ########################  SPARSIFY DENSE SUBGRAPH  #########################
    subgraph = subgraph.clone()

    # remove self-loops from dense adjacency matrices
    subgraph.edge_adjmat = dense.dense_remove_self_loops(
        subgraph.edge_adjmat
    )

    # remove no edge class from dense adjacency
    # matrices
    subgraph.edge_adjmat = dense.remove_no_edge(
        subgraph.edge_adjmat,
        sparse = False,
        collapsed = False
    )

    # transform the new graph to sparse format
    new_subgraph = dense.dense_graph_to_sparse_graph(
        dense_graph =	subgraph,
        num_nodes =		subgraph_nodes_num
    )

    ##########################  SPARSIFY DENSE EDGES  ##########################

    if ext_edges is not None:

        ext_edges.edge_adjmat = dense.remove_no_edge(
            ext_edges.edge_adjmat,
            sparse = False,
            collapsed = False
        )

        new_edges = dense.dense_edges_to_sparse_edges(
            dense_edges =		ext_edges,
            cum_num_nodes_s =	new_subgraph.ptr,
            cum_num_nodes_t =	ext_ptr
        )

    else:
        new_edges = None

    return new_subgraph, new_edges


def merge_sparse_dense_graphs_to_sparse(
        sparse_subgraph: SparseGraph,
        dense_subgraph: DenseGraph,
        dense_ext_edges: DenseEdges,
        dense_nodes_num: IntTensor
    ) -> SparseGraph:

    # sparsify the dense graph
    new_subgraph, new_edges_ba = sparsify_data(
        subgraph =				dense_subgraph,
        ext_edges =				dense_ext_edges,
        subgraph_nodes_num =	dense_nodes_num,
        ext_ptr = 				sparse_subgraph.ptr
    )

    if new_edges_ba is not None:

        # get both directions of the new edges
        new_edges_ab = new_edges_ba.clone().transpose()

        # merge the sparse graph with the sparsified dense graph
        merged_graph = split.merge_subgraphs(
            graph_a =	sparse_subgraph,
            graph_b =	new_subgraph,
            edges_ab =	new_edges_ab,
            edges_ba =	new_edges_ba
        )

    else:
        merged_graph = new_subgraph

    return merged_graph


###########################  BULK OPERATION METHODS  ###########################

def to_onehot_all(*data, **classes_nums):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = F.one_hot(
                d.long(), num_classes = classes_nums[k]
            ).float()

        elif isinstance(d, DenseEdges):
            ret_d = d.to_onehot(
                num_classes_e =	classes_nums[DIM_E]
            )
        
        elif isinstance(d, (DenseGraph, SparseGraph)):
            ret_d = d.to_onehot(
                num_classes_x =	classes_nums[DIM_X],
                num_classes_e =	classes_nums[DIM_E]
            )

        elif isinstance(d, Tensor):
            if d.dtype == torch.bool:
                ret_d = d.unsqueeze(-1)

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during to_onehot_all')
        
        ret_data.append(ret_d)

    return ret_data


def mask_all(*data, **masks):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = d * masks[k].unsqueeze(-1)
        
        elif isinstance(d, DenseGraph):
            ret_d = d.apply_mask()

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during mask_all')

        ret_data.append(ret_d)

    return ret_data


#########################  SMALL REPEATED OPERATIONS  ##########################
# the following methods are meant to abstract away some small operations that
# are repeated in the code

def append_time_to_graph_globals(
        graph: Union[DenseGraph, SparseGraph],
        time: Union[IntTensor, LongTensor],
    ) -> Union[DenseGraph, SparseGraph]:
    """Append the time to the graph globals vector y
    with the following criteria:
    - if the graph has no y, set y = time
    - if the graph has y, set y = [time, y], that is,
        the time is appended to the beginning of the vector

    Parameters
    ----------
    graph : Union[DenseGraph, SparseGraph]
        any kind of graph with batched y vector of size [batch_size, *] or None
    time : Union[IntTensor, LongTensor]
        time tensor of size [batch_size], this method will unsqueeze to [batch_size, 1]

    Returns
    -------
    same_graph : Union[DenseGraph, SparseGraph]
        same graph as the input, but with the updated y vector
    """

    time = time.float().unsqueeze(-1)

    if graph.y is None:
        graph.y = time
    else:
        if graph.y.ndim == 1:
            graph.y = graph.y.unsqueeze(-1)
        graph.y = torch.cat([time, graph.y], dim = -1)

    return graph





#################################  ASSERTIONS  #################################

def assert_is_onehot(*data):

    tensor_dims = {
        'xd': ('dense nodes', 3),
        'xs': ('sparse nodes', 2),
        'ed': ('dense edges', 4),
        'es': ('sparse edges', 2)
    }

    for i, d in enumerate(data):
        if isinstance(d, tuple):

            k: str
            d: Tensor
            k, d = d
            
            assert d.ndim == tensor_dims[k][1], \
                f'Expected {tensor_dims[k][0]} to be of dimension {tensor_dims[k][1]}, got {d.ndim}'

        elif isinstance(d, DenseGraph):
            assert not d.collapsed, \
                'Expected the dense graph to be onehot'
        
        elif isinstance(d, SparseGraph):
            assert_is_onehot(
                ('xs', d.x),
                ('es', d.edge_attr)
            )

        else:
            raise NotImplementedError(f'Expected {i}-th data to be of type tuple, DenseGraph or SparseGraph, got {type(d)}')