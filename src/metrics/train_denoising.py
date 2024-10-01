##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/metrics/train_metrics.py
#
##########################################################################################################

from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import src.metrics.metrics as m_list

import torch.nn.functional as F

def generate_weights(mask: Tensor):
    num_elems = mask.flatten(start_dim=1).sum(dim=-1)
    alive_batches = (num_elems > 0).sum()
    weights_per_batch_elem = 1 / (num_elems * alive_batches)
    weights = torch.repeat_interleave(weights_per_batch_elem, num_elems)

    return weights
    

class SimpleTrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train_E: float = 1., lambda_train_ext_E: float = 1., concat_edges: bool = False, **kwargs):
        super().__init__()
        self.lambda_train_E = lambda_train_E
        self.lambda_train_ext_E = lambda_train_ext_E
        self.concat_edges = concat_edges

    def forward(
            self,
            pred_values: List[Tensor],
            true_values: List[Tensor],
            weighted: bool=False,
            class_weighted: bool=False,
            reduce: bool=True, ret_log: bool=False):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        assert not weighted or (weighted and len(pred_values) == 6), "If weighted, pred_values must contain masks"

        if len(pred_values) == 3:
            pred_x, pred_e, pred_ext_e = pred_values
        elif len(pred_values) == 6:
            pred_x, pred_e, pred_ext_e, nodes_mask, edges_mask, ext_edges_mask = pred_values

        true_x, true_e, true_ext_e = true_values

        # compute cross entropy loss
        reduction = 'mean' if reduce else 'none'

        reduction_to_do = reduction if not weighted else 'none'

        if class_weighted:
            edge_class_weights = torch.full((pred_e.shape[-1],), fill_value=5., device=pred_e.device)
            edge_class_weights[0] = 1.
        else:
            edge_class_weights = None


        loss_x = F.cross_entropy(pred_x, true_x, reduction=reduction_to_do) if true_x.numel() > 0 else torch.zeros(1, device=pred_x.device)
        loss_e = F.cross_entropy(pred_e, true_e, reduction=reduction_to_do, weight=edge_class_weights) if true_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
        loss_ext_e = F.cross_entropy(pred_ext_e, true_ext_e, reduction=reduction_to_do, weight=edge_class_weights) if true_ext_e.numel() > 0 else torch.zeros(1, device=pred_x.device)

        if weighted:
            nodes_weights = generate_weights(nodes_mask)
            edges_weights = generate_weights(edges_mask)
            ext_edges_weights = generate_weights(ext_edges_mask)
            loss_x = loss_x * nodes_weights
            loss_e = loss_e * edges_weights
            loss_ext_e = loss_ext_e * ext_edges_weights
            if reduction == 'mean':
                loss_x = loss_x.sum()
                loss_e = loss_e.sum()
                loss_ext_e = loss_ext_e.sum()

        if self.concat_edges:
            pred_e = torch.cat([pred_e, pred_ext_e], dim=0)
            true_e = torch.cat([true_e, true_ext_e], dim=0)
            loss_e = F.cross_entropy(pred_e, true_e, reduction=reduction_to_do, weight=edge_class_weights) if true_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
            if weighted:
                edges_weights = generate_weights(torch.cat([edges_mask, ext_edges_mask], dim=2))
                loss_e = loss_e * edges_weights
                if reduction == 'mean':
                    loss_e = loss_e.sum()

            if reduction == 'mean':
                total_loss: Tensor = loss_x + self.lambda_train_E * loss_e
            else:
                total_loss: Tensor = loss_x.mean() + self.lambda_train_E * loss_e.mean()

        else:
            if reduction == 'mean':
                total_loss: Tensor = loss_x + self.lambda_train_E * loss_e + self.lambda_train_ext_E * loss_ext_e
            else:
                total_loss: Tensor = loss_x.mean() + self.lambda_train_E * loss_e.mean() + self.lambda_train_ext_E * loss_ext_e.mean()

        if ret_log:
            to_log = {
                m_list.KEY_DENOISING_LOSS_X_CE: loss_x.detach(),
                m_list.KEY_DENOISING_LOSS_E_CE: loss_e.detach(),
                m_list.KEY_DENOISING_LOSS_EXT_E_CE: loss_ext_e.detach(),
                m_list.KEY_DENOISING_LOSS_TOTAL_CE: total_loss.detach(),
            }
            return total_loss, to_log
        else:
            return total_loss

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train_E: float = 1., lambda_train_ext_E: float = 1.):
        super().__init__()
        self.lambda_train_E = lambda_train_E
        self.lambda_train_ext_E = lambda_train_ext_E

    def forward(
            self,
            masked_pred_X: Tensor, masked_pred_E: Tensor, masked_pred_ext_E: Tensor,
            true_X: Tensor, true_E: Tensor, true_ext_E: Tensor,
            reduce: bool=True, ret_log: bool=False):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        # flatten all examples along a single dimension
        flatten = lambda x : torch.reshape(x, (-1, x.size(-1)))

        true_X = flatten(true_X)						# (bs * nq, dx)
        true_E = flatten(true_E)						# (bs * nq * nq, de)
        true_ext_E = flatten(true_ext_E)				# (bs * nq * nk, de)
        
        masked_pred_X = flatten(masked_pred_X)			# (bs * nq, dx)
        masked_pred_E = flatten(masked_pred_E)			# (bs * nq * nq, de)
        masked_pred_ext_E = flatten(masked_pred_ext_E)	# (bs * nq * nk, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)
        mask_ext_E = (true_ext_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        flat_true_ext_E = true_ext_E[mask_ext_E, :]
        flat_pred_ext_E = masked_pred_ext_E[mask_ext_E, :]

        # compute cross entropy loss
        reduction = 'mean' if reduce else 'none'

        loss_X = F.cross_entropy(flat_pred_X, flat_true_X, reduction=reduction) if true_X.numel() > 0 else 0.0
        loss_E = F.cross_entropy(flat_pred_E, flat_true_E, reduction=reduction) if true_E.numel() > 0 else 0.0
        loss_ext_E = F.cross_entropy(flat_pred_ext_E, flat_true_ext_E, reduction=reduction) if true_ext_E.numel() > 0 else 0.0

        total_loss: Tensor = loss_X + self.lambda_train_E * loss_E + self.lambda_train_ext_E * loss_ext_E

        if ret_log:
            to_log = {
                m_list.KEY_DENOISING_LOSS_X_CE: loss_X.detach() if true_X.numel() > 0 else -1,
                m_list.KEY_DENOISING_LOSS_E_CE: loss_E.detach() if true_E.numel() > 0 else -1,
                m_list.KEY_DENOISING_LOSS_EXT_E_CE: loss_ext_E.detach() if true_ext_E.numel() > 0 else -1,
                m_list.KEY_DENOISING_LOSS_TOTAL_CE: total_loss.detach() if total_loss.numel() > 0 else -1,
            }
            return total_loss, to_log
        else:
            return total_loss