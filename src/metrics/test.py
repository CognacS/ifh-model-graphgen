##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/metrics/abstract_metrics.py
#
##########################################################################################################

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError
from torch_geometric.utils import to_dense_batch

import numpy as np
from scipy.stats import wasserstein_distance


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        if values.ndim == 0:
            self.total_samples += 1
        else:
            self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples
    

def reduce_metric(values: Tensor, reduction: str, fill_value: float=0.0):
    if reduction == 'mean':
        return torch.mean(values) if values.numel() > 0 else torch.full((1,), fill_value, device=values.device)
    elif reduction == 'sum':
        return torch.sum(values)
    elif reduction == 'none':
        return values
    else:
        raise ValueError(f"Unknown reduction {reduction}")


def regression_accuracy(pred: Tensor, true: Tensor, reduction: str = 'mean') -> Tensor:
    """ Compute accuracy for regression task. """
    # compute error
    error = torch.abs(pred - true)
    # compute accuracy
    correct = (error < 0.5).float()
    # reduce
    accuracy = reduce_metric(correct, reduction)
    return accuracy

def classification_accuracy(pred_logits: Tensor, true: Tensor, reduction: str = 'mean') -> Tensor:
    """ Compute accuracy for classification task. """
    # compute accuracy
    pred = torch.argmax(pred_logits, dim=-1)
    correct = (pred == true).float()
    # reduce
    accuracy = reduce_metric(correct, reduction, fill_value=0.5)
    return accuracy

def binary_classification_accuracy(pred_logits: Tensor, true: Tensor, reduction: str = 'mean') -> Tensor:
    """ Compute accuracy for binary classification task. Inputs have final dimension 1. """
    # compute accuracy
    pred = (pred_logits > 0).float()
    correct = (pred == true).float()
    # reduce
    accuracy = reduce_metric(correct, reduction, fill_value=0.5)
    return accuracy

def binary_classification_recall(pred_logits: Tensor, true: Tensor, reduction: str = 'mean') -> Tensor:
    """ Compute recall for binary classification task. Inputs have final dimension 1. """

    # compute mask
    mask = true > 0.5

    # compute accuracy
    pred = (pred_logits > 0).float()
    positive = (pred == true)[mask].float()
    # reduce
    recall = reduce_metric(positive, reduction, fill_value=0.5)
    return recall


def binary_classification_precision(pred_logits: Tensor, true: Tensor, reduction: str = 'mean') -> Tensor:
    """ Compute precision for binary classification task. Inputs have final dimension 1. """

    # compute mask
    mask = pred_logits > 0

    # compute accuracy
    pred = (pred_logits > 0).float()
    positive = (pred == true)[mask].float()
    # reduce
    precision = reduce_metric(positive, reduction, fill_value=0.5)
    return precision
    

def halting_prior_emd(
        pred_logits: Tensor,
        true: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        max_seq_len: int,
        reduction: str = 'mean'
    ) -> Tensor:
    """ Compute EMD between predicted halting prior probability and true halting signal. """

    # compute batch from pred_logits and true
    if pred_logits.ndim == 1:
        pred_logits = pred_logits.unsqueeze(-1)
    if true.ndim == 1:
        true = true.unsqueeze(-1)

    # here shapes are: (batch_size_flat, 1)
    # to_dense_batch requires that the batch is ordered
    ord = torch.argsort(batch_idx)
    batch_idx = batch_idx[ord].long()
    pred_logits = pred_logits[ord]
    true = true[ord]
    
    # compute dense batch
    pred_logits, _ = to_dense_batch(pred_logits, batch=batch_idx, max_num_nodes=max_seq_len, batch_size=batch_size)
    true, mask =     to_dense_batch(true,        batch=batch_idx, max_num_nodes=max_seq_len, batch_size=batch_size)

    pred_logits = pred_logits.squeeze(-1)
    true = true.squeeze(-1)
    # here shapes are: (batch_size, max_seq_len)

    # compute halting prior
    pred_probs = torch.sigmoid(pred_logits)

    # compute prior as prod(1 - p_j) * p_i
    neg_cumprod = torch.cumprod(1 - pred_probs, dim=-1)
    neg_cumprod = torch.cat([torch.ones_like(neg_cumprod[..., :1]), neg_cumprod[..., :-1]], dim=-1)
    prior = neg_cumprod * pred_probs

    # correct prior to sum to 1 (remaining mass is placed first to penalize)
    prior[..., 0] = prior[..., 0] + 1 - prior.sum(-1)
    prior = torch.clamp(prior, 0, 1)

    # compute EMD
    true_np = true.cpu().numpy()
    prior_np = prior.cpu().numpy()
    values = np.arange(true_np.shape[-1])

    # compute pad mask
    mask = mask.cpu().numpy().astype(bool)

    # encapsulate in function for computing masked EMD
    fn_emd = lambda v, p_x, p_y: wasserstein_distance(v, v, p_x, p_y)
    fn_masked_emd = lambda v, p_x, p_y, m: fn_emd(v[m], p_x[m], p_y[m])

    # compute EMD
    emd = [fn_masked_emd(values, true_np[i], prior_np[i], mask[i]) for i in range(true_np.shape[0])]

    emd = torch.tensor(emd, device=pred_logits.device)

    # reduce
    emd = reduce_metric(emd, reduction)
    return emd