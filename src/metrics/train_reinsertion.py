from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import src.metrics.metrics as m_list


class DistributionReinsertionLoss(nn.Module):
    """
    This loss uses KL-divergence to match a distribution
    on the number of added nodes to the true distribution
    """

    def forward(
            self,
            pred_params: Tensor,
            true_params: Tensor,
            reduce: bool=True, ret_log: bool=False
        ):
        """compute the KL-divergence between the predicted and true
        distributions of the number of added nodes

        Parameters
        ----------
        pred_params : List[Tensor]
            _description_
        true_params : List[Tensor]
            true params are expected to be standard probabilities
        reduce : bool, optional
            _description_, by default True
        ret_log : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        pred_logits = pred_params
        true_probs = true_params

        # convert softmax weights to log probabilities (needed for KL-divergence)
        pred_logprobs = torch.log_softmax(pred_logits, dim=-1)


        kl_div: Tensor = F.kl_div(
            input = 	pred_logprobs,
            target = 	torch.clip(true_probs, min=1e-8), # account for 0 probabilities
            reduction = 'batchmean' if reduce else 'none'
        )

        # kl_div: Tensor = F.cross_entropy(
        #     input = 	pred_logits,
        #     target = 	true_probs.argmax(dim=-1),
        #     reduction = 'mean' if reduce else 'none'
        # )

        if not reduce:
            kl_div = kl_div.sum(dim=-1)

        total_loss = kl_div

        if ret_log:
            to_log = {
                m_list.KEY_REINSERTION_LOSS_KLDIV: kl_div.detach(),
            }
            return total_loss, to_log

        else:
            return total_loss

class RegressionReinsertionLoss(nn.Module):
    """
    This variant only uses the MSE loss for predicting the correct number
    of remaining nodes
    """

    def forward(
            self,
            pred_params: List[Tensor],
            true_params: List[Tensor],
            reduce: bool=True, ret_log: bool=False
        ):

        pred_remaining_nodes = pred_params
        true_remaining_nodes = true_params

        support_term: Tensor = F.mse_loss(
            input = 	pred_remaining_nodes,
            target = 	true_remaining_nodes,
            reduction = 'mean' if reduce else 'none'
        )

        total_loss = support_term

        if ret_log:
            to_log = {
                m_list.KEY_REINSERTION_LOSS_MSE: support_term.detach(),
            }
            return total_loss, to_log

        else:
            return total_loss