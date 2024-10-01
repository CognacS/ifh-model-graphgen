from typing import List

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import src.metrics.metrics as m_list
from src.metrics.utils.losses import sigmoid_focal_loss


class HaltingLoss(nn.Module):
    """
    This loss uses the binary cross entropy loss to match the halt signal
    """

    def __init__(self, use_focal_loss: bool = False):
        super().__init__()
        self.use_focal_loss = use_focal_loss

    def forward(
            self,
            pred_halt_logits: Tensor,
            true_halt_signal: Tensor,
            reduce: bool=True, ret_log: bool=False
        ):

        if not self.use_focal_loss:
            halt_signal_loss = F.binary_cross_entropy_with_logits(
                input =		pred_halt_logits,
                target =	true_halt_signal.float(),
                reduction =	'mean' if reduce else 'none'
            )
        else:
            halt_signal_loss = sigmoid_focal_loss(
                input =		pred_halt_logits,
                target =	true_halt_signal.float(),
                reduction =	'mean' if reduce else 'none',
                alpha = -1,
                gamma = 1
            )

        if ret_log:
            to_log = {
                m_list.KEY_HALTING_LOSS_BCE: halt_signal_loss.detach(),
            }
            return halt_signal_loss, to_log

        else:
            return halt_signal_loss