from typing import Union, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import (
    RGCNConv,
    MessagePassing,
)


class RGCNConvLayer(RGCNConv):
    """This class is simply an adapter of RGCNConv to use it
    as a layer in a BasicGNN model, where only edge_weight
    and edge_attr can be passed. Then, edge_type from RGCNConv
    is passed as edge_attr to the super class.
    """

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]],
            edge_index: Tensor, edge_attr: Tensor = None):
        
        # if ndim == 2, then edge_attr is a one-hot encoding
        if edge_attr.ndim == 2:
            edge_type = torch.argmax(edge_attr, dim=-1)
        else:
            edge_type = edge_attr
        
        return super().forward(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type
        )


class RGCN(BasicGNN):

    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        return RGCNConvLayer(in_channels, out_channels, **kwargs)