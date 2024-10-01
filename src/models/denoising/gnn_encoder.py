from typing import Optional, Dict, List

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as gnn

from src.models.core.rgcn import RGCN

class GNNNodeEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_dim: int,
            global_dim: int,
            return_all_layers: bool=False,
            architecture_type: str='gat',
            **gnn_encoder_config
        ):

        super().__init__()

        self.return_all_layers = return_all_layers

        if architecture_type == 'gat':
            gnn_type = gnn.GAT
            add_args = {'edge_dim': edge_dim}

        elif architecture_type == 'rgcn':
            gnn_type = RGCN
            add_args = {'num_relations': edge_dim}


        if return_all_layers:

            hidden_channels = gnn_encoder_config['hidden_channels']
            num_layers = gnn_encoder_config['num_layers']

            self.lin_inp = nn.Linear(
                in_channels + global_dim,
                hidden_channels
            )


            self.encoder = gnn_type(
                in_channels =		hidden_channels,
                out_channels =		hidden_channels,
                **add_args,
                **gnn_encoder_config
            )
            # self.encoder = gnn.PNA(
            #     in_channels =		hidden_channels,
            #     out_channels =		hidden_channels,
            #     edge_dim =          edge_dim,
            #     **gnn_encoder_config
            # )
                
            # hack into gnn encoder to return all layers
            self.encoder.jk = lambda xs: xs

            self.lins_out = nn.ModuleList(
                [nn.Linear(hidden_channels, out_channels) for _ in range(num_layers)]
            )

            self.norms_out = nn.ModuleList(
                [nn.LayerNorm(out_channels) for _ in range(num_layers)]
            )

        else:
        
            self.encoder = gnn_type(
                in_channels =		in_channels + global_dim,
                out_channels =		out_channels,
                **add_args,
                **gnn_encoder_config
            )

            self.norm = nn.LayerNorm(out_channels)
            



    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            y: Optional[Tensor]=None,
            num_nodes: Optional[Tensor]=None
        ):

        # concatenate the global features to the node features
        if y is not None:
            # repeat y for each node in the examples
            y = y.repeat_interleave(num_nodes, dim=0)
            x = torch.cat([x, y], dim=-1)


        # return the hidden state at each layer
        if self.return_all_layers:

            # encode only nodes
            x0 = self.lin_inp(x)

            # encode nodes and edges
            xs: List[Tensor] = self.encoder(
                x =				x0,
                edge_index =	edge_index,
                edge_attr =		edge_attr
            )

            # add the input to the hidden states
            xs.insert(0, x0)

            # for each hidden state, apply a linear projection
            # to the output state
            for i, (layer, norm) in enumerate(zip(self.lins_out, self.norms_out)):
                xs[i] = norm(layer(xs[i]))

            # stack all hidden states on a new dimension
            # if it is put after the node dimension, it is
            # treated as a feature dimension by torch_geometric
            # so all good in using their methods
            return torch.stack(xs, dim=1)
        
        # return only the network output
        else:
            x = self.encoder(
                x =				x,
                edge_index =	edge_index,
                edge_attr =		edge_attr
            )

            return self.norm(x)
            