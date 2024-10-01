from typing import Optional, List

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as gnn
from src.models.core.rgcn import RGCN

class GNNRegressor(nn.Module):
    
    def __init__(
            self,
            node_in_channels: int,
            edge_dim: int,
            gnn_params: dict,
            gnn_out_channels: int,
            readout_hidden_channels: int,
            out_properties: int|List[int],
            architecture_type: str='gat',
            globals_dim: Optional[int]=None
        ):
        super().__init__()

        # no model means that to predict the output
        # there is no need for a model. This is the case
        # in the classification case with one class
        # which is done (for now) through a list
        # of one element with 1 property
        self.no_model = isinstance(out_properties, list) \
            and len(out_properties) == 1 \
            and out_properties[0] == 1
        
        if not self.no_model:

            if architecture_type == 'gat':
                gnn_type = gnn.GAT
                add_args = {'edge_dim': edge_dim}

            elif architecture_type == 'rgcn':
                gnn_type = RGCN
                add_args = {'num_relations': edge_dim}
            
            self.conv = gnn_type(
                in_channels =       node_in_channels,
                out_channels =      gnn_out_channels,
                **add_args,
                **gnn_params
            )

            readout_in_channels = gnn_out_channels

            self.using_y = globals_dim is not None
            self.out_properties = out_properties if isinstance(out_properties, list) else [out_properties]

            if self.using_y:
                readout_in_channels += globals_dim

            self.readouts = nn.ModuleList()

            for out_props_num in self.out_properties:

                self.readouts.append(
                    nn.Sequential(
                        nn.Linear(readout_in_channels, readout_hidden_channels),
                        nn.ReLU(),
                        nn.Linear(readout_hidden_channels, out_props_num)
                    )
                )

            # prediction for empty graphs
            self.empty_graphs_embedding = nn.Parameter(torch.rand(gnn_out_channels))


    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor]=None,
            batch_size: Optional[int]=None,
            y: Optional[Tensor]=None
        ):

        if not self.no_model:
            # graph attention convolution
            out_x = self.conv(
                x = 			x,
                edge_index =	edge_index,
                edge_attr =		edge_attr
            )

            # graph global pooling (sum all nodes)
            out_features = gnn.global_add_pool(out_x, batch=batch, size=batch_size)

            # replace empty graphs with the parameter vector
            # this was changed due to 
            mask = (out_features.sum(-1) == 0)
            idx = mask.nonzero(as_tuple=False).view(-1)
            empty_graphs_embedding_expand = self.empty_graphs_embedding.unsqueeze(0).expand(idx.shape[0], -1)
            out_features[idx] = empty_graphs_embedding_expand

            # concatenate global properties if needed
            if y is not None and self.using_y:
                out_features = torch.cat((out_features, y), dim=-1)
            
            out_properties = []

            for readout in self.readouts:
                # graph readout -> compute output properties
                prop = readout(out_features)
                if prop.shape[-1] == 1:
                    prop = prop.squeeze(-1)
                out_properties.append(prop)
            
            # if there is only one group of properties, return it directly
            ret_properties = out_properties if len(out_properties) > 1 else out_properties[0]
            
            return ret_properties

        else:
            return torch.ones(batch_size, 1, dtype=torch.float, device=x.device)