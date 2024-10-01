from typing import Optional, Dict, Tuple

from torch import Tensor
import torch.nn as nn

import torch_geometric.nn as gnn

from src.datatypes.dense import DenseGraph, DenseEdges
from src.datatypes.sparse import SparseGraph

from src.models.denoising.graph_transformer import GraphTransformer, DIM_X, DIM_E, DIM_Y
from src.models.denoising.gnn_encoder import GNNNodeEncoder


class ConditionalGraphTransformer(nn.Module):

    def __init__(
            self,
            input_dims: Dict,
            output_dims: Dict,
            transf_config: Dict,
            gnn_encoder_config: Optional[Dict]=None,
            use_exp_encoder: bool=False
        ):

        super().__init__()

        # transformer: generates the new subgraph prediction
        # - input:
        # 	- dense new subgraph: X, E, y, node_mask,
        #   - dense external graph: ext_X, ext_node_mask
        # 	- dense adjmat from the new to external graph: ext_E
        # - output:
        # 	- dense generated new subgraph: new_X, new_E, new_y
        # 	- dense generated adjmat from the new to external graph: new_ext_E
        self.transformer = GraphTransformer(
            input_dims = input_dims,
            output_dims = output_dims,
            **transf_config
        )

        self.use_ext_encoder = gnn_encoder_config is not None
        self.use_exp_encoder = use_exp_encoder
        # external graph encoder: encode the external graph representation
        # - input: external sparse graph: x, edge_index, edge_attr
        # - output: encoded nodes: out_x, of the same size as the input of the transformer layers
        if self.use_ext_encoder:
            if self.use_exp_encoder:
                self.ext_encoder = GNNNodeEncoder(
                    in_channels =		input_dims[DIM_X],
                    out_channels =		transf_config['transf_inout_dims'][DIM_X],
                    edge_dim =          input_dims[DIM_E]-1, # recall: edge dims are given considering the empty edge,
                    global_dim =        input_dims[DIM_Y]-1, # recall: global dims are given considering the diff. step
                    **gnn_encoder_config
                )

            else:
                self.ext_encoder = gnn.GAT(
                    in_channels =		input_dims[DIM_X],
                    out_channels =		transf_config['transf_inout_dims'][DIM_X],
                    edge_dim =          input_dims[DIM_E]-1, # recall: edge dims are given considering the empty edge
                    **gnn_encoder_config
                )

    
    def forward_encoding(
            self,
            ext_graph: SparseGraph
        ) -> Tensor:

        if not self.use_ext_encoder:
            return ext_graph.x
        
        # encode the external graph
        if self.use_exp_encoder:
            out_x = self.ext_encoder(
                x =				ext_graph.x,
                edge_index =	ext_graph.edge_index,
                edge_attr =		ext_graph.edge_attr,
                y =             ext_graph.y,
                num_nodes =     ext_graph.num_nodes_per_sample
            )
        else:
            out_x = self.ext_encoder(
                x =				ext_graph.x,
                edge_index =	ext_graph.edge_index,
                edge_attr =		ext_graph.edge_attr
            )

        return out_x
    
    def forward_transformer(
            self,
            subgraph: DenseGraph,
            ext_edges_new_to_ext: DenseEdges,
            ext_X: Tensor,
            ext_node_mask: Tensor
        ) -> Tuple[DenseGraph, Tensor]:

        # process the subgraph into the generated subgraph 
        new_subgraph, new_ext_edges = self.transformer(
            graph =         subgraph,
            ext_X =			ext_X,
            ext_node_mask =	ext_node_mask,
            ext_edges =     ext_edges_new_to_ext
        )

        return new_subgraph, new_ext_edges
    


    def forward(
            self,
            subgraph: DenseGraph,
            ext_E: Tensor,
            ext_graph: SparseGraph,
            num_iter: int

        ) -> Tuple[DenseGraph, Tensor]:

        pass