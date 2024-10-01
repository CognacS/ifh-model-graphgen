from typing import Tuple, Dict
from collections import defaultdict

import torch
from torch import Tensor

from src.datatypes.sparse import SparseGraph
from torch_geometric.data.batch import Batch
from torch_geometric.utils import scatter

from copy import deepcopy

# same function above, but with "elems" instead of "nodes"
def compute_cum_elems(
        batch: Tensor,
        batch_size: int
    ) -> Tuple[Tensor, Tensor]:
    """Compute the number of elements per graph and the cumulative number of elements per graph in a batch.
    """
    one = batch.new_ones(batch.size(0))
    num_elems = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_elems = torch.cat([batch.new_zeros(1), num_elems.cumsum(dim=0)])
    return num_elems, cum_elems


def build_graphs_batch(
        graph: SparseGraph,
        batch: Tensor,
        batch_size: int,
        original_slice_dict: Dict,
        original_inc_dict: Dict
    ) -> SparseGraph:
    """Fill out batch information from a sparse graph (actually a batch of graphs), the batch index and the batch size.
    This entails:
    1 - fill information into BatchData
    2 - adding the batch index, slice_dict and inc_dict to the BatchData object

    Parameters
    ----------
    graph : Data
        batch of graphs composed of many disconnected graphs
    batch : Tensor
        batch index of each node
    batch_size : int
        size of the considered batch

    Returns
    -------
    Data
        batched version of graph, e.g. BatchData
    """

    # 1 - fill information into BatchData
    # copy slice_dict and inc_dict
    slice_dict, inc_dict = deepcopy(original_slice_dict), deepcopy(original_inc_dict)
    # recall that:
    # slice_dict: for each key, it contains the start and end index of the corresponding data in its datastructure
    # inc_dict: for each key, it contains the increment in node index for each graph in the batch, this is only filled for edge_index

    # fill out slice_dict
    _, cum_nodes = compute_cum_elems(batch, batch_size)
    for attr in graph.get_all_node_attrs():
        if attr in slice_dict:
            slice_dict[attr] = cum_nodes
    # get the index of the graph of which the edge belongs to
    edge_index_batch = batch[graph.edge_index[0]]
    _, cum_edges = compute_cum_elems(edge_index_batch, batch_size)
    slice_dict['edge_index'] = cum_edges
    slice_dict['edge_attr'] = cum_edges


    # fill out inc_dict
    inc_dict['edge_index'] = cum_nodes[:-1]

    
    # 2 - adding the batch index, slice_dict and inc_dict to the BatchData object
    graph.batch = batch
    graph.ptr = cum_nodes
    graph._num_graphs = batch_size
    graph._slice_dict = slice_dict
    graph._inc_dict = inc_dict


    return graph