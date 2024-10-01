from typing import List

import src.data.transforms as base_t
from src.data.transforms import (
    preprocessing as prep_t,
    splitting as split_t,
    conditions as cond_t,
    graphs as grph_t,
    molecular as chem_t,
    qm9 as qm9_t
)
import src.data.dataset as ds


from torch_geometric.datasets.qm9 import conversion
import numpy as np


def graph_list_to_one_hot_transform(
        graph_list_df: str,
        num_classes_node_df: str=None,
        num_classes_edge_df: str=None,
        enable: bool=True
    ):

    if not enable:
        return None

    pipeline = base_t.DFIterateOver(
        datafield =         graph_list_df,
        iter_idx_df =       'curr_idx',
        iter_elem_df =      'curr_graph',

        transform=base_t.DFCompose([
            grph_t.DFGraphToOneHot( # should be inplace
                graph_df =      'curr_graph',
                num_classes_node_df = num_classes_node_df,
                num_classes_edge_df = num_classes_edge_df
            )
        ])
    )

    return pipeline