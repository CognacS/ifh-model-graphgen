from typing import Dict, List, Callable, Iterable, Optional

import pickle as pkl
import networkx as nx

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse, remove_self_loops
from torch_geometric.utils.convert import from_networkx

from . import DFBaseTransform
from src.datatypes.sparse import SparseGraph


class DFCollectGraphNodesStatistics(DFBaseTransform):

    def __init__(
            self,
            list_of_graphs_df: str,
            df_stats_dict: Optional[Dict[str, Callable[[Iterable], float]]]=None,
            histogram_df: str=None
        ):

        self.list_of_graphs_df = list_of_graphs_df
        self.df_stats_dict = df_stats_dict
        self.histogram_df = histogram_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # compute number of nodes
        graphs: List[Data] = data[self.list_of_graphs_df]
        num_nodes = np.array([g.num_nodes for g in graphs])

        # compute statistics
        if self.df_stats_dict is not None:
            # for each given statistic function
            # store it into data
            for df, stat_f in self.df_stats_dict.items():
                data[df] = stat_f(num_nodes).item()

        if self. histogram_df is not None:
            data[self.histogram_df] = get_dict_histogram(
                array=num_nodes
            )

        return data
    
    @property
    def input_df_list(self) -> List[str]:

        return [self.list_of_graphs_df]

    @property
    def output_df_list(self) -> List[str]:
        if self.df_stats_dict is None:
            df_stats_dict = []
        else:
            df_stats_dict = list(self.df_stats_dict.keys())

        if self.histogram_df is None:
            histogram_df = []
        else:
            histogram_df = [self.histogram_df]

        return df_stats_dict + histogram_df


class DFCollectGraphValuesStatistics(DFBaseTransform):

    def __init__(
            self,
            list_of_graphs_df: str,
            df_node_stats_dict: Optional[Dict[str, Callable[[Iterable], float]]]=None,
            df_edge_stats_dict: Optional[Dict[str, Callable[[Iterable], float]]]=None,
        ):

        self.list_of_graphs_df = list_of_graphs_df
        self.df_node_stats_dict = df_node_stats_dict
        self.df_edge_stats_dict = df_edge_stats_dict



    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # compute number of nodes
        graphs: List[Data] = data[self.list_of_graphs_df]

        if self.df_node_stats_dict is not None:
            # gather all node features
            x_list = [g.x for g in graphs]
            # compute statistics
            for df, stat_f in self.df_node_stats_dict.items():
                data[df] = stat_f(x_list).item()

        if self.df_edge_stats_dict is not None:
            # gather all edge features
            edge_attr_list = [g.edge_attr for g in graphs]
            # compute statistics
            for df, stat_f in self.df_edge_stats_dict.items():
                data[df] = stat_f(edge_attr_list).item()

        return data
    
    @property
    def input_df_list(self) -> List[str]:

        return [self.list_of_graphs_df]

    @property
    def output_df_list(self) -> List[str]:
        if self.df_node_stats_dict is None:
            df_node_stats_dict = []
        else:
            df_node_stats_dict = list(self.df_node_stats_dict.keys())

        if self.df_edge_stats_dict is None:
            df_edge_stats_dict = []
        else:
            df_edge_stats_dict = list(self.df_edge_stats_dict.keys())

        return df_node_stats_dict + df_edge_stats_dict
    


class DFNetworkxToGraphs(DFBaseTransform):
    
        def __init__(
                self,
                list_of_nxgraphs_df: str,
                list_of_graphs_df: str,
                to_one_hot: bool=False,
                remove_self_loops: bool=True
            ):
    
            self.list_of_nxgraphs_df = list_of_nxgraphs_df
            self.list_of_graphs_df = list_of_graphs_df
            self.to_one_hot = to_one_hot
            self.remove_self_loops = remove_self_loops
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
            """transform a networkx graph into a torch_geometric Data object
            """

            # get networkx graphs
            nxgraphs: List[nx.Graph] = data[self.list_of_nxgraphs_df]

            # convert to torch_geometric Data objects
            torch_graphs = [from_networkx(g) for g in nxgraphs]

            # get number of nodes
            num_nodes: List[int] = [g.num_nodes for g in torch_graphs]

            # convert to sparse edge_index
            get_edge_index = lambda g: g.edge_index if not self.remove_self_loops else remove_self_loops(g.edge_index)[0]
            edge_index_list = [get_edge_index(g) for g in torch_graphs]

            # function which returns a tensor of zero category or one-hot zero category
            # i.e. one component which is 1
            zeros_tensor = lambda n: torch.ones(n, 1) if self.to_one_hot else torch.zeros(n)

            data[self.list_of_graphs_df] = [
                SparseGraph(
                    x =             zeros_tensor(n),
                    edge_index =    edge_index,
                    edge_attr =     zeros_tensor(edge_index.shape[1]),
                    y =             None
                )
                for n, edge_index in zip(num_nodes, edge_index_list)
            ]

            return data
            



class DFAdjMatsToGraphs(DFBaseTransform):

    def __init__(
            self,
            list_of_adjmats_df: str,
            list_of_graphs_df: str,
            to_one_hot: bool=False
        ):

        self.list_of_adjmats_df = list_of_adjmats_df
        self.list_of_graphs_df = list_of_graphs_df
        self.to_one_hot = to_one_hot


    def __call__(
            self,
            data: Dict,
        ) -> Dict:
        """transform a adjacency matrices (torch Tensors)
        into a sparse edge_index, and finally compose
        the Data object
        """

        # get adjacency matrices
        adjmats: List[Tensor] = data[self.list_of_adjmats_df]

        num_nodes: List[int] = [adjmat.shape[0] for adjmat in adjmats]

        # convert to sparse edge_index
        edge_index_list = [dense_to_sparse(adjmat)[0] for adjmat in adjmats]

        # function which returns a tensor of zero category or one-hot zero category
        # i.e. one component which is 1
        zeros_tensor = lambda n: torch.ones(n, 1) if self.to_one_hot else torch.zeros(n)

        # compose Data objects
        data[self.list_of_graphs_df] = [
            SparseGraph(
                x =             zeros_tensor(n),
                edge_index =    edge_index,
                edge_attr =     zeros_tensor(edge_index.shape[1]),
                y =             None
            )
            for n, edge_index in zip(num_nodes, edge_index_list)
        ]
        
        return data
        



class DFSaveGraphListTorch(DFBaseTransform):

    def __init__(
            self,
            graph_list_df: str,
            save_path_df: str
        ):

        # save list of torch_geometric Data graphs
        self.graph_list_df = graph_list_df
        self.save_path_df = save_path_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        data_list = data[self.graph_list_df]
        path = data[self.save_path_df]

        torch.save(
            obj=InMemoryDataset.collate(data_list),
            f=path
        )

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.graph_list_df, self.save_path_df]
    


class DFLoadGraphListTorch(DFBaseTransform):
    
        def __init__(
                self,
                load_path_df: str,
                graph_list_df: str
            ):
    
            # load list of torch_geometric Data graphs
            self.load_path_df = load_path_df
            self.graph_list_df = graph_list_df
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            path = data[self.load_path_df]
    
            data[self.graph_list_df] = torch.load(
                f=path
            )
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.load_path_df]
    
        @property
        def output_df_list(self) -> List[str]:
            return [self.graph_list_df]
        

class DFLoadGraphListPickle(DFBaseTransform):
    
        def __init__(
                self,
                load_path_df: str,
                graph_list_df: str
            ):
    
            # load list of torch_geometric Data graphs
            self.load_path_df = load_path_df
            self.graph_list_df = graph_list_df
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            path = data[self.load_path_df]
    
            # data[self.graph_list_df] = torch.load(
            #     f=path
            # )

            with open(path, 'rb') as f:
                data[self.graph_list_df] = pkl.load(f)
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.load_path_df]
    
        @property
        def output_df_list(self) -> List[str]:
            return [self.graph_list_df]




def get_dict_histogram(array: np.ndarray):

    unique, counts = np.unique(
        array,
        return_counts=True
    )

    return dict(zip(unique.tolist(), counts.tolist()))


from src.datatypes.utils import one_hot

class DFGraphToOneHot(DFBaseTransform):

    def __init__(
            self,
            graph_df: str,
            one_hot_df: str=None, # if this is None, inplace
            num_classes_node_df: str=None,
            num_classes_edge_df: str=None
        ):

        self.graph_df = graph_df
        self.one_hot_df = one_hot_df
        self.num_classes_node_df = num_classes_node_df
        self.num_classes_edge_df = num_classes_edge_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        graph: SparseGraph = data[self.graph_df]

        # make a copy if not inplace
        if self.one_hot_df is not None:
            graph = graph.clone()

        # convert nodes to onehot
        if self.num_classes_node_df is not None:
            graph.x = one_hot(
                graph.x,
                num_classes=data[self.num_classes_node_df]
            )
        
        # convert edges to onehot
        if self.num_classes_edge_df is not None:
            graph.edge_attr = one_hot(
                graph.edge_attr,
                num_classes=data[self.num_classes_edge_df]
            )
        

        if self.one_hot_df is not None:
            data[self.one_hot_df] = graph

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.graph_df, self.num_classes_node_df, self.num_classes_edge_df]
    
    @property
    def output_df_list(self) -> List[str]:
        return [self.one_hot_df]
    
