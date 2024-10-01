from __future__ import annotations

from typing import Tuple, List, Optional, Union, Callable, Dict, Any

import torch
from torch import Tensor, BoolTensor, IntTensor
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, remove_self_loops, dense_to_sparse, scatter
from src.datatypes.utils import one_hot

from src.datatypes.sparse import SparseGraph, SparseEdges

################################################################################
#                                  DENSE GRAPH                                 #
################################################################################
class DenseGraphException(Exception):
    pass

class DenseGraph:
    def __init__(
            self,
            x: Tensor,
            edge_adjmat: Tensor,
            y: Tensor=None,
            node_mask: BoolTensor=None,
            edge_mask: Optional[Tensor]=None,
            masked=False
        ):
        self.x = x
        self.edge_adjmat = edge_adjmat
        self.y = y
        self.node_mask = node_mask
        self.edge_mask = edge_mask
        self.masked = masked
        self.collapsed = self.check_collapsed()


    @property
    def device(self):
        return self.x.device
    
    def to(self, device: str) -> DenseGraph:
        """ Changes the device of x, edge_adjmat, y. """
        ret_graph = DenseGraph(
            x = 			self.x.to(device),
            edge_adjmat = 	self.edge_adjmat.to(device),
            y =				None if self.y is None else self.y.to(device),
            node_mask =		None if self.node_mask is None else self.node_mask.to(device),
            edge_mask =		None if self.edge_mask is None else self.edge_mask.to(device),
        )

        ret_graph.masked = self.masked
        ret_graph.collapsed = self.collapsed

        return ret_graph

    def __setattr__(self, key: str, value: Any):
        """if x or edge_adjmat are set, then set masked to False"""
        if key in ['x', 'edge_adjmat']:
            self.masked = False
        super().__setattr__(key, value)
        if key in ['x', 'edge_adjmat'] and value is not None:
            self.collapsed = self.check_collapsed()


    def check_collapsed(self):
        return self.x.ndim == 2


    def type_as(self, x: Tensor):
        """ Changes the device and dtype of x, edge_adjmat, y. """
        self.x = 			self.x.type_as(x)
        self.edge_adjmat = 	self.edge_adjmat.type_as(x)
        self.y = 			self.y.type_as(x)
        return self


    def set_node_mask(self, node_mask: BoolTensor):
        self.node_mask = node_mask
    
    def set_edge_mask(self, edge_mask: Tensor):
        self.edge_mask = edge_mask

    
    def _apply_mask_nodes(self) -> DenseGraph:

        if self.collapsed:
            self.x = self.x * self.node_mask
        else:
            self.x = self.x * self.node_mask.unsqueeze(-1)

        return self

    
    def _apply_mask_edges(self) -> DenseGraph:

        # convert masks to be used on nodes and adjmats
        self.get_edge_mask_dense()
            
        if self.collapsed:
            self.edge_adjmat = self.edge_adjmat * self.edge_mask
        else:
            self.edge_adjmat = self.edge_adjmat * self.edge_mask.unsqueeze(-1)

        return self


    def apply_mask(self) -> DenseGraph:

        if self.node_mask is None:
            raise DenseGraphException(
                'Trying to mask dense graph, but it has no mask set. Call set_mask() to set it'
            )
        
        if not self.masked:
            self._apply_mask_nodes()._apply_mask_edges()
            self.masked = True

        return self
    

    def _collapse_nodes(self) -> DenseGraph:
        self.x = torch.argmax(self.x, dim=-1)
        return self

    def _collapse_edges(self) -> DenseGraph:
        self.edge_adjmat = torch.argmax(self.edge_adjmat, dim=-1)
        return self


    def collapse(self) -> DenseGraph:
        """returns a DenseGraph where each entry is a class instead of a feature
        vector

        Returns
        -------
        collapsed_graph : DenseGraph
            this graph but with classes instead of feature vectors
        """

        if not self.collapsed:

            # collapse to classes
            self._collapse_nodes()._collapse_edges()
            self.collapsed = True

        return self
    

    def _to_onehot_nodes(self, num_classes: int) -> DenseGraph: 
        self.x = one_hot(self.x, num_classes = num_classes, dtype=torch.float)
        return self
    
    def _to_onehot_edges(self, num_classes: int) -> DenseGraph:
        self.edge_adjmat = one_hot(self.edge_adjmat, num_classes = num_classes, dtype=torch.float)
        return self


    def to_onehot(self, num_classes_x: int, num_classes_e: int) -> DenseGraph:

        if self.collapsed:
            self._to_onehot_nodes(num_classes_x)._to_onehot_edges(num_classes_e)

            self.collapsed = False
    
        return self
    

    def get_edge_mask_dense(self) -> Tensor:
        if self.edge_mask is None:
            self.edge_mask = get_edge_mask_dense(node_mask=self.node_mask)

        return self.edge_mask


    def clone(self) -> DenseGraph:
        ret_graph = DenseGraph(
            x = 			self.x.clone(),
            edge_adjmat = 	self.edge_adjmat.clone(),
            y =				None if self.y is None else self.y.clone(),
            node_mask =		None if self.node_mask is None else self.node_mask.clone(),
            edge_mask =		None if self.edge_mask is None else self.edge_mask.clone()
        )

        ret_graph.masked = self.masked
        ret_graph.collapsed = self.collapsed

        return ret_graph
    
    def __getitem__(self, idx) -> Any:

        graph = DenseGraph(
            x =             self.x[idx],
            edge_adjmat =   self.edge_adjmat[idx],
            y =             None if self.y is None else self.y[idx],
            node_mask =     self.node_mask[idx],
            edge_mask =     self.get_edge_mask_dense()[idx],
            masked =        self.masked
        )

        return graph
    
    @property
    def num_graphs(self) -> int:
        return self.x.shape[0]
    
    @property
    def num_nodes(self) -> int:
        return self.node_mask.sum().item()
    
    @property
    def num_nodes_per_sample(self) -> Tensor:
        return self.node_mask.sum(dim=-1)
    
    @property
    def outdegree(self) -> Tensor:
        """Get the outdegree of the graph or batch of graphs."""
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='out') * self.node_mask
    
    @property
    def indegree(self) -> Tensor:
        """Get the indegree of the graph or batch of graphs."""
        self.apply_mask()
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='in') * self.node_mask
    
    
    def __repr__(self):
        return f'DenseGraph(x={self.x.shape}, edge_adjmat={self.edge_adjmat.shape}, y={self.y.shape}, node_mask={self.node_mask.shape}, masked={self.masked}, collapsed={self.collapsed})'
    

def compute_degree_adjmat(edge_adjmat: Tensor, collapsed: bool, inout='in') -> Tensor:

    if inout == 'in':
        dim = 1
    elif inout == 'out':
        dim = 2
    else:
        raise ValueError(f'Invalid value for inout: {inout}')
    
    if collapsed:
        return torch.sum(edge_adjmat > 0, dim=dim)
    else:
        return torch.sum(edge_adjmat[..., 1:].sum(-1), dim=dim)
    



class DenseEdges(DenseGraph):

    def __init__(
            self,
            edge_adjmat: Tensor,
            edge_mask: Optional[Tensor] = None,
            masked: bool = False
        ):
        super().__init__(
            x = None,
            edge_adjmat = edge_adjmat,
            y = None,
            node_mask = None,
            edge_mask = edge_mask,
            masked = masked
        )


    @property
    def device(self):
        return self.edge_adjmat.device
    

    def to(self, device: str) -> DenseEdges:
        """ Changes the device of x, edge_adjmat, y. """
        ret_graph = DenseEdges(
            edge_adjmat = 	self.edge_adjmat.to(device),
            edge_mask =		self.edge_mask.to(device)
        )

        ret_graph.masked = self.masked
        ret_graph.collapsed = self.collapsed

        return ret_graph
    

    def check_collapsed(self):
        return self.edge_adjmat.ndim == 3


    def type_as(self, x: Tensor):
        """ Changes the device and dtype of x, edge_adjmat, y. """
        self.edge_adjmat = 	self.edge_adjmat.type_as(x)
        return self


    def apply_mask(self) -> DenseGraph:

        if self.edge_mask is None:
            raise DenseGraphException(
                'Trying to mask dense graph, but it has no mask set. Call set_mask() to set it'
            )
        
        if not self.masked:
            self._apply_mask_edges()
            self.masked = True

        return self
    

    def collapse(self) -> DenseGraph:
        """returns a DenseGraph where each entry is a class instead of a feature
        vector

        Returns
        -------
        collapsed_graph : DenseGraph
            this graph but with classes instead of feature vectors
        """

        if not self.collapsed:

            # collapse to classes
            self._collapse_edges()
            self.collapsed = True

        return self


    def to_onehot(self, num_classes_e: int) -> DenseGraph:

        if self.collapsed:
            self._to_onehot_edges(num_classes_e)

            self.collapsed = False
    
        return self


    def clone(self) -> DenseGraph:
        ret_graph = DenseEdges(
            edge_adjmat = 	self.edge_adjmat.clone(),
            edge_mask =		None if self.edge_mask is None else self.edge_mask.clone()
        )

        ret_graph.masked = self.masked
        ret_graph.collapsed = self.collapsed

        return ret_graph
    
    @property
    def num_graphs(self) -> int:
        return self.edge_adjmat.shape[0]
    
    @property
    def outdegree(self) -> Tensor:
        """Get the outdegree of the graph or batch of graphs."""
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='out')
    
    @property
    def indegree(self) -> Tensor:
        """Get the indegree of the graph or batch of graphs."""
        self.apply_mask()
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='in')
    
    
    def __repr__(self):
        return f'DenseEdges(edge_adjmat={self.edge_adjmat.shape}, edge_mask={self.edge_mask.shape}, masked={self.masked}, collapsed={self.collapsed})'




################################################################################
#                         DENSE GRAPH UTILITY METHODS                          #
################################################################################


###############################  SPARSE TO DENSE  ##############################

def to_dense_adj_bipartite(
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_s: Optional[Tensor] = None,
        batch_t: Optional[Tensor] = None,
        max_num_nodes_s: Optional[int] = None,
        max_num_nodes_t: Optional[int] = None,
        batch_size: int = None,
        handle_one_hot: bool = False,
        edge_mask: Optional[Tensor] = None
    ) -> Tensor:
    """Convert the edge_index and edge_attr for edges from a source graph to a
    target graph to a dense adjacency matrix. This method is an adaptation of
    the to_dense_adj method from the PyTorch Geometric library, taking into account
    the fact that the source and target graphs are different. This way, the
    adjacency matrix is of shape (B, N_s, N_t, F) instead of (B, N, N, F), i.e.
    a square matrix.

    Parameters
    ----------
    edge_index : Tensor
        Tensor (2, E) containing the node indices of the source and target nodes
    edge_attr : Tensor
        Tensor (E, F) containing the edge attributes
    batch_s : Optional[Tensor], optional
        Tensor (N_s,) containing the batch example index for each node in the source
        graph, by default None
    batch_t : Optional[Tensor], optional
        Tensor (N_t,) containing the batch example index for each node in the target
        graph, by default None
    max_num_nodes_s : Optional[int], optional
        maximum number of nodes of the source graph examples in the batch, by default None
    max_num_nodes_t : Optional[int], optional
        maximum number of nodes of the target graph examples in the batch, by default None
    batch_size : int, optional
        number of examples in the batch, by default None

    Returns
    -------
    Tensor
        Tensor (B, N_s, N_t, F) containing the dense adjacency matrix for the batch
    """
    
    # setup initial variables
    if batch_s is None:
        num_nodes_s = int(edge_index[0].max()) + 1 if edge_index.numel() > 0 else 0
        batch_s = edge_index.new_zeros(num_nodes_s)
    if batch_t is None:
        num_nodes_t = int(edge_index[1].max()) + 1 if edge_index.numel() > 0 else 0
        batch_t = edge_index.new_zeros(num_nodes_t)

    if batch_size is None:
        batch_size = 1


    # function for computing cum_nodes for batch_s and batch_t, with type hinting
    def compute_cum_nodes(batch: Tensor) -> Tuple[Tensor, Tensor]:
        one = batch.new_ones(batch.size(0))
        num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
        return num_nodes, cum_nodes
    
    num_nodes_s, cum_nodes_s = compute_cum_nodes(batch_s)
    num_nodes_t, cum_nodes_t = compute_cum_nodes(batch_t)
    
    # for each edge:
    # idx0: batch index
    idx0 = batch_s[edge_index[0]]
    # idx1: row index (source node)
    idx1 = edge_index[0] - cum_nodes_s[batch_s][edge_index[0]]
    # idx2: column index (target node)
    idx2 = edge_index[1] - cum_nodes_t[batch_t][edge_index[1]]
    

    # get maximum number of nodes for both source and target
    if max_num_nodes_s is None:
        max_num_nodes_s = int(num_nodes_s.max())
    if max_num_nodes_t is None:
        max_num_nodes_t = int(num_nodes_t.max())
    
    # set edge_attr to ones if not given
    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)


    # setup size of the adjacency matrix
    size = [batch_size, max_num_nodes_s, max_num_nodes_t]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes_s * max_num_nodes_t

    # idx0 gives the batch index, idx1 the row index, idx2 the column index
    idx = idx0 * max_num_nodes_s * max_num_nodes_t + idx1 * max_num_nodes_t + idx2
    if handle_one_hot and edge_mask is not None:
        # pad one_hot vectors, e.g.: [1, 0, 0] -> [0, 1, 0, 0]
        # this creates a new class for the "no edge" case
        edge_attr = F.pad(edge_attr, (1, 0))
        # allocate the adj matrix
        adj = F.pad(edge_mask.float().flatten().unsqueeze(-1), (0, edge_attr.shape[-1]-1))
        # fill up the adj matrix with the edge_attr
        adj[idx, :] = edge_attr
        size[-1] += 1
    else:
        # build adjacency matrix by putting edge_attr in the right place
        adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = adj.view(size)

    return adj


##############################  DENSE TO SPARSE  ###############################

def dense_to_sparse(
        adj: Tensor,
        cum_num_nodes_s: Optional[Tensor] = None,
        cum_num_nodes_t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
    """Converts a dense bipartite adjacency matrix into a sparse edge index and
    edge attributes. The number of returned edges is equal to the number of
    non-zero entries in the adjacency matrix.

    Parameters
    ----------
    adj : Tensor
        Tensor (B, N_s, N_t, F), adjacency matrix for the edges between the source and target graphs
    cum_num_nodes_s : Optional[Tensor], optional
        cumulative number of nodes for each example in the source graph batch, by default None
    cum_num_nodes_t : Optional[Tensor], optional
        cumulative number of nodes for each example in the target graph batch, by default None

    Returns
    -------
    edge_index : Tensor
        Tensor (2, E) containing the node indices of the source and target nodes
    edge_attr : Tensor
        Tensor (E, F) containing the edge attributes

    Raises
    ------
    ValueError
        Dense adjacency matrix 'adj' must be 2- or 3-dimensional (got {adj.dim()} dimensions)
    """
    
    if adj.dim() < 2:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or "
                         f"3- or 4-dimensional (got {adj.dim()} dimensions)")

    if adj.dim() == 4:
        adj_collapsed = adj.sum(dim=-1)

    edge_index = adj_collapsed.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        if cum_num_nodes_s is None:
            node_idx_offset_s = adj.size(1) * edge_index[0]
        else:
            node_idx_offset_s = cum_num_nodes_s[edge_index[0]]

        if cum_num_nodes_t is None:
            node_idx_offset_t = adj.size(2) * edge_index[0]
        else:
            node_idx_offset_t = cum_num_nodes_t[edge_index[0]]

        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2], ...]
        row = edge_index[1] + node_idx_offset_s
        col = edge_index[2] + node_idx_offset_t
        return torch.stack([row, col], dim=0), edge_attr


###############  NO EDGE HANDLING  ###############

def encode_no_edge_dense_onehot(adjmat: Tensor) -> Tensor:
    """transform

    Parameters
    ----------
    adjmat : Tensor
        Tensor of shape (B, N, N, F), one-hot encoded adjacency matrix

    Returns
    -------
    adjmat : Tensor
        Tensor of shape (B, N, N, F), with encoded no edge in the first component
    """
    # check if matrix denote an empty graph
    if adjmat.shape[-1] == 0:
        return adjmat

    # find out where there is no edge
    no_edge = torch.sum(adjmat, dim=-2) == 0

    # turn on first component for onehot
    first_elt = adjmat[:, :, :, 0]
    first_elt[no_edge] = 1
    adjmat[:, :, :, 0] = first_elt

    # change diagonal to 0 again
    diag = torch.eye(adjmat.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    adjmat[diag] = 0
    return adjmat

def encode_no_edge(
        edge_features: Tensor,
        sparse: bool,
        collapsed: bool
    ) -> Tensor:

    if sparse:
        if collapsed:
            edge_features = edge_features + 1
        else:
            edge_features = torch.cat([edge_features, torch.ones(edge_features.size(0), 1, device=edge_features.device)], dim=-1)
        
    else:
        if collapsed:
            pass
        else:
            raise NotImplementedError

    return edge_features


def remove_no_edge(
        edge_features: Tensor,
        sparse: bool,
        collapsed: bool
    ) -> Tensor:

    if sparse:
        if collapsed:
            edge_features = edge_features - 1
        else:
            edge_features = edge_features[..., 1:]
    
    else:
        if collapsed:
            raise NotImplementedError
        else:
            edge_features = edge_features[..., 1:]

    return edge_features


##################  CONVERSIONS  #################

def sparse_graph_to_dense_graph(sparse_graph: SparseGraph, handle_one_hot=False) -> DenseGraph:

    # 1 - build dense node features tensor and node mask (if batch)
    node_mask = None
    batch_size = None
    if hasattr(sparse_graph, 'batch') and sparse_graph.batch is not None:
        batch = sparse_graph.batch
        batch_size = sparse_graph.num_graphs

        x, node_mask = to_dense_batch(
            x =		sparse_graph.x,
            batch =	batch,
            batch_size=batch_size
        )

        max_num_nodes = x.size(1)
        
    else:
        batch = None
        x: Tensor = sparse_graph.x
        max_num_nodes = x.size(0)

    # 2- preprocess edges
    edge_index, edge_attr = remove_self_loops(
        edge_index =	sparse_graph.edge_index,
        edge_attr =		sparse_graph.edge_attr
    )

    if handle_one_hot:
        edge_mask = get_edge_mask_dense(node_mask)

    # 3 - build dense adjacency matrix
    edge_adjmat = to_dense_adj_bipartite(
        edge_index =		edge_index,
        edge_attr =			edge_attr,
        batch_s =			batch,
        batch_t =			batch,
        max_num_nodes_s =	max_num_nodes,
        max_num_nodes_t =	max_num_nodes,
        batch_size =		batch_size,
        handle_one_hot =	handle_one_hot,
        edge_mask =			edge_mask
    )

    # 4 - build dense target vector
    y: Tensor = sparse_graph.y

    # 5 - build final dense graph with nodes, edges and targets
    # (optional node mask if it should be a batch)
    dense_graph = DenseGraph(
        x =				x,
        edge_adjmat =	edge_adjmat,
        y =				y,
        node_mask =		node_mask,
        edge_mask =		edge_mask
    )

    return dense_graph

def _subset_adjmatrix(adj: Tensor, node_mask: BoolTensor):
    """Notice: the adjacency matrix and node_mask refer to one single
    sample, not a batch of samples, so adj is 2D, and node_mask is 1D
    """
    # prepare mask
    num_true_nodes = node_mask.sum()
    mask_2d = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)

    # apply mask and reshape as a square matrix
    return adj[mask_2d].reshape(num_true_nodes, num_true_nodes)


def _dense_to_sparse_single_graph(x: Tensor, adj: Tensor, y: Tensor, node_mask: Tensor):

    # the following two operations are done in order to avoid skipping
    # indices during assignment
    # remove fake nodes
    sparse_x = x[node_mask]

    # remove fake edges
    adj = _subset_adjmatrix(adj, node_mask)

    # transform dense adj to edge_index and edge_attr
    edge_index, edge_attr = dense_to_sparse(adj=adj)

    return Data(
        x =				sparse_x,
        edge_index =	edge_index,
        edge_attr =		edge_attr,
        y =				y
    )


def dense_remove_self_loops(
        adjmat: Tensor
    ) -> Tensor:
    """Modifies inplace the adjacency matrix by removing self loops"""

    mask = torch.eye(adjmat.shape[1], dtype=torch.bool, device=adjmat.device)

    # if adjmat is a batch of adjacency matrices
    if adjmat.shape[1] == adjmat.shape[2]:
        mask = mask.unsqueeze(0).expand(adjmat.shape[0], -1, -1)

    # remove self loops
    adjmat[mask] = 0

    return adjmat


def dense_graph_to_sparse_graph(
        dense_graph: DenseGraph,
        num_nodes: IntTensor
    ) -> SparseGraph:

    # collect node tensors
    x = dense_graph.x[dense_graph.node_mask]
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), num_nodes.cumsum(dim=0)])
    batch = torch.repeat_interleave(torch.arange(len(num_nodes), device=x.device), num_nodes)

    # collect edge tensors
    edge_index, edge_attr = dense_to_sparse(
        adj =				dense_graph.edge_adjmat,
        cum_num_nodes_s	=	ptr,
        cum_num_nodes_t	=	ptr
    )

    # build sparse graph
    sparse_graph = SparseGraph(
        x =				x,
        edge_index =	edge_index,
        edge_attr =		edge_attr,
        y =				dense_graph.y,
        batch =			batch,
        ptr =			ptr
    )

    return sparse_graph


def dense_edges_to_sparse_edges(
        dense_edges: DenseEdges,
        cum_num_nodes_s: IntTensor,
        cum_num_nodes_t: IntTensor
    ) -> SparseGraph:

    edge_index, edge_attr = dense_to_sparse(
        adj =				dense_edges.edge_adjmat,
        cum_num_nodes_s =	cum_num_nodes_s,
        cum_num_nodes_t =	cum_num_nodes_t
    )

    sparse_edges = SparseEdges(
        edge_index =	edge_index,
        edge_attr =		edge_attr
    )

    return sparse_edges


def get_bipartite_edge_mask_dense(
        node_mask_a: Tensor,
        node_mask_b: Tensor
    ) -> Tensor:

    e_mask1 = node_mask_a.unsqueeze(-1)		# *, na, 1
    e_mask2 = node_mask_b.unsqueeze(-2)		# *, 1, nb
    e_mask = e_mask1 * e_mask2				# *, na, nb

    return e_mask

def get_edge_mask_dense(
        node_mask: Tensor=None,
        edge_mask: Tensor=None,
        remove_self_loops: bool = False,
        only_triangular: bool = False
    ) -> Tensor:
    if edge_mask is None:
        e_mask = get_bipartite_edge_mask_dense(node_mask, node_mask)
    else:
        e_mask = edge_mask
    
    if only_triangular:
        # remove upper triangular part and diagonal
        e_mask = torch.tril(e_mask, diagonal=-1)

    elif remove_self_loops:
        # remove self loops
        e_mask = e_mask * (1 - torch.eye(e_mask.size(-1), device=e_mask.device)).unsqueeze(0)

    return e_mask


def get_node_edge_mask_dense(node_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """build the node and edge masks from a squeezed node mask

    Parameters
    ----------
    node_mask : Tensor
        boolean mask of shape (*, num_nodes)

    Returns
    -------
    x_mask : Tensor
        boolean tensor of shape (*, num_nodes) to be applied on node features
        tensors
    e_mask : Tensor
        boolean tensor of shape (*, num_nodes, num_nodes) to be applied on
        edge features adjacency matrices
    """
    x_mask = node_mask						# bs, n
    e_mask = get_edge_mask_dense(node_mask)	# bs, n, n

    return x_mask, e_mask


################################################################################
#                                    CHECKS                                    #
################################################################################
def is_matrix_symmetric(mat: Tensor) -> bool:
    return torch.allclose(mat, torch.transpose(mat, 1, 2))