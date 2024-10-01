##########################################################################################################
#
# adapted from https://github.com/cvignac/DiGress/blob/main/dgd/models/transformer_model.py
#
##########################################################################################################

from typing import Optional, Dict, Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from torch.nn.modules.linear import Linear

from src.datatypes.dense import DenseGraph, get_bipartite_edge_mask_dense


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            cross_attn: bool=True,
            heads: int = 8,
            dim_ffX: int = 2048,
            dim_ffE: int = 128,
            dim_ffy: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5,
            act_fn = nn.ReLU
        ):
        """Builds graph transformer layer
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : Optional[int]
            number of global features. Optional for allowing the absence of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        dim_ffX : int, optional
            size of intermediate features in nodes FFN, by default 2048
        dim_ffE : int, optional
            size of intermediate features in edges FFN, by default 128
        dim_ffy : int, optional
            size of intermediate features in global FFN, by default 2048
        dropout : float, optional
            dropout probability, by default 0.1
        layer_norm_eps : float, optional
            layer normalization parameter epsilon, by default 1e-5
        """
        
        super().__init__()

        self.self_attn = XEySelfAttention(dx, de, dy, heads)
        self.cross_attn = cross_attn

        self.normX1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutE1 = Dropout(dropout)
        self.dropout_y1 = Dropout(dropout)

        # nodes FFN
        self.ffnX = nn.Sequential(
            Linear(dx, dim_ffX),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffX, dx),
            Dropout(dropout)
        )
        self.normX3 = LayerNorm(dx, eps=layer_norm_eps)

        # edges FFN
        self.ffnE = nn.Sequential(
            Linear(de, dim_ffE),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffE, de),
            Dropout(dropout)
        )
        self.normE3 = LayerNorm(de, eps=layer_norm_eps)

        # global FFN
        self.ffny = nn.Sequential(
            Linear(dy, dim_ffy),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffy, dy),
            Dropout(dropout)
        )
        self.norm_y3 = LayerNorm(dy, eps=layer_norm_eps)


    def forward(
            self,
            X: Tensor,
            E: Tensor,
            y: Tensor,
            node_mask: Tensor,
            ext_X: Optional[Tensor]=None,
            ext_E: Optional[Tensor]=None,
            ext_node_mask: Optional[Tensor]=None

        ) -> tuple[Tensor, Tensor, Tensor]:
        """Transformer layer for nodes, edges and global features, applying the usual:
            - self attention + residual
            - FFN + residual
        Parameters
        ----------
        X : Tensor
            node features of shape (bs, nq, d)
        E : Tensor
            edge features of shape (bs, nq, nq, d)
        y : Tensor
            global features of shape (bs, dy)
        node_mask : Tensor
            node masks for non-existing nodes (due to padding in dense representation) of shape (bs, nq)
        ext_X : Tensor
            node features of shape (bs, nk, d)
        ext_E : Tensor
            edge features of shape (bs, nq, nk, d)
        node_mask : Tensor
            node masks for non-existing nodes (due to padding in dense representation) of shape (bs, nk)
        Returns
        -------
        newX : Tensor
            new node features of shape (bs, n, d)
        newE : Tensor
            new edge features of shape (bs, nq, nq, d)
        new_y : Tensor
            new global features of shape (bs, dy)
        """

        do_merge = self.cross_att and ext_E is not None and ext_E.shape[2] > 0

        if do_merge:
            Xq = X
            Xk = torch.cat([X, ext_X], dim=1)
            node_mask_q = node_mask
            node_mask_k = torch.cat([node_mask, ext_node_mask], dim=1)
            bs, nq, nk, _ = ext_E.shape
            E = torch.cat([E, ext_E], dim=2)

        ####################  SELF-ATTENTION-RESIDUAL BLOCK  ###################
        # self-attention
        newX, newE, new_y = self.self_attn(
            Xq = Xq,
            Xk = Xk,
            node_mask_q = node_mask_q,
            node_mask_k = node_mask_k,
            E = E,
            y = y
        )

        # residual on nodes
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        # residual on edges
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        # residual on global
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        #########################  FFN-RESIDUAL BLOCK  #########################
        # X = norm(X + FFN(X))
        ff_outputX = self.ffnX(X)
        newX = self.normX3(X + ff_outputX)

        # E = norm(E + FFN(E))
        ff_outputE = self.ffnE(E)
        newE = self.normE3(E + ff_outputE)

        # y = norm(y + FFN(y))
        ff_output_y = self.ffny(y)
        new_y = self.norm_y3(y + ff_output_y)

        # split if necessary
        if do_merge:
            ext_newE = newE[:, :, nq:]
            newE = newE[:, :, :nq]


        if self.cross_attn:
            return newX, newE, new_y, ext_newE
        else:
            return newX, newE, new_y


def masked_softmax(x, mask, dim):
    x_masked = x.clone()
    x_masked[mask == 0] = NEG_INF
    return torch.softmax(x_masked, dim)

class XEySelfAttention(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            n_head: int
        ):
        """Self-attention block for computing new edges, nodes and global features
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : int
            number of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        """
        super().__init__()

        assert dx % n_head == 0, f"Cannot divide nodes features size by number of heads: dx: {dx} -- nhead: {n_head}"

        self.dx = dx
        self.de = de
        self.dy = dy

        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q_proj = Linear(dx, dx)
        self.k_proj = Linear(dx, dx)
        self.v_proj = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_proj = Linear(dy, dy)
        self.reduce_xq = Xtoy(dx, dy)	# projection of (mean, std, min, max)
        self.reduce_xk = Xtoy(dx, dy)	# projection of (mean, std, min, max)
        self.reduce_e = Etoy(de, dy)	# projection of (mean, std, min, max)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(4 * dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(
            self,
            Xq: Tensor,
            Xk: Tensor,
            node_mask_q: Tensor,
            node_mask_k: Tensor,
            E: Tensor,
            y: Tensor
        ) -> tuple[Tensor, Tensor, Tensor]:
        """Updates the nodes, edges and global representations
        Parameters
        ----------
        X : Tensor
            node features of shape (bs, n, d)
        E : Tensor
            edge features of shape (bs, nq, nk, d)
        y : Tensor
            global features of shape (bs, dy)
        node_mask : Tensor
            node masks for non-existing nodes (due to padding in dense representation) of shape (bs, n)
        Returns
        -------
        newX : Tensor
            new node features of shape (bs, n, d)
        newE : Tensor
            new edge features of shape (bs, n, n, d)
        new_y : Tensor
            new global features of shape (bs, dy)
        """

        bs, nq, nk, _ = E.shape

        #######################  FAKE NODES MASKS SETUP  #######################

        # unsqueeze to enable masking (dot product only if same number of dims)
        xq_mask = node_mask_q.unsqueeze(-1)		# (bs, nq, 1)
        xk_mask = node_mask_k.unsqueeze(-1)		# (bs, nk, 1)
        eq_mask = xq_mask.unsqueeze(2)			# (bs, nq, 1, 1)
        ek_mask = xk_mask.unsqueeze(1)			# (bs, 1, nk, 1)
        e_mask = eq_mask * ek_mask				# (bs, nq, nk, 1)


        # assert xq_mask.shape == (bs, nq, 1), f'xq_mask shape is {xq_mask.shape}, expected {(bs, nq, 1)}'
        # assert xk_mask.shape == (bs, nk, 1), f'xk_mask shape is {xk_mask.shape}, expected {(bs, nk, 1)}'
        # assert eq_mask.shape == (bs, nq, 1, 1), f'eq_mask shape is {eq_mask.shape}, expected {(bs, nq, 1, 1)}'
        # assert ek_mask.shape == (bs, 1, nk, 1), f'ek_mask shape is {ek_mask.shape}, expected {(bs, 1, nk, 1)}'
        # assert e_mask.shape == (bs, nq, nk, 1), f'e_mask shape is {e_mask.shape}, expected {(bs, nq, nk, 1)}'

        ####################  QUERIES, KEYS, VALUES SETUP  #####################

        #print(Xq)
        #print(Xq * xq_mask)

        Q = self.q_proj(Xq) * xq_mask			# (bs, nq, dx)
        #print(Q)
        K = self.k_proj(Xk) * xk_mask			# (bs, nk, dx)
        V = self.v_proj(Xk) * xk_mask			# (bs, nk, dx)
        # assert_correctly_masked(Q, xq_mask)
        # assert_correctly_masked(K, xk_mask)
        # assert_correctly_masked(V, xk_mask)

        # assert Q.shape == (bs, nq, self.dx), f'Q shape is {Q.shape}, expected {(bs, nq, self.dx)}'
        # assert K.shape == (bs, nk, self.dx), f'K shape is {K.shape}, expected {(bs, nk, self.dx)}'
        # assert V.shape == (bs, nk, self.dx), f'V shape is {V.shape}, expected {(bs, nk, self.dx)}'

        # Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((*Q.shape[:2], self.n_head, self.df))
        K = K.reshape((*K.shape[:2], self.n_head, self.df))
        V = V.reshape((*V.shape[:2], self.n_head, self.df))

        # assert Q.shape == (bs, nq, self.n_head, self.df), f'Q shape is {Q.shape}, expected {(bs, nq, self.n_head, self.df)}'
        # assert K.shape == (bs, nk, self.n_head, self.df), f'K shape is {K.shape}, expected {(bs, nk, self.n_head, self.df)}'
        # assert V.shape == (bs, nk, self.n_head, self.df), f'V shape is {V.shape}, expected {(bs, nk, self.n_head, self.df)}'

        # print('Checking Q, K, V shapes')
        # print('Q', Q.shape)
        # print('K', K.shape)
        # print('V', V.shape)

        # setup dimensions for outer product
        Q = Q.unsqueeze(2)			# (bs, nq, 1, n_head, df)
        K = K.unsqueeze(1)			# (bs, 1, nk, n head, df)
        V = V.unsqueeze(1)			# (bs, 1, nk, n_head, df)

        # assert Q.shape == (bs, nq, 1, self.n_head, self.df), f'Q shape is {Q.shape}, expected {(bs, nq, 1, self.n_head, self.df)}'
        # assert K.shape == (bs, 1, nk, self.n_head, self.df), f'K shape is {K.shape}, expected {(bs, 1, nk, self.n_head, self.df)}'
        # assert V.shape == (bs, 1, nk, self.n_head, self.df), f'V shape is {V.shape}, expected {(bs, 1, nk, self.n_head, self.df)}'

        ####################  OUTER PRODUCT SELF-ATTENTION  ####################

        # Compute unnormalized attentions. A is (bs, nq, nk, n_head, df)
        A = Q * K					# outer product
        A = A / math.sqrt(self.df)	# scaling by sqrt(df)
        # assert_correctly_masked(A, e_mask.unsqueeze(-1))

        # assert A.shape == (bs, nq, nk, self.n_head, self.df), f'A shape is {A.shape}, expected {(bs, nq, nk, self.n_head, self.df)}'

        # ----> node informed attention matrix A of shape (bs, nq, nk, n_head, df)

        #################  ATTENTION = FILM(EDGES, ATTENTION)  #################
        E1 = self.e_add(E) * e_mask				# bs, nq, nk, dx
        E1 = E1.reshape((*E.shape[:3], self.n_head, self.df))

        E2 = self.e_mul(E) * e_mask				# bs, nq, nk, dx
        E2 = E2.reshape((*E.shape[:3], self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        A = E1 + (E2 + 1) * A                   # (bs, nq, nk, n_head, df)

        # assert A.shape == (bs, nq, nk, self.n_head, self.df), f'A shape is {A.shape}, expected {(bs, nq, nk, self.n_head, self.df)}'

        # print('Checking A shape after FILM, and E1, E2')
        # print('A', A.shape)
        # print('E1', E1.shape)
        # print('E2', E2.shape)

        # ----> node/edge informed attention matrix A of shape (bs, n, n, n_head, df)

        ###############  OUT_EDGES = LIN(FILM(GLOBAL, EDGES))  #################

        # Incorporate y to E
        newE = A.flatten(start_dim=-2)						# (bs, nq, nk, dx)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)		# (bs, 1, 1, dx)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)		# (bs, 1, 1, dx)

        # print('Checking newE, ye1, ye2 shapes')
        # print('newE', newE.shape)
        # print('ye1', ye1.shape)
        # print('ye2', ye2.shape)
        newE = ye1 + (ye2 + 1) * newE

        # assert newE.shape == (bs, nq, nk, self.dx), f'newE shape is {newE.shape}, expected {(bs, nq, nk, self.dx)}'

        # Output E
        newE = self.e_out(newE) * e_mask					# (bs, nq, nk, de)
        # assert_correctly_masked(newE, e_mask)

        # ----> END OF EDGES BRANCH, new edges (bs, nq, nk, de)

        ####################  ATTENTION: AGGREGATE VALUES  #####################

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        # use masked softmax to avoid attention on non-existing nodes
        softmax_mask = ek_mask.expand(-1, nq, -1, self.n_head)	# bs, nq, nk, n_head
        attn_weights = A.sum(-1)                                # bs, nq, nk, n_head
        attn = masked_softmax(attn_weights, softmax_mask, dim=2)  	# bs, nq, nk, n_head

        # assert softmax_mask.shape == (bs, nq, nk, self.n_head), f'softmax_mask shape is {softmax_mask.shape}, expected {(bs, nq, nk, self.n_head)}'
        # assert attn_weights.shape == (bs, nq, nk, self.n_head), f'attn_weights shape is {attn_weights.shape}, expected {(bs, nq, nk, self.n_head)}'
        # assert attn.shape == (bs, nq, nk, self.n_head), f'attn shape is {attn.shape}, expected {(bs, nq, nk, self.n_head)}'

        # print('Checking mask and attn shape')
        # print('softmax_mask', softmax_mask.shape)
        # print('attn', attn.shape)
        
        # issue: for an example, all nodes could be fake, so no attention score
        # is actually computed => all values are -inf, and softmax returns nan
        # then transform nans into 0 weights, in order to not weight fake node values
        attn = torch.nan_to_num(attn) * e_mask			# bs, nq, nk, n_head
        #attn = attn * e_mask			                # bs, nq, nk, n_head
        #attn = torch.nan_to_num(attn)					# bs, nq, nk, n_head
        # print(attn)

        # Compute weighted values
        # print(V[..., 0])
        weighted_V = attn.unsqueeze(-1) * V				# (bs, nq, nk, n_head, df)
        # print('weighted_V:', weighted_V.shape)
        weighted_V = weighted_V.sum(dim=2)				# (bs, nq, n_head, df)
        # print(weighted_V[..., 0])
        # print('weighted_V after sum:', weighted_V.shape)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=-2)	# (bs, nq, dx)
        # print('weighted_V after flatten:', weighted_V.shape)

        # ----> self-attention output, aggregated node values (bs, nq, dx)

        ###############  OUT_NODES = LIN(FILM(GLOBAL, NODES))  #################

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        # print('Checking newX, yx1, yx2 shapes')
        # print('newX', newX.shape)
        # print('yx1', yx1.shape)
        # print('yx2', yx2.shape)

        # Output X
        newX = self.x_out(newX) * xq_mask
        # assert_correctly_masked(newX, xq_mask)
# 
        # assert newX.shape == (bs, nq, self.dx), f'newX shape is {newX.shape}, expected {(bs, nq, self.dx)}'

        # print('newX after x_out:', newX.shape)

        # ----> END OF NODES BRANCH, new nodes (bs, n, dx)

        ###############  GLOBAL BRANCH  #################

        # Process y based on X and E
        y = self.y_proj(y)			# (bs, dy)
        e_y = self.reduce_e(E, e_mask)		# (bs, dy)
        xq_y = self.reduce_xq(Xq, xq_mask)		# (bs, dy)
        xk_y = self.reduce_xk(Xk, xk_mask)		# (bs, dy)

        # assert y.shape == (bs, self.dy), f'y shape is {y.shape}, expected {(bs, self.dy)}'
        # assert e_y.shape == (bs, self.dy), f'e_y shape is {e_y.shape}, expected {(bs, self.dy)}'
        # assert xq_y.shape == (bs, self.dy), f'xq_y shape is {xq_y.shape}, expected {(bs, self.dy)}'
        # assert xk_y.shape == (bs, self.dy), f'xk_y shape is {xk_y.shape}, expected {(bs, self.dy)}'

        new_y = torch.cat([y, xq_y, xk_y, e_y], dim=-1)	# concat everything
        new_y = self.y_out(new_y)   # (bs, dy)

        # print('Checking new_y, y, e_y, xq_y, xk_y shapes')
        # print('new_y', new_y.shape)
        # print('y', y.shape)
        # print('e_y', e_y.shape)
        # print('xq_y', xq_y.shape)
        # print('xk_y', xk_y.shape)

        return newX, newE, new_y


#############  TRANSFORMER OPTIONS  ##############

DIM_X = 'x'
DIM_E = 'e'
DIM_Y = 'y'

if False:
    POS_INF = 1e9
    NEG_INF = -1e9
else:
    POS_INF = float('inf')
    NEG_INF = float('-inf')


class SimplerGraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(
            self,
            input_dims: Dict,
            output_dims: Dict,
            num_layers: int,
            encdec_hidden_dims: Dict,
            transf_inout_dims: Dict,
            transf_ffn_dims: Dict,
            transf_hparams: Dict,
            use_residuals_inout: bool = False,
            act_fn_in = nn.ReLU,
            act_fn_out = nn.ReLU
        ):

        super().__init__()

        self.num_layers = num_layers
        self.use_residuals_inout = use_residuals_inout

        self.in_dim_x = input_dims[DIM_X]
        self.in_dim_e = input_dims[DIM_E]
        self.in_dim_y = input_dims[DIM_Y]

        self.encdec_hidden_dims = encdec_hidden_dims
        self.transf_inout_dims = transf_inout_dims
        self.transf_ffn_dims = transf_ffn_dims

        self.using_y = self.in_dim_y is not None
        self.using_cross_attn = transf_hparams['cross_attn']

        self.out_dim_x = output_dims[DIM_X]
        self.out_dim_e = output_dims[DIM_E]
        self.out_dim_y = output_dims[DIM_Y]

        ###########################  INPUT ENCODERS  ###########################
        # nodes encoder
        self.mlp_in_X = nn.Sequential(
            nn.Linear(self.in_dim_x, encdec_hidden_dims[DIM_X]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_X], transf_inout_dims[DIM_X])
        )

        # edges encoder
        self.mlp_in_E = nn.Sequential(
            nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
        )

        if self.using_y:
            # global encoder
            self.mlp_in_y = nn.Sequential(
                nn.Linear(self.in_dim_y, encdec_hidden_dims[DIM_Y]),
                act_fn_in(),
                nn.Linear(encdec_hidden_dims[DIM_Y], transf_inout_dims[DIM_Y])
            )
        else:
            self.fixed_y = nn.Parameter(torch.randn(transf_inout_dims[DIM_Y]))

        if self.using_cross_attn:
            self.mlp_in_ext_E = nn.Sequential(
                nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
                act_fn_in(),
                nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
            )


        #######################  MAIN BODY: TRANSFORMER  #######################

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=transf_inout_dims[DIM_X],
                de=transf_inout_dims[DIM_E],
                dy=transf_inout_dims[DIM_Y],
                dim_ffX=transf_ffn_dims[DIM_X],
                dim_ffE=transf_ffn_dims[DIM_E],
                dim_ffy=transf_ffn_dims[DIM_Y],
                **transf_hparams
            )
            for _ in range(num_layers)
        ])

        ##########################  OUTPUT DECODERS  ###########################

        # nodes decoder
        self.mlp_out_X = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            act_fn_out(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_x)
        )

        # edges decoder
        self.mlp_out_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            act_fn_out(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )

        if self.using_y:
            # global decoder
            self.mlp_out_y = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_Y], encdec_hidden_dims[DIM_Y]),
                act_fn_out(),
                nn.Linear(encdec_hidden_dims[DIM_Y], self.out_dim_y)
            )

        if self.using_cross_attn:
            self.mlp_out_ext_E = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
                act_fn_out(),
                nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
            )



    def forward(
            self,
            X: Tensor,
            E: Tensor,
            node_mask:Tensor,
            y: Optional[Tensor]=None,
            ext_X: Optional[Tensor]=None,
            ext_E: Optional[Tensor]=None,
            ext_node_mask: Optional[Tensor]=None,
            return_all: bool=True
        ) -> Tuple[Tensor]:

        ########################  ASSERTIONS ON INPUT  #########################
        assert X.shape[-1] == self.in_dim_x, \
            f"X.shape[-1] = {X.shape[-1]}, self.in_dim_x = {self.in_dim_x}"
        assert E.shape[-1] == self.in_dim_e, \
            f"E.shape[-1] = {E.shape[-1]}, self.in_dim_e = {self.in_dim_e}"
        assert y is None or y.shape[-1] == self.in_dim_y, \
            f"y.shape[-1] = {y.shape[-1]}, self.in_dim_y = {self.in_dim_y}"
        if self.using_cross_attn:
            assert ext_X is None or ext_X.shape[-1] == self.transf_inout_dims[DIM_X], \
                f"ext_X.shape[-1] = {ext_X.shape[-1]}, self.transf_inout_dims[DIM_X] = {self.transf_inout_dims[DIM_X]}"
            assert ext_E is None or ext_E.shape[-1] == self.in_dim_e, \
                f"ext_E.shape[-1] = {ext_E.shape[-1]}, self.in_dim_e = {self.in_dim_e}"
            assert ext_node_mask is None or ext_node_mask.shape[1] == ext_X.shape[1], \
                f"ext_node_mask.shape[1] = {ext_node_mask.shape[1]}, ext_X.shape[1] = {ext_X.shape[1]}"
        
        ########################  AUXILIARY FUNCTIONS  #########################
        if self.using_cross_attn:
            ext_edge_mask = node_mask.unsqueeze(-1) * ext_node_mask.unsqueeze(-2)
        
        def mask_everything(curr_X, curr_E, curr_ext_E=None):
            masked = DenseGraph(
                x =				curr_X,
                edge_adjmat =	curr_E,
                node_mask =		node_mask
            ).apply_mask()

            if curr_ext_E is not None and self.using_cross_attn:
                curr_ext_E = curr_ext_E * ext_edge_mask.unsqueeze(-1)
            
            return masked.x, masked.edge_adjmat, curr_ext_E

        bs, n = X.shape[0], X.shape[1]

        ###############  SETUP SELFLOOP REMOVAL (DIAGONAL) MASK  ###############
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        ######################  SAVE RESIDUAL FOR LATER  #######################
        if self.use_residuals_inout:
            X_to_out = X[..., :self.out_dim_x]
            E_to_out = E[..., :self.out_dim_e]
            if self.using_y:
                y_to_out = y[..., :self.out_dim_y]
            if self.using_cross_attn:
                ext_E_to_out = ext_E[..., :self.out_dim_e]

        ###########################  ENCODE INPUTS  ############################
        # special treatment for edges (to make it symmetric (shouldn't this already be?))
        X = self.mlp_in_X(X)
        E = self.mlp_in_E(E)
        E = (E + E.transpose(1, 2)) / 2   # new_E should already be symmetric if E is symmetric!!!
        if self.using_y:
            y = self.mlp_in_y(y)
        else:
            y = self.fixed_y.clone().expand(bs, -1)

        if self.using_cross_attn:
            ext_E = self.mlp_in_ext_E(ext_E)

        # mask everything before feeding to transformer
        X, E, ext_E = mask_everything(X, E, ext_E)

        #######################  MAIN BODY: TRANSFORMER  #######################
        # if using cross attention
        if self.using_cross_attn:

            # if there is a set of external nodes for each layer (they have been encoded)
            if isinstance(ext_X, list):

                # for each layer use a different set of external encoded nodes
                for layer, curr_ext_X in zip(self.tf_layers, ext_X):
                    X, E, y, ext_E = layer(
                        X, E, y, node_mask,
                        curr_ext_X, ext_E, ext_node_mask
                    )
            # if only using one set of external nodes (no encoding)
            else:
                # for each layer use the only provided set of external encoded nodes
                for layer in self.tf_layers:
                    X, E, y, ext_E = layer(
                        X, E, y, node_mask,
                        ext_X, ext_E, ext_node_mask
                    )
        
        # if no external set of nodes and edges are provided
        else:
            for layer in self.tf_layers:
                X, E, y = layer(X, E, y, node_mask)


        ###########################  DECODE OUTPUT  ############################
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        if self.using_y:
            y = self.mlp_out_y(y)
        if self.using_cross_attn:
            ext_E = self.mlp_out_ext_E(ext_E)

        ###########################  FINAL RESIDUAL  ###########################
        if self.use_residuals_inout:
            X = X + X_to_out
            E = E + E_to_out
            
        # remove selfloop and make symmetric
        E = E * diag_mask
        E = (E + torch.transpose(E, 1, 2)) / 2 # here it's ok!
        
        if self.using_cross_attn:
            if self.use_residuals_inout:
                y = y + y_to_out
        if self.using_cross_attn:
            if self.use_residuals_inout:
                ext_E = ext_E + ext_E_to_out
        
        # mask everything before returning
        X, E, ext_E = mask_everything(X, E, ext_E)

        ###############################  RETURN  ###############################

        if return_all:
            return X, E, y, ext_E

        # unpack masked graph
        ret = [X, E]

        if self.using_y:
            ret.append(y)
        if self.using_cross_attn:
            ret.append(ext_E)

        return tuple(ret)
        


##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/models/layers.py
#
##########################################################################################################

def compute_masked_mean(X: Tensor, mask: Tensor, dim: int|Tuple[int]) -> Tensor:
    """ Computes the mean of X along dimension dim, ignoring masked values. """
    denominator = mask.sum(dim=dim)
    denominator[denominator == 0] = 1.
    return (X * mask).sum(dim=dim) / denominator


def compute_masked_std(X: Tensor, mean_X: Tensor, mask: Tensor, dim: int|Tuple[int]=1) -> Tensor:
    """ Computes the standard deviation of X along dimension dim, ignoring masked values. """
    if isinstance(dim, int):
        dim = (dim,)
    for i in dim:
        mean_X = mean_X.unsqueeze(i)
    
    std = (compute_masked_mean((X - mean_X) ** 2, mask, dim) + 1e-10).sqrt()
    std = std.clone()
    #std[std <= 1e-5] = 0. # really bad thing, but avoids nan in the gradients
    return std


def compute_masked_min(X: Tensor, mask: Tensor, dim: int|Tuple[int]=1) -> Tensor:
    """ Computes the min of X along dimension dim, ignoring masked values. """
    X = X.clone()
    X[~mask.squeeze(-1)] = float('inf')
    if isinstance(dim, int):
        dim = (dim,)
    for i in reversed(dim):
        X = X.min(dim=i)[0]
    return torch.nan_to_num(X, posinf=0)


def compute_masked_max(X: Tensor, mask: Tensor, dim: int|Tuple[int]=1) -> Tensor:
    """ Computes the max of X along dimension dim, ignoring masked values. """
    X = X.clone()
    X[~mask.squeeze(-1)] = float('-inf')
    if isinstance(dim, int):
        dim = (dim,)
    for i in reversed(dim):
        X = X.max(dim=i)[0]
    return torch.nan_to_num(X, neginf=0)


def compute_masked_mean_std_min_max(X: Tensor, mask: Tensor, dim: int|Tuple[int]=1) -> Tensor:

    if X.numel() == 0:
        return torch.zeros((X.shape[0], 4 * X.shape[-1]), device=X.device)

    else:
        if isinstance(dim, int):
            dim = (dim,)
        mean_X = compute_masked_mean(X, mask, dim)
        std_X = compute_masked_std(X, mean_X, mask, dim)
        min_X = compute_masked_min(X, mask, dim)
        max_X = compute_masked_max(X, mask, dim)

        return torch.hstack((mean_X, std_X, min_X, max_X))


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: Tensor, node_mask: Tensor):
        """ X: bs, n, dx. """
        z = compute_masked_mean_std_min_max(X, node_mask, dim=1)
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E: Tensor, edge_mask: Tensor):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        z = compute_masked_mean_std_min_max(E, edge_mask, dim=(1,2))
        out = self.lin(z)
        return out
    
def assert_correctly_masked(variable, node_mask):
    if variable.numel() == 0:
        return
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'