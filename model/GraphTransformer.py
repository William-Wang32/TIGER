# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 下午4:14
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : GraphTransformer.py
# @Software : PyCharm

import os
import sys

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import torch
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, ECConv, global_mean_pool, GINConv
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing

# We deliberately avoid `torch_geometric.utils.softmax` here.
# In some CUDA/PyTorch combinations (typically: newer GPU + older CUDA/NVRTC),
# the scripted softmax path can trigger NVRTC JIT compilation and fail with:
#   nvrtc: error: invalid value for --gpu-architecture (-arch)
# A pure eager scatter-softmax avoids TorchScript/NVRTC entirely.
try:
    from torch_scatter import scatter_max, scatter_add
except Exception as e:  # pragma: no cover
    scatter_max = scatter_add = None
    _torch_scatter_import_error = e


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Neighborhood softmax over edges grouped by `index`.

    Args:
        src: [E] or [E, H] logits.
        index: [E] target node indices.
        num_nodes: number of target nodes.

    Returns:
        softmax(src) grouped by index, same shape as src.
    """
    if scatter_max is None or scatter_add is None:  # pragma: no cover
        raise ImportError(
            "torch_scatter is required for segment_softmax, but import failed: "
            f"{_torch_scatter_import_error}"
        )
    # scatter_max supports src with extra dims, returning [num_nodes, ...]
    max_per, _ = scatter_max(src, index, dim=0, dim_size=num_nodes)
    out = torch.exp(src - max_per[index])
    denom = scatter_add(out, index, dim=0, dim_size=num_nodes)
    return out / (denom[index] + 1e-16)


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout,
                 attn_norm: str = 'ratio', attn_dropout: float = 0.0, edge_dropout: float = 0.0):
        super(GraphTransformerEncode, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim)
        )

        self.multiHeadAttention = MultiheadAttention(
            dim_model=self.in_dim,
            num_heads=self.num_heads,
            rel_encoder=rel_encoder,
            spatial_encoder=spatial_encoder,
            attn_norm=attn_norm,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
        )

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()

        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):

        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(x_norm, sp_edge_index, sp_value, edge_rel)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight

class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()

        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1) ##[n_edges, 1]
        out = self.fnn(lap_)

        return out


class MultiheadAttention(MessagePassing):
    """Relation-aware multi-head attention on a (sub)graph.

    This repo's original implementation produced an edge-wise score and then used a
    global denominator ("ratio" mode). It also passed edge weights into propagate()
    without overriding message(), meaning the weights were **not applied**.

    Improvement here:
      1) Correctly apply edge weights via a message() function.
      2) Provide a principled neighborhood-wise softmax normalization ("softmax" mode)
         that matches standard Transformer/GAT-style attention.
      3) Optional DropEdge and attention dropout for regularization.
    """

    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder,
                 attn_norm: str = 'ratio', attn_dropout: float = 0.0, edge_dropout: float = 0.0, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(
            Linear(dim_model, 1),
            ReLU()
        )

        self.spatial_encoding = spatial_encoder

        if attn_norm not in {'ratio', 'softmax'}:
            raise ValueError(f"attn_norm must be one of {{'ratio','softmax'}}, got {attn_norm}")
        self.attn_norm = attn_norm
        self.attn_dropout = float(attn_dropout)
        self.edge_dropout = float(edge_dropout)


        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)

        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.rel_embedding.reset_parameters()
        self.rel_encoding[0].reset_parameters()
        self.spatial_encoding.reset_parameters()

        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def message(self, x_j, edge_weight):
        # Apply edge-wise attention weights.
        return x_j * edge_weight.view(-1, 1)

    def softmax_kernel_transformation(self, data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
        data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
        data = data_normalizer * data
        ratio = data_normalizer
        data_dash = projection_matrix(data) ##[node_num, dim]
        diag_data = torch.square(data)
        diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
        diag_data = diag_data / 2.0
        diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
        last_dims_t = len(data_dash.shape) - 1
        attention_dims_t = len(data_dash.shape) - 3
        if is_query:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                        0]) + numerical_stabilizer
            )
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                                dim=attention_dims_t, keepdim=True)[
                        0]) + numerical_stabilizer
            )
        return data_dash

    def denominator(self, qs, ks):
        ##qs [num_node, num_heads, depth]
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)  # ks_sum refers to O_k in the paper
        return torch.einsum("nhm,hm->nh", qs, ks_sum)

    def forward(self, x, sp_edge_index, sp_value, edge_rel):

        # Optional DropEdge (applied consistently to index/value/rel).
        if self.edge_dropout > 0 and self.training:
            e = sp_edge_index.shape[1]
            keep = torch.rand(e, device=sp_edge_index.device) >= self.edge_dropout
            sp_edge_index = sp_edge_index[:, keep]
            sp_value = sp_value[keep]
            edge_rel = edge_rel[keep]

        rel_embedding = self.rel_embedding(edge_rel)  # [E, d_model]

        n = x.shape[0]
        # Project to multi-head
        q = self.wq(x).view(n, self.num_heads, self.depth)
        k = self.wk(x).view(n, self.num_heads, self.depth)
        v = self.wv(x).view(n, self.num_heads, self.depth)

        row, col = sp_edge_index  # row: source, col: target
        rel = rel_embedding.view(-1, self.num_heads, self.depth)

        q_e = q[col] + rel
        k_e = k[row] + rel

        # Scaled dot-product attention logits
        logits = torch.einsum('ehd,ehd->eh', q_e, k_e) / math.sqrt(self.depth)  # [E, H]
        logits = logits + self.spatial_encoding(sp_value)  # broadcast [E,1] -> [E,H]


        if self.attn_norm == 'softmax':
            # Neighborhood-wise softmax grouped by target node 'col'.
            # Use eager scatter-softmax to avoid TorchScript/NVRTC issues.
            attn_w = segment_softmax(logits, col, num_nodes=n)
        else:
            # Backward-compatible "ratio" mode: normalize with a global denominator.
            # Note: This is *not* a probability simplex; use softmax for principled attention.
            attn_normalizer = self.denominator(q, k)  # [N, H]
            attn_w = logits / (attn_normalizer[col] + 1e-9)

        if self.attn_dropout > 0:
            attn_w = torch.nn.functional.dropout(attn_w, p=self.attn_dropout, training=self.training)

        outputs = []
        for h in range(self.num_heads):
            out_h = self.propagate(edge_index=sp_edge_index, x=v[:, h, :], edge_weight=attn_w[:, h], size=None)
            outputs.append(out_h)

        out = torch.cat(outputs, dim=-1)
        return self.dense(out), attn_w


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num=3, embedding_dim=64, num_heads=4, num_rel=10, dropout=0.2, type='graph',
                 attn_norm: str = 'ratio', attn_dropout: float = 0.0, edge_dropout: float = 0.0):
        """GraphTransformer backbone.

        Args:
            attn_norm: 'ratio' (original) or 'softmax' (recommended).
            attn_dropout: dropout on attention weights.
            edge_dropout: DropEdge rate on sp_edge_index.
        """
        super(GraphTransformer, self).__init__()

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)  ##权重共享的
        self.spatial_encoder = SpatialEncoding(embedding_dim)  ##这两个是权重共享的

        self.encoder = torch.nn.ModuleList()
        for i in range(layer_num - 1):
            self.encoder.append(GraphTransformerEncode(num_heads = num_heads, in_dim = embedding_dim, dim_forward = embedding_dim*2,
                                                       rel_encoder = self.rel_encoder, spatial_encoder = self.spatial_encoder, dropout=dropout,
                                                       attn_norm=attn_norm, attn_dropout=attn_dropout, edge_dropout=edge_dropout))

    def reset_parameters(self):
        for e in self.encoder:
            e.reset_parameters()


    def forward(self, feature, data):

        ##首先就是按照edge index计算attn_weight, 然后按照权重聚合就可以了！！
        x = feature
        graph_embedding_layer = []
        attn_layer = []
        for graphEncoder in self.encoder:
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            graph_embedding_layer.append(x)
            attn_layer.append(attn)

        #all_out = torch.stack([x for x in graph_embedding_layer])


        if self.type == 'graph':
            ##pooling
            sub_representation = []
            for index, drug_mol_graph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]  ##第index个图中的各个节点的表示，[atom_number, emd_dim]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)  ##每个drug分子的图的表示
        else:
            ##只返回第一个
            sub_representation = []
            for index, drug_subgraph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                #print(sub_embedding.shape)
                sub_representation.append(sub_embedding) ##只取那个节点的embedding
            #print(x.shape)
            #print(data.id.shape)
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer

        ##对于节点级别的表示，需要每一层的级联，然后做最后的互信息最大化，这个层级的优化可能要考虑一下，但是最终落到的还是节点和图
