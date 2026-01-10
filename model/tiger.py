# -*- coding: utf-8 -*-
# @Time    : 2023/5/28 下午6:23
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : tiger.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
try:
    # When tiger.py is used as part of a package (e.g., model/tiger.py)
    from .GraphTransformer import GraphTransformer
except ImportError:  # pragma: no cover
    # When the code is executed as standalone scripts from the repo root.
    from GraphTransformer import GraphTransformer
import os



class NodeFeatures(torch.nn.Module):
    def __init__(self, degree, feature_num, embedding_dim, layer=2, type='graph'):
        super(NodeFeatures, self).__init__()

        if type == 'graph': ##代表有feature num
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)  ##将度的值映射成embedding
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):

        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())

        return node_feature

class TIGER(torch.nn.Module):
    """TIGER: dual-channel DDI prediction model.

    This repo's baseline fuses the two channel-level drug embeddings via
    simple concatenation followed by an MLP (self.fc1).

    Improvement added here (backward compatible): **adaptive gated fusion**.
    Instead of a fixed concatenation, a learnable gate g\in(0,1)^d is computed
    per-drug to reweight the two channel embeddings before fusion.
    """

    def __init__(
            self,
            max_layer=6,
            num_features_drug=78,
            num_nodes=200,
            num_relations_mol=10,
            num_relations_graph=10,
            output_dim=64,
            max_degree_graph=100,
            max_degree_node=100,
            sub_coeff=0.2,
            mi_coeff=0.5,
            dropout=0.2,
            device='cuda',
            fusion: str = 'concat',
            gate_coeff: float = 0.0,
            attn_norm: str = 'softmax',
            attn_dropout: float = 0.1,
            edge_dropout: float = 0.0,
            mi_objective: str = 'bce',
            mi_temp: float = 0.2,
    ):
        super(TIGER, self).__init__()

        print("TIGER Loaded")
        self.device = device

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        # Mutual-information objective between the final drug embedding and local sub-embeddings.
        # - 'bce'    : repo's original discriminator-based BCE loss
        # - 'infonce': in-batch contrastive InfoNCE (recommended for stability)
        if mi_objective not in {'bce', 'infonce'}:
            raise ValueError("mi_objective must be 'bce' or 'infonce'")
        self.mi_objective = mi_objective
        self.mi_temp = float(mi_temp)

        # Fusion mode.
        # - 'concat' (default): [node_emb || mol_emb] -> fc1
        # - 'gated'           : gate([node_emb||mol_emb]) to reweight channels before fc1
        if fusion not in {'concat', 'gated'}:
            raise ValueError(f"Unsupported fusion='{fusion}'. Use 'concat' or 'gated'.")
        self.fusion = fusion
        self.gate_coeff = gate_coeff

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, feature_num=num_features_drug, embedding_dim=output_dim, type='graph')
        self.drug_node_feature = NodeFeatures(degree=max_degree_node, feature_num=num_nodes, embedding_dim=output_dim, type='node')

        ##学习的模块
        # GraphTransformer attention improvements are controlled by:
        #  - attn_norm: 'softmax' (recommended) or 'ratio' (legacy)
        #  - attn_dropout: dropout on attention weights
        #  - edge_dropout: DropEdge rate on sp_edge_index
        self.mol_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_mol,
            dropout=dropout,
            type='graph',
            attn_norm=attn_norm,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
        )
        self.node_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_graph,
            dropout=dropout,
            type='node',
            attn_norm=attn_norm,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
        )
        ##Net用统一的代码就可以了，用type指示是哪种类型的学习，或者分开两个模块，然后两个模块里面集合一些公共的模块

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        # Adaptive gate for dual-channel fusion (only used when fusion == 'gated').
        # Output is a per-dimension gate in (0,1)^d.
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    def to(self, device):

        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)

        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)

        self.fc1.to(device)
        self.gate.to(device)
        self.fc2.to(device)

        self.disc.to(device)
        self.b_xent.to(device)

    def reset_parameters(self):

        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()

        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()


    def forward(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph):

        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        ##在获得了特征之后，就是学习相应的表示，一个是节点级别的表示，一个是图级别的表示！！
        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)


        # ---- Dual-channel fusion (concat or gated) ----
        gate_loss = torch.tensor(0.0, device=self.device)

        if self.fusion == 'concat':
            drug1_fuse_in = torch.concat([drug1_node_embedding, mol1_graph_embedding], dim=-1)
            drug2_fuse_in = torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1)
        else:  # self.fusion == 'gated'
            gate1 = self.gate(torch.concat([drug1_node_embedding, mol1_graph_embedding], dim=-1))
            gate2 = self.gate(torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1))

            # Reweight the two channels (element-wise gate).
            # gate ~= 1 -> rely more on node/subgraph channel
            # gate ~= 0 -> rely more on molecule/SMILES channel
            drug1_fuse_in = torch.concat([gate1 * drug1_node_embedding, (1.0 - gate1) * mol1_graph_embedding], dim=-1)
            drug2_fuse_in = torch.concat([gate2 * drug2_node_embedding, (1.0 - gate2) * mol2_graph_embedding], dim=-1)

            # Optional entropy regularization to avoid early collapse of gates.
            # Minimizing -H(g) is equivalent to maximizing entropy.
            if self.gate_coeff and self.gate_coeff > 0:
                eps = 1e-8
                ent1 = -(gate1 * torch.log(gate1 + eps) + (1.0 - gate1) * torch.log(1.0 - gate1 + eps))
                ent2 = -(gate2 * torch.log(gate2 + eps) + (1.0 - gate2) * torch.log(1.0 - gate2 + eps))
                gate_loss = -(ent1.mean() + ent2.mean())

        drug1_embedding = self.fc1(drug1_fuse_in)
        drug2_embedding = self.fc1(drug2_fuse_in)

        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))

        loss_s_m = self.mi_loss(drug1_embedding, mol1_atom_embedding) + self.mi_loss(drug2_embedding, mol2_atom_embedding)
        loss_s_d = self.mi_loss(drug1_embedding, drug1_sub_embedding) + self.mi_loss(drug2_embedding, drug2_sub_embedding)


        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        loss = loss_label + self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d
        if self.fusion == 'gated' and self.gate_coeff and self.gate_coeff > 0:
            loss = loss + self.gate_coeff * gate_loss

        return torch.exp(predicts_drug)[:,1], loss

    def MI(self, graph_embeddings, sub_embeddings):
        # Mutual information maximization between graph-level embeddings and local sub-embeddings.
        # This requires *negative* samples: shuffle graph_embeddings across the batch.
        # NOTE: the original implementation used a hand-crafted reverse index and could
        # break for small batches (e.g., last batch size == 1) and may introduce duplicates.
        batch_size = graph_embeddings.shape[0]
        if batch_size < 2:
            # No valid negative samples.
            return torch.zeros((0,), device=graph_embeddings.device)

        device = graph_embeddings.device
        idx = torch.randperm(batch_size, device=device)
        # Fix any accidental fixed points (idx[i] == i) by swapping with the next.
        base = torch.arange(batch_size, device=device)
        for i in range(batch_size):
            if idx[i].item() == base[i].item():
                j = (i + 1) % batch_size
                tmp = idx[i].clone()
                idx[i] = idx[j]
                idx[j] = tmp

        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx)
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub)) ##pos
            c_1_list.append(c_1.expand_as(sub)) ##neg
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def infonce_MI(self, graph_embeddings, sub_embeddings):
        """In-batch InfoNCE between local sub-embeddings and graph-level embeddings.

        For each local vector s belonging to sample i in the batch, we treat its
        positive as graph_embeddings[i] and negatives as all other graph embeddings
        in the batch. This is more stable than the discriminator-based BCE loss.
        """
        batch_size = graph_embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=graph_embeddings.device)

        g = F.normalize(graph_embeddings, dim=-1)  # [B, d]
        total_loss = torch.tensor(0.0, device=graph_embeddings.device)
        total_cnt = 0
        tau = max(self.mi_temp, 1e-6)

        for i, sub in enumerate(sub_embeddings):
            if sub is None or sub.numel() == 0:
                continue
            s = F.normalize(sub, dim=-1)  # [n_i, d]
            logits = (s @ g.T) / tau  # [n_i, B]
            labels = torch.full((s.size(0),), i, dtype=torch.long, device=s.device)
            total_loss = total_loss + F.cross_entropy(logits, labels, reduction='sum')
            total_cnt += s.size(0)

        if total_cnt == 0:
            return torch.tensor(0.0, device=graph_embeddings.device)
        return total_loss / total_cnt

    def mi_loss(self, graph_embeddings, sub_embeddings):
        """Unified MI loss wrapper."""
        if self.mi_objective == 'infonce':
            return self.infonce_MI(graph_embeddings, sub_embeddings)
        # Default: BCE with discriminator
        return self.loss_MI(self.MI(graph_embeddings, sub_embeddings))

    def loss_MI(self, logits):

        if logits.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1,-1]), lbl.view([1, -1]))

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path



class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits



def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)