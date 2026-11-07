"""
src/models/gat_encoder.py

Spatial Graph Attention Network (GAT) and Spatial Organ Self-Attention (Transformer) modules:
- GATEncoder: Multi-head message passing over 15 directed physiological edges with residual connections.
- DynamicOrganSelfAttention: Cross-organ Multi-Head Attention capturing adaptive clinical cross-talk.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        in_dim = in_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_heads = 1 if is_last else heads
            concat = not is_last
            
            self.convs.append(
                GATConv(in_dim, hidden_dim, heads=out_heads, dropout=dropout, concat=concat, add_self_loops=True)
            )
            out_dim = hidden_dim if is_last else hidden_dim * heads
            self.norms.append(nn.BatchNorm1d(out_dim))
            self.residual_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())
            in_dim = out_dim
            
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = x
        for conv, norm, res in zip(self.convs, self.norms, self.residual_projs):
            res_val = res(h)
            h = conv(h, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + res_val
        return h


class DynamicOrganSelfAttention(nn.Module):
    """
    Multi-Head Organ Self-Attention (Spatial Transformer)
    Allows all 8 organ nodes to attend to each other adaptively during multi-organ shock cascades.
    """
    def __init__(self, hidden_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h):
        attn_out, _ = self.mha(h, h, h)
        return self.norm(h + attn_out)
