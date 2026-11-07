"""
src/models/physiograph_pro.py

Unified PhysioGraph Multi-Horizon Pro Architecture:
- Combines Spatial GAT + Dynamic Organ Self-Attention (Spatial Transformer).
- Node-Preserving 4-Layer Causal Dilated TCN (31h Receptive Field).
- Temporal Trajectory Attention Pooling + Organ Attention Pooling.
- Concatenates elapsed ICU stay time (hour_in_icu).
- Multi-Horizon Auxiliary Heads: H=8 (Primary), H=6 (Acute), H=12 (Subacute).
"""
import torch
import torch.nn as nn
from .gat_encoder import GATEncoder, DynamicOrganSelfAttention
from .tcn import NodeTCN31h
from .attention_pooling import TemporalTrajectoryAttention, OrganAttentionPooling
from ..data.graph_topology import EDGE_INDEX

class PhysioGraphMultiHorizonPro(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_nodes=8,
        gat_hidden_dim=64,
        tcn_hidden_dim=64,
        mlp_hidden_dim=32,
        dropout=0.1,
        edge_index=None
    ):
        super().__init__()
        if edge_index is None:
            edge_index = EDGE_INDEX
        self.register_buffer("edge_index", edge_index)
        self.gat_hidden_dim = gat_hidden_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        
        # 1. Spatial Processing
        self.gat = GATEncoder(in_channels=in_channels, hidden_dim=gat_hidden_dim, heads=4, dropout=dropout)
        self.dynamic_organ_attn = DynamicOrganSelfAttention(hidden_dim=gat_hidden_dim, heads=4, dropout=dropout)
        
        # 2. Temporal Processing
        self.tcn = NodeTCN31h(in_channels=gat_hidden_dim, hidden_dim=tcn_hidden_dim, num_layers=4, dropout=dropout)
        
        # 3. Attention Pooling
        self.temporal_pool = TemporalTrajectoryAttention(hidden_dim=tcn_hidden_dim)
        self.organ_pool = OrganAttentionPooling(hidden_dim=tcn_hidden_dim)
        
        rep_dim = tcn_hidden_dim + 1 # +1 for elapsed ICU stay time
        
        # 4. Multi-Horizon Prediction Heads
        self.head_h8 = nn.Sequential(
            nn.Linear(rep_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.head_h6 = nn.Sequential(
            nn.Linear(rep_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.head_h12 = nn.Sequential(
            nn.Linear(rep_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
    def forward(self, x, hour_in_icu):
        B, T, N, C = x.shape
        x_flat = x.reshape(B * T * N, C)
        
        batch_offsets = torch.arange(B * T, device=x.device) * N
        edge_index_batched = (self.edge_index.unsqueeze(1) + batch_offsets.reshape(1, -1, 1)).permute(0, 2, 1).reshape(2, -1)
        
        # 1. Spatial GAT & Dynamic Organ Transformer
        h_spatial = self.gat(x_flat, edge_index_batched)
        h_spatial = h_spatial.reshape(B * T, N, self.gat_hidden_dim)
        h_spatial = self.dynamic_organ_attn(h_spatial)
        
        # 2. 4-Layer 31-Hour TCN
        h_spatial = h_spatial.reshape(B, T, N, self.gat_hidden_dim).permute(0, 2, 3, 1)
        h_spatial = h_spatial.reshape(B * N, self.gat_hidden_dim, T)
        h_temp_seq = self.tcn(h_spatial)
        
        # 3. Trajectory Attention & Organ Pooling
        h_temp_pooled = self.temporal_pool(h_temp_seq)
        h_temp_nodes = h_temp_pooled.reshape(B, N, self.tcn_hidden_dim)
        pooled, organ_weights = self.organ_pool(h_temp_nodes)
        
        # 4. Concatenate ICU Stay Time Embedding
        norm_hour = (hour_in_icu.reshape(B, 1).float() / 100.0).clamp(0.0, 10.0)
        combined = torch.cat([pooled, norm_hour], dim=-1)
        
        # 5. Multi-Horizon Outputs
        logits_h8  = self.head_h8(combined).squeeze(-1)
        logits_h6  = self.head_h6(combined).squeeze(-1)
        logits_h12 = self.head_h12(combined).squeeze(-1)
        
        return logits_h8, logits_h6, logits_h12, organ_weights
