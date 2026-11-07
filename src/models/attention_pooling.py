"""
src/models/attention_pooling.py

Attention Pooling Mechanisms:
1. TemporalTrajectoryAttention: Learns inflection points and rates of acceleration across the 24h timeline.
2. OrganAttentionPooling: Dynamically weights organ contributions for patient-level aggregation and clinical explainability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTrajectoryAttention(nn.Module):
    """
    Trajectory attention across time: learns inflection points rather than taking a simple last-step snapshot.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: [B * N, hidden_dim, T=24]
        x_perm = x.permute(0, 2, 1) # [B * N, T=24, hidden_dim]
        scores = self.attn(x_perm)  # [B * N, T=24, 1]
        weights = F.softmax(scores, dim=1)
        pooled = (x_perm * weights).sum(dim=1) # [B * N, hidden_dim]
        return pooled


class OrganAttentionPooling(nn.Module):
    """
    Organ-level attention pooling: dynamically determines which organ system is in crisis.
    Returns the aggregated patient state and the organ attention distribution for clinical explainability.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: [B, N=8, hidden_dim]
        scores = self.attn(x)      # [B, N=8, 1]
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1) # [B, hidden_dim]
        return pooled, weights.squeeze(-1) # [B, hidden_dim], [B, N=8]
