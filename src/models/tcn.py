"""
src/models/tcn.py

Node-Preserving 4-Layer Causal Dilated Temporal Convolutional Network (TCN):
- Dilations D = [1, 2, 4, 8] with kernel size K = 3.
- Total Receptive Field: RF = 1 + 2 * (1 + 2 + 4 + 8) = 31 hours (exceeds the 24-hour lookback).
- Left-padded causal convolutions strictly prevent future information leakage.
- Processes each organ node's trajectory independently over time.
"""
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout(self.act(self.norm1(self.conv1(x))))
        out = self.dropout(self.act(self.norm2(self.conv2(out))))
        return self.act(out + res)


class NodeTCN31h(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        c = in_channels
        for i in range(num_layers):
            layers.append(TCNBlock(c, hidden_dim, kernel_size, dilation=2**i, dropout=dropout))
            c = hidden_dim
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input: [B * N, C, T=24] -> Output: [B * N, hidden_dim, T=24]
        return self.network(x)
