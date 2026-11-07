#!/usr/bin/env python3
"""
run_graph_ablation.py
---------------------
Tier 2: Complete Self-Contained Graph Ablation Experiment Runner.
Evaluates the core thesis question:
  "Does the physiologically-grounded 15-edge structure actually improve prediction
   over arbitrary connectivity or isolated temporal processing?"

Ablation Matrix:
  1. Full Clinical (15 edges: 10 forward cascade + 5 feedback reflex arcs)
  2. No Edges (0 edges / self-loops only: isolates temporal TCN from spatial message passing)
  3. Complete Graph (All 56 ordered directed pairs: tests dense connectivity vs sparse mechanism)
  4. Random-15 (15 random directed edges across seeds: tests specific vs arbitrary sparsity)
  5. Standalone H=8 Head (Isolates multi-horizon auxiliary regularisation benefit)

Execution Target:
  Google Colab (Free T4 GPU, ~5 mins per run, ~20 mins total for all 4 variants).
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("⚠️ torch_geometric not installed. Install via: pip install torch-geometric")

# ==============================================================================
# 1. GRAPH TOPOLOGY & ABLATION DEFINITIONS
# ==============================================================================
NODE_FEATURES = [
    "resp_rate",    # 0: Pulmonary
    "spo2",         # 1: Pulmonary
    "heart_rate",   # 2: Cardiovascular
    "sbp",          # 3: Cardiovascular
    "temperature",  # 4: Systemic / Thermal
    "wbc",          # 5: Systemic / Thermal
    "lactate",      # 6: Renal / Metabolic
    "creatinine"    # 7: Renal / Metabolic
]
NUM_NODES = len(NODE_FEATURES)

# 1. Full 15 Clinical Directed Edges
CLINICAL_EDGES_15 = [
    (0, 1), # Resp Rate -> SpO2
    (1, 2), # SpO2 -> Heart Rate
    (2, 3), # Heart Rate -> SBP
    (3, 7), # SBP -> Creatinine (Renal hypoperfusion)
    (6, 7), # Lactate -> Creatinine (Metabolic acidosis compounds renal stress)
    (4, 5), # Temperature -> WBC (Systemic inflammatory response)
    (5, 6), # WBC -> Lactate (Microcirculatory hypoperfusion)
    (0, 6), # Resp Rate -> Lactate (Anaerobic shift)
    (2, 6), # Heart Rate -> Lactate (Circulatory shock)
    (1, 6), # SpO2 -> Lactate (Tissue hypoxia)
    (1, 0), # SpO2 -> Resp Rate (Carotid reflex)
    (3, 2), # SBP -> Heart Rate (Baroreceptor reflex)
    (6, 0), # Lactate -> Resp Rate (Kussmaul hyperventilation)
    (7, 3), # Creatinine -> SBP (Renin-angiotensin hypertension)
    (6, 3), # Lactate -> SBP (Distributive vasodilation)
]

# 2. No Edges (Empty list - self-loops only via GAT add_self_loops=True)
NO_EDGES = []

# 3. Complete Graph (All 8 * 7 = 56 directed pairs)
COMPLETE_EDGES_56 = [
    (i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j
]

# 4. Random-15 Generator
def generate_random_edges_15(seed: int = 42):
    rng = random.Random(seed)
    all_possible = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]
    return rng.sample(all_possible, 15)


def get_edge_index(variant: str, seed: int = 42, device: str = "cpu"):
    if variant == "clinical":
        edges = CLINICAL_EDGES_15
    elif variant == "no_edges":
        edges = NO_EDGES
    elif variant == "complete":
        edges = COMPLETE_EDGES_56
    elif variant == "random":
        edges = generate_random_edges_15(seed)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


# ==============================================================================
# 2. DATASET PATH FINDER & LOADER
# ==============================================================================
def resolve_data_paths():
    candidate_dirs = [
        Path("/content/drive/MyDrive/MIMIC-IV-v3.1"),
        Path("/content/drive/MyDrive/mimic-iv-v3.1"),
        Path("/content/drive/MyDrive/FYP_ICU"),
        Path("/content/FYP_ICU"),
        Path("/content"),
        Path("data/processed"),
        Path("./")
    ]
    data_dir = None
    for cdir in candidate_dirs:
        if (cdir / "train_filled.parquet").exists() or (cdir / "checkpoints" / "train_filled.parquet").exists():
            data_dir = cdir if (cdir / "train_filled.parquet").exists() else cdir / "checkpoints"
            print(f"✓ Found dataset directory at: {data_dir}")
            break

    if data_dir is None:
        raise FileNotFoundError(
            "Could not find train_filled.parquet! Please ensure your parquet files are in Google Drive "
            "or uploaded to /content/"
        )

    def find_file(prefix, suffix):
        cands = [
            data_dir / f"{prefix}_{suffix}.parquet",
            data_dir.parent / suffix / f"{prefix}_{suffix}.parquet",
            data_dir / suffix / f"{prefix}_{suffix}.parquet",
        ]
        for c in cands:
            if c.exists():
                return c
        return cands[0]

    paths = {
        "train_feats": find_file("train", "filled"),
        "val_feats": find_file("val", "filled"),
        "test_feats": find_file("test", "filled"),
        "train_lbls": find_file("train", "labels_v2"),
        "val_lbls": find_file("val", "labels_v2"),
        "test_lbls": find_file("test", "labels_v2"),
    }
    return paths


class MultiHorizonPhysioDataset(Dataset):
    def __init__(self, feats_path, lbls_path, split="train", config="nosepsis", lookback=24, undersample="1:1", seed=42):
        self.lookback = lookback
        df_feats = pd.read_parquet(feats_path)
        df_feats["stay_id"] = pd.to_numeric(df_feats["stay_id"]).astype("int64")
        df_feats["hour"] = pd.to_datetime(df_feats["hour"])

        df_lbls = pd.read_parquet(lbls_path)
        df_lbls["stay_id"] = pd.to_numeric(df_lbls["stay_id"]).astype("int64")
        df_lbls["hour"] = pd.to_datetime(df_lbls["hour"])

        col_h8, val_h8 = f"label_{config}_h8", f"valid_{config}_h8"
        col_h6, val_h6 = f"label_{config}_h6", f"valid_{config}_h6"
        col_h12, val_h12 = f"label_{config}_h12", f"valid_{config}_h12"

        cols = ["stay_id", "hour", "hours", col_h8, val_h8]
        if col_h6 in df_lbls.columns: cols.extend([col_h6, val_h6])
        if col_h12 in df_lbls.columns: cols.extend([col_h12, val_h12])

        merged = df_feats.merge(df_lbls[cols], on=["stay_id", "hour"], how="inner")
        del df_feats, df_lbls
        merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)

        val_block = merged[NODE_FEATURES].to_numpy(dtype=np.float32)
        mis_cols = [f + "_missing" for f in NODE_FEATURES]
        mis_block = merged[mis_cols].to_numpy(dtype=np.float32) if all(c in merged.columns for c in mis_cols) else np.zeros_like(val_block)

        # 4-channel tensor [Value, Missing, 1h_Delta, 4h_Velocity]
        delta_1h = np.zeros_like(val_block)
        delta_1h[1:] = val_block[1:] - val_block[:-1]
        delta_4h = np.zeros_like(val_block)
        delta_4h[4:] = (val_block[4:] - val_block[:-4]) / 4.0

        all_channels = np.stack([val_block, mis_block, delta_1h, delta_4h], axis=-1)

        # Window indexing
        stay_arr = merged["stay_id"].to_numpy()
        valid_mask = (merged[val_h8] == 1).to_numpy()
        h8_labels = merged[col_h8].to_numpy(dtype=np.float32)
        h6_labels = merged[col_h6].to_numpy(dtype=np.float32) if col_h6 in merged.columns else h8_labels
        h12_labels = merged[col_h12].to_numpy(dtype=np.float32) if col_h12 in merged.columns else h8_labels
        hours_arr = merged["hours"].to_numpy(dtype=np.float32)

        # Precompute valid window indices (lookback=24)
        indices = []
        for i in range(lookback - 1, len(merged)):
            if valid_mask[i] and stay_arr[i] == stay_arr[i - lookback + 1]:
                indices.append(i)

        indices = np.array(indices)
        if undersample == "1:1":
            rng = np.random.default_rng(seed)
            pos_idx = indices[h8_labels[indices] == 1]
            neg_idx = indices[h8_labels[indices] == 0]
            sampled_neg = rng.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
            indices = np.sort(np.concatenate([pos_idx, sampled_neg]))
        
        prevalence = np.mean(h8_labels[indices])
        print(f"✓ [{split.upper()}] Final Windows: {len(indices):,} | Positives: {int(np.sum(h8_labels[indices])):,} | Prevalence: {prevalence*100:.2f}%", flush=True)

        self.all_channels = all_channels
        self.indices = indices
        self.h8_labels = h8_labels
        self.h6_labels = h6_labels
        self.h12_labels = h12_labels
        self.hours_arr = hours_arr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.lookback + 1
        x = self.all_channels[start:end+1] # [24, 8, 4]
        return {
            "x": torch.from_numpy(x),
            "hours": torch.tensor(self.hours_arr[end] / 100.0, dtype=torch.float32),
            "y8": torch.tensor(self.h8_labels[end], dtype=torch.float32),
            "y6": torch.tensor(self.h6_labels[end], dtype=torch.float32),
            "y12": torch.tensor(self.h12_labels[end], dtype=torch.float32),
        }


# ==============================================================================
# 3. PHYSIOGRAPH NEURAL NETWORK MODEL
# ==============================================================================
class GATEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res = nn.ModuleList()
        in_dim = in_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_heads = 1 if is_last else heads
            concat = not is_last
            self.convs.append(GATConv(in_dim, hidden_dim, heads=out_heads, dropout=dropout, concat=concat, add_self_loops=True))
            out_dim = hidden_dim if is_last else hidden_dim * heads
            self.norms.append(nn.BatchNorm1d(out_dim))
            self.res.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())
            in_dim = out_dim
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: [B*T*8, 4]
        h = x
        for conv, norm, res in zip(self.convs, self.norms, self.res):
            res_val = res(h)
            h = conv(h, edge_index)
            h = norm(h)
            h = self.drop(self.act(h)) + res_val
        return h


class CausalTCNBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x):
        res = x
        out = self.conv1(x)[:, :, :-self.pad] if self.pad > 0 else self.conv1(x)
        out = self.drop(self.act(self.norm1(out)))
        out = self.conv2(out)[:, :, :-self.pad] if self.pad > 0 else self.conv2(out)
        out = self.drop(self.act(self.norm2(out)))
        return out + res


class PhysioGraphModel(nn.Module):
    def __init__(self, edge_index, hidden_dim=64):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.gat = GATEncoder(in_channels=4, hidden_dim=hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.mha_norm = nn.LayerNorm(hidden_dim)

        self.tcn = nn.Sequential(
            CausalTCNBlock(hidden_dim, dilation=1),
            CausalTCNBlock(hidden_dim, dilation=2),
            CausalTCNBlock(hidden_dim, dilation=4),
            CausalTCNBlock(hidden_dim, dilation=8),
        )
        self.time_pool = nn.Linear(24, 1)
        self.node_pool = nn.Linear(8, 1)

        # Multi-Horizon Heads (H=8, H=6, H=12)
        in_feat = hidden_dim + 1
        self.head_8 = nn.Sequential(nn.Linear(in_feat, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.head_6 = nn.Sequential(nn.Linear(in_feat, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.head_12 = nn.Sequential(nn.Linear(in_feat, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, hours):
        # x: [B, 24, 8, 4]
        B, T, N, C = x.shape
        x_flat = x.view(B * T * N, C)

        # 1. GAT message passing
        h_gat = self.gat(x_flat, self.edge_index) # [B*T*N, 64]
        h_gat = h_gat.view(B * T, N, -1)

        # 2. Cross-organ MHA
        h_mha, _ = self.mha(h_gat, h_gat, h_gat)
        h_mha = self.mha_norm(h_gat + h_mha) # [B*T, N, 64]

        # 3. Temporal TCN
        h_tcn_in = h_mha.view(B, T, N, -1).permute(0, 2, 3, 1).contiguous().view(B * N, 64, T)
        h_tcn_out = self.tcn(h_tcn_in) # [B*N, 64, T]

        # 4. Pooling
        h_time = self.time_pool(h_tcn_out).squeeze(-1).view(B, N, 64).permute(0, 2, 1) # [B, 64, N]
        h_node = self.node_pool(h_time).squeeze(-1) # [B, 64]

        # 5. Elapsed ICU time injection
        feat = torch.cat([h_node, hours.unsqueeze(-1)], dim=-1)

        l8 = self.head_8(feat).squeeze(-1)
        l6 = self.head_6(feat).squeeze(-1)
        l12 = self.head_12(feat).squeeze(-1)
        return l8, l6, l12


# ==============================================================================
# 4. TRAINING & EVALUATION LOOP
# ==============================================================================
def train_and_eval_variant(variant: str, paths: dict, seed: int = 42, epochs: int = 8, batch_size: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*80)
    print(f"RUNNING GRAPH ABLATION VARIANT: {variant.upper()} (Seed={seed}) ON {device.upper()}")
    print("="*80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    edge_index = get_edge_index(variant, seed=seed, device=device)
    print(f"Topology: {edge_index.shape[1]} directed edges.")

    train_ds = MultiHorizonPhysioDataset(paths["train_feats"], paths["train_lbls"], split="train", undersample="1:1", seed=seed)
    val_ds = MultiHorizonPhysioDataset(paths["val_feats"], paths["val_lbls"], split="val", undersample="1:1", seed=seed)
    test_ds = MultiHorizonPhysioDataset(paths["test_feats"], paths["test_lbls"], split="test", undersample="1:1")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = PhysioGraphModel(edge_index=edge_index).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, hours = batch["x"].to(device), batch["hours"].to(device)
            y8, y6, y12 = batch["y8"].to(device), batch["y6"].to(device), batch["y12"].to(device)

            optimizer.zero_grad()
            l8, l6, l12 = model(x, hours)
            loss = criterion(l8, y8) + 0.3 * criterion(l6, y6) + 0.3 * criterion(l12, y12)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, hours = batch["x"].to(device), batch["hours"].to(device)
                l8, l6, l12 = model(x, hours)
                prob = torch.sigmoid(0.70 * l8 + 0.15 * l6 + 0.15 * l12)
                val_preds.extend(prob.cpu().numpy())
                val_trues.extend(batch["y8"].numpy())

        v_auroc = roc_auc_score(val_trues, val_preds)
        v_auprc = average_precision_score(val_trues, val_preds)
        print(f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val AUROC: {v_auroc:.4f} | Val AUPRC: {v_auprc:.4f}")

        if v_auprc > best_val_auprc:
            best_val_auprc = v_auprc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Test Evaluation
    model.load_state_dict(best_state)
    model.eval()
    t_preds_blend, t_preds_h8, t_trues = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, hours = batch["x"].to(device), batch["hours"].to(device)
            l8, l6, l12 = model(x, hours)
            p_blend = torch.sigmoid(0.70 * l8 + 0.15 * l6 + 0.15 * l12)
            p_h8 = torch.sigmoid(l8)
            t_preds_blend.extend(p_blend.cpu().numpy())
            t_preds_h8.extend(p_h8.cpu().numpy())
            t_trues.extend(batch["y8"].numpy())

    test_auroc = roc_auc_score(t_trues, t_preds_blend)
    test_auprc = average_precision_score(t_trues, t_preds_blend)
    h8_auroc = roc_auc_score(t_trues, t_preds_h8)
    h8_auprc = average_precision_score(t_trues, t_preds_h8)

    print(f"\n[FINAL TEST RESULT — {variant.upper()}]")
    print(f"Blended Heads : AUROC = {test_auroc*100:.2f}% | AUPRC = {test_auprc*100:.2f}%")
    print(f"H=8 Head Alone: AUROC = {h8_auroc*100:.2f}% | AUPRC = {h8_auprc*100:.2f}%")
    return {
        "variant": variant,
        "edges": edge_index.shape[1],
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "h8_alone_auroc": h8_auroc,
        "h8_alone_auprc": h8_auprc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["clinical", "no_edges", "complete", "random", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    args, _ = parser.parse_known_args()

    paths = resolve_data_paths()

    if args.variant == "all":
        results = []
        for v in ["clinical", "no_edges", "complete", "random"]:
            res = train_and_eval_variant(v, paths, seed=args.seed, epochs=args.epochs)
            results.append(res)

        print("\n" + "="*80)
        print("SUMMARY: PHYSIOGRAPH GRAPH ABLATION BENCHMARK")
        print("="*80)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
    else:
        train_and_eval_variant(args.variant, paths, seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
