"""
src/data/dataset.py

Implements the multi-horizon balanced PyTorch Dataset for PhysioGraph:
1. Constructs 4-channel node tensors: [Value, Missing_Flag, 1h_Delta, 4h_Velocity].
2. Injects the Clinical Shock Index (HR / SBP) into cardiovascular node representations.
3. Supports 1:1 Case-Control balanced sampling and unselected real-world population evaluation.
4. Enforces zero cross-patient data leakage (window endpoints strictly from the same stay).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import gc

from .graph_topology import NODE_FEATURES

class MultiHorizonPhysioDataset(Dataset):
    def __init__(
        self,
        features_path,
        labels_path,
        split_name="train",
        config="nosepsis",
        lookback=24,
        undersample_ratio="1:1",
        seed=42
    ):
        self.lookback = lookback
        self.config = config
        self.undersample_ratio = undersample_ratio
        
        # Load features & timestamps
        feats = pd.read_parquet(features_path)
        feats["stay_id"] = pd.to_numeric(feats["stay_id"]).astype("int64")
        feats["hour"] = pd.to_datetime(feats["hour"])
        
        # Load multi-horizon labels (H=8h, 6h, 12h)
        labels = pd.read_parquet(labels_path)
        labels["stay_id"] = pd.to_numeric(labels["stay_id"]).astype("int64")
        labels["hour"] = pd.to_datetime(labels["hour"])
        
        col_h8, valid_h8   = f"label_{config}_h8", f"valid_{config}_h8"
        col_h6, valid_h6   = f"label_{config}_h6", f"valid_{config}_h6"
        col_h12, valid_h12 = f"label_{config}_h12", f"valid_{config}_h12"
        
        req_cols = ["stay_id", "hour", "hours", col_h8, valid_h8]
        if col_h6 in labels.columns:  req_cols.extend([col_h6, valid_h6])
        if col_h12 in labels.columns: req_cols.extend([col_h12, valid_h12])
        
        merged = feats.merge(labels[req_cols], on=["stay_id", "hour"], how="inner")
        del feats, labels
        merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        
        # 1. Base Node Values & Missingness
        val_block = merged[NODE_FEATURES].to_numpy(dtype=np.float32)
        mis_cols = [f + "_missing" if (f + "_missing") in merged.columns else f"{f}_mask" for f in NODE_FEATURES]
        mis_block = merged[mis_cols].to_numpy(dtype=np.float32) if all(c in merged.columns for c in mis_cols) else np.zeros_like(val_block)
        
        # 2. Physiological Shock Index Injection (HR_raw / SBP_raw)
        hr_bpm = val_block[:, 2] * 18.14 + 85.99
        sbp_mmHg = np.clip(val_block[:, 3] * 21.66 + 119.84, 40.0, 250.0)
        shock_index = np.clip(hr_bpm / sbp_mmHg, 0.2, 3.0)
        norm_si = (shock_index - 0.70) / 0.25
        
        val_block[:, 2] = val_block[:, 2] * 0.75 + norm_si * 0.25
        val_block[:, 3] = val_block[:, 3] * 0.75 - norm_si * 0.25
        
        # 3. Dual-Scale Dynamics: 1-hour delta and 4-hour acute velocity
        stay_arr = merged["stay_id"].to_numpy(dtype=np.int64)
        delta_1h = np.zeros_like(val_block)
        delta_4h = np.zeros_like(val_block)
        
        same_stay_1 = (stay_arr[1:] == stay_arr[:-1])
        delta_1h[1:][same_stay_1] = val_block[1:][same_stay_1] - val_block[:-1][same_stay_1]
        
        if len(stay_arr) > 4:
            same_stay_4 = (stay_arr[4:] == stay_arr[:-4])
            delta_4h[4:][same_stay_4] = (val_block[4:][same_stay_4] - val_block[:-4][same_stay_4]) / 4.0
            
        self.dense_features = np.stack([val_block, mis_block, delta_1h, delta_4h], axis=-1) # [N, 8, 4]
        self.hours_arr = merged["hours"].to_numpy(dtype=np.float32)
        
        # Targets: Primary H=8, Auxiliary H=6 and H=12
        lbl_h8  = merged[col_h8].to_numpy(dtype=np.int8)
        lbl_h6  = merged[col_h6].to_numpy(dtype=np.int8) if col_h6 in merged.columns else lbl_h8
        lbl_h12 = merged[col_h12].to_numpy(dtype=np.int8) if col_h12 in merged.columns else lbl_h8
        val_arr = merged[valid_h8].to_numpy(dtype=bool)
        
        pos_windows, neg_windows = [], []
        
        for sid, grp in merged.groupby("stay_id", sort=False):
            idxs = grp.index.to_numpy()
            n = len(idxs)
            if n < lookback:
                continue
            for i in range(lookback - 1, n):
                global_idx = idxs[i]
                if val_arr[global_idx]:
                    start_idx = idxs[i - lookback + 1]
                    y8  = int(lbl_h8[global_idx])
                    y6  = int(lbl_h6[global_idx])
                    y12 = int(lbl_h12[global_idx])
                    w = (start_idx, global_idx, y8, y6, y12, sid, float(self.hours_arr[global_idx]))
                    if y8 == 1:
                        pos_windows.append(w)
                    else:
                        neg_windows.append(w)
                        
        del merged
        gc.collect()
        
        if undersample_ratio not in ["none", None]:
            ratio_map = {"1:1": 1, "1:2": 2, "1:4": 4, "1:10": 10}
            multiplier = ratio_map.get(undersample_ratio, 1)
            num_neg_to_sample = min(len(neg_windows), len(pos_windows) * multiplier)
            
            rng = np.random.RandomState(seed)
            sampled_neg_idxs = rng.choice(len(neg_windows), size=num_neg_to_sample, replace=False)
            sampled_neg_windows = [neg_windows[i] for i in sampled_neg_idxs]
            
            final_windows = pos_windows + sampled_neg_windows
            rng.shuffle(final_windows)
            self.prevalence = len(pos_windows) / max(len(final_windows), 1)
        else:
            final_windows = pos_windows + neg_windows
            self.prevalence = len(pos_windows) / max(len(final_windows), 1)
            
        self.windows = final_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx, end_idx, y8, y6, y12, sid, h_icu = self.windows[idx]
        node_tensor = torch.from_numpy(self.dense_features[start_idx:end_idx + 1]).contiguous() # [24, 8, 4]
        targets = torch.tensor([y8, y6, y12], dtype=torch.float32) # [3] -> (H=8, H=6, H=12)
        return node_tensor, targets, sid, h_icu


def collate_multi_horizon(batch):
    tensors, targets, sids, h_icus = zip(*batch)
    return {
        "x": torch.stack(tensors, dim=0),
        "y": torch.stack(targets, dim=0),
        "stay_id": torch.tensor(sids, dtype=torch.int64),
        "hour_in_icu": torch.tensor(h_icus, dtype=torch.float32)
    }
