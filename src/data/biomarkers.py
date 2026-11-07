"""
src/data/biomarkers.py

Extracts 70 high-leverage non-linear clinical interaction biomarkers from the 24-hour lookback window:
1. Summary Statistics across all 8 organs (Mean, Min, Max, Std, Last, 24h Drift, 4h Delta).
2. Clinical Shock Index & Velocity (HR / SBP, Delta SI).
3. ROX Index Proxy (SpO2 / RR).
4. Heart Rate Volatility Proxy (6h rolling standard deviation).
5. Cardio-Metabolic Crisis Index (Max Lactate / Min SBP).
6. Acidosis-Hypoxia Interaction (Lactate * Hypoxemia severity).
7. Renal Injury Velocity (Creatinine Delta).
8. ICU Elapsed Stay Time.
"""
import numpy as np

def extract_clinical_interaction_features(dataset, max_samples=None):
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    feature_matrix = []
    labels = []
    
    for i in range(n):
        node_tensor, targets, sid, h_icu = dataset[i]
        arr = node_tensor[:, :, 0].numpy() # [24, 8] values
        
        # 1. Base Summary Statistics
        mean_vals = np.mean(arr, axis=0)
        min_vals = np.min(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        std_vals = np.std(arr, axis=0)
        last_vals = arr[-1]
        drift_24h = arr[-1] - arr[0]
        recent_4h = arr[-1] - arr[-5] if len(arr) >= 5 else np.zeros(8)
        
        # 2. Raw Clinical Physical Units
        rr_raw = arr[:, 0] * 5.93 + 20.18
        spo2_raw = arr[:, 1] * 3.02 + 96.74
        hr_raw = arr[:, 2] * 18.14 + 85.99
        sbp_raw = np.clip(arr[:, 3] * 21.66 + 119.84, 40.0, 250.0)
        lactate_raw = arr[:, 6] * 2.52 + 2.54
        creat_raw = arr[:, 7] * 1.49 + 1.51
        
        # 3. Clinical Shock Index (SI) & Velocity
        si = hr_raw / sbp_raw
        si_mean = np.mean(si)
        si_max = np.max(si)
        si_last = si[-1]
        si_velocity_4h = si[-1] - (si[-5] if len(si) >= 5 else si[0])
        
        # 4. ROX Index Proxy: SpO2 / RR
        rox = spo2_raw / np.clip(rr_raw, 8.0, 60.0)
        rox_mean = np.mean(rox)
        rox_min = np.min(rox)
        rox_last = rox[-1]
        
        # 5. Heart Rate Volatility (HRV proxy)
        hr_volatility_6h = np.std(hr_raw[-6:]) if len(hr_raw) >= 6 else np.std(hr_raw)
        
        # 6. Cardio-Metabolic Crisis Index
        cm_crisis_index = np.max(lactate_raw) / max(np.min(sbp_raw), 40.0)
        
        # 7. Acidosis-Hypoxia Strain
        acidosis_hypoxia = lactate_raw[-1] * max(100.0 - spo2_raw[-1], 0.0)
        
        # 8. Renal Injury Velocity
        creat_velocity = creat_raw[-1] - creat_raw[0]
        
        row = np.concatenate([
            mean_vals, min_vals, max_vals, std_vals, last_vals, drift_24h, recent_4h,
            [si_mean, si_max, si_last, si_velocity_4h],
            [rox_mean, rox_min, rox_last],
            [hr_volatility_6h, cm_crisis_index, acidosis_hypoxia, creat_velocity, h_icu]
        ])
        feature_matrix.append(row)
        labels.append(targets[0].item()) # Primary H=8 target
        
    return np.array(feature_matrix, dtype=np.float32), np.array(labels, dtype=np.float32)
