"""
src/models/mews.py

Modified Early Warning Score (MEWS) Clinical Rule-Based Baseline Model.
Implements the standardized bedside vital-sign scoring protocol:
- SBP (mmHg): <=70 (+3), 71-80 (+2), 81-100 (+1), 101-199 (0), >=200 (+2)
- Heart Rate (bpm): <40 (+2), 41-50 (+1), 51-100 (0), 101-110 (+1), 111-129 (+2), >=130 (+3)
- Resp Rate (bpm): <9 (+2), 9-14 (0), 15-20 (+1), 21-29 (+2), >=30 (+3)
- Temperature (°C): <35 (+2), 35.0-38.4 (0), 38.5-38.9 (+1), >=39.0 (+2)
"""
import numpy as np
import pandas as pd


def compute_single_mews(sbp, hr, rr, temp):
    """Computes vital-sign-based MEWS score (0-11) for a single observation."""
    score = 0

    # 1. Systolic Blood Pressure (mmHg)
    if not np.isnan(sbp):
        if sbp <= 70:
            score += 3
        elif sbp <= 80:
            score += 2
        elif sbp <= 100:
            score += 1
        elif sbp >= 200:
            score += 2

    # 2. Heart Rate (bpm)
    if not np.isnan(hr):
        if hr < 40:
            score += 2
        elif hr <= 50:
            score += 1
        elif hr <= 100:
            score += 0
        elif hr <= 110:
            score += 1
        elif hr <= 129:
            score += 2
        elif hr >= 130:
            score += 3

    # 3. Respiratory Rate (breaths/min)
    if not np.isnan(rr):
        if rr < 9:
            score += 2
        elif rr <= 14:
            score += 0
        elif rr <= 20:
            score += 1
        elif rr <= 29:
            score += 2
        elif rr >= 30:
            score += 3

    # 4. Temperature (°C)
    if not np.isnan(temp):
        if temp > 50:
            temp = (temp - 32) * 5.0 / 9.0
        if temp < 35.0:
            score += 2
        elif temp <= 38.4:
            score += 0
        elif temp <= 38.9:
            score += 1
        elif temp >= 39.0:
            score += 2

    return score


class MEWSBaseline:
    """
    Hospital Bedside MEWS Baseline Classifier.
    Computes rule-based early warning scores and maps them to normalized risk probabilities.
    """
    def __init__(self, critical_threshold=4.0):
        self.critical_threshold = critical_threshold

    def score_windows(self, sbp_arr, hr_arr, rr_arr, temp_arr):
        """Vectorized computation of MEWS scores across patient arrays."""
        scores = []
        for sbp, hr, rr, temp in zip(sbp_arr, hr_arr, rr_arr, temp_arr):
            scores.append(compute_single_mews(sbp, hr, rr, temp))
        return np.array(scores, dtype=np.float32)

    def predict_proba(self, sbp_arr, hr_arr, rr_arr, temp_arr):
        """Converts MEWS scores (0-11) into normalized risk probabilities via sigmoid calibration."""
        scores = self.score_windows(sbp_arr, hr_arr, rr_arr, temp_arr)
        # Standard clinical logistic calibration centered at MEWS threshold 4
        probs = 1.0 / (1.0 + np.exp(-(scores - self.critical_threshold)))
        return probs

    def predict(self, sbp_arr, hr_arr, rr_arr, temp_arr, threshold=0.50):
        probs = self.predict_proba(sbp_arr, hr_arr, rr_arr, temp_arr)
        return (probs >= threshold).astype(int)

