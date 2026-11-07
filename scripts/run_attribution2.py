#!/usr/bin/env python3
"""
run_attribution2.py — corrected faithfulness test + time-scalar check.

Fixes three problems in v1:
  1. Masking set values to the median AND flagged all 24 hours missing, which
     is itself a signal and pushes the input off-distribution. Predicted risk
     ROSE for every ranking including random, so the test measured the
     perturbation, not the model's reliance. Replaced with in-distribution
     permutation: a node's 24-hour trace is swapped with the same node's trace
     from a randomly chosen other window in the cohort. Marginal distribution
     of that node is preserved exactly; only its link to this patient breaks.
  2. The drop ratio divided by max(random_drop, 1e-6) and blew up when the
     random drop was negative. Now reports signed absolute drops only.
  3. grad*input is reported but not used for ranking: it scales with |x|, so
     the 94%-imputed temperature node accumulates attribution without
     carrying information. Plain gradient (the quantity OB3 specifies) ranks.

Also new: gradient w.r.t. the ICU elapsed-time scalar, to test whether the
non-physiological input carries a large share of the model's sensitivity.

    python run_attribution2.py --ckpt best_physiograph_multi_horizon_1_1.pt \
        --data data/processed --labels data/processed/labels_v2 --cases 256
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from run_attribution import (PhysioGraphMultiHorizonPro, build_windows,
                             NODE_FEATURES, NODE_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cases", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3,
                    help="permutation repeats per k (averaged)")
    args = ap.parse_args()

    data, lab = Path(args.data), Path(args.labels)
    prep = json.loads((data / "preprocessing_params.json").read_text())
    torch.manual_seed(0)

    print("Building test windows ...")
    dense, wins = build_windows(data / "test_filled.parquet",
                                lab / "test_labels_v2.parquet", prep)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    model = PhysioGraphMultiHorizonPro()
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint (epoch {ck.get('epoch','?')})")

    cand = [i for i, w in enumerate(wins) if w[2] == 1][:args.cases]

    def get_x(idxs):
        xb = torch.stack([torch.from_numpy(dense[wins[i][0]:wins[i][1] + 1]).contiguous()
                          for i in idxs])
        hb = torch.tensor([wins[i][6] for i in idxs], dtype=torch.float32)
        return xb, hb

    def score(x, h):
        l8, l6, l12, a = model(x, h)
        return 0.70 * l8 + 0.15 * l6 + 0.15 * l12, a

    # ---------------- attribution, including the time scalar ----------------
    print(f"\nAttributing {len(cand)} positive windows ...")
    S, A, P, XB, HB, GT = [], [], [], [], [], []
    for s in range(0, len(cand), args.batch):
        xb, hb = get_x(cand[s:s + args.batch])
        xb = xb.clone().float().requires_grad_(True)
        hb = hb.clone().float().requires_grad_(True)
        sc, a = score(xb, hb)
        gx, gh = torch.autograd.grad(sc.sum(), [xb, hb])
        S.append(gx.abs().sum(dim=(1, 3)).detach())       # [B,8] node sensitivity
        GT.append(gh.abs().detach())                      # [B]   time sensitivity
        A.append(a.detach()); P.append(torch.sigmoid(sc).detach())
        XB.append(xb.detach()); HB.append(hb.detach())
        print(f"  {min(s+args.batch, len(cand))}/{len(cand)}", end="\r", flush=True)
    S, A, P = torch.cat(S), torch.cat(A), torch.cat(P)
    XB, HB, GT = torch.cat(XB), torch.cat(HB), torch.cat(GT)

    # ---------------- is the time scalar doing the work? --------------------
    # Compare like with like: sensitivity per input unit, node channel-0 vs hour.
    node_unit = S.sum(dim=1) / (8 * 24 * 4)
    print("\n\n" + "=" * 70)
    print("ICU ELAPSED-TIME SCALAR vs PHYSIOLOGICAL NODES")
    print("=" * 70)
    print(f"mean |d score / d hour_in_icu|      : {GT.mean():.6f}")
    print(f"mean |d score / d x| per input elem : {node_unit.mean():.6f}")
    print(f"mean |d score / d x| summed (all)   : {S.sum(1).mean():.6f}")
    print(f"ratio  time : per-element node      : {GT.mean()/max(node_unit.mean(),1e-12):.1f}x")
    print("A large ratio means a non-physiological input drives much of the")
    print("sensitivity, which would also explain a null graph ablation.")

    # ---------------- in-distribution permutation faithfulness --------------
    rk_attr = torch.argsort(S, dim=1, descending=True)
    rk_attn = torch.argsort(A, dim=1, descending=True)
    rng = np.random.default_rng(0)
    rk_rand = torch.tensor(np.array([rng.permutation(8) for _ in range(len(S))]))

    def permuted_risk(rank, k, seed):
        g = np.random.default_rng(seed)
        vals = []
        with torch.no_grad():
            for s in range(0, len(S), args.batch):
                xb = XB[s:s + args.batch].clone()
                hb = HB[s:s + args.batch]
                B = xb.shape[0]
                donor = g.integers(0, len(XB), size=B)
                for b in range(B):
                    for nnode in rank[s + b, :k].tolist():
                        xb[b, :, nnode, :] = XB[donor[b], :, nnode, :]
                sc, _ = score(xb, hb)
                vals.append(torch.sigmoid(sc))
        return float(torch.cat(vals).mean())

    print("\n" + "=" * 70)
    print("FAITHFULNESS — in-distribution permutation of the top-k nodes")
    print("=" * 70)
    print("A node's 24-hour trace is replaced by the SAME node's trace from a")
    print("random other window. Distribution preserved; patient link broken.")
    print(f"\n{'k':>3}{'gradient':>12}{'attention':>12}{'random':>10}")
    curves = {"attr": [], "attn": [], "rand": []}
    for k in range(5):
        row = []
        for key, rk in (("attr", rk_attr), ("attn", rk_attn), ("rand", rk_rand)):
            v = float(np.mean([permuted_risk(rk, k, 100 + r) for r in range(args.repeats)]))
            curves[key].append(v); row.append(v)
        print(f"{k:>3}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>10.4f}")

    base = curves["attr"][0]
    print(f"\nRisk change from baseline after permuting top-3 nodes:")
    for key, lbl in (("attr", "gradient "), ("attn", "attention"), ("rand", "random   ")):
        print(f"  {lbl}: {curves[key][3] - base:+.4f}")
    print("\nA LARGER DROP for gradient than for random means the attribution")
    print("identifies nodes the model actually relies on. Similar magnitudes")
    print("mean the ranking carries no information about reliance.")

    out = Path(args.data) / "attribution_results_v2.npz"
    np.savez(out, sens=S.numpy(), attn=A.numpy(), prob=P.numpy(),
             grad_time=GT.numpy(),
             curve_attr=curves["attr"], curve_attn=curves["attn"],
             curve_rand=curves["rand"], node_names=np.array(NODE_NAMES))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()