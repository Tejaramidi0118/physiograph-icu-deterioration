"""
src/train/train_gnn.py

Multi-Horizon Deep Auxiliary Supervision Training Engine for PhysioGraph Multi-Horizon Pro.
Implements:
1. Multi-Horizon Loss: L_total = L(H=8) + 0.3 * L(H=6) + 0.3 * L(H=12)
2. AdamW Optimizer with Cosine Annealing Learning Rate Scheduler
3. Mixed Precision Training (torch.amp.autocast + GradScaler)
4. Gradient clipping (max_norm = 2.0)
5. Blended Multi-Horizon Inference: 0.70 * L8 + 0.15 * L6 + 0.15 * L12
6. Optimal F1 threshold tuning on validation set & early stopping
7. Checkpoint persistence (.pt payload)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.graph_topology import EDGE_INDEX
from ..data.dataset import MultiHorizonPhysioDataset, collate_multi_horizon
from ..models.physiograph_pro import PhysioGraphMultiHorizonPro
from ..evaluation.metrics import compute_all_metrics, optimize_decision_threshold


def train_physiograph_multi_horizon(
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    features_path_train=None,
    labels_path_train=None,
    features_path_val=None,
    labels_path_val=None,
    features_path_test=None,
    labels_path_test=None,
    config="nosepsis",
    undersample_ratio="1:1",
    lookback=24,
    batch_size=256,
    lr=3e-4,
    weight_decay=1e-4,
    epochs=10,
    patience=4,
    progress_every=100,
    checkpoint_dir="./models",
    checkpoint_name=None,
    device=None,
    force_retrain=False
):
    """
    Executes the multi-horizon deep supervision training loop for PhysioGraph.
    Can accept either pre-instantiated Datasets or file paths to parquets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratio_str = str(undersample_ratio).replace(":", "_")
    if checkpoint_name is None:
        checkpoint_name = f"best_physiograph_multi_horizon_{ratio_str}.pt"

    ckpt_path = Path(checkpoint_dir) / checkpoint_name
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING PHYSIOGRAPH MULTI-HORIZON TRAINING: Ratio={undersample_ratio} | Device={device.type.upper()}")
    print("Auxiliary Loss Supervision: Primary L(H=8) + 0.3 * L(H=6) + 0.3 * L(H=12)")
    print("=" * 80)

    # 1. Dataset Initialization
    if train_dataset is None:
        if features_path_train is None or labels_path_train is None:
            raise ValueError("Either train_dataset or (features_path_train, labels_path_train) must be provided.")
        train_dataset = MultiHorizonPhysioDataset(
            features_path=features_path_train,
            labels_path=labels_path_train,
            split_name="train",
            config=config,
            lookback=lookback,
            undersample_ratio=undersample_ratio
        )

    if val_dataset is None and features_path_val is not None and labels_path_val is not None:
        val_dataset = MultiHorizonPhysioDataset(
            features_path=features_path_val,
            labels_path=labels_path_val,
            split_name="val",
            config=config,
            lookback=lookback,
            undersample_ratio=undersample_ratio
        )

    if test_dataset is None and features_path_test is not None and labels_path_test is not None:
        test_dataset = MultiHorizonPhysioDataset(
            features_path=features_path_test,
            labels_path=labels_path_test,
            split_name="test",
            config=config,
            lookback=lookback,
            undersample_ratio=undersample_ratio
        )

    use_cuda = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multi_horizon,
        num_workers=0,
        pin_memory=use_cuda
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_multi_horizon,
            num_workers=0,
            pin_memory=use_cuda
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_multi_horizon,
            num_workers=0,
            pin_memory=use_cuda
        )

    # 2. Model Instantiation
    model = PhysioGraphMultiHorizonPro(
        in_channels=4,
        num_nodes=8,
        gat_hidden_dim=64,
        tcn_hidden_dim=64,
        mlp_hidden_dim=32,
        dropout=0.1,
        edge_index=EDGE_INDEX
    ).to(device)

    # 3. Check for Existing Checkpoint
    if not force_retrain and ckpt_path.exists():
        print(f"✓ Found existing checkpoint at {ckpt_path}! Loading weights...")
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_threshold = float(ckpt.get("best_threshold", 0.50))
        best_val_auprc = float(ckpt.get("best_val_auprc", 0.0))
        print(f"✓ Successfully loaded trained weights! (Best Val AUPRC: {best_val_auprc*100:.2f}%)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        best_val_auprc = -1.0
        best_threshold = 0.50
        patience_cnt = 0
        total_train_batches = len(train_loader)

        # 4. Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            n_samples = 0
            t0 = time.time()

            print(f"\n--- Epoch {epoch:02d}/{epochs} (Train Batches: {total_train_batches:,}) ---", flush=True)

            for batch_i, batch in enumerate(train_loader, 1):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)  # [B, 3] -> (H=8, H=6, H=12)
                h = batch["hour_in_icu"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits_h8, logits_h6, logits_h12, _ = model(x, h)
                    # Multi-Horizon Loss Formulation
                    loss_8 = criterion(logits_h8, y[:, 0])
                    loss_6 = criterion(logits_h6, y[:, 1])
                    loss_12 = criterion(logits_h12, y[:, 2])
                    total_loss = loss_8 + 0.3 * loss_6 + 0.3 * loss_12

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += total_loss.item() * len(y)
                n_samples += len(y)

                if progress_every and (batch_i == 1 or batch_i % progress_every == 0 or batch_i == total_train_batches):
                    speed = batch_i / max(time.time() - t0, 0.1)
                    eta = (total_train_batches - batch_i) / max(speed, 0.1)
                    pct = (batch_i / total_train_batches) * 100
                    print(
                        f"  [Epoch {epoch:02d} Train] Batch {batch_i:,}/{total_train_batches:,} ({pct:4.1f}%) | "
                        f"Loss: {total_loss.item():.4f} | Speed: {speed:4.1f} b/s | ETA: {eta:3.0f}s",
                        flush=True
                    )

            scheduler.step()
            train_loss /= max(n_samples, 1)
            train_time = time.time() - t0

            # 5. Validation Evaluation
            if val_loader is not None:
                model.eval()
                val_y, val_p = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(device, non_blocking=True)
                        h = batch["hour_in_icu"].to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=use_cuda):
                            logits_8, logits_6, logits_12, _ = model(x, h)
                            # Blended multi-horizon temporal prediction
                            blended_logits = 0.70 * logits_8 + 0.15 * logits_6 + 0.15 * logits_12
                        val_y.extend(batch["y"][:, 0].cpu().numpy())
                        val_p.extend(torch.sigmoid(blended_logits).cpu().numpy())

                val_metrics = compute_all_metrics(val_y, val_p, threshold=0.50)
                print(
                    f"Epoch {epoch:02d} Complete | Train Loss: {train_loss:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']*100:.2f}% | Val AUPRC: {val_metrics['auprc']*100:.2f}% | "
                    f"Accuracy: {val_metrics['accuracy']*100:.2f}% | Train Time: {train_time:.1f}s",
                    flush=True
                )

                if val_metrics["auprc"] > best_val_auprc:
                    best_val_auprc = float(val_metrics["auprc"])
                    opt_th, best_f1 = optimize_decision_threshold(val_y, val_p)
                    best_threshold = opt_th

                    payload = {
                        "model_state": model.state_dict(),
                        "best_val_auprc": best_val_auprc,
                        "best_threshold": best_threshold,
                        "epoch": int(epoch),
                        "ratio": str(undersample_ratio)
                    }
                    torch.save(payload, str(ckpt_path))
                    print(
                        f"   ★ Saved new best Multi-Horizon model! Val AUPRC: {best_val_auprc*100:.2f}% | "
                        f"F1: {best_f1*100:.2f}% | Optimal Th: {best_threshold:.4f} -> {ckpt_path}",
                        flush=True
                    )
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        print(f" Early stopping triggered at epoch {epoch}.")
                        break

        # Reload best weights
        if ckpt_path.exists():
            try:
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model_state"])

    # 6. Final Test Evaluation
    test_metrics = None
    test_y, test_p = [], []
    if test_loader is not None:
        print("\n" + "=" * 80)
        print("RUNNING FINAL BALANCED TEST EVALUATION")
        print("=" * 80)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device, non_blocking=True)
                h = batch["hour_in_icu"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits_8, logits_6, logits_12, _ = model(x, h)
                    blended_logits = 0.70 * logits_8 + 0.15 * logits_6 + 0.15 * logits_12
                test_y.extend(batch["y"][:, 0].cpu().numpy())
                test_p.extend(torch.sigmoid(blended_logits).cpu().numpy())

        test_metrics = compute_all_metrics(test_y, test_p, threshold=0.50)
        print(f"Balanced Test AUROC    : {test_metrics['auroc']*100:.2f}%")
        print(f"Balanced Test AUPRC    : {test_metrics['auprc']*100:.2f}%")
        print(f"Balanced Test Accuracy : {test_metrics['accuracy']*100:.2f}%")
        print(f"Balanced Test F1 Score : {test_metrics['f1']*100:.2f}%")
        print(f"Clinical Sensitivity   : {test_metrics['sensitivity']*100:.2f}%")
        print(f"Clinical Specificity   : {test_metrics['specificity']*100:.2f}%")
        print("=" * 80)

    return model, test_y, test_p, test_metrics

