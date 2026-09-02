"""
Centralized (Non-Federated, Non-DP) Training Baseline for Track B.
Establishes the empirical utility ceiling (U_max).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from research_track.data import PhysioNetDataset, get_federated_dataloaders
from research_track.models import evaluate_model_performance, get_loss_function, get_model, ClinicalEvaluationReport


def run_centralized_training(
    processed_dir: str | Path = "research_track/data/processed",
    architecture: str = "DPLSTM",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str | None = None,
    save_checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[torch.nn.Module, ClinicalEvaluationReport, list[dict[str, Any]]]:
    """
    Train a centralized baseline detector on combined Site A + Site B data.
    """
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loaders = get_federated_dataloaders(processed_dir, batch_size=batch_size, seed=seed)
    pos_weight = loaders["pos_weight"]

    # Combine Site A and Site B training sets
    w_a_tr, y_a_tr = loaders["site_a_train"].dataset.windows.numpy(), loaders["site_a_train"].dataset.labels.numpy()
    w_b_tr, y_b_tr = loaders["site_b_train"].dataset.windows.numpy(), loaders["site_b_train"].dataset.labels.numpy()
    w_comb = torch.tensor(np_stack := __import__("numpy").vstack([w_a_tr, w_b_tr]), dtype=torch.float32)
    y_comb = torch.tensor(__import__("numpy").concatenate([y_a_tr, y_b_tr]), dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PhysioNetDataset(w_comb.numpy(), y_comb.numpy()),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    test_loader = loaders["combined_test"]

    model = get_model(architecture).to(dev)
    criterion = get_loss_function(pos_weight=pos_weight).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_auprc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on test set
        rep = evaluate_model_performance(model, test_loader, device=dev)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "test_auprc": rep.auprc,
            "test_auroc": rep.auroc,
            "test_fnr": rep.false_negative_rate,
            "test_f1": rep.f1,
        })

        if rep.auprc > best_auprc:
            best_auprc = rep.auprc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    wall_clock = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    final_report = evaluate_model_performance(
        model, test_loader, device=dev, wall_clock_s=wall_clock
    )

    if save_checkpoint_path:
        chk_path = Path(save_checkpoint_path)
        chk_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), chk_path)

    return model, final_report, history
