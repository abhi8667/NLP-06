"""
Flower FedAvg Multi-Site Server & Simulation Harness for Track B.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research_track.data import get_federated_dataloaders
from research_track.federation.client import ClinicalFlowerClient
from research_track.models import evaluate_model_performance, get_model, ClinicalEvaluationReport


def aggregate_weights(client_updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Standard FedAvg federated parameter averaging weighted by sample count."""
    total_samples = sum(num_samples for _, num_samples in client_updates)
    if total_samples == 0:
        return client_updates[0][0]

    num_layers = len(client_updates[0][0])
    aggregated: list[np.ndarray] = []

    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_updates[0][0][layer_idx], dtype=np.float64)
        for weights, num_samples in client_updates:
            weighted_sum += weights[layer_idx].astype(np.float64) * (num_samples / total_samples)
        aggregated.append(weighted_sum.astype(client_updates[0][0][layer_idx].dtype))

    return aggregated


def run_federated_simulation(
    processed_dir: str | Path = "research_track/data/processed",
    architecture: str = "DPLSTM",
    target_epsilon: float | None = None,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    rounds: int = 10,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    save_checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[torch.nn.Module, ClinicalEvaluationReport, list[dict[str, Any]]]:
    """
    Run full federated training across Hospital Site A and Hospital Site B with Opacus DP-SGD.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loaders = get_federated_dataloaders(processed_dir, batch_size=batch_size, seed=seed)
    pos_weight = loaders["pos_weight"]

    # Instantiate Site A and Site B clients
    client_a = ClinicalFlowerClient(
        site_id="site_a",
        train_loader=loaders["site_a_train"],
        test_loader=loaders["site_a_test"],
        architecture=architecture,
        target_epsilon=target_epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        total_rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        pos_weight=pos_weight,
        device=dev,
    )
    client_b = ClinicalFlowerClient(
        site_id="site_b",
        train_loader=loaders["site_b_train"],
        test_loader=loaders["site_b_test"],
        architecture=architecture,
        target_epsilon=target_epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        total_rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        pos_weight=pos_weight,
        device=dev,
    )
    clients = [client_a, client_b]

    # Global Server Model
    global_model = get_model(architecture).to(dev)
    global_weights = [val.cpu().numpy() for _, val in global_model.state_dict().items()]

    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_auprc = -1.0
    best_state = None
    latest_eps = None

    for r in range(1, rounds + 1):
        client_updates: list[tuple[list[np.ndarray], int]] = []
        round_losses: list[float] = []
        eps_values: list[float] = []

        # Train each client
        for client in clients:
            updated_weights, num_samples, metrics = client.fit(copy.deepcopy(global_weights))
            client_updates.append((updated_weights, num_samples))
            round_losses.append(metrics["train_loss"])
            if metrics["achieved_epsilon"] is not None:
                eps_values.append(metrics["achieved_epsilon"])

        # Aggregate parameters via FedAvg
        global_weights = aggregate_weights(client_updates)

        # Update global evaluation model
        state_dict = {}
        for (k, _), v in zip(global_model.state_dict().items(), global_weights):
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
        global_model.load_state_dict(state_dict)

        # Worst-case epsilon across all hospital sites
        latest_eps = max(eps_values) if eps_values else None

        # Evaluate on combined test set
        rep = evaluate_model_performance(
            global_model,
            loaders["combined_test"],
            device=dev,
            achieved_epsilon=latest_eps,
            delta=delta if target_epsilon else None,
        )

        history.append({
            "round": r,
            "mean_train_loss": round(float(np.mean(round_losses)), 4),
            "test_auprc": rep.auprc,
            "test_auroc": rep.auroc,
            "test_fnr": rep.false_negative_rate,
            "test_f1": rep.f1,
            "achieved_epsilon": latest_eps,
        })

        if rep.auprc > best_auprc:
            best_auprc = rep.auprc
            best_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    wall_clock = time.perf_counter() - start_time

    if best_state is not None:
        global_model.load_state_dict(best_state)

    final_report = evaluate_model_performance(
        global_model,
        loaders["combined_test"],
        device=dev,
        achieved_epsilon=latest_eps,
        delta=delta if target_epsilon else None,
        wall_clock_s=wall_clock,
    )

    if save_checkpoint_path:
        chk_path = Path(save_checkpoint_path)
        chk_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), chk_path)

    return global_model, final_report, history
