"""
Flower NumPyClient with Opacus DP-SGD & Cumulative Rényi Privacy Accounting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from research_track.models import get_loss_function, get_model


class ClinicalFlowerClient:
    """
    Simulates or executes a single hospital site's federated training client.
    Supports standard FedAvg and Differentially Private FedAvg with Opacus DP-SGD.
    """

    def __init__(
        self,
        site_id: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        architecture: str = "DPLSTM",
        target_epsilon: float | None = None,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        total_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 1e-3,
        pos_weight: float | None = None,
        device: str | torch.device = "cpu",
    ):
        self.site_id = site_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.architecture = architecture
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.total_rounds = total_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device(device)

        self.model = get_model(architecture).to(self.device)
        self.criterion = get_loss_function(pos_weight=pos_weight).to(self.device)

        # Pre-compute DP noise multiplier sigma once over total planned steps across all rounds
        self.use_dp = target_epsilon is not None and target_epsilon < float("inf")
        self.accountant_state: dict[str, Any] | None = None
        self.sigma: float | None = None

        if self.use_dp:
            dataset_size = len(train_loader.dataset)
            batch_size = train_loader.batch_size or 32
            steps_per_round = len(train_loader) * local_epochs
            total_steps = steps_per_round * total_rounds
            sample_rate = batch_size / max(dataset_size, 1)

            self.sigma = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=total_steps,
                accountant="rdp",
            )

    def get_parameters(self) -> list[np.ndarray]:
        """Extract model weights as a list of numpy arrays for Flower aggregation."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Update model weights from aggregated server parameters."""
        state_dict = {}
        for (k, _), v in zip(self.model.state_dict().items(), parameters):
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[np.ndarray]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """
        Execute local training round with continuous DP accounting.
        """
        # Recreate fresh model instance each round to attach clean Opacus hooks
        model = get_model(self.architecture).to(self.device)
        self.model = model
        self.set_parameters(parameters)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        current_epsilon = None

        if self.use_dp:
            privacy_engine = PrivacyEngine()
            if self.accountant_state is not None:
                privacy_engine.accountant.load_state_dict(self.accountant_state)

            model_priv, opt_priv, dl_priv = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_grad_norm,
            )

            total_loss = 0.0
            n_batches = 0

            for _ in range(self.local_epochs):
                for x_b, y_b in dl_priv:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    opt_priv.zero_grad()
                    logits = model_priv(x_b)
                    loss = self.criterion(logits, y_b)
                    loss.backward()
                    opt_priv.step()
                    total_loss += loss.item()
                    n_batches += 1

            self.accountant_state = privacy_engine.accountant.state_dict()
            current_epsilon = privacy_engine.get_epsilon(delta=self.delta)

            # Strip _module prefix from private module
            unwrapped = model_priv._module if hasattr(model_priv, "_module") else model_priv
            self.model = unwrapped
        else:
            total_loss = 0.0
            n_batches = 0
            for _ in range(self.local_epochs):
                for x_b, y_b in self.train_loader:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    optimizer.zero_grad()
                    logits = model(x_b)
                    loss = self.criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
            self.model = model

        num_samples = len(self.train_loader.dataset)
        metrics = {
            "site_id": self.site_id,
            "train_loss": round(total_loss / max(n_batches, 1), 4),
            "achieved_epsilon": round(current_epsilon, 4) if current_epsilon is not None else None,
        }

        return self.get_parameters(), num_samples, metrics
