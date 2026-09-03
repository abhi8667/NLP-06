"""
PyTorch Dataset and Federated Dataloaders for Track B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from research_track.data.dataset_builder import DatasetBuilder, load_split_arrays


class PhysioNetDataset(Dataset):
    """
    Encapsulates pre-processed sliding windows for a single site and split.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        patient_ids: np.ndarray | None = None,
    ):
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.patient_ids = patient_ids if patient_ids is not None else np.array([])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def get_federated_dataloaders(
    processed_dir: str | Path = "research_track/data/processed",
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Builds PyTorch DataLoaders for Site A and Site B partitions.
    Automatically calls DatasetBuilder if processed directory is empty.
    """
    p_dir = Path(processed_dir)
    if not (p_dir / "site_a_train.npz").exists():
        builder = DatasetBuilder(output_dir=p_dir)
        builder.build_dataset(max_patients_per_site=50, seed=seed)

    datasets = {}
    total_pos = 0.0
    total_samples = 0

    for site in ["site_a", "site_b"]:
        for split in ["train", "test"]:
            w, y, pids = load_split_arrays(p_dir, site, split)
            ds = PhysioNetDataset(w, y, pids)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=False,
            )
            datasets[f"{site}_{split}"] = loader
            if split == "train":
                total_pos += float(y.sum())
                total_samples += len(y)

    # Combined test loader across Site A and Site B
    w_a_te, y_a_te, p_a_te = load_split_arrays(p_dir, "site_a", "test")
    w_b_te, y_b_te, p_b_te = load_split_arrays(p_dir, "site_b", "test")
    w_comb_te = np.concatenate([w_a_te, w_b_te], axis=0)
    y_comb_te = np.concatenate([y_a_te, y_b_te], axis=0)
    p_comb_te = np.concatenate([p_a_te, p_b_te], axis=0)
    combined_ds = PhysioNetDataset(w_comb_te, y_comb_te, p_comb_te)
    datasets["combined_test"] = DataLoader(
        combined_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    pos_ratio = total_pos / max(total_samples, 1)
    pos_weight = float((1.0 - pos_ratio) / max(pos_ratio, 1e-4))
    datasets["pos_weight"] = pos_weight

    return datasets
