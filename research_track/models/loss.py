"""
Prevalence-Weighted Loss Function for Clinical Anomaly Detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_loss_function(pos_weight: float | torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """
    Construct BCEWithLogitsLoss with positive class weighting.
    The ~12.6% deterioration prevalence requires pos_weight ≈ (1 - p) / p ≈ 6.94
    so that missed deteriorations (false negatives) are penalized symmetrically.
    """
    if pos_weight is not None:
        if not isinstance(pos_weight, torch.Tensor):
            weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
        else:
            weight_tensor = pos_weight.clone().detach().float()
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    return nn.BCEWithLogitsLoss()
