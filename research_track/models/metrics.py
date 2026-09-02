"""
Clinical Evaluation Metrics & Statistical Summaries for Track B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClinicalEvaluationReport:
    """Standardized results record for an experiment run."""
    auprc: float
    auroc: float
    false_negative_rate: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int
    total_samples: int
    positive_rate: float
    optimal_threshold: float = 0.5
    achieved_epsilon: float | None = None
    delta: float | None = None
    wall_clock_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_model_performance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device = "cpu",
    threshold: float | str = "optimal",
    achieved_epsilon: float | None = None,
    delta: float | None = None,
    wall_clock_s: float | None = None,
) -> ClinicalEvaluationReport:
    """
    Evaluate detector on a DataLoader, computing raw probabilities via sigmoid.
    Supports optimal threshold calibration via Youden's J statistic.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_targets: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_b = x_batch.to(dev)
            raw_logits = model(x_b)
            probs = torch.sigmoid(raw_logits).cpu().numpy()

            all_probs.extend(probs.flatten().tolist())
            all_targets.extend(y_batch.numpy().flatten().tolist())

    y_true = np.array(all_targets, dtype=np.float32)
    y_scores = np.array(all_probs, dtype=np.float32)

    # Calculate headline discrimination & utility
    try:
        auprc = float(average_precision_score(y_true, y_scores))
    except Exception:
        auprc = 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auroc = 0.5

    # Determine decision threshold
    if threshold == "optimal" or threshold == "youden":
        from sklearn.metrics import roc_curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresh_arr = roc_curve(y_true, y_scores)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            chosen_thresh = float(thresh_arr[best_idx])
            chosen_thresh = max(min(chosen_thresh, 0.85), 0.15)
        else:
            chosen_thresh = 0.5
    else:
        chosen_thresh = float(threshold)

    y_pred = (y_scores >= chosen_thresh).astype(np.int32)

    # Confusion matrix & clinical false negative rate
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    pos_count = tp + fn
    fnr = float(fn / pos_count) if pos_count > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return ClinicalEvaluationReport(
        auprc=round(auprc, 4),
        auroc=round(auroc, 4),
        false_negative_rate=round(fnr, 4),
        f1=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total_samples=len(y_true),
        positive_rate=round(float(np.mean(y_true)), 4) if len(y_true) > 0 else 0.0,
        optimal_threshold=round(chosen_thresh, 4),
        achieved_epsilon=round(achieved_epsilon, 4) if achieved_epsilon is not None else None,
        delta=delta,
        wall_clock_s=round(wall_clock_s, 2) if wall_clock_s is not None else None,
    )
