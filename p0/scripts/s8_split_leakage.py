"""
P0 · S8 — How much does window-level splitting inflate the results?

With stride-1 sliding windows, consecutive windows share 11 of their 12
timesteps. Splitting those windows randomly puts near-duplicates in both train
and test: the model does not generalise, it recognises. Splitting by PATIENT is
the only honest option, because the deployed system predicts for patients it has
never seen, not for unseen hours of patients it already trained on.

This script trains the same model twice on the same data, changing only the
split, and reports the gap. The gap is the size of the lie.

    python scripts/s8_split_leakage.py --root ../physioNet --patients 600
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

VITALS = ["HR", "SBP", "O2Sat", "Resp", "Temp", "Glucose"]
WINDOW = 12
SEED = 42
THRESHOLD = 5


def score_resp(v):
    if pd.isna(v): return 0
    if v <= 8 or v >= 25: return 3
    if v >= 21: return 2
    if v <= 11: return 1
    return 0


def score_spo2(v):
    if pd.isna(v): return 0
    if v <= 91: return 3
    if v <= 93: return 2
    if v <= 95: return 1
    return 0


def score_sbp(v):
    if pd.isna(v): return 0
    if v <= 90 or v >= 220: return 3
    if v <= 100: return 2
    if v <= 110: return 1
    return 0


def score_hr(v):
    if pd.isna(v): return 0
    if v <= 40 or v >= 131: return 3
    if v <= 50: return 1
    if v <= 90: return 0
    if v <= 110: return 1
    return 2


def score_temp(v):
    if pd.isna(v): return 0
    if v <= 35.0: return 3
    if v <= 36.0: return 1
    if v <= 38.0: return 0
    if v <= 39.0: return 1
    return 2


SCORERS = {"Resp": score_resp, "O2Sat": score_spo2, "SBP": score_sbp,
           "HR": score_hr, "Temp": score_temp}


def build(folder: Path, n_patients: int):
    """Returns X [n,12,6], y [n], pid [n] - patient id per window."""
    files = sorted(folder.glob("*.psv"))
    random.Random(SEED).shuffle(files)
    files = files[:n_patients]

    X, y, pid = [], [], []
    for f in files:
        d = pd.read_csv(f, sep="|")
        if len(d) < WINDOW + 1:
            continue
        filled = d[VITALS].ffill().bfill()
        if filled.isna().any().any():
            continue
        news = sum(filled[c].map(fn) for c, fn in SCORERS.items())
        lab = (news >= THRESHOLD).astype(int).values
        arr = filled.values.astype(np.float32)
        for i in range(len(arr) - WINDOW):
            X.append(arr[i:i + WINDOW])
            y.append(lab[i + WINDOW])
            pid.append(f.stem)
    return np.asarray(X), np.asarray(y, dtype=np.float32), np.asarray(pid)


class Net(nn.Module):
    def __init__(self, inp=6, hid=48):
        super().__init__()
        self.rnn = nn.LSTM(inp, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :])


def train_eval(Xtr, ytr, Xte, yte, device, epochs=6):
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(Xte), 4096):
            xb = torch.from_numpy(Xte[i:i + 4096]).to(device)
            logits.append(model(xb).squeeze(-1).cpu().numpy())
        p = 1 / (1 + np.exp(-np.concatenate(logits)))
    return {
        "auprc": float(average_precision_score(yte, p)),
        "auroc": float(roc_auc_score(yte, p)),
        "positive_rate": float(yte.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../physioNet"))
    ap.add_argument("--patients", type=int, default=600)
    args = ap.parse_args()

    folder = None
    for cand in (args.root / "training_setA" / "training_setA", args.root / "training_setA"):
        if cand.is_dir() and any(cand.glob("*.psv")):
            folder = cand
            break
    if folder is None:
        raise SystemExit(f"no .psv under {args.root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("S8 - WINDOW-LEVEL vs PATIENT-LEVEL SPLIT")
    print("=" * 70)
    print(f"  device: {device}\n  building windows from {args.patients} patients...")

    X, y, pid = build(folder, args.patients)
    patients = np.unique(pid)
    print(f"  {len(X):,} windows from {len(patients):,} patients "
          f"({y.mean():.1%} positive)\n")

    rng = np.random.default_rng(SEED)

    # ---- A: window-level split (the mistake) ----
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    shared = len(set(pid[tr]) & set(pid[te]))
    print("-" * 70)
    print("  A. WINDOW-LEVEL split (random over windows)")
    print("-" * 70)
    print(f"    train {len(tr):,} | test {len(te):,}")
    print(f"    patients appearing in BOTH sets: {shared:,}/{len(patients):,} "
          f"({shared / len(patients):.0%})   <-- the leak")
    res_w = train_eval(X[tr], y[tr], X[te], y[te], device)
    print(f"    AUPRC {res_w['auprc']:.4f}   AUROC {res_w['auroc']:.4f}")

    # ---- B: patient-level split (correct) ----
    pshuf = rng.permutation(patients)
    n_tr = int(0.8 * len(pshuf))
    tr_p, te_p = set(pshuf[:n_tr]), set(pshuf[n_tr:])
    m_tr = np.fromiter((p in tr_p for p in pid), bool, len(pid))
    m_te = np.fromiter((p in te_p for p in pid), bool, len(pid))
    print("\n" + "-" * 70)
    print("  B. PATIENT-LEVEL split (no patient in both sets)")
    print("-" * 70)
    print(f"    train {m_tr.sum():,} windows / {len(tr_p):,} patients")
    print(f"    test  {m_te.sum():,} windows / {len(te_p):,} patients")
    print(f"    patients appearing in BOTH sets: 0   <-- correct")
    res_p = train_eval(X[m_tr], y[m_tr], X[m_te], y[m_te], device)
    print(f"    AUPRC {res_p['auprc']:.4f}   AUROC {res_p['auroc']:.4f}")

    # ---- the gap ----
    infl_ap = (res_w["auprc"] - res_p["auprc"]) / max(res_p["auprc"], 1e-9)
    infl_auc = (res_w["auroc"] - res_p["auroc"]) / max(res_p["auroc"], 1e-9)
    print("\n" + "=" * 70)
    print("  THE GAP")
    print("=" * 70)
    print(f"    AUPRC  window {res_w['auprc']:.4f}  vs  patient {res_p['auprc']:.4f}"
          f"   -> {infl_ap:+.1%}")
    print(f"    AUROC  window {res_w['auroc']:.4f}  vs  patient {res_p['auroc']:.4f}"
          f"   -> {infl_auc:+.1%}")
    print()
    if infl_ap > 0.02:
        print("    Window-level splitting reports a better model than exists.")
        print("    Every epsilon-utility point measured that way would be wrong")
        print("    in the same direction, so the shape of the curve survives but")
        print("    the clinical threshold crossing - the paper's headline claim -")
        print("    would land at the wrong epsilon.")
    else:
        print("    Gap is small on this sample, but patient-level splitting is")
        print("    still the only defensible choice: the deployed system predicts")
        print("    for unseen PATIENTS, and reviewers will check this.")
    print()
    print("    BINDING: split by patient everywhere - train/test, any")
    print("    cross-validation (grouped by patient), and the held-out set.")
    print("=" * 70)

    out = RESULTS / "s8_split_leakage.json"
    out.write_text(json.dumps({
        "windows": int(len(X)), "patients": int(len(patients)),
        "positive_rate": round(float(y.mean()), 4),
        "window_level": res_w, "patient_level": res_p,
        "patients_leaked": int(shared),
        "auprc_inflation": round(float(infl_ap), 4),
        "auroc_inflation": round(float(infl_auc), 4),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
