"""
P0 · S2a — Does a recurrent model train under DP-SGD at all?

Layer one of three. No federation yet: one node, one model, Opacus. If this does
not work, nothing above it can.

Two traps this script exists to demonstrate:

  TRAP 1  nn.LSTM / nn.GRU are NOT supported by Opacus. Per-sample gradients are
          undefined for the fused cuDNN kernels. Opacus ships DPLSTM / DPGRU as
          drop-in replacements, and ModuleValidator.fix() will NOT substitute
          them for you. The model classes in the NLP-06 Research Guide fail here
          as written.

  TRAP 2  The guide's models end with torch.sigmoid(...), while the data plan
          specifies BCEWithLogitsLoss with pos_weight. That loss applies sigmoid
          internally - feeding it already-sigmoided output squashes predictions
          into roughly [0.5, 0.73] and gradients nearly vanish. Return raw logits.

    python scripts/s2a_dp_single_node.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_SAMPLES = 2048
WINDOW, N_VITALS = 12, 6
BATCH = 32
EPOCHS = 5
TARGET_EPS = 2.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
POS_RATE = 0.15  # matches the ~10-15% anomaly rate the guides expect


class DPLSTMClassifier(nn.Module):
    """The corrected model: DPLSTM, and raw logits out."""

    def __init__(self, inp: int = N_VITALS, hid: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = DPLSTM(inp, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # logits - BCEWithLogitsLoss applies sigmoid


def synthetic_batch(n: int = N_SAMPLES) -> TensorDataset:
    """Shaped like the real thing; content is irrelevant to this spike."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(n, WINDOW, N_VITALS, generator=g)
    y = (torch.rand(n, generator=g) < POS_RATE).float()
    # give the label a faint signal so loss can actually move
    x[y == 1] += 0.6
    return TensorDataset(x, y)


def main() -> None:
    findings: dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    findings["device"] = str(device)

    # ------------------------------------------------------------------
    # TRAP 1 - demonstrate that the guide's nn.LSTM is rejected
    # ------------------------------------------------------------------
    print("-" * 62)
    print("TRAP 1 - is nn.LSTM usable under Opacus?")
    print("-" * 62)
    plain = nn.LSTM(N_VITALS, 64, num_layers=2, batch_first=True)
    errs = ModuleValidator.validate(plain, strict=False)
    print(f"  ModuleValidator on nn.LSTM -> {len(errs)} incompatibility(ies)")
    for e in errs[:3]:
        print(f"    {type(e).__name__}: {e}")
    findings["nn_lstm_incompatibilities"] = len(errs)

    dp_model = DPLSTMClassifier()
    errs_dp = ModuleValidator.validate(dp_model, strict=False)
    print(f"  ModuleValidator on DPLSTM  -> {len(errs_dp)} incompatibility(ies)")
    findings["dplstm_incompatibilities"] = len(errs_dp)
    if errs_dp:
        for e in errs_dp[:3]:
            print(f"    {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # TRAP 2 - show the double-sigmoid gradient collapse
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("TRAP 2 - sigmoid output fed to BCEWithLogitsLoss")
    print("-" * 62)
    logits = torch.randn(256, 1, requires_grad=True)
    targets = (torch.rand(256, 1) < POS_RATE).float()
    crit = nn.BCEWithLogitsLoss()

    correct = crit(logits, targets)
    correct.backward()
    grad_ok = logits.grad.abs().mean().item()

    logits2 = logits.detach().clone().requires_grad_(True)
    wrong = crit(torch.sigmoid(logits2), targets)  # the bug
    wrong.backward()
    grad_bug = logits2.grad.abs().mean().item()

    print(f"  mean |grad| with raw logits (correct) : {grad_ok:.6f}")
    print(f"  mean |grad| with sigmoid applied (bug): {grad_bug:.6f}")
    print(f"  gradient shrinks by {grad_ok / max(grad_bug, 1e-12):.1f}x")
    print("  >> training looks flat for no obvious reason. Return logits.")
    findings["grad_correct"] = round(grad_ok, 6)
    findings["grad_double_sigmoid"] = round(grad_bug, 6)
    findings["grad_shrink_factor"] = round(grad_ok / max(grad_bug, 1e-12), 2)

    # ------------------------------------------------------------------
    # The actual spike: DP training end to end
    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("DP-SGD training on one node")
    print("-" * 62)
    ds = synthetic_batch()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = DPLSTMClassifier().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    pe = PrivacyEngine()
    model, opt, dl = pe.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=dl,
        target_epsilon=TARGET_EPS,
        target_delta=DELTA,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
    )
    sigma = opt.noise_multiplier
    print(f"  target eps {TARGET_EPS} over {EPOCHS} epochs -> noise multiplier {sigma:.4f}")
    findings["noise_multiplier"] = round(float(sigma), 4)

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"    epoch {ep + 1}/{EPOCHS}  loss {tot / max(nb, 1):.4f}"
              f"  eps-so-far {pe.get_epsilon(DELTA):.3f}")
    elapsed = time.perf_counter() - t0

    eps = pe.get_epsilon(DELTA)
    print(f"\n  achieved epsilon: {eps:.4f} (delta={DELTA})")
    print(f"  wall-clock: {elapsed:.1f}s for {EPOCHS} epochs on {N_SAMPLES} samples")
    findings["achieved_epsilon"] = round(float(eps), 4)
    findings["train_seconds"] = round(elapsed, 2)
    findings["verdict"] = "PASS" if eps <= TARGET_EPS * 1.05 else "CHECK"

    print()
    print("=" * 62)
    print(f"VERDICT: {findings['verdict']} - DP-SGD runs on a recurrent model.")
    print("Feed the wall-clock into S4. Next: s2b (federation, no DP).")
    print("=" * 62)

    out = RESULTS / "s2a_dp_single_node.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
