"""
P0 · S2c — Does the privacy budget mean what the paper will claim?

Layer three of three, and the trap most likely to invalidate a headline claim.

C4 claims a tight Renyi-DP bound across 100 federated rounds. That claim is only
true if the accountant accumulates across those rounds. The natural way to write
a Flower client - construct a PrivacyEngine inside fit() - starts a FRESH
accountant every round, so the epsilon you print is per-round, not cumulative.
The training still runs. The number is just wrong, and wrong in the safe-looking
direction: it under-reports privacy loss.

This script runs the same federated training three ways and prints the epsilon
each one reports:

    NAIVE   fresh PrivacyEngine per round        -> epsilon flat across rounds
    FIX A   accountant state carried across      -> epsilon grows correctly
    FIX B   sigma sized once for total steps     -> epsilon lands on target

    python scripts/s2c_fed_dp.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.layers import DPLSTM
from torch.utils.data import DataLoader, TensorDataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_WARDS = 3          # keep small - this spike is about accounting, not accuracy
ROUNDS = 6
LOCAL_EPOCHS = 2
BATCH = 32
PER_WARD = 384
WINDOW, N_VITALS = 12, 6
TARGET_EPS = 2.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
SEED = 42


class DPLSTMClassifier(nn.Module):
    def __init__(self, inp: int = N_VITALS, hid: int = 32, layers: int = 1):
        super().__init__()
        self.rnn = DPLSTM(inp, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def make_ward(i: int, n: int = PER_WARD) -> TensorDataset:
    g = torch.Generator().manual_seed(SEED + i)
    x = torch.randn(n, WINDOW, N_VITALS, generator=g)
    y = (torch.rand(n, generator=g) < 0.15).float()
    x[y == 1] += 0.6
    return TensorDataset(x, y)


def fedavg(updates: list[tuple[dict, int]]) -> dict:
    total = sum(n for _, n in updates)
    out: dict = {}
    for key in updates[0][0]:
        ref = updates[0][0][key]
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, n in updates:
            acc += sd[key].to(torch.float32) * (n / total)
        out[key] = acc.to(ref.dtype)
    return out


def _strip(sd: dict) -> dict:
    """Opacus wraps the module as GradSampleModule, prefixing keys with '_module.'."""
    return {k.replace("_module.", "", 1): v.detach().cpu() for k, v in sd.items()}


def local_round(global_sd: dict, data: TensorDataset, sigma: float,
                device: torch.device, accountant_state: dict | None):
    """One ward's local training. Returns (params, n, epsilon, accountant_state)."""
    model = DPLSTMClassifier()
    model.load_state_dict(global_sd)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    dl = DataLoader(data, batch_size=BATCH, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()

    pe = PrivacyEngine()
    # carry prior privacy loss forward when asked to
    if accountant_state is not None:
        pe.accountant.load_state_dict(accountant_state)

    model, opt, dl = pe.make_private(
        module=model, optimizer=opt, data_loader=dl,
        noise_multiplier=sigma, max_grad_norm=MAX_GRAD_NORM,
    )

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()

    return _strip(model.state_dict()), len(data), pe.get_epsilon(DELTA), pe.accountant.state_dict()


def run(mode: str, sigma: float, device: torch.device) -> list[float]:
    """mode: 'naive' resets the accountant each round; 'carry' persists it."""
    torch.manual_seed(SEED)
    wards = [make_ward(i) for i in range(N_WARDS)]
    global_sd = DPLSTMClassifier().state_dict()
    states: list[dict | None] = [None] * N_WARDS
    trace: list[float] = []

    for rnd in range(1, ROUNDS + 1):
        updates, epsilons = [], []
        for w in range(N_WARDS):
            prior = None if mode == "naive" else states[w]
            sd, n, eps, acc_state = local_round(global_sd, wards[w], sigma, device, prior)
            states[w] = acc_state
            updates.append((sd, n))
            epsilons.append(eps)
        global_sd = fedavg(updates)
        worst = max(epsilons)  # report the worst-case ward
        trace.append(worst)
        print(f"    round {rnd:>2}  epsilon = {worst:.4f}")
    return trace


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"{N_WARDS} wards, {ROUNDS} rounds, {LOCAL_EPOCHS} local epochs, "
          f"delta={DELTA}\n")

    steps_per_epoch = -(-PER_WARD // BATCH)
    total_steps = ROUNDS * LOCAL_EPOCHS * steps_per_epoch
    sample_rate = BATCH / PER_WARD

    findings: dict = {
        "device": str(device), "wards": N_WARDS, "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps, "sample_rate": round(sample_rate, 5),
    }

    # ------------------------------------------------------------------
    # Size the noise ONCE for the whole campaign, not per round.
    # ------------------------------------------------------------------
    print("-" * 62)
    print("SIZING NOISE FOR THE FULL RUN")
    print("-" * 62)
    sigma = get_noise_multiplier(
        target_epsilon=TARGET_EPS, target_delta=DELTA,
        sample_rate=sample_rate, steps=total_steps, accountant="rdp",
    )
    print(f"  target epsilon {TARGET_EPS} over {total_steps} total steps")
    print(f"  -> noise multiplier sigma = {sigma:.4f}")

    sigma_1round = get_noise_multiplier(
        target_epsilon=TARGET_EPS, target_delta=DELTA,
        sample_rate=sample_rate, steps=LOCAL_EPOCHS * steps_per_epoch, accountant="rdp",
    )
    print(f"  (sizing for ONE round only would give sigma = {sigma_1round:.4f} "
          f"- {sigma / sigma_1round:.2f}x less noise)")
    findings["sigma_full_run"] = round(float(sigma), 4)
    findings["sigma_one_round"] = round(float(sigma_1round), 4)

    # ------------------------------------------------------------------
    print()
    print("-" * 62)
    print("NAIVE - fresh PrivacyEngine every round (the trap)")
    print("-" * 62)
    naive = run("naive", sigma, device)

    print()
    print("-" * 62)
    print("FIX A - accountant state carried across rounds")
    print("-" * 62)
    carried = run("carry", sigma, device)

    findings["naive_epsilon_trace"] = [round(e, 4) for e in naive]
    findings["carried_epsilon_trace"] = [round(e, 4) for e in carried]

    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    print("RESULT")
    print("=" * 62)
    print(f"  naive   : first round {naive[0]:.4f} -> last round {naive[-1]:.4f}")
    print(f"  carried : first round {carried[0]:.4f} -> last round {carried[-1]:.4f}")
    print()

    naive_flat = abs(naive[-1] - naive[0]) < 1e-6
    carried_grows = carried[-1] > carried[0] + 1e-6
    understated = carried[-1] / max(naive[-1], 1e-12)

    if naive_flat:
        print("  CONFIRMED: the naive pattern reports a FLAT epsilon. Every round")
        print("  looks equally private because the accountant restarts each time.")
    if carried_grows:
        print(f"  CONFIRMED: carrying the accountant grows epsilon to {carried[-1]:.4f}.")
        print(f"  The naive number under-reports privacy loss by {understated:.2f}x")
        print(f"  after only {ROUNDS} rounds. At 100 rounds the gap is far larger.")
    print()
    print("  ACTION for P3B: size sigma once for ROUNDS x LOCAL_EPOCHS x steps,")
    print("  hold it fixed, and persist each client's accountant across rounds via")
    print("  pe.accountant.state_dict() / load_state_dict(). Report the worst-case")
    print("  ward's epsilon, not the mean.")
    print()
    print("  ALSO RECORD: Opacus gives SAMPLE-level DP, and your samples are")
    print("  overlapping windows. One patient contributes many windows, so this")
    print("  epsilon is NOT a per-patient guarantee. State the unit in the paper.")
    print("=" * 62)

    findings["naive_reports_flat_epsilon"] = bool(naive_flat)
    findings["carried_epsilon_grows"] = bool(carried_grows)
    findings["understatement_factor"] = round(float(understated), 3)
    findings["verdict"] = "TRAP CONFIRMED" if (naive_flat and carried_grows) else "INCONCLUSIVE"

    out = RESULTS / "s2c_fed_dp.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
