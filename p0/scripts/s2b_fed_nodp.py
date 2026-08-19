"""
P0 · S2b — Does federation work before DP is added?

Layer two of three. A minimal, framework-free FedAvg loop across simulated ward
nodes, with NO differential privacy. The point is to confirm that parameters make
the round trip and weighted aggregation improves a shared model, so that when
s2c adds DP and something breaks, you know the break came from DP.

Deliberately does not use Flower. The question here is federated *semantics*, and
a 40-line loop answers it without betting on a framework API. Flower's own
viability on this machine is probed separately in s2d.

    python scripts/s2b_fed_nodp.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_WARDS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH = 32
PER_WARD = 400
WINDOW, N_VITALS = 12, 6
SEED = 42


class LSTMClassifier(nn.Module):
    """Plain nn.LSTM is fine here - no DP yet. s2c swaps in DPLSTM."""

    def __init__(self, inp: int = N_VITALS, hid: int = 64, layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(inp, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # logits


def make_ward(ward_id: int, n: int = PER_WARD) -> TensorDataset:
    """Non-IID by construction: each ward has a different anomaly rate and shift."""
    g = torch.Generator().manual_seed(SEED + ward_id)
    pos_rate = [0.25, 0.08, 0.18, 0.30, 0.12][ward_id % 5]
    x = torch.randn(n, WINDOW, N_VITALS, generator=g)
    y = (torch.rand(n, generator=g) < pos_rate).float()
    x[y == 1] += 0.6 + 0.1 * ward_id  # ward-specific signal strength
    return TensorDataset(x, y)


def local_train(model: nn.Module, data: TensorDataset, device: torch.device) -> tuple[dict, int]:
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    dl = DataLoader(data, batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, len(data)


def fedavg(updates: list[tuple[dict, int]]) -> dict:
    """Weighted average by local dataset size - the FedAvg rule."""
    total = sum(n for _, n in updates)
    out: dict = {}
    for key in updates[0][0]:
        ref = updates[0][0][key]
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()  # counters etc. - do not average
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, n in updates:
            acc += sd[key].to(torch.float32) * (n / total)
        out[key] = acc.to(ref.dtype)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, data: TensorDataset, device: torch.device) -> tuple[float, float]:
    model = model.to(device).eval()
    dl = DataLoader(data, batch_size=256)
    loss_fn = nn.BCEWithLogitsLoss()
    tot, correct, n, nb = 0.0, 0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze(-1)
        tot += loss_fn(logits, yb).item()
        correct += ((logits > 0).float() == yb).sum().item()
        n += len(yb)
        nb += 1
    return tot / max(nb, 1), correct / max(n, 1)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"{N_WARDS} wards x {PER_WARD} windows, {ROUNDS} rounds, "
          f"{LOCAL_EPOCHS} local epochs, no DP\n")

    wards = [make_ward(i) for i in range(N_WARDS)]
    heldout = make_ward(99, n=500)
    rates = [float(w.tensors[1].mean()) for w in wards]
    print("  ward anomaly rates (non-IID by design):")
    print("   ", "  ".join(f"w{i}={r:.0%}" for i, r in enumerate(rates)))
    print()

    global_model = LSTMClassifier()
    history = []

    loss0, acc0 = evaluate(global_model, heldout, device)
    print(f"  round  0 (init)   loss {loss0:.4f}  acc {acc0:.3f}")

    for rnd in range(1, ROUNDS + 1):
        updates = [local_train(global_model, w, device) for w in wards]
        global_model.load_state_dict(fedavg(updates))
        loss, acc = evaluate(global_model, heldout, device)
        history.append({"round": rnd, "loss": round(loss, 4), "acc": round(acc, 4)})
        print(f"  round {rnd:>2}          loss {loss:.4f}  acc {acc:.3f}")

    improved = history[-1]["loss"] < loss0
    print()
    print("=" * 62)
    if improved:
        print("VERDICT: PASS - parameters round-trip and FedAvg improves the model.")
        print("Federated semantics are sound. Proceed to s2c (add DP).")
    else:
        print("VERDICT: FAIL - loss did not improve. Fix this before adding DP,")
        print("or you will be debugging two problems at once.")
    print("=" * 62)

    out = RESULTS / "s2b_fed_nodp.json"
    out.write_text(json.dumps({
        "device": str(device), "wards": N_WARDS, "rounds": ROUNDS,
        "ward_anomaly_rates": [round(r, 4) for r in rates],
        "init_loss": round(loss0, 4), "history": history,
        "verdict": "PASS" if improved else "FAIL",
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
