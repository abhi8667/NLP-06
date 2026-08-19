"""
P0 · S4 — Does the experimental grid fit the compute available?

Times one federated DP round at the REAL configuration, extrapolates to the full
grid, and measures peak GPU memory so the Ollama contention question can be
answered with a number instead of a guess.

Cut the grid here, deliberately, rather than in Phase 4 in a panic.

    python scripts/s4_timing.py                    # default: 2 timed rounds
    python scripts/s4_timing.py --rounds 3 --wards 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.layers import DPGRU, DPLSTM
from torch.utils.data import DataLoader, TensorDataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# --- the campaign configuration these timings extrapolate to ---
CAMPAIGN_ROUNDS = 100
CAMPAIGN_LOCAL_EPOCHS = 5
WINDOWS_PER_WARD = 4000        # ~200 patients x ~20 windows; adjust after S1
BATCH = 32
WINDOW, N_VITALS = 12, 6
MAX_GRAD_NORM = 1.0
SIGMA = 1.0

GRID = [
    ("core / clean Synthea", 3, 6, 5),      # models, epsilons, seeds
    ("noise-injected Synthea", 1, 6, 5),
    ("PTB-XL", 1, 6, 5),
]

# Llama 3.1 8B at Q4_K_M needs roughly this much VRAM resident.
OLLAMA_VRAM_GB = 5.0


class DPLSTMClassifier(nn.Module):
    def __init__(self, inp=N_VITALS, hid=64, layers=2):
        super().__init__()
        self.rnn = DPLSTM(inp, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class DPGRUClassifier(nn.Module):
    def __init__(self, inp=N_VITALS, hid=64, layers=2):
        super().__init__()
        self.rnn = DPGRU(inp, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class CNNClassifier(nn.Module):
    def __init__(self, inp=N_VITALS, filters=64):
        super().__init__()
        self.c1 = nn.Conv1d(inp, filters, 3, padding=1)
        self.c2 = nn.Conv1d(filters, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.fc(self.pool(x).squeeze(-1))


MODELS = {"DPLSTM": DPLSTMClassifier, "DPGRU": DPGRUClassifier, "CNN": CNNClassifier}


def make_data(n: int, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, WINDOW, N_VITALS, generator=g)
    y = (torch.rand(n, generator=g) < 0.15).float()
    return TensorDataset(x, y)


def time_local_epochs(model_cls, data, epochs, device) -> tuple[float, float]:
    """Returns (seconds, peak_vram_gb) for `epochs` of DP local training."""
    model = model_cls().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dl = DataLoader(data, batch_size=BATCH, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()

    pe = PrivacyEngine()
    model, opt, dl = pe.make_private(
        module=model, optimizer=opt, data_loader=dl,
        noise_multiplier=SIGMA, max_grad_norm=MAX_GRAD_NORM,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    return elapsed, peak


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h:.1f} h ({h / 24:.1f} days)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2, help="rounds to actually time")
    ap.add_argument("--wards", type=int, default=5)
    ap.add_argument("--windows", type=int, default=WINDOWS_PER_WARD)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    findings: dict = {"device": str(device), "windows_per_ward": args.windows}

    print("=" * 66)
    print("S4 - COMPUTE BUDGET")
    print("=" * 66)
    print(f"  device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024**3
        print(f"  gpu   : {props.name}  ({total_vram:.1f} GB)")
        findings["gpu"] = props.name
        findings["total_vram_gb"] = round(total_vram, 2)
    else:
        total_vram = 0.0
        print("  WARNING: timing on CPU. These numbers will not reflect the campaign.")
    print(f"  timing {args.rounds} round(s) x {args.wards} wards x "
          f"{CAMPAIGN_LOCAL_EPOCHS} local epochs on {args.windows:,} windows/ward\n")

    data = make_data(args.windows)
    per_model: dict = {}

    for name, cls in MODELS.items():
        print(f"  {name} ...", end="", flush=True)
        # one ward's local training, timed
        secs, peak = time_local_epochs(cls, data, CAMPAIGN_LOCAL_EPOCHS, device)
        round_s = secs * args.wards
        run_h = round_s * CAMPAIGN_ROUNDS / 3600
        per_model[name] = {
            "local_train_s": round(secs, 2),
            "round_s": round(round_s, 2),
            "full_run_hours": round(run_h, 3),
            "peak_vram_gb": round(peak, 3),
        }
        print(f"  local {secs:6.2f}s | round {round_s:7.2f}s | "
              f"full {CAMPAIGN_ROUNDS}-round run {fmt_hours(run_h):>16} | "
              f"peak VRAM {peak:.2f} GB")

    findings["per_model"] = per_model

    # ------------------------------------------------------------------
    print()
    print("-" * 66)
    print("GRID ARITHMETIC")
    print("-" * 66)
    slowest = max(per_model.values(), key=lambda d: d["full_run_hours"])["full_run_hours"]
    mean_h = sum(d["full_run_hours"] for d in per_model.values()) / len(per_model)

    total_runs, total_h = 0, 0.0
    print(f"  {'grid':<26} {'models':>7} {'eps':>5} {'seeds':>6} {'runs':>6} {'est. time':>18}")
    for label, m, e, s in GRID:
        runs = m * e * s
        # core grid mixes architectures; validation grids use the slowest single model
        h = runs * (mean_h if m > 1 else slowest)
        total_runs += runs
        total_h += h
        print(f"  {label:<26} {m:>7} {e:>5} {s:>6} {runs:>6} {fmt_hours(h):>18}")
    print(f"  {'TOTAL':<26} {'':>7} {'':>5} {'':>6} {total_runs:>6} {fmt_hours(total_h):>18}")

    findings["total_runs"] = total_runs
    findings["total_hours"] = round(total_h, 2)

    # ------------------------------------------------------------------
    # Which lever actually moves the number?
    # ------------------------------------------------------------------
    print()
    print("-" * 66)
    print("SENSITIVITY - what each cut is worth")
    print("-" * 66)
    base = total_h
    lstm_h = per_model.get("DPLSTM", {}).get("full_run_hours", mean_h)
    cnn_h = per_model.get("CNN", {}).get("full_run_hours", mean_h)

    def grid_hours(models: int, eps: int, seeds: int, val_grids: int,
                   val_model_h: float, scale: float = 1.0) -> float:
        core = models * eps * seeds * (mean_h if models > 1 else lstm_h)
        val = val_grids * eps * seeds * val_model_h
        return (core + val) * scale

    scenarios = [
        ("baseline (as planned)", grid_hours(3, 6, 5, 2, lstm_h)),
        ("drop DPGRU (2 models)", grid_hours(2, 6, 5, 2, lstm_h)),
        ("drop noised Synthea", grid_hours(3, 6, 5, 1, lstm_h)),
        ("50 rounds not 100", grid_hours(3, 6, 5, 2, lstm_h, scale=0.5)),
        ("3 seeds not 5", grid_hours(3, 6, 3, 2, lstm_h)),
        ("4 eps levels not 6", grid_hours(3, 4, 5, 2, lstm_h)),
        ("half the windows/ward", grid_hours(3, 6, 5, 2, lstm_h, scale=0.5)),
        ("drop GRU + noised + 50 rounds", grid_hours(2, 6, 5, 1, lstm_h, scale=0.5)),
    ]
    print(f"  {'scenario':<32} {'total':>16} {'saved':>10}")
    for label, h in scenarios:
        saved = "" if h >= base else f"-{(1 - h / base) * 100:.0f}%"
        print(f"  {label:<32} {fmt_hours(h):>16} {saved:>10}")
    findings["sensitivity"] = {label: round(h, 1) for label, h in scenarios}

    print()
    print("  NOTE: windows/ward is currently an ASSUMPTION "
          f"({args.windows:,}). S1 determines the")
    print("  real figure, and it scales this table linearly. Re-run S4 after S1.")
    print("  Rounds is the other big lever - check whether FedAvg has actually")
    print("  converged by round 50 before paying for 100.")

    # ------------------------------------------------------------------
    print()
    print("-" * 66)
    print("GPU CONTENTION WITH OLLAMA")
    print("-" * 66)
    if device.type == "cuda":
        peak_train = max(d["peak_vram_gb"] for d in per_model.values())
        headroom = total_vram - peak_train
        print(f"  total VRAM              {total_vram:6.2f} GB")
        print(f"  peak DP training        {peak_train:6.2f} GB")
        print(f"  headroom                {headroom:6.2f} GB")
        print(f"  Llama 3.1 8B Q4 needs  ~{OLLAMA_VRAM_GB:6.2f} GB resident")
        fits = headroom >= OLLAMA_VRAM_GB
        findings["ollama_coexists"] = bool(fits)
        if fits:
            print("\n  -> both fit. Still verify under real load before relying on it.")
        else:
            print("\n  -> THEY DO NOT BOTH FIT. Person A's Ollama demo and Person B's")
            print("     campaign cannot share this GPU. Decide in P0:")
            print("       a) separate time windows (campaign overnight, demo by day)")
            print("       b) run Ollama on CPU for development, GPU only for the demo")
            print("       c) second machine for the campaign")
    else:
        findings["ollama_coexists"] = None
        print("  no CUDA device - rerun once the CUDA build of torch is installed")

    # ------------------------------------------------------------------
    print()
    print("=" * 66)
    print("DECISIONS TO RECORD")
    print("=" * 66)
    print(f"  measured total     : {total_runs} runs, {fmt_hours(total_h)} of pure compute")
    print("  available wall-clock: ______ h   <- fill this in")
    print()
    if device.type == "cuda":
        ratio_note = "If total exceeds available, descend the de-scoping ladder:"
        print(f"  {ratio_note}")
        print("    1. drop DPGRU        -> core grid 90 -> 60 runs")
        print("    2. drop noised Synthea -> -30 runs")
        print("    3. drop dropout experiment")
        print("  Never trade away PTB-XL or the seed count.")
    print()
    print("  SEED COUNT IS ALSO A STATISTICS DECISION. Five seeds per group with")
    print("  12 Bonferroni-corrected pairwise tests can never reach significance:")
    print("  the smallest attainable two-sided p at n=5 v 5 is ~0.0079, above the")
    print("  corrected threshold of 0.05/12 ~ 0.0042. Adopt a single omnibus model")
    print("  over architecture x epsilon, or raise seeds and cut comparisons.")
    print("=" * 66)

    out = RESULTS / "s4_timing.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
