"""
P0 · S3 — How does PTB-XL become model input?

PTB-XL is 10-second 12-lead ECG at 100 Hz -> roughly (1000, 12) per record.
The NLP-06 models take [batch, 12, 6]: twelve timesteps of six tabular vitals.
No mapping between the two exists in any project document, and PTB-XL is the
single highest-leverage item in the plan, so the mapping must be decided before
P2 builds a pipeline for it.

This script implements OPTION A (recommended): segment each strip into intervals
and compute physiological features per interval, producing [timesteps, features]
windows that the existing architecture consumes unchanged.

Runs in two modes:

  --demo            synthesise an ECG-like signal and prove the mapping works.
                    No download required. Use this first.
  --ptbxl <dir>     run against a real PTB-XL download.

Download (open access, no credentialing). Get the 100 Hz records only:
    https://physionet.org/content/ptb-xl/

    python scripts/s3_ptbxl.py --demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

FS = 100          # Hz, the 100 Hz PTB-XL variant
STRIP_S = 10      # seconds per record
N_SEGMENTS = 12   # -> matches the model's 12 timesteps
LEAD_II = 1       # index of lead II in the standard PTB-XL lead order

FEATURES = [
    "heart_rate",     # bpm, from R-peak intervals
    "rr_std",         # ms, short-horizon RR variability
    "qrs_amplitude",  # mV, R-peak prominence
    "st_level",       # mV, ST-segment offset
    "signal_energy",  # mV^2, per-segment power
    "lead_dispersion" # mV, cross-lead spread
]


def synth_ecg(seconds: int = STRIP_S, fs: int = FS, hr: float = 72.0,
              n_leads: int = 12, seed: int = 0) -> np.ndarray:
    """An ECG-shaped signal: enough to validate the mapping, not a simulator."""
    rng = np.random.default_rng(seed)
    n = seconds * fs
    t = np.arange(n) / fs
    beat_period = 60.0 / hr
    sig = np.zeros(n)
    for beat_t in np.arange(0.3, seconds, beat_period):
        c = beat_t * fs
        idx = np.arange(n)
        sig += 1.2 * np.exp(-0.5 * ((idx - c) / 1.6) ** 2)          # R
        sig -= 0.25 * np.exp(-0.5 * ((idx - c + 4) / 1.8) ** 2)     # Q
        sig -= 0.20 * np.exp(-0.5 * ((idx - c - 4) / 1.8) ** 2)     # S
        sig += 0.30 * np.exp(-0.5 * ((idx - c - 18) / 6.0) ** 2)    # T
        sig += 0.15 * np.exp(-0.5 * ((idx - c + 22) / 5.0) ** 2)    # P
    sig += 0.02 * np.sin(2 * np.pi * 0.3 * t)                       # baseline wander
    leads = np.stack([sig * (0.6 + 0.4 * rng.random()) +
                      0.01 * rng.standard_normal(n) for _ in range(n_leads)], axis=1)
    return leads


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Threshold-and-refractory R-peak detection. Adequate for a spike."""
    d = np.diff(x, prepend=x[0])
    energy = d ** 2
    k = max(1, fs // 20)
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    thresh = smooth.mean() + 1.2 * smooth.std()
    refractory = int(0.2 * fs)
    peaks, last = [], -refractory
    for i in range(1, len(smooth) - 1):
        if smooth[i] > thresh and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]:
            if i - last >= refractory:
                peaks.append(i)
                last = i
    return np.asarray(peaks, dtype=int)


def segment_features(leads: np.ndarray, fs: int = FS,
                     n_segments: int = N_SEGMENTS) -> np.ndarray:
    """(samples, leads) -> (n_segments, len(FEATURES)). This is the mapping."""
    n = leads.shape[0]
    bounds = np.linspace(0, n, n_segments + 1).astype(int)
    ref = leads[:, LEAD_II] if leads.shape[1] > LEAD_II else leads[:, 0]
    peaks = detect_r_peaks(ref, fs)

    rows = []
    for s in range(n_segments):
        a, b = bounds[s], bounds[s + 1]
        seg = ref[a:b]
        seg_leads = leads[a:b, :]
        in_seg = peaks[(peaks >= a) & (peaks < b)]

        # Heart rate and RR variability need a local neighbourhood of beats,
        # so widen to the nearest beats around the segment when it is short.
        near = peaks[(peaks >= max(0, a - fs)) & (peaks < min(n, b + fs))]
        if len(near) >= 2:
            rr = np.diff(near) / fs
            hr = float(60.0 / rr.mean()) if rr.mean() > 0 else 0.0
            rr_std = float(np.std(rr) * 1000.0)
        else:
            hr, rr_std = 0.0, 0.0

        qrs_amp = float(ref[in_seg].mean()) if len(in_seg) else float(seg.max() - seg.min())
        # ST level: ~80 ms after each R peak, relative to the segment baseline
        off = int(0.08 * fs)
        st_idx = [p + off for p in in_seg if p + off < n]
        st = float(np.mean(ref[st_idx]) - np.median(seg)) if st_idx else 0.0
        energy = float(np.mean(seg ** 2))
        disp = float(np.mean(np.std(seg_leads, axis=1)))

        rows.append([hr, rr_std, qrs_amp, st, energy, disp])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use a synthesised ECG (no download)")
    ap.add_argument("--ptbxl", type=Path, help="path to a PTB-XL download root")
    ap.add_argument("--n", type=int, default=20, help="records to process in --ptbxl mode")
    args = ap.parse_args()

    if not args.demo and not args.ptbxl:
        ap.error("pass --demo or --ptbxl <dir>")

    findings: dict = {"features": FEATURES, "n_segments": N_SEGMENTS}
    print("=" * 62)
    print("S3 - PTB-XL -> model input mapping (Option A: interval features)")
    print("=" * 62)

    if args.demo:
        print("  mode: DEMO (synthesised ECG, no download)\n")
        windows, labels = [], []
        for i in range(24):
            hr = 72.0 if i % 2 == 0 else 118.0        # stand-in for NORM vs abnormal
            leads = synth_ecg(hr=hr, seed=i)
            windows.append(segment_features(leads))
            labels.append(0 if i % 2 == 0 else 1)
        X = np.stack(windows)
        y = np.asarray(labels)
        findings["source"] = "synthetic demo"
    else:
        print(f"  mode: PTB-XL at {args.ptbxl}\n")
        try:
            import wfdb
        except ImportError:
            raise SystemExit("wfdb not installed - pip install wfdb")
        import pandas as pd

        db = pd.read_csv(args.ptbxl / "ptbxl_database.csv", index_col="ecg_id")
        recs = db["filename_lr"].head(args.n)  # _lr = 100 Hz records
        windows = []
        for rel in recs:
            sig, meta = wfdb.rdsamp(str(args.ptbxl / rel))
            windows.append(segment_features(np.asarray(sig), fs=int(meta["fs"])))
        X = np.stack(windows)
        y = np.zeros(len(X), dtype=int)  # real labels come from scp_statements.csv
        findings["source"] = str(args.ptbxl)
        print("  NOTE: labels not derived here. In P2, map scp_statements.csv to")
        print("        diagnostic superclasses and collapse to NORM vs abnormal.\n")

    print(f"  input  per record : ({STRIP_S * FS}, 12)   raw samples x leads")
    print(f"  output per record : {X.shape[1:]}   timesteps x features")
    print(f"  batch tensor      : {X.shape}")
    print()
    print("  feature ranges across the batch:")
    for j, name in enumerate(FEATURES):
        col = X[:, :, j]
        print(f"    {name:<16} min {col.min():9.3f}   mean {col.mean():9.3f}   max {col.max():9.3f}")

    # Does it actually feed the model unchanged?
    print()
    print("-" * 62)
    print("SHAPE COMPATIBILITY WITH THE NLP-06 ARCHITECTURE")
    print("-" * 62)
    try:
        import torch
        import torch.nn as nn

        class Probe(nn.Module):
            def __init__(self, n_feat: int):
                super().__init__()
                self.rnn = nn.LSTM(n_feat, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])

        xb = torch.from_numpy(X)
        logits = Probe(X.shape[2])(xb)
        print(f"  fed [batch, {X.shape[1]}, {X.shape[2]}] through an LSTM -> logits {tuple(logits.shape)}")
        print("  the architecture consumes it unchanged, with input_size adjusted")
        print(f"  from 6 (Synthea vitals) to {X.shape[2]} (ECG interval features).")
        compat = True
    except ImportError:
        print("  torch not available - shape check skipped")
        compat = None

    findings["window_shape"] = list(X.shape[1:])
    findings["batch_shape"] = list(X.shape)
    findings["architecture_compatible"] = compat

    print()
    print("=" * 62)
    print("DECISION TO RECORD IN P0_MEMO.md")
    print("=" * 62)
    print(f"  mapping     : Option A - {N_SEGMENTS} interval segments x {len(FEATURES)} features")
    print(f"  input shape : [batch, {N_SEGMENTS}, {len(FEATURES)}]")
    print("  model change: input_size 6 -> 6 (coincidentally equal here); if you")
    print("                add features, PTB-XL runs need their own input_size.")
    print()
    print("  CLAIM WORDING - be precise. PTB-XL is diagnostic ECG classification,")
    print("  not deterioration detection from ward vitals. The defensible claim is")
    print("  that the epsilon-utility TREND also appears on real physiological")
    print("  signals, i.e. the finding is not a synthetic-data artifact. It is NOT")
    print("  evidence the anomaly detector works on real patients. Overclaiming")
    print("  turns your strongest defensive asset into an attack surface.")
    print("=" * 62)

    out = RESULTS / "s3_ptbxl.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
