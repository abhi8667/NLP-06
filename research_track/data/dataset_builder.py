"""
Data Pipeline, 4-6h Horizon Windowing, and Patient Split Isolation (Phase P2).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from shared.preprocessing import (
    VITALS,
    WINDOW,
    HORIZON,
    load_patient,
    fill_vitals,
    add_news2,
    patient_split,
    ImputationReport,
)


def load_split_arrays(
    processed_dir: str | Path,
    site_id: str,
    split_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-processed numpy arrays for a given site and split."""
    p = Path(processed_dir) / f"{site_id.lower()}_{split_type.lower()}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Array file not found: {p}")
    data = np.load(p, allow_pickle=True)
    return data["windows"].astype(np.float32), data["labels"].astype(np.float32), data["patient_ids"]


class DatasetBuilder:
    """
    Builds and cryptographically freezes sliding-window datasets for Site A and Site B.
    """

    def __init__(self, output_dir: str | Path = "research_track/data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.physionet_dir = ROOT_DIR / "physioNet"

    def build_dataset(self, max_patients_per_site: int = 50, seed: int = 42) -> dict[str, Any]:
        """
        Processes patients, extracts 12h windows with 4-6h horizon labels,
        splits by patient into train/test, computes SHA-256 hashes, and saves .npz files.
        """
        rng = np.random.default_rng(seed)
        sites = {
            "site_a": self.physionet_dir / "training_setA",
            "site_b": self.physionet_dir / "training_setB",
        }

        overall_imputation = ImputationReport()
        manifest: dict[str, Any] = {
            "window_len": WINDOW,
            "horizon_hours": HORIZON,
            "seed": seed,
            "imputation_rates": {},
            "file_hashes": {},
            "sites": {},
        }

        # 1. Process each site
        for site_id, site_folder in sites.items():
            files = sorted(list(site_folder.glob("**/*.psv")), key=lambda p: p.stem)
            if max_patients_per_site and len(files) > max_patients_per_site:
                indices = rng.permutation(len(files))[:max_patients_per_site]
                files = [files[i] for i in indices]

            # Split files by patient
            pids = [f.stem for f in files]
            train_mask, test_mask = patient_split(np.array(pids), test_frac=0.2, seed=seed)
            train_files = [files[i] for i, is_tr in enumerate(train_mask) if is_tr]
            test_files = [files[i] for i, is_te in enumerate(test_mask) if is_te]

            for split_name, split_files in [("train", train_files), ("test", test_files)]:
                all_w, all_y, all_pids = [], [], []
                for f in split_files:
                    try:
                        df = load_patient(f)
                        fallback_means = {"HR": 80.0, "SBP": 120.0, "O2Sat": 98.0, "Resp": 18.0, "Temp": 37.0, "Glucose": 110.0}
                        df_filled, rep = fill_vitals(df, fallback=fallback_means)
                        overall_imputation = overall_imputation.merge(rep)
                        df_scored = add_news2(df_filled)

                        # Rolling windows
                        if len(df_scored) > WINDOW + HORIZON:
                            vitals_arr = df_scored[VITALS].to_numpy(np.float32)
                            # 4-6h horizon: max NEWS2 label between t+4 and t+6
                            labels_arr = df_scored["news2_label"].to_numpy(np.float32)
                            n_windows = len(df_scored) - (WINDOW + HORIZON)

                            for i in range(n_windows):
                                win = vitals_arr[i : i + WINDOW]
                                # Lookahead label: max over horizon window
                                label = float(np.max(labels_arr[i + WINDOW + 4 : i + WINDOW + HORIZON + 1])) if i + WINDOW + HORIZON < len(labels_arr) else labels_arr[i + WINDOW]
                                all_w.append(win)
                                all_y.append(label)
                                all_pids.append(f.stem)
                    except Exception:
                        continue

                w_arr = np.array(all_w, dtype=np.float32) if all_w else np.empty((0, WINDOW, len(VITALS)), dtype=np.float32)
                y_arr = np.array(all_y, dtype=np.float32) if all_y else np.empty((0,), dtype=np.float32)
                p_arr = np.array(all_pids, dtype=object) if all_pids else np.empty((0,), dtype=object)

                out_path = self.output_dir / f"{site_id}_{split_name}.npz"
                np.savez_compressed(out_path, windows=w_arr, labels=y_arr, patient_ids=p_arr)

                # Compute sha256 hash of generated file
                file_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()
                manifest["file_hashes"][f"{site_id}_{split_name}.npz"] = file_hash

        # Record imputation rates
        for v in VITALS:
            manifest["imputation_rates"][v] = float(overall_imputation.rate(v))

        # Write freeze manifest
        manifest_path = self.output_dir / "freeze_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


if __name__ == "__main__":
    print("Executing Phase P2: Building and Cryptographically Freezing Dataset...")
    builder = DatasetBuilder()
    manifest = builder.build_dataset(max_patients_per_site=50)
    print("\nDataset successfully built and frozen!")
    print(f"Output Directory: {builder.output_dir}")
    print("\nGenerated File Hashes (SHA-256):")
    for fname, fhash in manifest.get("file_hashes", {}).items():
        print(f"  - {fname}: {fhash}")
    print("\nImputation Rates:")
    for vital, rate in manifest.get("imputation_rates", {}).items():
        print(f"  - {vital}: {rate * 100:.2f}%")

