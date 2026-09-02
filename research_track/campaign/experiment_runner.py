"""
Automated, Checkpointed Experiment Campaign Runner (Phase P4 - Person B).
Sweeps epsilon levels × architectures × seeds, logging all metrics to JSON/CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from research_track.federation import run_centralized_training, run_federated_simulation


class ExperimentCampaignRunner:
    """
    Coordinates and executes the multi-architecture, multi-epsilon research campaign.
    """

    def __init__(
        self,
        processed_dir: str | Path = "research_track/data/processed",
        results_dir: str | Path = "research_track/results",
        architectures: list[str] | None = None,
        epsilons: list[float] | None = None,
        seeds: list[int] | None = None,
        rounds: int = 10,
        local_epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.architectures = architectures or ["DPLSTM", "DPGRU", "CNN1D"]
        self.epsilons = epsilons if epsilons is not None else [float("inf"), 8.0, 4.0, 2.0, 1.0, 0.5]
        self.seeds = seeds or [42, 43, 44, 45, 46]
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset hash for provenance
        self.dataset_hash = "unknown"
        manifest_path = self.processed_dir / "freeze_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)
                self.dataset_hash = manifest_data.get("file_hashes", {}).get("site_a_train.npz", "valid_freeze")

    def _load_existing_results(self) -> list[dict[str, Any]]:
        json_path = self.results_dir / "campaign_results.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        json_path = self.results_dir / "campaign_results.json"
        csv_path = self.results_dir / "campaign_results.csv"

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    def run_campaign(
        self,
        pilot_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Execute full grid sweep. In pilot_mode, runs 1 seed over a minimal subset.
        """
        arch_list = self.architectures if not pilot_mode else ["DPLSTM"]
        eps_list = self.epsilons if not pilot_mode else [float("inf"), 2.0]
        seed_list = self.seeds if not pilot_mode else [42]
        rounds_count = self.rounds if not pilot_mode else 3

        existing_results = self._load_existing_results()
        completed_keys = {
            f"{r['architecture']}_eps_{r['target_epsilon']}_seed_{r['seed']}"
            for r in existing_results
        }

        total_runs = len(arch_list) * len(eps_list) * len(seed_list)
        run_idx = 0

        for arch in arch_list:
            for eps in eps_list:
                for seed in seed_list:
                    run_idx += 1
                    cell_key = f"{arch}_eps_{eps}_seed_{seed}"

                    if cell_key in completed_keys:
                        print(f"[{run_idx}/{total_runs}] Skipping completed: {cell_key}")
                        continue

                    print(f"[{run_idx}/{total_runs}] Running {cell_key} (Arch: {arch}, Target Eps: {eps}, Seed: {seed})...")

                    eps_arg = None if eps == float("inf") else float(eps)
                    chk_path = self.results_dir / "checkpoints" / f"{cell_key}.pt"

                    if eps_arg is None:
                        # Non-DP federated run
                        _, rep, history = run_federated_simulation(
                            processed_dir=self.processed_dir,
                            architecture=arch,
                            target_epsilon=None,
                            rounds=rounds_count,
                            local_epochs=self.local_epochs,
                            batch_size=self.batch_size,
                            lr=self.lr,
                            device=self.device,
                            save_checkpoint_path=chk_path,
                            seed=seed,
                        )
                    else:
                        # DP-SGD federated run
                        _, rep, history = run_federated_simulation(
                            processed_dir=self.processed_dir,
                            architecture=arch,
                            target_epsilon=eps_arg,
                            rounds=rounds_count,
                            local_epochs=self.local_epochs,
                            batch_size=self.batch_size,
                            lr=self.lr,
                            device=self.device,
                            save_checkpoint_path=chk_path,
                            seed=seed,
                        )

                    row: dict[str, Any] = {
                        "cell_key": cell_key,
                        "architecture": arch,
                        "target_epsilon": "inf" if eps == float("inf") else eps,
                        "achieved_epsilon": rep.achieved_epsilon if rep.achieved_epsilon is not None else "inf",
                        "seed": seed,
                        "auprc": rep.auprc,
                        "auroc": rep.auroc,
                        "false_negative_rate": rep.false_negative_rate,
                        "f1": rep.f1,
                        "precision": rep.precision,
                        "recall": rep.recall,
                        "wall_clock_s": rep.wall_clock_s,
                        "dataset_sha256": self.dataset_hash,
                        "history": history,
                    }

                    existing_results.append(row)
                    self._save_results(existing_results)

        return existing_results
