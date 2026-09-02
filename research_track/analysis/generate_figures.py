"""
Publication Figure Generator for Track B Research Paper (Phase P7/P8).
Produces high-resolution figures for empirical analysis of DP-SGD across sequence architectures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_all_figures(
    results_json: str | Path = "research_track/results/campaign_results.json",
    output_dir: str | Path = "research_track/results/figures",
) -> list[str]:
    """
    Load campaign results and render publication figures.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = Path(results_json)
    if not r_path.exists():
        raise FileNotFoundError(f"Campaign results not found at {r_path}")

    with open(r_path) as f:
        data = json.load(f)

    if not data:
        print("No results to plot.")
        return []

    df = pd.DataFrame(data)
    df["eps_numeric"] = df["target_epsilon"].apply(lambda x: 16.0 if x == "inf" or x == float("inf") else float(x))

    generated_files: list[str] = []

    # Styling
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    # --- FIGURE 1: AUPRC vs Epsilon ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for arch, grp in df.groupby("architecture"):
        agg = grp.groupby("eps_numeric")["auprc"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("eps_numeric")
        ax.errorbar(
            agg["eps_numeric"],
            agg["mean"],
            yerr=agg["std"].fillna(0),
            label=f"{arch}",
            marker="o",
            capsize=4,
        )

    ax.set_xlabel("Privacy Budget (ε) [log scale]")
    ax.set_ylabel("Headline Utility (AUPRC)")
    ax.set_title("Figure 1: Predictive Utility across Differential Privacy Budgets")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["0.5", "1.0", "2.0", "4.0", "8.0", "∞ (Non-DP)"])
    ax.axhline(0.20, color="gray", linestyle="--", alpha=0.7, label="Baseline Prevalence")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    f1_path = out_dir / "fig1_auprc_vs_epsilon.png"
    fig.savefig(f1_path)
    plt.close(fig)
    generated_files.append(str(f1_path))

    # --- FIGURE 2: False Negative Rate vs Epsilon ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for arch, grp in df.groupby("architecture"):
        agg = grp.groupby("eps_numeric")["false_negative_rate"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("eps_numeric")
        ax.errorbar(
            agg["eps_numeric"],
            agg["mean"],
            yerr=agg["std"].fillna(0),
            label=f"{arch}",
            marker="s",
            capsize=4,
        )

    ax.set_xlabel("Privacy Budget (ε) [log scale]")
    ax.set_ylabel("Clinical Harm Metric: False Negative Rate (FNR)")
    ax.set_title("Figure 2: Missed Deterioration Risk under Privacy Noise")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["0.5", "1.0", "2.0", "4.0", "8.0", "∞ (Non-DP)"])
    ax.axhline(0.35, color="red", linestyle="--", alpha=0.7, label="Clinical Action Boundary (35% FNR)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    f2_path = out_dir / "fig2_fnr_vs_epsilon.png"
    fig.savefig(f2_path)
    plt.close(fig)
    generated_files.append(str(f2_path))

    # --- FIGURE 3: Architecture Resilience (Recurrent vs CNN) ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for arch, grp in df.groupby("architecture"):
        agg = grp.groupby("eps_numeric")["auroc"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("eps_numeric")
        ax.plot(
            agg["eps_numeric"],
            agg["mean"],
            label=f"{arch} AUROC",
            marker="^",
        )

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Discrimination (AUROC)")
    ax.set_title("Figure 3: Sequence vs Convolutional Architecture Resilience")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["0.5", "1.0", "2.0", "4.0", "8.0", "∞"])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    f3_path = out_dir / "fig3_architecture_resilience.png"
    fig.savefig(f3_path)
    plt.close(fig)
    generated_files.append(str(f3_path))

    print(f"Successfully generated {len(generated_files)} figures in {out_dir}")
    return generated_files
