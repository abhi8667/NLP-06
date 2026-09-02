"""
LaTeX and Markdown Results Table Exporter for Track B Paper (Phase P8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_results_tables(
    results_json: str | Path = "research_track/results/campaign_results.json",
    output_dir: str | Path = "research_track/results/tables",
) -> dict[str, str]:
    """
    Export Markdown and LaTeX summary tables grouped by architecture and epsilon.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = Path(results_json)
    if not r_path.exists():
        raise FileNotFoundError(f"Campaign results not found at {r_path}")

    with open(r_path) as f:
        data = json.load(f)

    if not data:
        return {}

    df = pd.DataFrame(data)

    def _fmt(m: float, s: float) -> str:
        return f"{m:.3f} ± {s:.3f}" if not pd.isna(s) else f"{m:.3f}"

    # Group by architecture and epsilon
    summary_rows = []
    for (arch, eps), grp in df.groupby(["architecture", "target_epsilon"]):
        summary_rows.append({
            "Architecture": arch,
            "Target ε": "∞ (Non-DP)" if str(eps) in ("inf", "float('inf')") else f"ε = {eps}",
            "AUPRC": _fmt(grp["auprc"].mean(), grp["auprc"].std()),
            "AUROC": _fmt(grp["auroc"].mean(), grp["auroc"].std()),
            "FNR (Missed Det)": _fmt(grp["false_negative_rate"].mean(), grp["false_negative_rate"].std()),
            "F1-Score": _fmt(grp["f1"].mean(), grp["f1"].std()),
            "Seeds": len(grp),
        })

    sum_df = pd.DataFrame(summary_rows)

    # Export Markdown
    md_path = out_dir / "table1_main_results.md"
    md_content = sum_df.to_markdown(index=False)
    with open(md_path, "w") as f:
        f.write("# Table 1: Main Empirical Results — Predictive Utility & Clinical Safety across Privacy Budgets\n\n")
        f.write(md_content)
        f.write("\n")

    # Export LaTeX
    tex_path = out_dir / "table1_main_results.tex"
    tex_content = sum_df.to_latex(index=False, caption="Main Empirical Results across Privacy Budgets", label="tab:main_results")
    with open(tex_path, "w") as f:
        f.write(tex_content)

    return {
        "markdown": str(md_path),
        "latex": str(tex_path),
    }
