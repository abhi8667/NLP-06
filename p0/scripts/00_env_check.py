"""
P0 · Step 0 — Environment record.

Captures the exact environment the spikes run in. Every number S1-S4 produce is
only meaningful alongside this record, so run it first and keep the output.

    python scripts/00_env_check.py
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def _probe(mod: str) -> str | None:
    try:
        m = __import__(mod)
    except Exception:
        return None
    return getattr(m, "__version__", "unknown")


def _tool(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    return (out.stdout + out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "present"


def main() -> None:
    rec: dict = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {m: _probe(m) for m in
                     ("torch", "opacus", "flwr", "numpy", "pandas",
                      "sklearn", "scipy", "wfdb", "neurokit2")},
        "tools": {
            "java": _tool(["java", "-version"]),
            "git": _tool(["git", "--version"]),
            "ollama": _tool(["ollama", "--version"]),
        },
    }

    # --- GPU: the number that sizes S4 ---
    gpu: dict = {"available": False}
    try:
        import torch

        gpu["torch_cuda_build"] = torch.version.cuda
        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            props = torch.cuda.get_device_properties(0)
            gpu["name"] = props.name
            gpu["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
            gpu["capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        gpu["error"] = repr(exc)
    rec["gpu"] = gpu

    out_path = RESULTS / "00_env.json"
    out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # --- human-readable ---
    print("=" * 62)
    print("NLP-06 - P0 environment record")
    print("=" * 62)
    print(f"  captured   : {rec['captured_utc']}")
    print(f"  platform   : {rec['platform']}")
    print(f"  python     : {rec['python']}")
    print()
    print("  packages")
    for name, ver in rec["packages"].items():
        print(f"    {name:<12} {ver or '-- NOT INSTALLED --'}")
    print()
    print("  external tools")
    for name, ver in rec["tools"].items():
        print(f"    {name:<12} {ver or '-- NOT FOUND --'}")
    print()
    print("  gpu")
    if gpu.get("available"):
        print(f"    {gpu['name']}  ({gpu['total_vram_gb']} GB, sm_{gpu['capability'].replace('.', '')})")
        print(f"    torch CUDA build: {gpu.get('torch_cuda_build')}")
    else:
        print("    no CUDA device visible to torch  <-- S4 timings will be CPU-bound")
    print()

    # --- blockers ---
    blockers = []
    if rec["tools"]["java"] is None:
        blockers.append("java missing -> S1 cannot run (Synthea needs a JDK)")
    for pkg in ("torch", "opacus", "flwr"):
        if rec["packages"][pkg] is None:
            blockers.append(f"{pkg} missing -> S2 cannot run")
    for pkg in ("wfdb", "neurokit2"):
        if rec["packages"][pkg] is None:
            blockers.append(f"{pkg} missing -> S3 cannot run")
    if not gpu.get("available"):
        blockers.append("no GPU -> S4 numbers will not reflect the real campaign")

    if blockers:
        print("  BLOCKERS")
        for b in blockers:
            print(f"    - {b}")
    else:
        print("  no blockers - all four spikes can run")
    print()
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
