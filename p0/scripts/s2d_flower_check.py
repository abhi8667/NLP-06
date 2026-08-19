"""
P0 · S2d — Is Flower viable on this machine, and by which path?

s2b and s2c deliberately answer the federated-semantics and DP-accounting
questions with a framework-free loop, so that a framework problem cannot be
mistaken for a DP problem. This script answers the remaining question: which
Flower execution path P3B should actually build on.

Flower offers two, and they cost very different things:

  SIMULATION   flwr.simulation.run_simulation - defaults to a Ray backend.
               Convenient, but Ray is a heavy extra dependency on Windows.

  DEPLOYMENT   SuperLink + SuperNode processes over gRPC.
               Heavier to stand up, but it IS the Dockerised ward-node setup
               the execution plan already commits to for "genuine process-level
               federation". Building simulation first means building twice.

    python scripts/s2d_flower_check.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def main() -> None:
    findings: dict = {}

    import flwr

    print("=" * 62)
    print("FLOWER VIABILITY PROBE")
    print("=" * 62)
    print(f"  flwr version: {flwr.__version__}")
    findings["flwr_version"] = flwr.__version__

    # --- API surface actually present in this version ---
    print("\n  API surface")
    api = {}
    for label, (mod, name) in {
        "ClientApp": ("flwr.client", "ClientApp"),
        "ServerApp": ("flwr.server", "ServerApp"),
        "run_simulation": ("flwr.simulation", "run_simulation"),
        "start_simulation (legacy)": ("flwr.simulation", "start_simulation"),
        "NumPyClient": ("flwr.client", "NumPyClient"),
    }.items():
        try:
            m = __import__(mod, fromlist=[name])
            ok = hasattr(m, name)
        except Exception:
            ok = False
        api[label] = ok
        print(f"    {label:<26} {'yes' if ok else 'NO'}")
    findings["api"] = api

    # --- strategies relevant to the plan ---
    print("\n  strategies")
    import flwr.server.strategy as st
    for s in ("FedAvg", "FaultTolerantFedAvg", "DPFedAvgFixed", "DPFedAvgAdaptive"):
        print(f"    {s:<26} {'yes' if hasattr(st, s) else 'NO'}")
    findings["has_fedavg"] = hasattr(st, "FedAvg")

    # --- backends ---
    print("\n  execution backends")
    ray_ok = have("ray")
    print(f"    ray (simulation backend)   {'installed' if ray_ok else 'NOT INSTALLED'}")
    flwr_cli = shutil.which("flwr") or shutil.which("flower-superlink")
    print(f"    flwr CLI / superlink       {'found' if flwr_cli else 'not on PATH'}")
    findings["ray_installed"] = ray_ok
    findings["flwr_cli"] = bool(flwr_cli)

    # --- recommendation ---
    print()
    print("=" * 62)
    print("RECOMMENDATION")
    print("=" * 62)
    if not ray_ok:
        print("  Ray is absent, so run_simulation will not work as-is.")
        print()
        print("  Do NOT install Ray just to unblock a spike. The execution plan")
        print("  already commits P3B to Dockerised ward nodes over gRPC for the")
        print("  'genuine process-level federation' claim - which is the DEPLOYMENT")
        print("  path, not the simulation path. Building simulation first means")
        print("  building the federation layer twice and throwing one away.")
        print()
        print("  P0 decision to record: build P3B directly on the deployment path.")
        print("  s2b/s2c already prove the semantics and the DP accounting, so")
        print("  nothing is blocked by skipping the simulation engine.")
        rec = "build P3B on deployment path (SuperLink/SuperNode over gRPC); skip Ray"
    else:
        print("  Ray is present - simulation is available as a fast inner loop.")
        print("  Still plan the Docker/gRPC deployment path for P3B, since the")
        print("  paper's federation claim rests on it.")
        rec = "simulation available; still target deployment path for P3B"
    findings["recommendation"] = rec
    print("=" * 62)

    out = RESULTS / "s2d_flower_check.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
