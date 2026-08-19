"""
P0 · S5 — Which local LLM fits this laptop, and is the small one good enough?

Answers two questions with measurements instead of arithmetic:

  1. SPEED   tokens/sec for each candidate model on this exact GPU, plus how
             much of the model Ollama actually managed to keep on the GPU.
  2. QUALITY side-by-side output on a real NLP-06 alert-summary prompt, so the
             size decision is made on the task that matters rather than in
             the abstract.

The clinical summary task here is deliberately narrow - read retrieved records,
write 3-5 grounded sentences, invent nothing. Small models tend to hold up well
on constrained extraction tasks, which is exactly why this is worth testing
rather than assuming bigger is better.

    python scripts/s5_llm_benchmark.py --models llama3.2:3b llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434"

# A realistic NLP-06 alert summary prompt: the FL detector has flagged a
# patient, and the RAG layer has retrieved their context.
CLINICAL_PROMPT = """You are a clinical decision support assistant.
The anomaly detection system has flagged this patient for elevated risk.

FLAGGED VITALS (abnormal readings):
respiratory_rate: 26 breaths/min (normal: 12-20)
spo2: 91% (normal: 94-100)
heart_rate: 118 bpm (normal: 60-100)

RISK SCORE: 0.87 / 1.00

PATIENT CLINICAL CONTEXT (retrieved from records):
68-year-old male, admitted 3 days ago for elective hip replacement.
History of COPD, diagnosed 2019, managed with tiotropium inhaler.
Ex-smoker, 40 pack-years, quit 2016.
Post-operative day 2 note: patient reports mild breathlessness on exertion,
chest clear on auscultation, mobilising with physiotherapy.
Medications: tiotropium, enoxaparin, paracetamol, oxycodone PRN.
No documented fever. Last recorded temperature 37.1 C.

Provide a concise 3-5 sentence clinical summary for the attending clinician.
Focus on: (1) what is abnormal and by how much, (2) relevant patient history
that may explain or worsen the risk, (3) any patterns in recent records.
Do not recommend specific treatments. Do not fabricate information."""

# Facts that ARE in the context - a good summary should draw on these.
GROUNDED_FACTS = ["copd", "hip", "smok", "tiotropium", "oxycodone", "91", "26", "118"]
# Things that are NOT in the context - mentioning them is a hallucination.
ABSENT_FACTS = ["fever", "pneumonia", "x-ray", "chest radiograph", "sepsis",
                "embolism", "antibiotic", "diabetes"]


def ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=5).ok
    except Exception:
        return False


def installed_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def ps_placement(model: str) -> str:
    """`ollama ps` reports how much of the model sits on GPU vs CPU."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            if model.split(":")[0] in line:
                return " ".join(line.split())
    except Exception:
        pass
    return "unavailable"


def run_model(model: str, prompt: str, num_predict: int = 400) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=900)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()

    eval_count = d.get("eval_count", 0)
    eval_ns = d.get("eval_duration", 1) or 1
    load_ns = d.get("load_duration", 0) or 0
    prompt_n = d.get("prompt_eval_count", 0)

    return {
        "response": d.get("response", "").strip(),
        "tokens_generated": eval_count,
        "tokens_per_sec": round(eval_count / (eval_ns / 1e9), 2),
        "prompt_tokens": prompt_n,
        "load_seconds": round(load_ns / 1e9, 2),
        "wall_seconds": round(wall, 2),
    }


def score_grounding(text: str) -> dict:
    low = text.lower()
    hit = [f for f in GROUNDED_FACTS if f in low]
    halluc = [f for f in ABSENT_FACTS if f in low]
    sentences = [s for s in text.replace("\n", " ").split(".") if s.strip()]
    return {
        "grounded_hits": len(hit),
        "grounded_of": len(GROUNDED_FACTS),
        "grounded_terms": hit,
        "possible_hallucinations": halluc,
        "sentence_count": len(sentences),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:3b", "llama3.1:8b"])
    ap.add_argument("--warmup", action="store_true", default=True)
    args = ap.parse_args()

    print("=" * 68)
    print("S5 - LOCAL LLM BENCHMARK")
    print("=" * 68)

    if not ollama_up():
        raise SystemExit("Ollama is not responding at localhost:11434.\n"
                         "Start it with:  ollama serve")

    have = installed_models()
    print(f"  models installed: {', '.join(have) if have else '(none)'}")
    baseline_vram = gpu_used_mib()
    print(f"  GPU in use before loading anything: {baseline_vram} MiB\n")

    findings: dict = {"baseline_vram_mib": baseline_vram, "models": {}}

    for model in args.models:
        if not any(model.split(":")[0] in h and model.split(":")[-1] in h for h in have):
            print(f"  !! {model} not installed - skipping. Pull with: ollama pull {model}\n")
            continue

        print("-" * 68)
        print(f"  {model}")
        print("-" * 68)

        # warm-up load so the timed run excludes cold-start disk read
        if args.warmup:
            try:
                run_model(model, "Reply with the single word: ready", num_predict=8)
            except Exception as exc:
                print(f"    warm-up failed: {exc}")

        vram_loaded = gpu_used_mib()
        placement = ps_placement(model)

        try:
            res = run_model(model, CLINICAL_PROMPT)
        except Exception as exc:
            print(f"    generation failed: {exc}\n")
            continue

        grounding = score_grounding(res["response"])
        vram_peak = gpu_used_mib()

        print(f"    speed            {res['tokens_per_sec']:.1f} tokens/sec")
        print(f"    generated        {res['tokens_generated']} tokens in {res['wall_seconds']:.1f}s wall")
        print(f"    prompt tokens    {res['prompt_tokens']}")
        print(f"    VRAM after load  {vram_loaded} MiB  (delta {vram_loaded - baseline_vram:+} MiB)")
        print(f"    placement        {placement}")
        print(f"    grounded terms   {grounding['grounded_hits']}/{grounding['grounded_of']}  {grounding['grounded_terms']}")
        if grounding["possible_hallucinations"]:
            print(f"    !! UNSUPPORTED   {grounding['possible_hallucinations']}")
        else:
            print("    unsupported      none detected")
        print(f"    sentences        {grounding['sentence_count']}")
        print()
        print("    --- output ---")
        for line in res["response"].splitlines():
            print(f"    {line}")
        print()

        findings["models"][model] = {
            **{k: v for k, v in res.items() if k != "response"},
            "response": res["response"],
            "vram_after_load_mib": vram_loaded,
            "vram_delta_mib": (vram_loaded - baseline_vram) if vram_loaded and baseline_vram else None,
            "vram_peak_mib": vram_peak,
            "placement": placement,
            **grounding,
        }

    # ------------------------------------------------------------------
    print("=" * 68)
    print("COMPARISON")
    print("=" * 68)
    if len(findings["models"]) >= 2:
        print(f"  {'model':<18} {'tok/s':>8} {'VRAM MiB':>10} {'grounded':>10} {'unsupported':>13}")
        for name, d in findings["models"].items():
            print(f"  {name:<18} {d['tokens_per_sec']:>8.1f} {str(d['vram_delta_mib']):>10} "
                  f"{str(d['grounded_hits']) + '/' + str(d['grounded_of']):>10} "
                  f"{len(d['possible_hallucinations']):>13}")
        print()
        print("  Decide on the task, not the parameter count. If the smaller model")
        print("  matches on grounded terms and produces no unsupported claims, the")
        print("  extra parameters are buying nothing for THIS job - and the VRAM")
        print("  they free up is worth more to the project.")
    else:
        print("  Need at least two models for a comparison.")

    out = RESULTS / "s5_llm_benchmark.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
