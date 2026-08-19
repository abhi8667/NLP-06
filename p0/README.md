# NLP-06 · P0 Feasibility Spikes

Four questions to answer before committing to the build. The deliverable is
[`P0_MEMO.md`](P0_MEMO.md) filled in — not four working systems. Resist the urge to
start building the real pipeline inside a spike; everything here is throwaway
scaffolding that exists to produce a number and a decision.

## Setup

The environment is already created at `.venv` (Python 3.11). To recreate it:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --index-strategy unsafe-best-match
```

### Already present

**Java 17.0.12** is installed and on PATH, so Synthea and S1 are unblocked. (A
PowerShell check will report it missing — `java -version` writes to stderr, which
PowerShell surfaces as a `NativeCommandError`. Verify from bash or with
`where.exe java` instead.)

### Still to install

Ollama is not needed for S1–S4, but install it during P0 so S4 can measure real GPU
contention against the campaign:

```bash
winget install --id Ollama.Ollama -e
```

### GPU note

`torch` currently resolves to the **CPU** build on this machine. S4's timings are
meaningless without CUDA, so install the CUDA build before running it:

```bash
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

Then confirm:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Running the spikes

Run in order. Each writes a JSON record to `results/` and prints a verdict.

```bash
.venv/Scripts/python.exe scripts/00_env_check.py
```

| Script | Answers | Needs |
|---|---|---|
| `00_env_check.py` | what environment produced these numbers | — |
| `s1_cadence.py` | does Synthea emit vitals at ward cadence? | Java + a Synthea cohort |
| `s2a_dp_single_node.py` | does a recurrent model train under DP-SGD? | — |
| `s2b_fed_nodp.py` | does federation work before DP is added? | — |
| `s2c_fed_dp.py` | does ε mean what the paper will claim? | — |
| `s2d_flower_check.py` | which Flower execution path should P3B use? | — |
| `s3_ptbxl.py --demo` | how does PTB-XL become model input? | — |
| `s4_timing.py` | does the grid fit the compute available? | CUDA torch |

### S1 needs a Synthea cohort first

```bash
git clone https://github.com/synthetichealth/synthea.git
```

Then, from the `synthea` directory:

```bash
./run_synthea.bat -p 50 --exporter.csv.export true --exporter.fhir.export false
```

Fifty patients is plenty — you are measuring a gap distribution, not training anything.
Then point the spike at the output:

```bash
.venv/Scripts/python.exe scripts/s1_cadence.py --obs ../synthea/output/csv/observations.csv
```

### S3 against real PTB-XL

`--demo` validates the mapping with a synthesised signal and needs no download. Once
you have the 100 Hz PTB-XL records:

```bash
.venv/Scripts/python.exe scripts/s3_ptbxl.py --ptbxl data/ptb-xl
```

## What the spikes already confirmed

These ran on this machine and the findings are recorded in `results/`.

**`nn.LSTM` and `nn.GRU` do not work under Opacus.** Opacus raises
`ShouldReplaceModuleError` and points at `DPLSTM`/`DPGRU`, which `ModuleValidator.fix()`
will *not* substitute for you. The model classes in the Research Guide fail as written.

**The models double-apply sigmoid.** They end with `torch.sigmoid(...)` while the data
plan specifies `BCEWithLogitsLoss`, which applies sigmoid internally. Measured gradient
shrink: **4.2×**. Training looks mysteriously flat. Return raw logits.

**The privacy budget is under-reported by the natural implementation.** Constructing a
`PrivacyEngine` inside each federated round restarts the accountant, so ε reads flat
across rounds. Measured on 6 rounds: naive reports **0.7524** every round while correct
accounting reaches **1.8255** — a **2.43× understatement**, and the gap widens with
round count. C4's "tight bound across 100 rounds" depends on getting this right.

**The import path in the Research Guide does not exist.** `opacus.accountant.analysis`
raises `ModuleNotFoundError`; the module is `opacus.accountants` (plural), and
`get_noise_multiplier` lives in its `utils`.

**Ray is not installed**, so Flower's `run_simulation` will not work as-is. Do not
install it just to unblock a spike — P3B is already committed to Dockerised ward nodes
over gRPC, which is the deployment path, not the simulation path.

**The planned grid does not fit this machine.** Measured on the RTX 4050: a full
100-round run takes **4.1 h** for DPLSTM, **4.4 h** for DPGRU, **43 min** for CNN — so
DP-SGD costs roughly **5.7×** on a recurrent model versus the convolutional baseline.
The 150-run grid therefore comes to **~530 h, about 22 days** of uninterrupted compute.

Peak VRAM during training is only **0.03 GB**, so Ollama and the campaign *can* share
the GPU — that contention worry turned out to be unfounded.

### The de-scoping ladder targets the wrong levers for compute

The ladder in the execution plan is ordered by what is cheapest to lose *scientifically*.
Measured against compute, it barely helps:

| Cut | Saves |
|---|---|
| Drop DPGRU | 17% |
| Drop noise-injected Synthea | 24% |
| 4 ε levels instead of 6 | 33% |
| 3 seeds instead of 5 | 40% |
| **50 rounds instead of 100** | **50%** |
| **Half the windows per ward** | **50%** |

Rounds and dataset size dominate, and neither is a scope cut — they are methodological
questions. **Check whether FedAvg has actually converged by round 50 before paying for
100.** Combining the scope cuts with 50 rounds brings the grid to ~6.5 days, which fits.

Note that windows-per-ward is currently an assumption (4,000). S1 determines the real
figure and it scales the whole table linearly, so **re-run S4 after S1**.

## Layout

```
p0/
├── README.md              this file
├── P0_MEMO.md             the actual deliverable - fill it in
├── requirements.txt
├── scripts/               spike harnesses, throwaway by design
├── results/               JSON records written by each spike
└── data/                  put the PTB-XL download here
```
