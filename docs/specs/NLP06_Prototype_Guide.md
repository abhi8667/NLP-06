# NLP-06 Prototype Guide
## Privacy-Preserving Personalized Healthcare & Wellness Assistant
### Everything needed to build the working prototype

---

## 1. What the Prototype Is

A fully on-premise, two-interface clinical decision support system deployed within a simulated hospital ward network. No data leaves the local machine or local network at any point.

**Two interfaces:**
- Patient-facing chat assistant — answers questions grounded in the patient's own record
- Doctor-facing alert dashboard — shows anomaly alerts with RAG-generated patient summaries

**Two backend systems:**
- RAG pipeline — retrieves patient-specific context and passes it to a local LLM
- FL anomaly model — runs inference on incoming vitals and generates risk scores

---

## 2. What Is Cut From the Original PRD

The following are explicitly out of scope for the prototype:

| Cut feature | Reason |
|---|---|
| Voice pipeline (Whisper + TTS) | Adds zero research value, high integration risk |
| Meal / exercise logging forms | Not in Synthea, no research use |
| Medication reminders | Out of scope for the paper |
| Federated LLM fine-tuning | Computationally impossible in timeline |
| Full UI polish | Streamlit is sufficient — rigor lives in experiments |

---

## 3. Technology Stack

| Layer | Component | Tool / Library | Notes |
|---|---|---|---|
| LLM inference | Local model runtime | Ollama | Pull Llama 3.1 8B GGUF quantized |
| LLM model | Primary | Llama 3.1 8B (Q4_K_M quantized) | Fallback: Mistral 7B |
| Embeddings | Sentence embedder | BGE-small-en or MiniLM-L6-v2 | Run locally, no API |
| Vector DB | Patient-scoped store | ChromaDB | Local file-based, no server needed |
| Backend API | REST layer | FastAPI (Python) | Serves both interfaces |
| Frontend | Patient + doctor UI | Streamlit | Two separate pages |
| Database | Encrypted storage | SQLite with AES-256 | Via sqlcipher or cryptography lib |
| FL framework | Federated orchestration | Flower (flwr) | Docker containers via gRPC (genuine federation) |
| DP implementation | DP-SGD | Opacus (PyTorch) | Rényi DP accountant built in |
| Anomaly model | Sequence classifier | PyTorch LSTM / GRU | CNN as baseline |
| Data generation | Synthetic EHR | Synthea (Java) | Runs locally, no credentials |
| Real-signal validation | Research track only | PTB-XL (PhysioNet) | Real ECGs, open access, no credentialing |
| Encryption at rest | Full data volume | LUKS / encrypted Docker volume | Encrypts ChromaDB + SQLite together |
| PDF parsing | Lab report ingestion | pdfplumber | Extract text from uploaded PDFs |
| Environment | Python | 3.10+ | venv or conda |
| Version control | Repo | Git + GitHub | Branch per track |

---

## 4. Repository Structure

```
nlp06/
├── data/
│   ├── synthea_output/          # raw Synthea JSON/CSV output
│   ├── processed/               # cleaned per-patient records
│   ├── ward_partitions/         # ward1/ ward2/ ... ward5/ splits
│   ├── embeddings/              # ChromaDB persistent store
│   └── anomaly_labels/          # labeled windows for FL training
│
├── product_track/               # Person A
│   ├── rag/
│   │   ├── embedder.py          # chunk + embed patient records
│   │   ├── retriever.py         # ChromaDB semantic search
│   │   └── prompt_builder.py    # assemble context + query → prompt
│   ├── llm/
│   │   └── ollama_client.py     # calls local Ollama API
│   ├── ingestion/
│   │   └── lab_report.py        # PDF → text → embed → store
│   ├── interfaces/
│   │   ├── patient_app.py       # Streamlit patient chat
│   │   └── doctor_app.py        # Streamlit doctor dashboard
│   └── api/
│       └── main.py              # FastAPI entry point
│
├── research_track/              # Person B
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── cnn_model.py
│   ├── federated/
│   │   ├── client.py            # Flower client (ward node)
│   │   ├── server.py            # Flower server (FedAvg aggregator)
│   │   └── strategy.py          # custom FedAvg strategy if needed
│   ├── privacy/
│   │   └── dp_trainer.py        # Opacus DP-SGD wrapper
│   ├── experiments/
│   │   ├── epsilon_sweep.py     # main experiment runner
│   │   └── results/             # CSV logs, figures
│   └── inference/
│       └── risk_scorer.py       # runs trained model on new vitals
│
├── shared/
│   ├── data_pipeline.py         # Synthea → processed records
│   ├── anomaly_labeler.py       # NEWS2-grounded threshold labeling
│   ├── ward_simulator.py        # partition patients into ward nodes
│   ├── schema.py                # per-patient record schema
│   └── encryption.py            # AES-256 at rest helpers
│
├── integration/
│   └── alert_rag_bridge.py      # C3 — alert triggers RAG summary
│
├── tests/
│   ├── test_rag.py
│   ├── test_fl_loop.py
│   └── test_alert_bridge.py
│
├── requirements.txt
├── README.md
└── docker-compose.yml           # optional, for reproducible local deploy
```

---

## 5. Data Pipeline — Step by Step

### Step 1 — Generate Synthea patients

```bash
# Install Synthea (Java required)
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate 1000 patients
./run_synthea -p 1000 --exporter.fhir.export false --exporter.csv.export true

# Output lands in output/csv/
```

Key CSV files you need:
- `observations.csv` — vitals and lab values with timestamps
- `patients.csv` — demographics (age, gender, race)
- `conditions.csv` — diagnoses (for non-IID ward skewing)
- `encounters.csv` — visit timestamps

### Step 2 — Extract and clean vitals

Target LOINC codes from `observations.csv`:

| Vital | LOINC code |
|---|---|
| Heart rate | 8867-4 |
| Systolic BP | 8480-6 |
| Diastolic BP | 8462-4 |
| SpO2 | 59408-5 |
| Respiratory rate | 9279-1 |
| Body temperature | 8310-5 |
| Blood glucose | 2339-0 |

```python
# shared/data_pipeline.py — core extraction logic
VITAL_CODES = {
    '8867-4': 'heart_rate',
    '8480-6': 'bp_systolic',
    '8462-4': 'bp_diastolic',
    '59408-5': 'spo2',
    '9279-1': 'resp_rate',
    '8310-5': 'temperature',
    '2339-0': 'glucose',
}

def extract_vitals(obs_df, patient_id):
    patient_obs = obs_df[obs_df['PATIENT'] == patient_id]
    vitals = patient_obs[patient_obs['CODE'].isin(VITAL_CODES)]
    vitals = vitals[['DATE', 'CODE', 'VALUE']].copy()
    vitals['vital_name'] = vitals['CODE'].map(VITAL_CODES)
    vitals['DATE'] = pd.to_datetime(vitals['DATE'])
    vitals['VALUE'] = pd.to_numeric(vitals['VALUE'], errors='coerce')
    return vitals.pivot_table(
        index='DATE', columns='vital_name',
        values='VALUE', aggfunc='mean'
    ).reset_index().sort_values('DATE')
```

### Step 3 — Apply anomaly labels (NEWS2-grounded)

```python
# shared/anomaly_labeler.py
ANOMALY_THRESHOLDS = {
    'heart_rate':   (40, 130),    # bpm
    'bp_systolic':  (80, 180),    # mmHg
    'spo2':         (90, 100),    # % (lower bound only)
    'resp_rate':    (8, 30),      # breaths/min
    'temperature':  (35.0, 39.5), # Celsius
    'glucose':      (3.0, 16.7),  # mmol/L
}

def label_anomaly(row):
    for vital, (low, high) in ANOMALY_THRESHOLDS.items():
        if vital in row and pd.notna(row[vital]):
            if row[vital] < low or row[vital] > high:
                return 1
    return 0
```

Reference: NEWS2 (Royal College of Physicians, 2017) — cite this in the paper to justify thresholds.

**Strengthen the labels — use full NEWS2 aggregate scoring + MEWS cross-check.**
Rather than single-vital thresholds, implement the full NEWS2 aggregate score (each vital contributes 0-3 points; total ≥ 5 triggers anomaly). This mirrors the nationally-adopted NHS protocol and makes "not clinician-validated" become "derived from the NEWS2 protocol validated across NHS hospitals." Then cross-check with MEWS (Modified Early Warning Score) and confirm your results hold under both schemes — this shows the finding isn't an artifact of one arbitrary threshold choice.

```python
# shared/anomaly_labeler.py — NEWS2 aggregate scoring
def news2_score(row):
    score = 0
    # Respiratory rate
    rr = row.get('resp_rate')
    if rr is not None:
        if rr <= 8 or rr >= 25: score += 3
        elif rr >= 21: score += 2
        elif rr <= 11: score += 1
    # SpO2
    spo2 = row.get('spo2')
    if spo2 is not None:
        if spo2 <= 91: score += 3
        elif spo2 <= 93: score += 2
        elif spo2 <= 95: score += 1
    # (repeat for HR, BP systolic, temperature per NEWS2 chart)
    return score

def label_anomaly_news2(row, threshold=5):
    return 1 if news2_score(row) >= threshold else 0
```

### Step 4 — Create sliding windows

```python
# 12 timesteps x 6 vitals per window, stride 1
WINDOW_SIZE = 12
VITALS = ['heart_rate', 'bp_systolic', 'spo2', 'resp_rate', 'temperature', 'glucose']

def create_windows(patient_df):
    windows, labels = [], []
    for i in range(len(patient_df) - WINDOW_SIZE):
        window = patient_df[VITALS].iloc[i:i+WINDOW_SIZE].values
        label = patient_df['anomaly'].iloc[i+WINDOW_SIZE]
        if not np.isnan(window).any():
            windows.append(window)
            labels.append(label)
    return np.array(windows), np.array(labels)
```

### Step 5 — Ward partitioning (non-IID by design)

```python
# shared/ward_simulator.py
WARD_CONFIG = {
    'ward1': {'age_min': 70, 'conditions': ['diabetes', 'hypertension']},
    'ward2': {'age_max': 50, 'conditions': ['post_surgical']},
    'ward3': {'conditions': ['cardiac', 'hypertension']},
    'ward4': {'conditions': ['copd', 'respiratory']},
    'ward5': {'conditions': []},  # mixed general medicine
}
```

Each ward gets approximately 200 patients. The skew is deliberate — ward 1 will have more glucose anomalies, ward 4 more SpO2 anomalies. This creates genuine non-IID data heterogeneity for the FL experiments.

### Step 5b — Inject realistic noise (robustness variant)

Create a noised copy of the Synthea data so the research track can test robustness to real-world messiness. Store as a parallel dataset, don't overwrite the clean one.

```python
# shared/noise_injection.py
def inject_realistic_noise(vitals_df, noise_std_pct=0.05,
                            missingness_rate=0.10, artifact_rate=0.02):
    noisy = vitals_df.copy()
    for col in VITALS:
        noise = np.random.normal(0, noisy[col].std() * noise_std_pct, len(noisy))
        noisy[col] = noisy[col] + noise
        mask = np.random.random(len(noisy)) < missingness_rate
        noisy.loc[mask, col] = np.nan
        artifact_mask = np.random.random(len(noisy)) < artifact_rate
        noisy.loc[artifact_mask, col] *= np.random.uniform(1.5, 3.0)
    return noisy
```

### Step 5c — Prepare PTB-XL real-signal validation set (research track only)

Download PTB-XL from PhysioNet (open access, no credentialing). This is the real-data validation set that makes the paper defensible. It does NOT go into the RAG/product track — it is only for the research track's ε-sweep validation.

```bash
# PTB-XL — real 12-lead ECGs with diagnostic labels
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Process into the same window format as Synthea so the same model code runs on it. Use the SCP-ECG diagnostic statements as anomaly labels (normal vs abnormal). See the Research Guide for how this feeds the validation experiment.

### Step 6 — Generate synthetic conversation notes

Use the local LLM (Ollama) to generate realistic clinical conversation snippets for each patient. These go into the RAG knowledge base.

```python
# Template prompt for LLM-generated notes
TEMPLATE = """
Generate a realistic 3-sentence clinical encounter note for a patient with the following profile:
- Age: {age}, Gender: {gender}
- Primary diagnosis: {diagnosis}
- Recent vitals: HR {hr}, BP {bp}, SpO2 {spo2}
- Medications: {meds}
Output only the clinical note, no preamble.
"""
```

Generate 3-5 notes per patient covering different encounter types (admission, follow-up, discharge). Store alongside vitals in the patient record.

---

## 6. Per-Patient Record Schema

Each patient in the system has a unified record with two storage layers:

**Structured layer (SQLite, AES-encrypted):**
```python
{
    "patient_id": "uuid",
    "age": int,
    "gender": str,
    "ward_id": str,           # ward1-ward5
    "diagnoses": [str],
    "medications": [str],
    "vitals_history": [       # list of timestamped readings
        {
            "timestamp": "ISO8601",
            "heart_rate": float,
            "bp_systolic": float,
            "bp_diastolic": float,
            "spo2": float,
            "resp_rate": float,
            "temperature": float,
            "glucose": float,
            "anomaly_label": int   # 0 or 1
        }
    ],
    "lab_results": [          # extracted from Synthea + any uploaded PDFs
        {
            "timestamp": "ISO8601",
            "test_name": str,
            "value": float,
            "unit": str,
            "reference_range": str
        }
    ]
}
```

**Unstructured layer (ChromaDB, per-patient collection):**
- Encounter notes (LLM-generated)
- Lab report text (from PDF ingestion)
- Diagnosis summaries
- Each chunk tagged with `patient_id` metadata for strict filtering

---

## 7. RAG Pipeline — Implementation Detail

### Embedding and indexing

```python
# product_track/rag/embedder.py
from chromadb import Client
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # runs locally

def embed_patient_record(patient_id, text_chunks):
    client = Client()  # persistent local store
    collection = client.get_or_create_collection(
        name=f"patient_{patient_id}",
        metadata={"patient_id": patient_id}
    )
    embeddings = model.encode(text_chunks).tolist()
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=[f"{patient_id}_{i}" for i in range(len(text_chunks))],
        metadatas=[{"patient_id": patient_id}] * len(text_chunks)
    )
```

### Retrieval with strict patient filtering

```python
# product_track/rag/retriever.py
def retrieve(patient_id, query, top_k=4):
    collection = client.get_collection(f"patient_{patient_id}")
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"patient_id": patient_id}   # strict isolation
    )
    return results['documents'][0]
```

The `where` filter is critical — it is the privacy boundary within the RAG layer. No patient can ever receive context from another patient's record.

### Prompt construction

```python
# product_track/rag/prompt_builder.py
def build_prompt(patient_id, query, retrieved_chunks):
    context = "\n---\n".join(retrieved_chunks)
    return f"""You are a clinical decision support assistant.
You have access only to the following information about this patient.
Do not infer, assume, or generate information not present in the context.
If the context does not contain enough information to answer safely, say so.

PATIENT CONTEXT:
{context}

PATIENT OR CLINICIAN QUESTION:
{query}

RESPONSE:"""
```

### LLM call via Ollama

```python
# product_track/llm/ollama_client.py
import requests

def generate(prompt, model="llama3.1:8b-instruct-q4_K_M"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }
    )
    return response.json()["response"]
```

Temperature 0.1 — low for clinical use to minimize creative hallucination.

---

## 8. RAG-FL Alert Interface (C3) — Implementation Detail

This is the research-significant connection between the two tracks.

```python
# integration/alert_rag_bridge.py

ALERT_THRESHOLD = 0.75   # risk score above this triggers alert

ALERT_RAG_PROMPT_TEMPLATE = """
You are a clinical decision support assistant.
The anomaly detection system has flagged this patient for elevated risk.

FLAGGED VITALS (abnormal readings):
{flagged_vitals}

RISK SCORE: {risk_score:.2f} / 1.00

PATIENT CLINICAL CONTEXT (retrieved from records):
{retrieved_context}

Provide a concise 3-5 sentence clinical summary for the attending clinician.
Focus on: (1) what is abnormal and by how much, (2) relevant patient history
that may explain or worsen the risk, (3) any patterns in recent records.
Do not recommend specific treatments. Do not fabricate information.
"""

def process_alert(patient_id, risk_score, vital_readings, retriever, llm):
    if risk_score < ALERT_THRESHOLD:
        return None

    # identify which vitals are anomalous
    flagged = []
    for vital, (low, high) in ANOMALY_THRESHOLDS.items():
        value = vital_readings.get(vital)
        if value and (value < low or value > high):
            flagged.append(f"{vital}: {value} (normal: {low}-{high})")

    # build alert-specific RAG query
    query = f"clinical history relevant to: {', '.join(flagged)}"
    context_chunks = retriever.retrieve(patient_id, query, top_k=4)

    # generate summary
    prompt = ALERT_RAG_PROMPT_TEMPLATE.format(
        flagged_vitals="\n".join(flagged),
        risk_score=risk_score,
        retrieved_context="\n---\n".join(context_chunks)
    )
    summary = llm.generate(prompt)

    return {
        "patient_id": patient_id,
        "risk_score": risk_score,
        "flagged_vitals": flagged,
        "rag_summary": summary,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Evaluating C3 (rubric for the paper)

Generate 20-30 alert scenarios from held-out Synthea patients. Score each RAG-generated summary on:

| Criterion | Score 0-3 | What 3 means |
|---|---|---|
| Factual accuracy | 0-3 | All facts in summary are present in patient record |
| Relevance to flagged vitals | 0-3 | Summary directly addresses the flagged abnormality |
| Clinical completeness | 0-3 | Relevant history surfaced (conditions, meds, prior episodes) |
| Hallucination absence | 0/3 | No fabricated information (binary pass/fail, weighted) |
| Conciseness | 0-3 | 3-5 sentences, no padding |

Two raters (both team members) score independently. Report inter-rater agreement (Cohen's kappa).

**Strengthen the evaluation with two additional layers** (see Research Guide C3 for detail):
1. **LLM-as-judge** — have GPT-4o and Claude score the same summaries against the same rubric, following b4's methodology. Grounds the evaluation in a peer-reviewed approach, not just student ratings.
2. **Programmatic fact-verification** — since you control the source records, automatically check that every fact in each summary appears in the patient record. Gives an objective, verifiable hallucination rate.

```python
# integration/fact_verifier.py
def verify_summary_facts(summary, source_record):
    claims = extract_claims(summary)
    hallucinated = [c for c in claims if not claim_supported_by_record(c, source_record)]
    return len(hallucinated) / max(len(claims), 1), hallucinated
```

Together these three layers give C3 a robust, publishable evaluation.

---

## 9. Patient Interface — Streamlit

```python
# product_track/interfaces/patient_app.py
import streamlit as st
from product_track.rag.retriever import retrieve
from product_track.rag.prompt_builder import build_prompt
from product_track.llm.ollama_client import generate

st.set_page_config(page_title="Your Health Assistant", layout="centered")
st.title("Your Health Assistant")
st.caption("Your questions are answered using only your own health records.")

# In production: patient_id comes from session auth
patient_id = st.text_input("Patient ID", value="demo_patient_001")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about your health records..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    context = retrieve(patient_id, query)
    prompt = build_prompt(patient_id, query, context)
    response = generate(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
```

**Key patient interface features to implement:**
- Conversation history within session
- Lab report PDF upload → ingestion → embedded into patient's ChromaDB collection
- Clear disclaimer: "This assistant provides information only, not medical advice"
- Session isolation — no cross-patient data visible

---

## 10. Doctor Interface — Streamlit

```python
# product_track/interfaces/doctor_app.py
import streamlit as st

st.set_page_config(page_title="Ward Alert Dashboard", layout="wide")
st.title("Ward Alert Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Active Alerts")
    # fetch alerts from FastAPI backend
    alerts = fetch_alerts()  # returns list of alert dicts
    for alert in sorted(alerts, key=lambda x: x['risk_score'], reverse=True):
        color = "red" if alert['risk_score'] > 0.9 else "orange"
        if st.button(
            f"Patient {alert['patient_id']} — Risk: {alert['risk_score']:.2f}",
            key=alert['patient_id']
        ):
            st.session_state.selected_alert = alert

with col2:
    if 'selected_alert' in st.session_state:
        alert = st.session_state.selected_alert
        st.subheader(f"Patient {alert['patient_id']}")
        st.metric("Risk Score", f"{alert['risk_score']:.2f}")

        st.markdown("**Flagged vitals:**")
        for v in alert['flagged_vitals']:
            st.markdown(f"- {v}")

        st.markdown("**Clinical context summary:**")
        st.info(alert['rag_summary'])

        st.markdown("**Ask follow-up:**")
        if followup := st.text_input("Query patient record", key="followup"):
            context = retrieve(alert['patient_id'], followup)
            prompt = build_prompt(alert['patient_id'], followup, context)
            response = generate(prompt)
            st.markdown(response)
```

---

## 11. FastAPI Backend — Core Endpoints

```python
# product_track/api/main.py
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query_patient(patient_id: str, question: str):
    context = retrieve(patient_id, question)
    prompt = build_prompt(patient_id, question, context)
    response = generate(prompt)
    return {"response": response, "context_used": context}

@app.post("/ingest/lab_report")
async def ingest_lab(patient_id: str, file: UploadFile):
    text = extract_pdf_text(file)
    chunks = chunk_text(text)
    embed_patient_record(patient_id, chunks)
    return {"status": "ingested", "chunks": len(chunks)}

@app.get("/alerts")
async def get_alerts():
    return fetch_active_alerts()  # from alert store

@app.post("/score")
async def score_vitals(patient_id: str, vitals: dict):
    risk_score = risk_scorer.predict(vitals)
    alert = process_alert(patient_id, risk_score, vitals, retriever, llm)
    if alert:
        store_alert(alert)
    return {"risk_score": risk_score, "alert_generated": alert is not None}
```

---

## 12. Encryption at Rest

```python
# shared/encryption.py
from cryptography.fernet import Fernet

KEY = Fernet.generate_key()  # store securely, not in code
fernet = Fernet(KEY)

def encrypt_record(data: dict) -> bytes:
    import json
    return fernet.encrypt(json.dumps(data).encode())

def decrypt_record(token: bytes) -> dict:
    import json
    return json.loads(fernet.decrypt(token).decode())
```

**Eliminating the ChromaDB plaintext limitation:**
ChromaDB stores embeddings in plaintext by default. Rather than noting this as a limitation, eliminate it entirely by deploying the whole system inside an encrypted volume:

```bash
# Linux — LUKS encrypted volume for all persistent data
sudo cryptsetup luksFormat /path/to/nlp06_data.img
sudo cryptsetup luksOpen /path/to/nlp06_data.img nlp06_secure
sudo mkfs.ext4 /dev/mapper/nlp06_secure
sudo mount /dev/mapper/nlp06_secure /mnt/nlp06_data

# Point ChromaDB, SQLite, and all data/ at the encrypted mount
export CHROMA_PERSIST_DIR=/mnt/nlp06_data/embeddings
```

Alternatively, use an encrypted Docker volume. With this in place, every persistent artifact — structured records, vector embeddings, conversation logs — is encrypted at rest. This removes the encryption limitation from the paper entirely. Note additionally that dense embeddings are not trivially invertible to source text (embedding inversion is an open research problem), so even the embeddings carry inherent protection.

---

## 13. Environment Setup

```bash
# Create environment
python -m venv nlp06_env
source nlp06_env/bin/activate  # Windows: nlp06_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.1:8b-instruct-q4_K_M

# Install Synthea (Java 11+ required)
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./run_synthea -p 1000 --exporter.csv.export true
cd ..

# Generate data pipeline
python shared/data_pipeline.py
python shared/anomaly_labeler.py
python shared/ward_simulator.py

# Embed patient records
python product_track/rag/embedder.py

# Start backend
uvicorn product_track.api.main:app --reload --port 8000

# Start interfaces (separate terminals)
streamlit run product_track/interfaces/patient_app.py --server.port 8501
streamlit run product_track/interfaces/doctor_app.py  --server.port 8502
```

**requirements.txt:**
```
# LLM + RAG
ollama>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain>=0.1.0

# Backend + frontend
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.28.0
pdfplumber>=0.10.0

# FL + DP
torch>=2.0.0
flwr>=1.6.0
opacus>=1.4.0

# Data
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Crypto + utilities
cryptography>=41.0.0
python-multipart>=0.0.6
requests>=2.31.0
```

---

## 14. End-to-End Demo Flow (Integration Test)

The full walkthrough that must run successfully for Phase 5 completion:

1. Doctor opens dashboard → sees ward-level anomaly alerts sorted by risk score
2. Doctor clicks on highest-risk patient → sees flagged vitals + RAG-generated clinical summary
3. Doctor types follow-up question → receives grounded response from patient record
4. Doctor opens patient conversation log → reads prior exchanges
5. Patient opens chat → asks "what do my latest labs show?" → receives grounded summary
6. Patient uploads a PDF lab report → system ingests it → patient asks about it → RAG retrieves and answers
7. New vital reading comes in (from FL scoring endpoint) → if anomalous, alert appears on doctor dashboard within one refresh cycle

**Privacy audit checklist (run before final demo):**
- [ ] No outbound network calls during LLM inference (Ollama is local)
- [ ] ChromaDB collections are strictly per-patient (test cross-patient query returns nothing)
- [ ] SQLite file is encrypted (verify raw file is not readable)
- [ ] FL noisy gradients contain no raw patient records (inspect gradient tensors)
- [ ] No patient data in API logs or Streamlit session state leaks

---

## 15. Limitations — What's Fixed, Mitigated, and Defended

The limitations strategy is handled in detail in the Research Guide (Section 7). Summary of what the prototype build must deliver to support it:

| Limitation | Status | What the prototype must do |
|---|---|---|
| Synthea realism | 🟡 Mitigated | Build noise-injection pipeline (Step 5b) + PTB-XL prep (Step 5c) |
| ChromaDB encryption | 🟢 Fixed | Deploy on encrypted LUKS/Docker volume — no longer a limitation |
| Simulated federation | 🟡 Mitigated | Dockerize ward nodes with gRPC + dropout experiment |
| Rule-based labels | 🟢 Fixed | Full NEWS2 aggregate + MEWS cross-check in anomaly_labeler.py |
| C3 clinician validation | 🟡 Mitigated | Add LLM-judge + programmatic fact-verifier |
| Voice interface | 🔴 Defended | Cut deliberately; state as future work |
| No EHR/FHIR integration | 🔴 Defended | Out of scope; state as future work |

**What remains genuinely limited and must be stated honestly:**
- PTB-XL validation is narrower than full ward vitals (real ECG, not full multi-vital telemetry)
- Ward federation is process-level (Docker), not multi-institution physical deployment
- No formal multi-clinician usability study of the live system
- Not deployed in a live hospital environment

See Research Guide Section 7 for the exact wording to use in the paper for each item.
