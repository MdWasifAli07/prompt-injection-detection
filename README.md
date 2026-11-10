# 🛡️ Sentinel — Prompt Injection Detection

A hybrid NLP ensemble for detecting prompt injection attacks against Large Language Model (LLM) applications. Combines a fine-tuned **DeBERTa-v3-base** transformer with a **TF-IDF + XGBoost** lexical classifier, fused by a logistic meta-learner trained on out-of-fold predictions.

---

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [Evaluation Protocol](#evaluation-protocol)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Large Language Models (LLMs) treat all input as a mix of data and instructions. A malicious payload embedded in a user prompt, a retrieved document, or a tool output can silently override the developer's system prompt — a class of attacks known as **prompt injection**.

This project builds an **upstream gatekeeper** that detects injection attempts *before* they reach the LLM. Given a free-form prompt `x`, the system predicts a binary label:

- `0` = **BENIGN**
- `1` = **ATTACK**

Two complementary models are combined:

1. **DeBERTa-v3-base** (semantic) — understands meaning and paraphrasing.
2. **TF-IDF + XGBoost** (lexical) — catches explicit keyword-based attacks.

Two fusion strategies are provided:

- **Weighted vote** — `0.6 × DeBERTa + 0.4 × XGBoost` (transparent, no training)
- **Logistic meta-learner** — learned weights on out-of-fold predictions

---

## Results at a Glance

### Test Set Performance (207 held-out examples)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|---------:|----------:|-------:|-----:|--------:|-------:|
| TF-IDF + XGBoost | 85.02% | 85.96% | 86.73% | 86.34% | 0.9377 | 0.9429 |
| **DeBERTa-v3** | **97.10%** | **99.08%** | 95.58% | **97.30%** | **0.9974** | **0.9978** |
| Weighted Vote (0.6/0.4) | 95.65% | 94.83% | **97.35%** | 96.07% | 0.9931 | 0.9948 |
| Logistic Meta-Ensemble | 95.65% | 99.06% | 92.92% | 95.89% | 0.9944 | 0.9957 |

### Key Findings

- **DeBERTa-v3 alone dominates** on accuracy, F1, ROC-AUC, and PR-AUC.
- The **weighted vote** achieves the highest attack recall (97.35%) — better for security-critical gates.
- The **meta-learner does not outperform DeBERTa** — stacking is not automatically beneficial when one base model dominates.
- DeBERTa-v3 requires FP16 to be disabled on T4 GPUs to avoid silent NaN failures.

---

## Architecture

```
                     ┌──────────────────────────────┐
                     │        INPUT PROMPT x        │
                     └──────────────┬───────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                              ▼
   ┌────────────────────┐                       ┌──────────────────────┐
   │  DeBERTa-v3-base   │                       │  TF-IDF Vectorizer   │
   │  (fine-tuned)      │                       │  (word 1-2 grams)    │
   └─────────┬──────────┘                       └──────────┬───────────┘
             │                                             ▼
             │                                  ┌────────────────────┐
             │                                  │  XGBoost (500 trees)│
             │                                  └──────────┬─────────┘
             ▼                                             ▼
        P_DeBERTa(x)                                  P_TFIDF(x)
             │                                             │
             └──────────────────┬──────────────────────────┘
                                ▼
                     ┌─────────────────────┐
                     │   Fusion Layer      │
                     │  • Weighted Vote    │
                     │  • Logistic Meta    │
                     └──────────┬──────────┘
                                ▼
                        ŷ = 1 if p ≥ τ else 0
```

---

## Repository Structure

```
softcom-prompt-injection/
│
├── README.md                              ← this file
├── requirements.txt                       ← Python dependencies
│
├── data/
│   ├── raw/                               ← original dataset CSV
│   └── processed/
│       ├── dev.csv                        ← 824 development examples
│       ├── test.csv                       ← 207 held-out examples
│       └── schema.json                    ← split metadata
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb        ← clean, dedupe, stratified split
│   ├── 02_train_tfidf_xgboost.ipynb       ← lexical baseline (OOF + test)
│   ├── 03_train_transformer_deberta.ipynb ← DeBERTa fine-tuning (OOF + test)
│   ├── 04_ensemble.ipynb                  ← weighted vote + meta-learner
│   └── 05_evaluation.ipynb                ← final metrics, ROC/PR, confusion
│
├── artifacts/                             ← model outputs (auto-created)
│   ├── tfidf_xgb.joblib                   ← TF-IDF + XGB bundle
│   ├── deberta_model/                     ← fine-tuned DeBERTa
│   ├── meta_logistic.joblib               ← logistic meta-learner
│   ├── metadata.json                      ← thresholds, weights, feature names
│   ├── tfidf_xgb_oof.csv                  ← OOF predictions (dev)
│   ├── deberta_oof.csv                    ← OOF predictions (dev)
│   ├── ensemble_oof.csv                   ← fused OOF predictions
│   ├── ensemble_test.csv                  ← fused test predictions
│   ├── roc_curves.png                     ← evaluation figures
│   ├── pr_curves.png
│   └── confusion_matrix_meta.png
│
└── streamlit_app/
    ├── app.py                             ← Streamlit frontend
    ├── rule_based.py                      ← (optional) rule-based signals
    ├── requirements.txt                   ← app-specific deps
    └── models/                            ← copy from artifacts/
```

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Raw rows | 2,000 |
| Unique prompts after cleaning | **1,031** |
| Class distribution | 469 BENIGN (45.5%) · 562 ATTACK (54.5%) |
| Development set | 824 examples |
| Test set | 207 examples |
| Split ratio | 80/20, stratified, seed = 42 |

**Preprocessing applied:**

1. Unicode NFKC normalization (preserves casing and punctuation)
2. Control character removal (`\x00`–`\x1F`, except `\n`, `\t`)
3. Conflicting label removal (same text, different labels)
4. Exact-duplicate removal

**Attack families covered:**

- Direct instruction override (`"Ignore previous instructions..."`)
- Role-play jailbreaks (`"Pretend you're a hacker..."`)
- Credential harvesting (`"Give me the admin password..."`)
- Meta-instruction attacks (`"Reveal your system prompt..."`)

---

## Installation

### Prerequisites

- Python 3.10+
- Google Colab with GPU (recommended: T4 or higher) for training
- Local machine for running the Streamlit app

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
torch>=2.0
transformers>=4.46,<5
accelerate>=0.34,<2
sentencepiece>=0.2
safetensors>=0.5
scikit-learn>=1.3
xgboost>=3.4,<4
pandas>=2.0
numpy>=1.24
joblib>=1.4
matplotlib>=3.9
streamlit>=1.30
```

---

## How to Run

### Step 1 — Prepare the Data (Colab, CPU)

Run `notebooks/01_data_preprocessing.ipynb`.

**Input:** `data/raw/dataset_prompt_injection - Dataset.csv`
**Output:** `data/processed/dev.csv`, `data/processed/test.csv`

### Step 2 — Train the TF-IDF + XGBoost Baseline (Colab, CPU)

Run `notebooks/02_train_tfidf_xgboost.ipynb`.

**Output:**

- `artifacts/tfidf_xgb.joblib`
- `artifacts/tfidf_xgb_oof.csv`
- `artifacts/tfidf_xgb_test.csv`

### Step 3 — Fine-Tune DeBERTa-v3 (Colab, GPU required)

Run `notebooks/03_train_transformer_deberta.ipynb`.

> ⚠️ **Runtime → Change runtime type → GPU (T4)**

**Output:**

- `artifacts/deberta_model/`
- `artifacts/deberta_oof.csv`
- `artifacts/deberta_test.csv`

**Training runs:** 3 OOF fold models + 1 final model = **4 fine-tuning runs total.**

### Step 4 — Train the Ensemble (Colab, CPU)

Run `notebooks/04_ensemble.ipynb`.

**Output:**

- `artifacts/ensemble_oof.csv`
- `artifacts/ensemble_test.csv`
- `artifacts/meta_logistic.joblib`
- `artifacts/metadata.json`

### Step 5 — Final Evaluation (Colab, CPU)

Run `notebooks/05_evaluation.ipynb`.

**Output:** Full metrics tables, ROC/PR curves, confusion matrices.

### Step 6 — Run the Streamlit App (Local)

```bash
# Copy artifacts into the app's models directory
cp -r artifacts/* streamlit_app/models/

# Launch the app
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Model Details

### DeBERTa-v3-base (Semantic Branch)

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/deberta-v3-base` |
| Layers | 12 |
| Max input length | 256 tokens |
| Task head | `AutoModelForSequenceClassification` (2 labels) |
| Learning rate | 1e-5 |
| Epochs | 2 |
| Train batch size | 4 |
| Gradient accumulation | 2 steps (effective batch = 8) |
| Optimizer | `adamw_torch` |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Max grad norm | 1.0 |
| **FP16 / BF16** | **Disabled** (T4 NaN fix) |
| Fine-tuning | Full (not LoRA/QLoRA) |

**Stability fix:** DeBERTa-v3's disentangled attention produces NaN logits under FP16 autocast on T4 GPUs. Mixed precision is disabled at *both* the `TrainingArguments` level (`fp16=False`) *and* the accelerate environment level (`ACCELERATE_MIXED_PRECISION="no"`).

### TF-IDF + XGBoost (Lexical Branch)

| Parameter | Value |
|-----------|-------|
| Analyzer | word |
| N-grams | 1–2 |
| min_df | 2 |
| max_df | 0.98 |
| Max features | 50,000 |
| Sublinear TF | Yes |
| XGBoost trees | 500 |
| Max depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.9 |
| Column sample | 0.9 |
| Reg. lambda | 1.0 |

**Leakage safety:** The vectorizer is refit **inside each OOF fold** so the validation fold's vocabulary and IDF weights do not influence the model that scores it.

### Fusion (Logistic Meta-Learner)

| Parameter | Value |
|-----------|-------|
| Model | LogisticRegression (L2, C=1.0) |
| Trained on | OOF predictions only |
| Learned weights | w_DeBERTa = 8.86 · w_TFIDF = 2.94 · b = −7.23 |

---

## Evaluation Protocol

The pipeline follows a **strict leakage-safe** evaluation:

1. **Test set is opened exactly once** — in notebook 05, for final reporting only.
2. **All model fitting, threshold selection, and meta-learner training** happen inside the development set.
3. **3-fold stratified OOF** — base models train on 2 folds, predict on the held-out fold.
4. **Meta-learner trains on OOF predictions only** — never on in-sample predictions.

### Reported Metrics

- Accuracy
- Precision (attack class)
- Recall (attack class)
- F1 (attack class)
- ROC-AUC
- PR-AUC
- Confusion matrix

---

## Deployment

The Streamlit app (`streamlit_app/app.py`) loads the following artifacts:

```
streamlit_app/models/
├── deberta_model/                ← fine-tuned DeBERTa
├── tfidf_xgb.joblib              ← TF-IDF vectorizer + XGBoost model
├── meta_logistic.joblib          ← logistic meta-learner
└── metadata.json                 ← thresholds, weights, max_length
```

### Inference Flow (per prompt)

1. Tokenize → DeBERTa → softmax → P_DeBERTa
2. Vectorize → XGBoost → P_TFIDF
3. Fuse (weighted vote or meta-learner)
4. Compare to threshold (0.5) → BENIGN or ATTACK

### Decision Source Switch

Change `metadata.json` → `decision_source` to one of:

- `"deberta"` — DeBERTa alone (default; best accuracy)
- `"weighted"` — 0.6 / 0.4 blend (best recall)
- `"meta"` — logistic meta-learner

---

## Known Limitations

| Limitation | Impact |
|------------|--------|
| **Small dataset** (1,031 examples) | Reported F1 should be treated as an upper bound |
| **Mild class imbalance** (54.5% attack) | Model may lean toward ATTACK on ambiguous inputs |
| **Single-turn only** | Multi-turn conversational drift not detected |
| **No indirect injection** | Malicious content in retrieved docs / tool outputs not covered |
| **No adversarial training** | Paraphrased, encoded, or cross-lingual variants may evade |
| **Rule layer removed** | Earlier rule-layer experiment (AUC 0.47) was a stub; properly tuned rules could help |

---

## Future Work

1. **Multi-turn detection** — aggregate per-turn probabilities to flag gradual drift.
2. **Adversarial augmentation** — paraphrased, back-translated, and encoded attack variants.
3. **Rule-layer rebuild** — tuned obfuscation engine (Base64, hex, zero-width characters).
4. **Cross-lingual** — evaluate XLM-RoBERTa on non-English injections.
5. **Calibration** — temperature scaling + human-review confidence band.
6. **Continual learning** — refit base models and meta-learner from production logs.

---

## References

1. Perez, F. & Ribeiro, I. (2022). *Ignore Previous Prompt: Attack Techniques for Language Models.* arXiv:2211.09527.
2. Greshake, K. et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* ACM AISec.
3. Wei, A., Haghtalab, N., & Steinhardt, J. (2023). *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS.
4. He, P., Liu, X., Gao, J., & Chen, W. (2021). *DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.* ICLR.
5. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD.
6. OWASP Foundation (2023). *OWASP Top 10 for Large Language Model Applications.*

---

## License

This project is developed for academic purposes as part of the Softcom course. All code and artifacts are the intellectual property of the listed authors.

*Last updated: 2026*
