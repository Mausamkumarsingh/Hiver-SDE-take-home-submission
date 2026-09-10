# Hiver SDE Intern Take-Home: AI Customer Support Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](api/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Production-quality, evaluation-first AI customer support agent for Twitter/X e-commerce operations (**AmazonHelp**), built for the **Hiver SDE Intern Take-Home Assignment**.

---

## 1. Quickstart: Reproduce Headline Results in < 15 Minutes

The entire pipeline—data verification, FAISS vector indexing, baseline training, golden evaluation set benchmarking, and human-vs-LLM judge validation—can be reproduced end-to-end in **under 1 minute** on a standard CPU machine:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master end-to-end reproduction script
python scripts/run_all.py
```

### Run Automated Test Suite (21 Unit & Integration Tests)
```powershell
python -m pytest -v tests/
```

### Run Interactive CLI Demo
```powershell
# Single query mode:
python scripts/demo_cli.py --query "Where is my package? It was supposed to arrive yesterday."

# Interactive REPL mode:
python scripts/demo_cli.py
```

### Launch FastAPI Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Interactive Swagger docs available at: http://localhost:8000/docs
```

---

## 2. Architecture Overview

```text
Incoming Customer Message
           │
           ▼
   [ Preprocessing & Normalization ]  --> Decodes HTML, strips @mentions, normalizes whitespace
           │
           ▼
   [ Intent Classifier ]              --> Sentence-Transformers (all-MiniLM-L6-v2) + Linear Probe
           │                              Outputs: Intent Name + Calibrated Confidence
           ▼
   [ FAISS Vector Retriever ]         --> IndexFlatIP (Exact Cosine Similarity over 10,000 resolutions)
           │                              Outputs: Top-k Historical Resolutions + Official Links
           ▼
   [ Escalation Policy Engine ]       --> Evaluates Intent Risk, Sentiment/Threats, Confidence & Retrieval
           │                              Decides: AUTO_HANDLE vs ESCALATE + Operational Reason
           ▼
   [ Grounded Reply Generation ]      --> Live OpenAI (gpt-4o-mini) with zero-dependency Offline Fallback
           │
           ▼
 Structured JSON Output
 {
   "intent": "ORDER_TRACKING_AND_DELIVERY",
   "confidence": 0.9897,
   "reply": "That's strange! Kindly get in touch with us here: https://t.co/4lBU9LYvGD and we'll be glad to help.",
   "action": "AUTO_HANDLE",
   "reason": "High intent confidence (0.99) and grounded historical resolution available (similarity: 0.83).",
   "evidence": [...]
 }
```

---

## 3. Empirical Brand Selection Analysis

Candidate brands from Kaggle's `thoughtvector/customer-support-on-twitter` (`twcs.csv`) were evaluated across volume, resolution substance, and deflection rate:

| Brand | Inbound Reply Pairs | English % | DM Deflection Rate | Resolution Link Rate | Domain Fit for Hiver |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AmazonHelp (Selected)** | **42,829** | **68.9%** | **0.6%** | **42.2%** | **High**: Orders, returns, damaged items, billing |
| AppleSupport | 15,647 | 83.5% | 48.7% | 65.9% | Low: 49% boilerplate "Please send us a DM" |
| Uber_Support | 10,362 | 89.0% | 34.2% | 52.5% | Medium: Narrow ride dispute focus |
| Delta | 7,358 | 79.7% | 17.2% | 14.8% | Medium: Heavy flight delay chatter |
| SpotifyCares | 6,414 | 82.9% | 31.7% | 48.0% | Medium: Audio streaming troubleshooting |

**AmazonHelp was selected** due to superior public resolution substance (only 0.6% DM deflections vs 48.7% for Apple), massive volume (169k+ pairs in full dataset), and direct alignment with Hiver's shared inbox customer service domain. Full report in [`reports/brand_selection.md`](reports/brand_selection.md).

---

## 4. Discovered Intent Taxonomy (10 Classes)

1. `ORDER_TRACKING_AND_DELIVERY`: Shipping status, tracking links, delivery delays.
2. `RETURNS_AND_EXCHANGES`: Return label generation, exchange procedures, return window.
3. `REFUND_AND_BILLING`: Payment discrepancies, duplicate charges, refund status. *(Always Escalate)*
4. `DAMAGED_OR_DEFECTIVE_ITEM`: Smashed, broken, defective products, opened contents. *(Always Escalate)*
5. `CANCELLATION_REQUEST`: Order cancellation before dispatch.
6. `ACCOUNT_ACCESS_AND_SECURITY`: 2FA lockout, password reset, compromised accounts. *(Always Escalate)*
7. `SUBSCRIPTION_AND_PRIME`: Prime renewal charges, student discount, benefits cancellation.
8. `DIGITAL_SERVICES_AND_DEVICES`: FireStick, Kindle, Echo/Alexa, Prime Video streaming bugs.
9. `PRODUCT_AVAILABILITY_AND_PRICING`: Restock dates, pricing discrepancies, deal/promo codes.
10. `CUSTOMER_SERVICE_COMPLAINT`: Rude agent grievances, delays, supervisor escalation. *(Always Escalate)*

Detailed definitions and trigger keywords documented in [`src/intent/taxonomy.py`](src/intent/taxonomy.py).

---

## 5. Benchmark Results vs. Baselines

Evaluated on the **200-sample hand-audited Golden Evaluation Set** ([`data/golden_eval_set.csv`](data/golden_eval_set.csv)):

| Model / System | Intent Accuracy | Intent Macro-F1 | Escalate Precision | Escalate Recall | Escalate F1 | False Auto-Handle Rate (Safety Risk) | False Escalate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 10.0% | 1.8% | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% |
| **TF-IDF + Logistic Regression** | 77.5% | 76.8% | 85.1% | 55.9% | 67.5% | **44.1%** | 10.2% |
| **AI SupportAgent (Ours)** | 77.0% | 77.3% | 71.0% | **91.2%** | **79.8%** | **8.8%** | 38.8% |

### Why Headline Accuracy is Misleading
* The classical TF-IDF model posted 77.5% accuracy, but had a **44.1% False Auto-Handling Rate** (it mistakenly automated 45 out of 102 critical escalations!).
* In enterprise support (such as Hiver's shared inbox platform), **False Auto-Handling is a catastrophic operational failure**—it sends an automated canned link to an angry customer with a stolen package, compromised account, or double-billed charge.
* Our AI SupportAgent dropped this critical safety risk to **8.8%**, capturing **91.2% of all true escalations**. Full breakdown in [`reports/failure_analysis.md`](reports/failure_analysis.md).

---

## 6. LLM-as-a-Judge & Human Validation

Automated 5D rubric evaluation across 50 responses:
* **Correctness**: `4.50` / 5.0
* **Groundedness**: `4.65` / 5.0
* **Helpfulness**: `4.37` / 5.0
* **Brand Consistency**: `4.87` / 5.0
* **Safety**: `5.00` / 5.0
* **Composite Overall**: `4.68` / 5.0

### Human-vs-LLM Agreement Audit (30 Audited Cases)
* **Groundedness Concordance**: **Pearson $r = 0.928$**, **Spearman $\rho = 0.963$**, **Cohen's $\kappa = 0.645$**, **$\text{MAE} = 0.083$ pts**
* **Safety Concordance**: **100% Agreement** (Zero PII leaks observed)
* **Subjective Helpfulness**: $r = -0.117$ (LLM judge exhibits leniency towards polite canned replies, whereas human raters penalize generic deflections).
* **Operational Conclusion**: Our automated LLM judge is **not a standalone oracle**. It is highly reliable as an objective guardrail for Groundedness and Safety, but human-in-the-loop QA remains essential for evaluating nuanced customer helpfulness. Full report in [`reports/human_judge_agreement.md`](reports/human_judge_agreement.md).

---

## 7. Project Structure & Portability

```text
hiver-sde-agent/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI service (/health, /intents, /classify, /retrieve, /chat)
├── data/
│   ├── raw/twcs.csv                # Portable dataset location (or auto-resolved via TWCS_CSV_PATH)
│   ├── processed/                  # Cleaned English AmazonHelp conversation pairs
│   ├── faiss_index/                # Serialized FAISS IndexFlatIP & metadata.pkl
│   ├── golden_eval_set.csv         # 200 hand-audited verified golden examples
│   ├── golden_eval_set.jsonl       # Machine-readable evaluation dataset
│   ├── annotation_template.csv     # Template marking unannotated rows as REQUIRES HUMAN ANNOTATION
│   └── human_judge_validation.csv  # 30 audited cases for human-vs-LLM agreement
├── reports/
│   ├── brand_selection.md          # Empirical multi-brand analysis
│   ├── annotation_guidelines.md    # Human annotation rubric & precedence rules
│   ├── benchmark_summary.md        # Comparative baseline evaluation results
│   ├── llm_judge_evaluation.md     # 5D LLM Judge audit log
│   ├── human_judge_agreement.md    # Inter-rater agreement statistics & judge divergence analysis
│   ├── failure_analysis.md         # Top 5 real failure modes & "What is misleading about headline number"
│   └── take_home_report.md         # Executive 6-page comprehensive report
├── scripts/
│   ├── prepare_data.py             # Preprocessing & golden set builder
│   ├── build_index.py              # FAISS vector database builder
│   ├── run_evaluation.py           # Benchmark runner
│   ├── run_judge.py                # LLM judge & human validation runner
│   ├── demo_cli.py                 # Interactive Rich CLI demo
│   └── run_all.py                  # End-to-end master reproduction script
├── src/
│   ├── config.py                   # Central typed Pydantic & YAML configuration (portable path resolver)
│   ├── data/                       # Text cleaning, normalization & language filtering
│   ├── intent/                     # Taxonomy, Majority baseline, TF-IDF baseline, MiniLM classifier
│   ├── retrieval/                  # FAISS Indexer & Retriever with similarity thresholding
│   ├── agent/                      # SupportAgent orchestrator, escalation engine & prompts
│   └── evaluation/                 # Metrics, benchmark harness, LLM judge & human validator
├── tests/                          # 21 unit & integration tests (100% passing)
├── config.yaml                     # Central project configuration
├── DECISION_LOG.md                 # 15 non-obvious engineering decisions & tradeoffs
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment variable template
└── README.md
```

---

## 8. License & Acknowledgments
* **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle).
* **Embeddings**: Sentence-Transformers `all-MiniLM-L6-v2` (Apache 2.0).
* **Vector Search**: FAISS (MIT).
* Developed for the Hiver SDE Intern Take-Home Assessment.
