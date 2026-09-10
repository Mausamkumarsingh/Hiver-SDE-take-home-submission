# Hiver SDE Intern Take-Home: AI Customer Support Agent

> Evaluation-first AI customer support agent for Twitter/X e-commerce support, built for the Hiver SDE Intern Take-Home Assignment.

## Project Overview

This project builds an AI customer-support agent for **AmazonHelp** using the Customer Support on Twitter dataset.

For every incoming customer message, the system:

1. Classifies the customer's intent.
2. Retrieves similar historical support conversations.
3. Generates a historically grounded response.
4. Decides whether to automatically handle the request or escalate it to a human.
5. Provides the reason and supporting historical evidence.

The primary design goal is **trustworthy automation**, rather than maximizing a single headline accuracy number.

---

## Quickstart

The complete pipeline is designed to reproduce the project's headline evaluation results in under 15 minutes after the dataset and API environment are configured.

### Install

    pip install -r requirements.txt

### Configure Environment

Create a `.env` file using `.env.example` and add your own OpenAI API key.

    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-4o-mini
    AGENT_PORT=8000
    LOG_LEVEL=INFO
    CONFIDENCE_THRESHOLD=0.65
    RETRIEVAL_SIMILARITY_THRESHOLD=0.55

**Never commit `.env` or expose your API key.**

### Run Complete Pipeline

    python scripts/run_all.py

### Run Tests

    python -m pytest -v tests/

### Run CLI Demo

    python scripts/demo_cli.py --query "Where is my package? It was supposed to arrive yesterday."

Interactive mode:

    python scripts/demo_cli.py

### Run FastAPI

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Swagger documentation:

    http://localhost:8000/docs

---

## Architecture

    Incoming Customer Message
              |
              v
    +-------------------------+
    | Preprocessing           |
    | & Normalization         |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | Intent Classifier       |
    | Sentence Transformers   |
    | + Linear Probe          |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | FAISS Retriever         |
    | Historical Resolutions  |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | Escalation Policy       |
    | Engine                  |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | Grounded Reply          |
    | Generation              |
    | OpenAI / Fallback       |
    +------------+------------+
                 |
                 v
          Structured Response

Example response:

    {
      "intent": "ORDER_TRACKING_AND_DELIVERY",
      "confidence": 0.9897,
      "reply": "...",
      "action": "AUTO_HANDLE",
      "reason": "...",
      "evidence": []
    }

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML Baselines | scikit-learn |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| LLM | OpenAI API |
| Structured Output | Pydantic |
| API | FastAPI |
| Testing | pytest |

The implementation intentionally avoids unnecessary frameworks and complex infrastructure so that the system remains easy to reproduce and explain.

---

## Dataset

Primary dataset:

**Customer Support on Twitter**

Kaggle:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The original dataset contains approximately 3 million tweets across multiple brands.

The full dataset is not committed to this repository.

Place the dataset locally under:

    data/raw/

or configure the dataset path using the project configuration.

---

## Brand Selection

Several brands were evaluated based on:

- conversation volume
- English-language coverage
- resolution substance
- DM deflection
- actionable resolution links
- relevance to customer-support workflows

The selected brand is:

### AmazonHelp

| Brand | Inbound Reply Pairs | English % | DM Deflection Rate | Resolution Link Rate |
|---|---:|---:|---:|---:|
| **AmazonHelp** | **42,829** | **68.9%** | **0.6%** | **42.2%** |
| AppleSupport | 15,647 | 83.5% | 48.7% | 65.9% |
| Uber_Support | 10,362 | 89.0% | 34.2% | 52.5% |
| Delta | 7,358 | 79.7% | 17.2% | 14.8% |
| SpotifyCares | 6,414 | 82.9% | 31.7% | 48.0% |

AmazonHelp was selected because its conversations contain substantial public resolution patterns and are closely aligned with common customer-support workflows such as orders, returns, billing, delivery, and damaged products.

Detailed analysis:

`reports/brand_selection.md`

---

## Intent Taxonomy

The system uses 10 intents derived from the selected brand's support interactions.

1. `ORDER_TRACKING_AND_DELIVERY` — Shipping status, tracking, and delivery delays.
2. `RETURNS_AND_EXCHANGES` — Return labels, exchange procedures, and return-related questions.
3. `REFUND_AND_BILLING` — Payment discrepancies, duplicate charges, and refund status.
4. `DAMAGED_OR_DEFECTIVE_ITEM` — Broken, damaged, defective, or opened products.
5. `CANCELLATION_REQUEST` — Requests to cancel an order.
6. `ACCOUNT_ACCESS_AND_SECURITY` — Account access, password, 2FA, or compromised-account issues.
7. `SUBSCRIPTION_AND_PRIME` — Prime-related charges, benefits, discounts, and cancellations.
8. `DIGITAL_SERVICES_AND_DEVICES` — Kindle, Fire TV, Echo/Alexa, Prime Video, and related technical issues.
9. `PRODUCT_AVAILABILITY_AND_PRICING` — Stock, pricing discrepancies, promotions, and deals.
10. `CUSTOMER_SERVICE_COMPLAINT` — Complaints about support quality, delays, or supervisor escalation.

Detailed definitions:

`src/intent/taxonomy.py`

---

## Historical Retrieval / RAG

The system uses:

- `all-MiniLM-L6-v2`
- FAISS `IndexFlatIP`
- cosine-similarity-based retrieval

Historical support resolutions are embedded and indexed.

For a new customer message:

    Customer Message
           |
           v
       Embedding
           |
           v
       FAISS Search
           |
           v
    Top-k Historical Cases
           |
           v
    Grounded Reply Generation

Retrieved cases are used as evidence rather than simply copying previous responses.

A retrieval similarity threshold is also applied to avoid using weak historical matches.

---

## Escalation Policy

The system distinguishes between:

**AUTO_HANDLE**

and

**ESCALATE**

Escalation can be triggered by:

- security-sensitive issues
- potentially unauthorized transactions
- damaged or high-risk cases
- customer requests for human support
- low intent confidence
- insufficient retrieval evidence
- conflicting or ambiguous requests

The system returns an operational reason for the decision.

The objective is to minimize unsafe automatic handling rather than maximize automation at any cost.

---

## Evaluation Methodology

Evaluation is performed using a **200-example Golden Evaluation Set**.

The evaluation set covers:

- common intents
- rare intents
- ambiguous messages
- short messages
- difficult customer requests
- escalation-sensitive cases

Annotation methodology and guidelines are documented in:

`reports/annotation_guidelines.md`

Evaluation data:

`data/golden_eval_set.csv`

---

## Baselines

Two baselines are implemented.

### Baseline 1 — Majority Class

Always predicts the most frequent intent.

### Baseline 2 — TF-IDF + Logistic Regression

A traditional text-classification baseline:

    TF-IDF
       |
       v
    Logistic Regression

These baselines provide reference points for evaluating whether the AI system provides meaningful improvement.

---

## Benchmark Results

Evaluation results on the Golden Evaluation Set:

| Model / System | Intent Accuracy | Intent Macro-F1 | Escalate Precision | Escalate Recall | Escalate F1 | False Auto-Handle Rate | False Escalate Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority-Class Baseline | 10.0% | 1.8% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| TF-IDF + Logistic Regression | 77.5% | 76.8% | 85.1% | 55.9% | 67.5% | 44.1% | 10.2% |
| **AI SupportAgent** | **77.0%** | **77.3%** | **71.0%** | **91.2%** | **79.8%** | **8.8%** | 38.8% |

---

## What Is Misleading About the Headline Number?

Intent accuracy alone does not adequately describe the quality of a support agent.

The TF-IDF baseline achieves **77.5% Intent Accuracy**, but has a **44.1% False Auto-Handle Rate**.

The AI SupportAgent achieves slightly lower intent accuracy at **77.0%**, but substantially improves escalation safety:

- **91.2% Escalation Recall**
- **8.8% False Auto-Handle Rate**

For customer support, incorrectly automating a sensitive case can be more costly than unnecessarily escalating a routine case.

Therefore, headline accuracy should not be interpreted as the overall quality or safety of the system.

Other limitations include:

- class imbalance
- rare-intent performance
- limited golden-set size
- dataset noise
- ambiguity in customer messages
- limitations of automated judging

---

## LLM-as-a-Judge

Generated replies are evaluated using a five-dimensional rubric:

- Correctness
- Groundedness
- Helpfulness
- Brand Consistency
- Safety

Current automated judge results:

| Dimension | Score |
|---|---:|
| Correctness | 4.50 / 5 |
| Groundedness | 4.65 / 5 |
| Helpfulness | 4.37 / 5 |
| Brand Consistency | 4.87 / 5 |
| Safety | 5.00 / 5 |
| Composite | 4.68 / 5 |

These scores are treated as evaluation signals rather than ground truth.

---

## Human vs LLM Judge Validation

A subset of responses was independently reviewed to evaluate agreement between human assessment and the LLM judge.

Results:

- Groundedness Pearson correlation: **0.928**
- Groundedness Spearman correlation: **0.963**
- Groundedness Cohen's kappa: **0.645**
- Groundedness MAE: **0.083**
- Safety agreement: **100%**
- Subjective helpfulness correlation: **-0.117**

The results suggest that the LLM judge is more reliable for objective properties such as groundedness and safety than for subjective helpfulness.

Therefore:

> **The LLM judge is treated as a supplementary evaluation tool, not as an authoritative replacement for human QA.**

Detailed analysis:

`reports/human_judge_agreement.md`

---

## Failure Analysis

The project includes detailed failure analysis covering:

- ambiguous customer messages
- short or context-dependent messages
- multi-intent requests
- weak historical retrieval
- incorrect escalation
- unsupported or overly generic responses

For each failure mode, the report documents:

1. Example
2. Expected behavior
3. Actual behavior
4. Hypothesis for failure
5. Potential improvement

See:

`reports/failure_analysis.md`

---

## Project Structure

    hiver-sde-agent/
    |
    ├── api/
    |   └── main.py
    |
    ├── data/
    |   ├── raw/
    |   ├── processed/
    |   ├── faiss_index/
    |   ├── golden_eval_set.csv
    |   ├── golden_eval_set.jsonl
    |   ├── annotation_template.csv
    |   └── human_judge_validation.csv
    |
    ├── reports/
    |   ├── brand_selection.md
    |   ├── annotation_guidelines.md
    |   ├── benchmark_summary.md
    |   ├── llm_judge_evaluation.md
    |   ├── human_judge_agreement.md
    |   ├── failure_analysis.md
    |   └── take_home_report.md
    |
    ├── scripts/
    |   ├── prepare_data.py
    |   ├── build_index.py
    |   ├── run_evaluation.py
    |   ├── run_judge.py
    |   ├── demo_cli.py
    |   └── run_all.py
    |
    ├── src/
    |   ├── data/
    |   ├── intent/
    |   ├── retrieval/
    |   ├── agent/
    |   └── evaluation/
    |
    ├── tests/
    |
    ├── config.yaml
    ├── DECISION_LOG.md
    ├── requirements.txt
    ├── .env.example
    └── README.md

---

## Testing

The project includes unit and integration tests covering:

- preprocessing
- intent classification
- retrieval
- escalation logic
- response validation
- configuration
- API behavior

Run:

    python -m pytest -v tests/

Current test suite:

**21 tests passing**

---

## API

Start the FastAPI service:

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Swagger:

`http://localhost:8000/docs`

The API returns structured agent outputs containing:

- intent
- confidence
- response
- action
- escalation reason
- historical evidence

---

## CLI Demo

Run:

    python scripts/demo_cli.py

Example input:

    Where is my package? It was supposed to arrive yesterday.

The system returns:

    Intent
    Confidence
    Suggested Reply
    AUTO_HANDLE / ESCALATE
    Reason
    Historical Evidence

---

## Reproducibility

The project uses configurable parameters for:

- LLM model
- embedding model
- retrieval top-k
- intent confidence threshold
- retrieval similarity threshold
- dataset path
- evaluation size

Configuration is available through:

`config.yaml`

and:

`.env`

No API keys are included in the repository.

---

## Decision Log

The project contains a 15-item decision log documenting important engineering choices and trade-offs.

See:

`DECISION_LOG.md`

Topics include:

- brand selection
- intent taxonomy
- evaluation sampling
- retrieval architecture
- embedding model
- escalation policy
- confidence thresholds
- LLM judge
- leakage prevention
- failure handling

---

## What I Would Do With One More Week

With one additional week, I would prioritize:

1. Increase the size and diversity of the human-labelled evaluation set.
2. Improve multi-intent classification.
3. Calibrate intent and retrieval confidence thresholds.
4. Improve retrieval using conversation-level context.
5. Add more rigorous human evaluation of reply helpfulness.
6. Investigate false escalations to increase safe automation without increasing false auto-handling.
7. Evaluate additional embedding models.
8. Add automated regression tests for previously observed failure cases.

---

## Limitations

This is an offline research/prototype system and does not perform real customer account actions.

It does not:

- access private customer accounts
- process real refunds
- modify orders
- communicate directly with Twitter/X
- guarantee real-world resolution
- replace human support agents

The historical Twitter dataset is noisy and may contain incomplete conversations or outdated support practices.

Retrieved historical conversations should therefore be treated as evidence rather than authoritative policy.

---

## Acknowledgements

### Dataset

Customer Support on Twitter:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

### Embeddings

Sentence Transformers:

`all-MiniLM-L6-v2`

### Vector Search

FAISS

### LLM

OpenAI API

---

## License

MIT
