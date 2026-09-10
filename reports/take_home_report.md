# Comprehensive Technical Report: AI Customer Support Agent
**Candidate Role**: Hiver SDE Intern Take-Home Assignment  
**Target Brand**: AmazonHelp (`thoughtvector/customer-support-on-twitter`)  
**Evaluation Set**: 200 Hand-Verified Stratified Examples  

---

## 1. Problem Framing & Scope

### 1.1 What "Good" Means for AmazonHelp
For an enterprise customer support platform powering high-velocity shared inboxes (like Hiver), "good" cannot simply mean generating a superficially polite bot response or maximizing raw classification accuracy. "Good" is defined by four operational pillars:
1. **Safety First (Zero Dangerous Auto-Handling)**: The agent must never auto-handle queries involving account takeover, fraudulent charges, or lost/stolen merchandise with generic FAQs. Misclassifying an escalation request causes customer churn, brand liability, and regulatory risk.
2. **Strict Grounding (Zero Hallucination)**: Any factual claim, return window, troubleshooting step, or URL presented to the customer must be strictly anchored in verified historical resolutions.
3. **Actionable Resolution Speed**: For safe, high-confidence queries (e.g. general return procedures, tracking status links, Prime benefits), the system must provide instant, self-service resolution without human intervention.
4. **Deterministic Auditing**: Every routing decision (`AUTO_HANDLE` vs. `ESCALATE`) must produce a transparent, machine-verifiable operational reason.

### 1.2 What We Chose NOT to Build (Deliberate Non-Goals)
To maintain an interview-friendly, production-grade architecture without unnecessary complexity:
* **No Multi-Agent Frameworks or Graph Loops (LangGraph/CrewAI)**: We explicitly avoided complex multi-agent frameworks. Multi-agent loops introduce non-deterministic latency, high token consumption, and debugging opacity that cannot survive SLA requirements. A clean, modular pipeline is faster, cheaper, and more reliable.
* **No Dynamic Tool Execution / Write Actions**: The agent does not execute active account mutations (e.g., cancelling an order in the database or issuing a bank refund). In production customer service, write actions require strict authenticated agent permissions and dual approval.
* **No External Vector Database Microservices (Pinecone/Milvus/Qdrant)**: For a 10,000-resolution corpus, FAISS `IndexFlatIP` running embedded in-process delivers exact cosine similarity search in $< 1$ millisecond with zero external infrastructure overhead.

---

## 2. Empirical Brand Selection Analysis

From the 3-million-tweet Kaggle dataset (`twcs.csv`), we empirically benchmarked the top 5 candidate brands:

| Brand | Inbound Reply Pairs | English Pairs | English % | DM Deflection Rate | Resolution Link Rate | Avg Customer Length | Avg Brand Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AmazonHelp (Selected)** | **42,829** | **29,413** | **68.9%** | **0.6%** | **42.2%** | **131.8 chars** | **129.1 chars** |
| AppleSupport | 15,647 | 13,057 | 83.5% | 48.7% | 65.9% | 124.1 chars | 137.5 chars |
| Uber_Support | 10,362 | 9,219 | 89.0% | 34.2% | 52.5% | 130.8 chars | 110.2 chars |
| Delta | 7,358 | 5,856 | 79.7% | 17.2% | 14.8% | 119.2 chars | 108.1 chars |
| SpotifyCares | 6,414 | 5,309 | 82.9% | 31.7% | 48.0% | 119.2 chars | 132.9 chars |

### Why AmazonHelp Won
* **Substantive Resolutions vs. "DM Deflection"**: While AppleSupport has high volume, **48.7% of its replies simply tell the user to "send a DM with your iOS version"**, offering zero public resolution value. AmazonHelp has a **0.6% DM rate** and a **42.2% actionable link rate**.
* **Direct Domain Match for Hiver**: AmazonHelp represents e-commerce orders, delivery delays, returns, damaged items, subscriptions, and payment disputes—the exact customer support workflows managed in Hiver shared inboxes.

---

## 3. Discovered Intent Taxonomy & Golden Evaluation Set

We identified 10 mutually exclusive, collectively exhaustive (MECE) intents directly from AmazonHelp customer interactions:

```text
1. ORDER_TRACKING_AND_DELIVERY      6. ACCOUNT_ACCESS_AND_SECURITY
2. RETURNS_AND_EXCHANGES            7. SUBSCRIPTION_AND_PRIME
3. REFUND_AND_BILLING               8. DIGITAL_SERVICES_AND_DEVICES
4. DAMAGED_OR_DEFECTIVE_ITEM        9. PRODUCT_AVAILABILITY_AND_PRICING
5. CANCELLATION_REQUEST            10. CUSTOMER_SERVICE_COMPLAINT
```

### Golden Evaluation Set Construction & Audit Process
* **Dataset Scale**: Exactly **200 stratified real customer messages** sampled from TWCS (`data/golden_eval_set.csv`).
* **Manual Verification by Author**: Every single example was individually audited to eliminate cross-intent leakage and verify the ground-truth action (`AUTO_HANDLE` vs. `ESCALATE`).
* **Operational Balance**: 98 examples are ground-truth `AUTO_HANDLE` (49%) and 102 are `ESCALATE` (51%).
* **Honest Attribution**: All 200 items were curated and verified by the author according to strict operational criteria documented in [`reports/annotation_guidelines.md`](annotation_guidelines.md).

---

## 4. Benchmark Results vs. Baselines

We evaluated three complete systems on the 200-sample golden evaluation set:
1. **Majority-Class Baseline**: Predicts the most frequent class and default action (`AUTO_HANDLE`).
2. **TF-IDF + Logistic Regression Baseline**: Classical NLP benchmark with sublinear TF-IDF (unigrams + bigrams, 5000 features) and balanced multinomial logistic regression.
3. **AI SupportAgent (Our System)**: Dense sentence embeddings (`all-MiniLM-L6-v2`) + regularized linear probe + FAISS `IndexFlatIP` retrieval + multi-factor escalation policy.

### Headline Benchmark Comparison

| Model / System | Intent Accuracy | Intent Macro-F1 | Escalate Precision | Escalate Recall | Escalate F1 | False Auto-Handle Rate (Safety Risk) | False Escalate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 10.0% | 1.8% | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% |
| **TF-IDF + Logistic Regression** | 77.5% | 76.8% | 85.1% | 55.9% | 67.5% | **44.1%** | 10.2% |
| **AI SupportAgent (Ours)** | 77.0% | 77.3% | 71.0% | **91.2%** | **79.8%** | **8.8%** | 38.8% |

### Key Takeaway: The False Auto-Handling Revelation
While the classical TF-IDF model posted a comparable 77.5% intent accuracy, it exhibited a catastrophic **44.1% False Auto-Handling Rate**—meaning it failed to escalate 45 out of 102 urgent customer complaints! 
Our AI SupportAgent slashed this safety hazard down to **8.8%**, capturing **91.2% of all true escalations** (93 out of 102).

---

## 5. Failure Analysis: Top 5 Real Failure Modes

All failure cases below are extracted directly from our empirical prediction logs (`reports/golden_evaluation_predictions.csv`):

### 1. Compound Inquiries & Delivery-Refund Cannibalization
* **Customer Inquiry**: *"delivery I paid for today, didn’t arrive. why not? i paid enough for it. where is it?? I’m unhappy. refund the delivery charge"* (`eval_ref_001`)
* **True**: `REFUND_AND_BILLING` / `ESCALATE` | **Predicted**: `ORDER_TRACKING_AND_DELIVERY` / `AUTO_HANDLE`
* **Root Cause**: The message contains both a delivery delay and a refund demand. The classifier prioritized the tracking tokens, generating an automated tracking link rather than escalating for a shipping refund.

### 2. Sarcastic Churn Threats Cannibalized by Subscription Intent
* **Customer Inquiry**: *"Saying I preordered to early, terrible customer service. Won’t renew prime if this isn’t fixed."* (`eval_cus_001`)
* **True**: `CUSTOMER_SERVICE_COMPLAINT` / `ESCALATE` | **Predicted**: `SUBSCRIPTION_AND_PRIME` / `AUTO_HANDLE`
* **Root Cause**: The customer threatened churn (*"Won't renew prime"*). The retriever matched standard Prime renewal management, misrouting an angry customer to self-service.

### 3. Procedural Inquiries Overriding Physical Damage
* **Customer Inquiry**: *"Hi, I’ve purchased something via your app and it’s broken! How can I get it replaced?"* (`eval_dam_002`)
* **True**: `DAMAGED_OR_DEFECTIVE_ITEM` / `ESCALATE` | **Predicted**: `DAMAGED_OR_DEFECTIVE_ITEM` / `AUTO_HANDLE`
* **Root Cause**: Phrased as a procedural question (*"How can I get it replaced?"*), the query matched an automated Returns FAQ, failing to enforce human inspection required for broken items.

### 4. Colloquial Abbreviations & Polysemy ("OTP")
* **Customer Inquiry**: *"Spent tday otp w . I heart them as a customer but seller support is hair pullingly difficult"* (`eval_acc_002`)
* **True**: `CUSTOMER_SERVICE_COMPLAINT` | **Predicted**: `PRODUCT_AVAILABILITY_AND_PRICING` (conf: 0.31)
* **Root Cause**: The customer used Twitter slang *"otp"* (*"on the phone"*), which collided with *One-Time Password*. Low confidence (0.31) safely triggered escalation.

### 5. Rhetorical Hyperbole & Sarcasm
* **Customer Inquiry**: *"- It is really poor service or has your online chat support been hacked ?"* (`eval_acc_001`)
* **True**: `CUSTOMER_SERVICE_COMPLAINT` | **Predicted**: `ACCOUNT_ACCESS_AND_SECURITY` (conf: 0.31)
* **Root Cause**: The customer used hyperbole (*"has your support been hacked?"*) to insult service quality. Low confidence (0.31) safely forced escalation.

---

## 6. “What is Misleading About My Headline Number?”

1. **Accuracy Masks Catastrophic Safety Failures**:
   TF-IDF (77.5%) and our AI Agent (77.0%) have nearly identical accuracy. But TF-IDF missed 44.1% of escalations, whereas our agent achieved 91.2% escalation recall. Headline accuracy completely obscures this crucial operational safety difference.
2. **Stratified Balance vs. Real Traffic Skew**:
   Our 200-sample test set is balanced (20 samples per intent). In live Twitter traffic, delivery and returns account for >65% of volume, while Account Security accounts for <3%. A naive model optimized for overall accuracy on skewed traffic would achieve high headline numbers while failing on rare security events.
3. **Sample Size Uncertainty ($N = 200$)**:
   With 20 samples per intent, a single misclassification shifts intent recall by 5.0%. The 95% Wilson Score confidence interval for our 77.0% accuracy is $[70.8\%, 82.2\%]$.
4. **Automated Judge Leniency**:
   The automated LLM judge awarded an overall composite score of **4.68 / 5.0**. Hand auditing revealed that the LLM judge suffers from politeness bias, rating generic deflections as 4.5 even when a human rater scored them 3.0.

---

## 7. LLM-as-a-Judge & Human Validation Findings

We audited replies across 5 rubric dimensions on a 1.0–5.0 scale on 30 hand-audited cases:

| Evaluation Dimension | Human Mean | LLM Mean | Pearson $r$ | Spearman $\rho$ | Cohen\'s $\kappa$ | MAE | Disagreement Rate ($|\Delta| \ge 1.0$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Groundedness** | `4.57` | `4.65` | **0.928** | **0.963** | **0.645** | **0.083** | **0.0%** | **Strong Concordance** |
| **Safety** | `5.00` | `5.00` | 1.000* | 1.000* | 1.000* | **0.000** | **0.0%** | **Perfect Agreement** |
| **Brand Consistency** | `4.50` | `4.87` | N/A* | N/A* | 1.000* | **0.373** | **0.0%** | **Near Concordance** |
| **Correctness** | `4.17` | `4.50` | N/A* | N/A* | 0.000 | **1.000** | **33.3%** | **LLM Leniency Gap** |
| **Helpfulness** | `3.65` | `4.37` | **-0.117** | **0.060** | **0.316** | **0.717** | **33.3%** | **Weak Subjective Correlation** |
| **Composite Overall** | `4.38` | `4.68` | **-0.033** | **0.140** | **-0.263** | **0.336** | **3.3%** | **Supplementary Only** |

*(Note: `*` indicates zero-variance constant concordance where both raters assigned identical passing scores).*

> **Crucial Insight**: Our automated LLM judge is **not reliable enough to serve as a standalone universal quality metric**. Agreement is strong for Groundedness ($r = 0.928$) and Safety (100%), but weak for subjective Helpfulness. We treat automated LLM-judge metrics strictly as supplementary regression guardrails.

---

## 8. What We’d Do Next With One More Week

1. **Hierarchical Multi-Label Goal Parser**: Decouple active problem statements (e.g. refund request) from secondary contextual entities (e.g. delayed package).
2. **Contextual Thread Modeling**: Incorporate multi-turn conversation context to track customer sentiment across sequential replies.
3. **Active Learning Queue Expansion**: Use uncertainty sampling on production traffic to scale the verified golden evaluation set from 200 to 1,000 examples.
4. **Hiver Shared Inbox Webhook Receiver**: Deploy an event-driven webhook that directly tags incoming customer emails with `#auto-handled` or `#escalate-human`.
