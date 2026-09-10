# Failure Analysis: Real-World Error Taxonomy and Metric Dissection

## 1. Overview
This document examines the empirical limitations, error modes, and edge cases of the Hiver AI Customer Support Agent.
All failure modes documented below are grounded in **actual misclassifications from the 200-sample Golden Evaluation Set** 
evaluated during the fresh benchmark run.

---

## 2. Top 5 Real-World Failure Modes

### Failure Mode 1: Compound Inquiries & Delivery-Refund Cannibalization
* **Customer Inquiry**:
  > *"delivery I paid for today, didn’t arrive. why not? i paid enough for it. where is it?? I’m unhappy. refund the delivery charge"* (`eval_ref_001`)
* **Ground Truth**: `REFUND_AND_BILLING` | `ESCALATE`
* **Agent Prediction**: `ORDER_TRACKING_AND_DELIVERY` (Confidence: 0.74) | `AUTO_HANDLE`
* **Root Cause**: The message features multiple overlapping signals: a delivery delay (*"didn't arrive"*, *"where is it"*) combined with an explicit financial demand (*"refund the delivery charge"*). The embedding classifier gave higher weight to the delivery tracking terms, misrouting the ticket to self-service tracking and auto-handling it without addressing the paid shipping refund demand.
* **Operational Impact**: High Safety Risk (False Auto-Handle). An unhappy customer demanding a refund is rebuffed with an automated order tracking link.
* **Mitigation**: Introduce a multi-intent goal parser where any explicit monetary keyword (*"refund"*, *"charge"*, *"reimburse"*) takes immediate routing precedence over delivery status.

---

### Failure Mode 2: Colloquial Abbreviations & Acronym Polysemy ("OTP")
* **Customer Inquiry**:
  > *"Spent tday otp w . I heart them as a customer but seller support is hair pullingly difficult"* (`eval_acc_002`)
* **Ground Truth**: `CUSTOMER_SERVICE_COMPLAINT` | `ESCALATE`
* **Agent Prediction**: `PRODUCT_AVAILABILITY_AND_PRICING` (Confidence: 0.31) | `ESCALATE`
* **Root Cause**: In customer service taxonomy, *"OTP"* is standard shorthand for *One-Time Password* (triggering Account Security). However, the customer used Twitter slang: *"otp"* = *"on the phone"*. The embedding model was uncertain (confidence 0.31), leading to an intent misclassification.
* **Operational Impact**: Neutral (Mitigated by Policy). Because confidence was 0.31 (< 0.65 threshold), the escalation policy forced human escalation anyway.
* **Mitigation**: Slang normalization dictionary in preprocessing (`otp` followed by `with` / `on` -> `on the phone`).

---

### Failure Mode 3: Sarcastic Churn Threat Cannibalized by Subscription Intent
* **Customer Inquiry**:
  > *"Saying I preordered to early, terrible customer service. Won’t renew prime if this isn’t fixed."* (`eval_cus_001`)
* **Ground Truth**: `CUSTOMER_SERVICE_COMPLAINT` | `ESCALATE`
* **Agent Prediction**: `SUBSCRIPTION_AND_PRIME` (Confidence: 0.68) | `AUTO_HANDLE`
* **Root Cause**: The customer threatened to cancel their Prime membership (*"Won't renew prime"*) due to terrible support. The dense retriever latched onto *"renew prime"* and classified the intent as Prime Subscription, allowing automated self-service.
* **Operational Impact**: Severe Safety Failure (False Auto-Handle). An escalated complaint with explicit churn risk received a canned FAQ on managing Prime renewals.
* **Mitigation**: Sentiment and churn threat detection head: any pattern like *"won't renew"*, *"switching to"*, or *"cancelling my subscription because"* triggers an immediate supervisor lock.

---

### Failure Mode 4: False Auto-Handling on Procedural Physical Breakage
* **Customer Inquiry**:
  > *"Hi , I’ve purchased something via your app and it’s broken! How can I get it replaced?"* (`eval_dam_002`)
* **Ground Truth**: `DAMAGED_OR_DEFECTIVE_ITEM` | `ESCALATE` (Requires damage claim / courier RMA inspection)
* **Agent Prediction**: `DAMAGED_OR_DEFECTIVE_ITEM` | `AUTO_HANDLE`
* **Root Cause**: The customer asked a procedural question (*"How can I get it replaced?"*). The vector retriever retrieved a helpful historical resolution explaining how to select 'Damaged' in the Online Returns Center, so the agent auto-handled.
* **Operational Impact**: Operational policy violation. Although polite, Amazon support policy for broken merchandise requires courier verification or photo inspection to prevent replacement fraud.
* **Mitigation**: Hard policy lock: all `DAMAGED_OR_DEFECTIVE_ITEM` tickets must default to `ESCALATE` regardless of customer politeness.

---

### Failure Mode 5: Rhetorical Sarcasm & Metaphorical Expressions
* **Customer Inquiry**:
  > *"- It is really poor service or has your online chat support been hacked ?"* (`eval_acc_001`)
* **Ground Truth**: `CUSTOMER_SERVICE_COMPLAINT` | `ESCALATE`
* **Agent Prediction**: `ACCOUNT_ACCESS_AND_SECURITY` (Confidence: 0.31) | `ESCALATE`
* **Root Cause**: The customer used hyperbole (*"has your support been hacked?"*) to insult the quality of the chat service. The lexical pattern *"hacked"* triggered the security taxonomy.
* **Operational Impact**: Low (Safely Escalated). The low confidence score (0.31) triggered the escalation engine, routing to human support.

---

## 3. “What is Misleading About My Headline Number?”

In technical customer support automation, headline metrics (e.g., *"77.0% Intent Accuracy"*, *"77.5% Baseline Accuracy"*) can create a dangerous illusion of operational readiness. Here is why the headline numbers are misleading:

### 1. The Accuracy vs. Safety Illusion (The False Auto-Handling Trap)
* Looking at raw intent accuracy alone, **TF-IDF (77.5%)** and the **AI SupportAgent (77.0%)** appear virtually identical.
* However, looking at the **False Auto-Handling Rate** reveals a critical divergence:
  - **TF-IDF Baseline False Auto-Handle Rate: 44.1%** (It missed 45 out of 102 critical escalations!).
  - **AI SupportAgent False Auto-Handle Rate: 8.8%** (Escalation Recall: 91.2%).
* In enterprise customer service (such as Hiver's shared inbox platform), **False Auto-Handling is a catastrophic failure mode**. Rebuffing an angry customer whose credit card was compromised or who received shattered merchandise with an automated bot reply causes immediate churn, chargebacks, and executive escalations.
* The AI SupportAgent intentionally accepts lower deflection efficiency (38.8% false escalation rate) to guarantee customer safety. Headline accuracy completely obscures this life-or-death tradeoff.

### 2. Stratified Golden Set vs. Natural Class Imbalance
* Our 200-sample Golden Set is **artificially balanced** with exactly 20 examples per intent to guarantee statistical representation of rare intents (`ACCOUNT_ACCESS_AND_SECURITY`, `CUSTOMER_SERVICE_COMPLAINT`).
* In live Twitter traffic, `ORDER_TRACKING_AND_DELIVERY` represents **over 45%** of inbound volume, while Account Security represents **under 3%**.
* A naive model that simply memorizes delivery phrases would show inflated accuracy in production while catastrophically failing on rare security events.

### 3. Sample Size Uncertainty ($N = 200$)
* In a 200-sample test set with 20 examples per class, **a single customer message shifts an intent's recall by 5.0%**.
* The 95% Wilson Score confidence interval for our 77.0% accuracy is $[70.8\%, 82.2\%]$. Headline numbers without confidence bounds overstate precision.

### 4. Automated Judge Leniency and Confirmation Bias
* The automated LLM judge awarded an overall composite score of **4.68 / 5.0**.
* Hand validation revealed that the LLM judge suffers from a **politeness and URL sycophancy bias**: it routinely awards 4.5 even when a reply gave a generic deflection that failed to solve the user's specific problem.
* Automated judge scores must never be trusted in isolation without an audited human validation baseline.
