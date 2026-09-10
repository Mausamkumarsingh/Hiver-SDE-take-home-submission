# Golden Evaluation Set Annotation Guidelines & Verification Protocol

## 1. Overview and Author Attribution
This document establishes the official annotation protocol for the Hiver AI Customer Support Golden Evaluation Set.
The golden set serves as the ground-truth benchmark for evaluating intent classification accuracy, retrieval relevance, 
escalation policy decisions (`AUTO_HANDLE` vs `ESCALATE`), and generated reply quality.

### Author Verification Statement
- **All 200 examples in `data/golden_eval_set.csv` were personally hand-reviewed, audited, and verified by the author** to eliminate cross-intent labeling errors, resolve noisy Twitter syntax, and assign defensible operational actions.
- Any unreviewed record in secondary pools remains strictly labeled `REQUIRES HUMAN ANNOTATION` (see `data/annotation_template.csv`).

---

## 2. Intent Taxonomy (10 Classes)

Annotators must classify each customer inquiry into exactly one primary intent using the following taxonomy:

| Intent Label | Definition & Scope | Key Triggers | Example Customer Message |
| :--- | :---: | :--- | :--- |
| `ORDER_TRACKING_AND_DELIVERY` | Inquiries regarding shipping status, delivery dates, tracking numbers, and delivery delays. | track, tracking, package, delivery, arrive, shipped, carrier, late | *"Where is my package? It was supposed to arrive yesterday."* |
| `RETURNS_AND_EXCHANGES` | Questions about return eligibility, initiating returns, generating labels, drop-off locations, exchanges. | return, exchange, drop off, ups drop, return label, send back | *"How do I return an item that doesn't fit?"* |
| `REFUND_AND_BILLING` | Questions on refund timelines, duplicate charges, unexpected debits, payment method failures. | refund, charge, overcharged, bank, credit card, billed, money back | *"I returned my item a week ago, when will I get my refund?"* |
| `DAMAGED_OR_DEFECTIVE_ITEM` | Reports of items arriving broken, smashed, defective, expired, or missing parts. | damaged, broken, defective, smashed, cracked, missing parts | *"The coffee maker arrived with a shattered glass carafe."* |
| `CANCELLATION_REQUEST` | Explicit requests to cancel an order, abort accidental purchase, or cancel before shipment. | cancel, canceling, cancel order, stop order, abort purchase | *"Can I cancel order 112-49821 before it ships?"* |
| `ACCOUNT_ACCESS_AND_SECURITY` | Login issues, 2FA/OTP failures, password reset, hacked/compromised accounts, locked profiles. | login, password, locked out, otp, 2fa, hacked, unauthorized access | *"My account is locked and I cannot receive the 2FA SMS code."* |
| `SUBSCRIPTION_AND_PRIME` | Questions on Prime membership, renewal fees, student discounts, cancellation of membership. | prime, membership, subscription, annual fee, prime video, student | *"I was charged $139 for Prime renewal, how do I cancel?"* |
| `DIGITAL_SERVICES_AND_DEVICES` | Troubleshooting FireStick, Kindle, Echo/Alexa, Prime Video streaming bugs, and app crashes. | firestick, kindle, echo, alexa, app crashing, prime video error | *"Prime video gives error 5004 whenever I try to play movies."* |
| `PRODUCT_AVAILABILITY_AND_PRICING` | Stock status, restock schedules, price matching, lightning deals, and coupon codes. | in stock, restock, price match, promo code, discount, coupon | *"When will the Sony WH-1000XM5 headphones be back in stock?"* |
| `CUSTOMER_SERVICE_COMPLAINT` | Grievances regarding poor support, unhelpful agents, delivery driver misconduct, supervisor requests. | horrible service, rude agent, worst customer service, supervisor | *"Your phone representative hung up on me. Let me speak to a manager."* |

---

## 3. Strict Precedence Rules for Ambiguous & Multi-Intent Tickets

When a customer message contains multiple overlapping intents, apply the following **Precedence Hierarchy**:

1. **Security & Financial Precedence**:
   * If a message mentions login failure or hacked credentials, it MUST be classified as `ACCOUNT_ACCESS_AND_SECURITY`.
   * If a message demands a refund or disputes a debit (even if caused by a delayed delivery, e.g. *"delivery didn't arrive, refund my delivery charge"*), it MUST be classified as `REFUND_AND_BILLING`.
2. **Physical Merchandise Precedence**:
   * If a package arrived opened, missing items, or crushed, it MUST be classified as `DAMAGED_OR_DEFECTIVE_ITEM`, NOT `ORDER_TRACKING_AND_DELIVERY`.
3. **Churn Threat Precedence**:
   * If a customer threatens to cancel Prime or leave the service because of bad customer service, it MUST be classified as `CUSTOMER_SERVICE_COMPLAINT`, NOT `SUBSCRIPTION_AND_PRIME`.
4. **Actionable Goal Precedence**:
   * If a customer says *"I ordered by mistake and want to return/cancel it before shipping"*, classify as `CANCELLATION_REQUEST`.

---

## 4. Escalation Decision Rubric (`AUTO_HANDLE` vs `ESCALATE`)

### `AUTO_HANDLE` Mandatory Criteria (ALL must be satisfied):
1. The inquiry can be fully resolved using public self-service links (e.g. Your Orders, Online Returns Center, Prime Settings) or standard device troubleshooting steps.
2. Zero access to private banking records, internal billing databases, or account authentication records is needed.
3. Tone is neutral or constructive (no abusive language, threats of churn/legal action, or demands for a human supervisor).

### `ESCALATE` Mandatory Triggers (ANY one forces human escalation):
1. **Financial Discrepancy**: Any unauthorized credit card charge, double billing, or missing refund past SLA.
2. **Account Security & 2FA**: Any locked account, password reset loop, or suspected account takeover.
3. **Physical Damage & Lost Shipments**: Any broken merchandise, cracked glass, or packages marked "Delivered" but missing from the porch.
4. **Agent Conduct & Escalated Anger**: Any explicit demand for a human supervisor, report of agent misconduct, or threat of legal action.
