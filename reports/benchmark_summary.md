# AI Customer Support Agent Benchmark Report

**Evaluation Set**: 200 stratified golden examples across 10 discovered intents.

## 1. Headline Comparative Benchmark

| Model / System | Intent Accuracy | Intent Macro-F1 | Escalate Precision | Escalate Recall | Escalate F1 | False Auto-Handle Rate (Risk) | False Escalate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 10.0% | 1.8% | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% |
| **TF-IDF + Logistic Regression** | 77.5% | 76.8% | 85.1% | 55.9% | 67.5% | **44.1%** | 10.2% |
| **AI SupportAgent (MiniLM + RAG + Policy)** | 77.0% | 77.3% | 71.0% | 91.2% | 79.8% | **8.8%** | 38.8% |

## 2. Per-Intent Performance (AI SupportAgent)

| Intent | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `ORDER_TRACKING_AND_DELIVERY` | 47.2% | 85.0% | 60.7% | 20 |
| `RETURNS_AND_EXCHANGES` | 85.0% | 85.0% | 85.0% | 20 |
| `REFUND_AND_BILLING` | 67.9% | 95.0% | 79.2% | 20 |
| `DAMAGED_OR_DEFECTIVE_ITEM` | 100.0% | 70.0% | 82.4% | 20 |
| `CANCELLATION_REQUEST` | 84.2% | 80.0% | 82.1% | 20 |
| `ACCOUNT_ACCESS_AND_SECURITY` | 100.0% | 75.0% | 85.7% | 20 |
| `SUBSCRIPTION_AND_PRIME` | 72.0% | 90.0% | 80.0% | 20 |
| `DIGITAL_SERVICES_AND_DEVICES` | 100.0% | 90.0% | 94.7% | 20 |
| `PRODUCT_AVAILABILITY_AND_PRICING` | 91.7% | 55.0% | 68.8% | 20 |
| `CUSTOMER_SERVICE_COMPLAINT` | 69.2% | 45.0% | 54.5% | 20 |

## 3. Confusion Matrix (AI SupportAgent)

| True \ Pred | ORDER_TRACKI | RETURNS_AND_ | REFUND_AND_B | DAMAGED_OR_D | CANCELLATION | ACCOUNT_ACCE | SUBSCRIPTION | DIGITAL_SERV | PRODUCT_AVAI | CUSTOMER_SER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **ORDER_TRACKI** | 17 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **RETURNS_AND_** | 1 | 17 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **REFUND_AND_B** | 1 | 0 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **DAMAGED_OR_D** | 2 | 2 | 0 | 14 | 0 | 0 | 1 | 0 | 0 | 1 |
| **CANCELLATION** | 3 | 0 | 1 | 0 | 16 | 0 | 0 | 0 | 0 | 0 |
| **ACCOUNT_ACCE** | 1 | 0 | 0 | 0 | 1 | 15 | 0 | 0 | 1 | 2 |
| **SUBSCRIPTION** | 0 | 0 | 1 | 0 | 1 | 0 | 18 | 0 | 0 | 0 |
| **DIGITAL_SERV** | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 18 | 0 | 0 |
| **PRODUCT_AVAI** | 6 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 11 | 0 |
| **CUSTOMER_SER** | 4 | 0 | 3 | 0 | 1 | 0 | 3 | 0 | 0 | 9 |

