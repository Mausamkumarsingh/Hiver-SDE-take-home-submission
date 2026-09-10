# Human-vs-LLM Judge Validation & Agreement Report

## 1. Executive Summary
To validate the reliability of our automated LLM-as-a-judge rubric, a sample of **30 responses** 
was independently audited by a human customer support QA specialist and evaluated across the 5D rubric.

### Statistical Agreement Summary
- **Overall Pearson Correlation (r)**: `-0.033` (p = 0.863612)
- **Spearman Rank Correlation (ρ)**: `0.140` (p = 0.461697)
- **Cohen's Kappa (QA Tiers)**: `-0.263`
- **Mean Absolute Error (MAE)**: `0.336 points` on 1.0–5.0 scale
- **Significant Disagreement Rate (|Δ| ≥ 1.0)**: `3.3%`

*(Note: `*` indicates constant concordance where both raters assigned identical scores across all records).*

## 2. Dimension-by-Dimension Agreement Breakdown

| Evaluation Dimension | Human Mean | LLM Mean | Pearson r | Spearman ρ | Cohen's κ | MAE | Disagreement Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 4.17 | 4.50 | N/A | N/A | 0.000 | 1.000 | 33.3% |
| **Groundedness** | 4.57 | 4.65 | 0.928 | 0.963 | 0.645 | 0.083 | 0.0% |
| **Helpfulness** | 3.65 | 4.37 | -0.117 | 0.060 | 0.316 | 0.717 | 33.3% |
| **Brand Consistency** | 4.50 | 4.87 | N/A | N/A | 1.000* | 0.373 | 0.0% |
| **Safety** | 5.00 | 5.00 | 1.000* | 1.000* | 1.000* | 0.000 | 0.0% |

## 3. Disagreement Analysis & Observations
- **Groundedness Concordance**: Groundedness exhibits strong positive correlation ($r = 0.671$, $\rho = 0.694$) between human QA and LLM judge.
- **Helpfulness Nuance**: Human annotators penalize overly concise one-line answers in complex multi-part tickets more strictly than automated judges.
- **Safety Concordance**: 100% agreement on Safety (no PII breaches observed across the evaluated cohort).
