from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

from src.config import config
from src.intent.taxonomy import ALL_INTENTS
from src.intent.baselines import MajorityClassBaseline, TfidfLogisticRegressionBaseline
from src.intent.classifier import EmbeddingIntentClassifier
from src.agent.core import SupportAgent
from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_escalation_metrics,
    format_confusion_matrix_markdown
)

class EvaluationHarness:
    """
    Automated benchmark harness comparing Majority-class baseline,
    TF-IDF + Logistic Regression, and the production AI SupportAgent.
    """
    def __init__(
        self,
        golden_set_path: str = "data/golden_eval_set.csv",
        training_corpus_path: str = "data/processed/amazon_conversations.csv"
    ):
        self.golden_set_path = golden_set_path
        self.training_corpus_path = training_corpus_path
        self.golden_df = pd.read_csv(golden_set_path)
        
        # Load training sample for baselines
        train_df = pd.read_csv(training_corpus_path)
        golden_ids = set(self.golden_df["customer_tweet_id"].tolist())
        self.train_pool = train_df[~train_df["customer_tweet_id"].isin(golden_ids)]
        
        # Baselines
        self.majority_baseline = MajorityClassBaseline()
        self.tfidf_baseline = TfidfLogisticRegressionBaseline()
        self.agent = SupportAgent()

    def train_baselines(self, max_train_samples: int = 4000):
        """Fits baseline models on non-overlapping historical training data."""
        print(f"Fitting baselines on training sample (max {max_train_samples})...")
        
        # Generate training pseudo-labels using heuristic alignment from taxonomy
        from src.data.golden_set_builder import heuristic_intent_score, determine_ground_truth_action
        from src.intent.taxonomy import IntentName
        
        train_sample = self.train_pool.sample(min(max_train_samples, len(self.train_pool)), random_state=42)
        
        train_texts = []
        train_intents = []
        train_actions = []
        
        for _, row in train_sample.iterrows():
            text = str(row["customer_text"])
            best_intent = None
            best_score = 0.0
            for iname in IntentName:
                s = heuristic_intent_score(text, iname)
                if s > best_score:
                    best_score = s
                    best_intent = iname
            if best_intent and best_score >= 2.0:
                act, _ = determine_ground_truth_action(best_intent, text)
                train_texts.append(text)
                train_intents.append(best_intent.value)
                train_actions.append(act)
                
        print(f"Prepared {len(train_texts)} high-confidence labeled training examples for baselines.")
        
        # 1. Fit Majority
        self.majority_baseline.fit(train_texts, train_intents, train_actions)
        
        # 2. Fit TF-IDF Logistic Regression
        self.tfidf_baseline.fit(train_texts, train_intents, train_actions)
        
        # 3. Fit Agent Classifier Linear Probe if needed
        if not self.agent.classifier.is_fitted:
            self.agent.classifier.fit(train_texts, train_intents)

    def run(self) -> Dict[str, Any]:
        """Runs comparative benchmark across all systems on the golden evaluation set."""
        print(f"\nRunning evaluation on {len(self.golden_df)} golden evaluation examples...")
        
        texts = self.golden_df["customer_text"].tolist()
        y_true_intents = self.golden_df["true_intent"].tolist()
        y_true_actions = self.golden_df["true_action"].tolist()
        
        # 1. Evaluate Majority Baseline
        maj_pred_intents = self.majority_baseline.predict_intent(texts)
        maj_pred_actions = self.majority_baseline.predict_action(texts)
        maj_intent_metrics = compute_intent_metrics(y_true_intents, maj_pred_intents, ALL_INTENTS)
        maj_esc_metrics = compute_escalation_metrics(y_true_actions, maj_pred_actions)
        
        # 2. Evaluate TF-IDF Baseline
        tfidf_pred_intents = self.tfidf_baseline.predict_intent(texts)
        tfidf_pred_actions = self.tfidf_baseline.predict_action(texts)
        tfidf_intent_metrics = compute_intent_metrics(y_true_intents, tfidf_pred_intents, ALL_INTENTS)
        tfidf_esc_metrics = compute_escalation_metrics(y_true_actions, tfidf_pred_actions)
        
        # 3. Evaluate Production AI SupportAgent
        agent_pred_intents = []
        agent_pred_actions = []
        agent_confidences = []
        agent_replies = []
        
        print("Evaluating SupportAgent pipeline...")
        for idx, text in enumerate(texts):
            resp = self.agent.process(text)
            agent_pred_intents.append(resp.intent)
            agent_pred_actions.append(resp.action.value)
            agent_confidences.append(resp.confidence)
            agent_replies.append(resp.reply)
            
        agent_intent_metrics = compute_intent_metrics(y_true_intents, agent_pred_intents, ALL_INTENTS)
        agent_esc_metrics = compute_escalation_metrics(y_true_actions, agent_pred_actions)
        
        # Store predictions back into DataFrame for inspection
        results_df = self.golden_df.copy()
        results_df["majority_pred_intent"] = maj_pred_intents
        results_df["tfidf_pred_intent"] = tfidf_pred_intents
        results_df["agent_pred_intent"] = agent_pred_intents
        results_df["agent_confidence"] = agent_confidences
        results_df["majority_pred_action"] = maj_pred_actions
        results_df["tfidf_pred_action"] = tfidf_pred_actions
        results_df["agent_pred_action"] = agent_pred_actions
        results_df["agent_reply"] = agent_replies
        
        results_df.to_csv("reports/golden_evaluation_predictions.csv", index=False, encoding="utf-8")
        
        benchmark_summary = {
            "dataset_size": len(self.golden_df),
            "num_intents": len(ALL_INTENTS),
            "systems": {
                "majority_baseline": {
                    "intent": maj_intent_metrics,
                    "escalation": maj_esc_metrics
                },
                "tfidf_logistic_regression": {
                    "intent": tfidf_intent_metrics,
                    "escalation": tfidf_esc_metrics
                },
                "support_agent": {
                    "intent": agent_intent_metrics,
                    "escalation": agent_esc_metrics
                }
            }
        }
        
        # Save JSON results
        with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)
            
        # Generate Markdown Summary
        self._write_summary_markdown(benchmark_summary)
        
        print("\nBenchmark completed successfully! Reports saved to reports/")
        return benchmark_summary

    def _write_summary_markdown(self, results: Dict[str, Any]):
        out_path = "reports/benchmark_summary.md"
        sys_res = results["systems"]
        
        lines = [
            "# AI Customer Support Agent Benchmark Report\n",
            f"**Evaluation Set**: {results['dataset_size']} stratified golden examples across {results['num_intents']} discovered intents.\n",
            "## 1. Headline Comparative Benchmark\n",
            "| Model / System | Intent Accuracy | Intent Macro-F1 | Escalate Precision | Escalate Recall | Escalate F1 | False Auto-Handle Rate (Risk) | False Escalate Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        row_map = {
            "Majority-Class Baseline": sys_res["majority_baseline"],
            "TF-IDF + Logistic Regression": sys_res["tfidf_logistic_regression"],
            "AI SupportAgent (MiniLM + RAG + Policy)": sys_res["support_agent"]
        }
        
        for name, data in row_map.items():
            im = data["intent"]
            em = data["escalation"]
            lines.append(
                f"| **{name}** | {im['accuracy']*100:.1f}% | {im['macro_f1']*100:.1f}% | "
                f"{em['escalate_precision']*100:.1f}% | {em['escalate_recall']*100:.1f}% | "
                f"{em['escalate_f1']*100:.1f}% | **{em['false_auto_handle_rate']*100:.1f}%** | "
                f"{em['false_escalation_rate']*100:.1f}% |"
            )
            
        lines.append("\n## 2. Per-Intent Performance (AI SupportAgent)\n")
        lines.append("| Intent | Precision | Recall | F1-Score | Support |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        
        rep = sys_res["support_agent"]["intent"]["classification_report"]
        for intent in ALL_INTENTS:
            if intent in rep:
                d = rep[intent]
                lines.append(f"| `{intent}` | {d['precision']*100:.1f}% | {d['recall']*100:.1f}% | {d['f1-score']*100:.1f}% | {int(d['support'])} |")
                
        lines.append("\n## 3. Confusion Matrix (AI SupportAgent)\n")
        cm = sys_res["support_agent"]["intent"]["confusion_matrix"]
        labels = sys_res["support_agent"]["intent"]["labels"]
        lines.append(format_confusion_matrix_markdown(cm, labels))
        lines.append("\n")
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
        print(f"Summary markdown written to {out_path}")
