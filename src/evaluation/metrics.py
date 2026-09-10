from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict[str, Any]:
    """Computes comprehensive intent classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    unique_labels = labels or sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels
    }

def compute_escalation_metrics(y_true_actions: List[str], y_pred_actions: List[str]) -> Dict[str, Any]:
    """
    Computes operational escalation metrics including False Auto-Handling Rate
    and False Escalation Rate.
    """
    # Binary representation: ESCALATE = 1, AUTO_HANDLE = 0
    y_true_bin = [1 if a == "ESCALATE" else 0 for a in y_true_actions]
    y_pred_bin = [1 if a == "ESCALATE" else 0 for a in y_pred_actions]
    
    acc = accuracy_score(y_true_actions, y_pred_actions)
    
    # Escalation Precision, Recall, F1
    esc_prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    esc_rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    esc_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # Confusion counts
    tp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 1 and yp == 1) # True Escalate
    fp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 0 and yp == 1) # False Escalate
    tn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 0 and yp == 0) # True Auto-Handle
    fn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 1 and yp == 0) # False Auto-Handle
    
    total_true_escalate = tp + fn
    total_true_autohandle = tn + fp
    
    # Critical Operational Metrics:
    # 1. False Auto-Handling Rate = FN / (TP + FN) -> bot auto-handled when human escalation was needed (High Risk!)
    false_autohandle_rate = (fn / total_true_escalate) if total_true_escalate > 0 else 0.0
    
    # 2. False Escalation Rate = FP / (TN + FP) -> bot escalated when auto-handle was safe (Operational Inefficiency)
    false_escalation_rate = (fp / total_true_autohandle) if total_true_autohandle > 0 else 0.0
    
    return {
        "accuracy": round(float(acc), 4),
        "escalate_precision": round(float(esc_prec), 4),
        "escalate_recall": round(float(esc_rec), 4),
        "escalate_f1": round(float(esc_f1), 4),
        "false_auto_handle_rate": round(float(false_autohandle_rate), 4),
        "false_escalation_rate": round(float(false_escalation_rate), 4),
        "counts": {
            "true_positive_escalate": tp,
            "false_positive_escalate": fp,
            "true_negative_autohandle": tn,
            "false_negative_autohandle": fn
        }
    }

def format_confusion_matrix_markdown(cm: List[List[int]], labels: List[str]) -> str:
    """Formats confusion matrix into a clean markdown table."""
    short_labels = [l[:12] for l in labels]
    header = "| True \\ Pred | " + " | ".join(short_labels) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(short_labels)) + " |"
    lines = [header, divider]
    for idx, row in enumerate(cm):
        row_str = f"| **{short_labels[idx]}** | " + " | ".join(str(c) for c in row) + " |"
        lines.append(row_str)
    return "\n".join(lines)
