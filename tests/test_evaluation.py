import pytest
from src.evaluation.metrics import compute_intent_metrics, compute_escalation_metrics
from src.evaluation.llm_judge import LLMJudge, JudgeScore
from src.evaluation.human_validation import calculate_agreement_metrics

def test_compute_intent_metrics():
    y_true = ["INTENT_A", "INTENT_B", "INTENT_A", "INTENT_B"]
    y_pred = ["INTENT_A", "INTENT_B", "INTENT_B", "INTENT_B"]
    res = compute_intent_metrics(y_true, y_pred, labels=["INTENT_A", "INTENT_B"])
    
    assert res["accuracy"] == 0.75
    assert res["macro_f1"] > 0.0
    assert len(res["confusion_matrix"]) == 2

def test_compute_escalation_metrics_false_autohandle():
    # 2 True Escalate, 2 True Auto-Handle
    y_true = ["ESCALATE", "ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE"]
    # Model mistakenly auto-handles one of the escalations
    y_pred = ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE", "AUTO_HANDLE"]
    
    res = compute_escalation_metrics(y_true, y_pred)
    assert res["counts"]["false_negative_autohandle"] == 1
    # False auto handle rate = 1 / 2 = 0.50
    assert res["false_auto_handle_rate"] == 0.50

def test_llm_judge_offline_evaluation():
    judge = LLMJudge()
    score = judge.evaluate_reply(
        query="Where is my package?",
        intent="ORDER_TRACKING_AND_DELIVERY",
        action="AUTO_HANDLE",
        reply="You can track your package here: https://t.co/track",
        evidence_text="https://t.co/track"
    )
    assert isinstance(score, JudgeScore)
    assert 1.0 <= score.correctness <= 5.0
    assert score.safety == 5.0

def test_calculate_agreement_metrics():
    h = [4.5, 5.0, 3.0, 4.0, 5.0]
    l = [4.5, 4.5, 3.5, 4.0, 5.0]
    m = calculate_agreement_metrics(h, l)
    assert m["sample_size"] == 5
    assert m["mean_absolute_error"] < 0.5
    assert m["disagreement_rate"] == 0.0
