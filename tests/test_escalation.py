import pytest
from src.agent.escalation import EscalationEngine
from src.agent.schema import ActionType
from src.intent.taxonomy import IntentName

def test_threat_keyword_escalation():
    engine = EscalationEngine()
    action, reason = engine.evaluate(
        message="This is fraud! I am contacting my lawyer and will sue you in court.",
        intent=IntentName.ORDER_TRACKING_AND_DELIVERY.value,
        confidence=0.95,
        evidence_similarities=[0.85]
    )
    assert action == ActionType.ESCALATE
    assert "lawyer" in reason or "sue" in reason

def test_sensitive_intent_escalation():
    engine = EscalationEngine()
    action, reason = engine.evaluate(
        message="Please help me reset my account 2FA password.",
        intent=IntentName.ACCOUNT_ACCESS_AND_SECURITY.value,
        confidence=0.92,
        evidence_similarities=[0.80]
    )
    assert action == ActionType.ESCALATE
    assert "credentials" in reason or "ACCOUNT_ACCESS_AND_SECURITY" in reason

def test_low_confidence_escalation():
    engine = EscalationEngine(confidence_threshold=0.65)
    action, reason = engine.evaluate(
        message="maybe order item thing",
        intent=IntentName.ORDER_TRACKING_AND_DELIVERY.value,
        confidence=0.42,
        evidence_similarities=[0.75]
    )
    assert action == ActionType.ESCALATE
    assert "below operational threshold" in reason

def test_safe_auto_handle():
    engine = EscalationEngine(confidence_threshold=0.65, min_evidence_similarity=0.55)
    action, reason = engine.evaluate(
        message="Where can I check the estimated delivery date for my order?",
        intent=IntentName.ORDER_TRACKING_AND_DELIVERY.value,
        confidence=0.92,
        evidence_similarities=[0.84]
    )
    assert action == ActionType.AUTO_HANDLE
    assert "safe for automated self-service" in reason
