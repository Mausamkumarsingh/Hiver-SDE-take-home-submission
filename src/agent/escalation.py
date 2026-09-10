from __future__ import annotations
import re
from typing import Tuple, List, Optional
from src.agent.schema import ActionType
from src.intent.taxonomy import IntentName

class EscalationEngine:
    """
    Multi-factor enterprise escalation policy engine.
    Evaluates intent risk, classifier confidence, retrieval grounding quality,
    sentiment threat signals, and sensitive PII requirements.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        min_evidence_similarity: float = 0.55,
        threat_keywords: Optional[List[str]] = None,
        sensitive_intents: Optional[List[str]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.min_evidence_similarity = min_evidence_similarity
        
        self.threat_keywords = threat_keywords or [
            "sue", "lawyer", "attorney", "fraud", "police", "legal",
            "court", "scam", "stolen", "fraudulent", "speak to a human",
            "speak to an agent", "talk to a human", "real person",
            "supervisor", "manager", "disgusted", "unacceptable",
            "report you", "better business bureau", "attorney general"
        ]
        
        self.sensitive_intents = sensitive_intents or [
            IntentName.ACCOUNT_ACCESS_AND_SECURITY.value,
            IntentName.REFUND_AND_BILLING.value,
            IntentName.DAMAGED_OR_DEFECTIVE_ITEM.value,
            IntentName.CUSTOMER_SERVICE_COMPLAINT.value
        ]

    def evaluate(
        self,
        message: str,
        intent: str,
        confidence: float,
        evidence_similarities: List[float]
    ) -> Tuple[ActionType, str]:
        """
        Evaluates message and context against operational escalation criteria.
        Returns (ActionType, human_readable_reason).
        """
        text_lower = message.lower()
        
        # 1. Sentiment & Supervisor Escalation Check
        for kw in self.threat_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return (
                    ActionType.ESCALATE,
                    f"Customer expressed high urgency/dissatisfaction or requested human supervisor (keyword: '{kw}')."
                )
                
        # 2. Lost/Stolen Package Special Case
        if intent == IntentName.ORDER_TRACKING_AND_DELIVERY.value:
            delivery_threats = ["stolen", "missing from porch", "says delivered but", "never received", "marked delivered"]
            if any(term in text_lower for term in delivery_threats):
                return (
                    ActionType.ESCALATE,
                    "Package reported missing despite 'delivered' tracking status; requires carrier investigation."
                )

        # 3. Inherently Sensitive Financial or Security Intents
        if intent in self.sensitive_intents:
            return (
                ActionType.ESCALATE,
                f"Intent '{intent}' involves private credentials, financial transactions, or claims requiring human agent authorization."
            )

        # 4. Low Intent Classifier Confidence
        if confidence < self.confidence_threshold:
            return (
                ActionType.ESCALATE,
                f"Intent classification confidence ({confidence:.2f}) is below operational threshold ({self.confidence_threshold:.2f}); routing to human to prevent misguidance."
            )

        # 5. Insufficient Retrieval Grounding Evidence
        max_sim = max(evidence_similarities) if evidence_similarities else 0.0
        if max_sim < self.min_evidence_similarity:
            return (
                ActionType.ESCALATE,
                f"No strongly relevant historical resolution found (top similarity: {max_sim:.2f} < {self.min_evidence_similarity:.2f}); escalating to human specialist."
            )

        # Passed all safety checks: Eligible for automated self-service resolution
        return (
            ActionType.AUTO_HANDLE,
            f"High intent confidence ({confidence:.2f}) and grounded historical resolution available (similarity: {max_sim:.2f}); safe for automated self-service."
        )
