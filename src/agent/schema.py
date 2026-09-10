from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"

class EvidenceItemSchema(BaseModel):
    query: str
    resolution: str
    similarity: float
    links: List[str] = Field(default_factory=list)
    customer_tweet_id: Optional[int] = None
    brand_tweet_id: Optional[int] = None

class AgentRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    top_k: int = 3

class AgentResponse(BaseModel):
    intent: str
    confidence: float
    reply: str
    action: ActionType
    reason: str
    evidence: List[EvidenceItemSchema] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reply": self.reply,
            "action": self.action.value,
            "reason": self.reason,
            "evidence": [e.model_dump() for e in self.evidence]
        }
