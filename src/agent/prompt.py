from __future__ import annotations
from typing import List, Dict, Any
from src.agent.schema import ActionType

SYSTEM_PROMPT = """You are an expert AI Customer Support Agent for AmazonHelp on Twitter/X, deployed via Hiver.
Your mission is to generate empathetic, highly concise, grounded, and actionable customer support replies.

STRICT OPERATIONAL RULES:
1. GROUNDEDNESS: Ground your instructions strictly in the provided historical resolutions and official support links. Do NOT invent policies, fake URLs, order IDs, or promises of refunds.
2. ACTION CONFORMANCE:
   - If the action is AUTO_HANDLE: Provide direct, clear troubleshooting or self-service instructions, including the official support link from the evidence.
   - If the action is ESCALATE: Politely acknowledge the customer's issue, reassure them that a specialist is reviewing their case, and explain the next step or link to real-time support.
3. BRAND VOICE: Professional, empathetic, concise, and direct (under 280 characters if possible, similar to Twitter/X support standards).
4. SAFETY: Never request customer passwords, full credit card numbers, or sensitive PII over public channels.
"""

def construct_agent_prompt(
    customer_message: str,
    intent: str,
    confidence: float,
    action: ActionType,
    reason: str,
    evidence: List[Dict[str, Any]]
) -> str:
    """Builds user prompt for LLM reply generation."""
    evidence_text = ""
    if evidence:
        for idx, item in enumerate(evidence, 1):
            evidence_text += f"\n--- Evidence #{idx} (Similarity: {item.get('similarity', 0.0):.2f}) ---\n"
            evidence_text += f"Historical Query: {item.get('query', '')}\n"
            evidence_text += f"Historical Resolution: {item.get('resolution', '')}\n"
            links = item.get('links', [])
            if links:
                evidence_text += f"Official Links: {', '.join(links)}\n"
    else:
        evidence_text = "No strong historical matches found.\n"

    prompt = f"""Customer Inquiry:
"{customer_message}"

Operational Context:
- Classified Intent: {intent} (Confidence: {confidence:.2f})
- Operational Action: {action.value}
- Decision Reason: {reason}

Historical Support Resolutions (Evidence):
{evidence_text}

Generate the customer support response following the operational rules:
"""
    return prompt
