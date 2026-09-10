import pytest
from src.agent.core import SupportAgent
from src.agent.schema import AgentRequest, AgentResponse, ActionType

def test_support_agent_process_schema():
    agent = SupportAgent()
    req = AgentRequest(message="Where is my delivery tracking number?", top_k=2)
    resp = agent.process(req)
    
    assert isinstance(resp, AgentResponse)
    assert resp.intent == "ORDER_TRACKING_AND_DELIVERY"
    assert resp.confidence > 0.0
    assert resp.action in [ActionType.AUTO_HANDLE, ActionType.ESCALATE]
    assert len(resp.reply) > 0
    assert len(resp.reason) > 0
    assert isinstance(resp.evidence, list)
    
    # Test dictionary export
    resp_dict = resp.to_dict()
    assert "intent" in resp_dict
    assert "action" in resp_dict
    assert "reply" in resp_dict
    assert "evidence" in resp_dict
