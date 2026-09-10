from __future__ import annotations
import os
import re
from typing import Optional, List, Dict, Any
from src.config import config, AppConfig
from src.data.preprocessor import clean_text
from src.intent.classifier import EmbeddingIntentClassifier
from src.retrieval.retriever import FaissRetriever, EvidenceItem
from src.agent.escalation import EscalationEngine
from src.agent.schema import AgentRequest, AgentResponse, ActionType, EvidenceItemSchema
from src.agent.prompt import SYSTEM_PROMPT, construct_agent_prompt

class SupportAgent:
    """
    End-to-end Production AI Customer Support Agent pipeline for Hiver.
    Orchestrates Preprocessing -> Intent Classification -> FAISS Retrieval ->
    Escalation Policy -> Grounded Reply Generation.
    """
    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        retriever: Optional[FaissRetriever] = None,
        classifier: Optional[EmbeddingIntentClassifier] = None,
        escalation_engine: Optional[EscalationEngine] = None
    ):
        self.config = app_config or config
        
        # 1. Intent Classifier
        self.classifier = classifier or EmbeddingIntentClassifier(
            model_name=self.config.model.embedding_model
        )
        
        # 2. FAISS Retriever
        self.retriever = retriever or FaissRetriever(
            index_dir=self.config.data.faiss_index_dir,
            model_name=self.config.model.embedding_model,
            similarity_threshold=self.config.retrieval.similarity_threshold
        )
        
        # 3. Escalation Engine
        self.escalation_engine = escalation_engine or EscalationEngine(
            confidence_threshold=self.config.escalation.confidence_threshold,
            min_evidence_similarity=self.config.escalation.min_evidence_similarity,
            threat_keywords=self.config.escalation.sentiment_threat_keywords,
            sensitive_intents=self.config.escalation.sensitive_intents
        )
        
        # 4. LLM Client Initialization
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")

    def generate_reply(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
        action: ActionType,
        reason: str,
        evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Generates grounded reply using OpenAI API if available,
        or deterministic grounded synthesis template if offline.
        """
        # A. Live LLM Generation via OpenAI API
        if self.client:
            try:
                prompt = construct_agent_prompt(
                    customer_message=customer_message,
                    intent=intent,
                    confidence=confidence,
                    action=action,
                    reason=reason,
                    evidence=evidence
                )
                response = self.client.chat.completions.create(
                    model=self.config.model.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.model.temperature,
                    max_tokens=150
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                print(f"Notice: OpenAI call failed ({e}). Using offline grounded fallback.")

        # B. Robust Offline Grounded Generator (Guaranteed Reliability)
        return self._generate_grounded_offline_reply(
            customer_message=customer_message,
            intent=intent,
            action=action,
            evidence=evidence
        )

    def _generate_grounded_offline_reply(
        self,
        customer_message: str,
        intent: str,
        action: ActionType,
        evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministic, grounded reply generator synthesizing retrieved evidence
        and official links when running offline or without OpenAI credentials.
        """
        top_link = None
        top_res = None
        if evidence:
            top_res = evidence[0].get("resolution", "")
            links = evidence[0].get("links", [])
            if links:
                top_link = links[0]

        if action == ActionType.ESCALATE:
            if top_link:
                return (
                    f"I'm very sorry for the frustration. To ensure this is handled securely with your account details, "
                    f"our specialized support team is reviewing this. Please connect directly with an agent here: {top_link}"
                )
            return (
                "I apologize for the inconvenience. Due to the sensitive nature of your request, I am escalating "
                "this to a dedicated customer support specialist who will assist you shortly."
            )
        else: # AUTO_HANDLE
            if top_res and len(top_res) > 20:
                # Ground reply directly on historical resolution
                clean_res = top_res.strip()
                if top_link and top_link not in clean_res:
                    return f"{clean_res} For self-service guidance, visit: {top_link}"
                return clean_res
            elif top_link:
                return f"You can resolve this directly in your account. Please follow the instructions at: {top_link}"
            else:
                return (
                    "Thank you for contacting support! You can manage this request directly from 'Your Orders' "
                    "in your account dashboard."
                )

    def process(self, request: AgentRequest | str) -> AgentResponse:
        """
        Processes an incoming customer message through the full pipeline.
        """
        if isinstance(request, str):
            request = AgentRequest(message=request)
            
        raw_msg = request.message
        cleaned_msg = clean_text(raw_msg)
        if not cleaned_msg:
            cleaned_msg = raw_msg
            
        # 1. Intent Classification
        pred_results = self.classifier.predict_with_confidence([cleaned_msg])
        intent, confidence = pred_results[0] if pred_results else ("ORDER_TRACKING_AND_DELIVERY", 0.5)
        
        # 2. FAISS Retrieval
        evidence_items = self.retriever.retrieve(cleaned_msg, top_k=request.top_k)
        evidence_dicts = [e.to_dict() for e in evidence_items]
        evidence_sims = [e.similarity for e in evidence_items]
        
        # 3. Escalation Decision
        action, reason = self.escalation_engine.evaluate(
            message=cleaned_msg,
            intent=intent,
            confidence=confidence,
            evidence_similarities=evidence_sims
        )
        
        # 4. Grounded Reply Generation
        reply = self.generate_reply(
            customer_message=cleaned_msg,
            intent=intent,
            confidence=confidence,
            action=action,
            reason=reason,
            evidence=evidence_dicts
        )
        
        # 5. Construct Pydantic Response
        schema_evidence = [
            EvidenceItemSchema(
                query=e.query,
                resolution=e.resolution,
                similarity=e.similarity,
                links=e.links,
                customer_tweet_id=e.customer_tweet_id,
                brand_tweet_id=e.brand_tweet_id
            )
            for e in evidence_items
        ]
        
        return AgentResponse(
            intent=intent,
            confidence=confidence,
            reply=reply,
            action=action,
            reason=reason,
            evidence=schema_evidence
        )
