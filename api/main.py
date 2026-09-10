from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import config
from src.agent.core import SupportAgent
from src.agent.schema import AgentRequest, AgentResponse
from src.intent.taxonomy import ALL_INTENTS, TAXONOMY

app = FastAPI(
    title="Hiver AI Customer Support Agent API",
    description="Production-grade AI customer support agent for Twitter/X e-commerce operations (AmazonHelp)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton agent
_agent: Optional[SupportAgent] = None

def get_agent() -> SupportAgent:
    global _agent
    if _agent is None:
        _agent = SupportAgent()
    return _agent

class ClassifyRequest(BaseModel):
    text: str = Field(..., example="Where is my package? It was supposed to arrive yesterday.")

class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    description: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

@app.on_event("startup")
def startup_event():
    # Warm up agent on startup
    get_agent()

@app.get("/health")
def health_check():
    agent = get_agent()
    ntotal = agent.retriever.indexer.index.ntotal if agent.retriever.indexer.index else 0
    return {
        "status": "healthy",
        "brand": config.brand.selected,
        "embedding_model": config.model.embedding_model,
        "faiss_vectors_indexed": ntotal,
        "num_intents": len(ALL_INTENTS)
    }

@app.get("/intents")
def list_intents():
    return {
        "count": len(ALL_INTENTS),
        "intents": [
            {
                "name": name,
                "description": TAXONOMY[name].description,
                "keywords": TAXONOMY[name].keywords,
                "auto_handle_eligible": TAXONOMY[name].auto_handle_eligible
            }
            for name in ALL_INTENTS
        ]
    }

@app.post("/classify", response_model=ClassifyResponse)
def classify_message(req: ClassifyRequest):
    agent = get_agent()
    results = agent.classifier.predict_with_confidence([req.text])
    if not results:
        raise HTTPException(status_code=500, detail="Classification failed")
    intent, conf = results[0]
    desc = TAXONOMY[intent].description if intent in TAXONOMY else ""
    return ClassifyResponse(intent=intent, confidence=conf, description=desc)

@app.post("/retrieve")
def retrieve_evidence(req: RetrieveRequest):
    agent = get_agent()
    evidence = agent.retriever.retrieve(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "count": len(evidence),
        "evidence": [e.to_dict() for e in evidence]
    }

@app.post("/chat", response_model=AgentResponse)
def process_message(request: AgentRequest):
    agent = get_agent()
    response = agent.process(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.server.host, port=config.server.port, reload=False)
