from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field

def resolve_dataset_path(configured_path: str) -> str:
    """
    Intelligently resolves the path to the raw twcs.csv dataset:
    1. Checks environment variable TWCS_CSV_PATH
    2. Checks the configured path relative to project root
    3. Checks ~/.cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/...
    4. Downloads automatically via kagglehub if available and missing
    """
    # 1. Check environment variable override
    env_path = os.environ.get("TWCS_CSV_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    # 2. Check local repo path
    p = Path(configured_path)
    if p.exists():
        return str(p.resolve())

    # 3. Check common kagglehub cache locations
    home = Path.home()
    kaggle_candidates = [
        home / ".cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv",
        home / ".cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/twcs.csv"
    ]
    for cand in kaggle_candidates:
        if cand.exists():
            return str(cand.resolve())

    # 4. Fallback to configured path
    return str(p)

class BrandConfig(BaseModel):
    selected: str = "AmazonHelp"
    domain: str = "E-Commerce Customer Support & Order Management"

class DataConfig(BaseModel):
    raw_twcs_path: str = "data/raw/twcs.csv"
    processed_corpus_path: str = "data/processed/amazon_conversations.csv"
    golden_set_csv: str = "data/golden_eval_set.csv"
    golden_set_jsonl: str = "data/golden_eval_set.jsonl"
    faiss_index_dir: str = "data/faiss_index"
    max_training_pairs: int = 25000
    golden_set_size: int = 200

    def model_post_init(self, __context):
        self.raw_twcs_path = resolve_dataset_path(self.raw_twcs_path)

class ModelConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    openai_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.1

class RetrievalConfig(BaseModel):
    top_k: int = 3
    similarity_threshold: float = 0.55

class EscalationConfig(BaseModel):
    confidence_threshold: float = 0.65
    min_evidence_similarity: float = 0.55
    sentiment_threat_keywords: List[str] = Field(default_factory=lambda: [
        "sue", "lawyer", "attorney", "fraud", "police", "legal action",
        "court", "scam", "stolen", "fraudulent", "speak to a human",
        "speak to an agent", "talk to a human", "real person", "supervisor", "manager"
    ])
    sensitive_intents: List[str] = Field(default_factory=lambda: [
        "ACCOUNT_ACCESS_AND_SECURITY", "REFUND_AND_BILLING", "CUSTOMER_SERVICE_COMPLAINT"
    ])

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class AppConfig(BaseModel):
    brand: BrandConfig = Field(default_factory=BrandConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()

config = load_config()
