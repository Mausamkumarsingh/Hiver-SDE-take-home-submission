from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.retrieval.indexer import FaissIndexer

class EvidenceItem:
    def __init__(
        self,
        query: str,
        resolution: str,
        similarity: float,
        links: List[str],
        customer_tweet_id: int,
        brand_tweet_id: int
    ):
        self.query = query
        self.resolution = resolution
        self.similarity = round(float(similarity), 4)
        self.links = links
        self.customer_tweet_id = customer_tweet_id
        self.brand_tweet_id = brand_tweet_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "resolution": self.resolution,
            "similarity": self.similarity,
            "links": self.links,
            "customer_tweet_id": self.customer_tweet_id,
            "brand_tweet_id": self.brand_tweet_id
        }

class FaissRetriever:
    """
    Retrieves top-k semantically relevant customer support resolutions
    from the FAISS index for grounding agent replies.
    """
    def __init__(
        self,
        index_dir: str = "data/faiss_index",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.55
    ):
        self.index_dir = index_dir
        self.similarity_threshold = similarity_threshold
        self.indexer = FaissIndexer(model_name=model_name)
        
        index_path = Path(index_dir) / "index.faiss"
        if index_path.exists():
            self.indexer.load(index_dir)
        else:
            print(f"Warning: FAISS index not found at {index_dir}. Call build_index first.")

    def retrieve(self, query: str, top_k: int = 3) -> List[EvidenceItem]:
        """
        Searches index for most similar historical customer queries.
        Returns evidence items above the similarity threshold.
        """
        if self.indexer.index is None or self.indexer.index.ntotal == 0:
            return []
            
        q_emb = self.indexer.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        D, I = self.indexer.index.search(q_emb, top_k)
        
        results: List[EvidenceItem] = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            sim = float(dist)
            # Filter matches that pass relevance threshold
            if sim >= self.similarity_threshold:
                meta = self.indexer.metadata[idx]
                results.append(EvidenceItem(
                    query=meta["customer_query"],
                    resolution=meta["brand_resolution"],
                    similarity=sim,
                    links=meta.get("links", []),
                    customer_tweet_id=meta.get("customer_tweet_id", 0),
                    brand_tweet_id=meta.get("brand_tweet_id", 0)
                ))
                
        return results
