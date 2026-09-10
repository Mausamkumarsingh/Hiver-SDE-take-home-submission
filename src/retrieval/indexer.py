from __future__ import annotations
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class FaissIndexer:
    """
    Builds and manages a FAISS IndexFlatIP vector database over historical customer-support
    conversations using L2-normalized dense embeddings for exact cosine similarity search.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.encoder = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "customer_text",
        reply_column: str = "brand_text",
        batch_size: int = 64
    ) -> FaissIndexer:
        """
        Embeds customer queries, normalizes vectors, and populates FAISS index with metadata.
        """
        print(f"Building FAISS index over {len(df)} conversation pairs...")
        texts = df[text_column].astype(str).tolist()
        
        # Compute dense embeddings
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Initialize Inner Product (Cosine Similarity) index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = []
        for idx, row in df.reset_index(drop=True).iterrows():
            links = str(row.get("links", "")).split(";") if pd.notna(row.get("links")) else []
            links = [l for l in links if l.strip()]
            self.metadata.append({
                "doc_id": idx,
                "customer_tweet_id": int(row.get("customer_tweet_id", 0)),
                "brand_tweet_id": int(row.get("brand_tweet_id", 0)),
                "customer_query": str(row[text_column]),
                "brand_resolution": str(row[reply_column]),
                "links": links
            })
            
        print(f"FAISS index built successfully with {self.index.ntotal} vectors.")
        return self

    def save(self, index_dir: str):
        """Serializes FAISS index and metadata to disk."""
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        index_file = out_dir / "index.faiss"
        meta_file = out_dir / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_file))
        with open(meta_file, "wb") as f:
            pickle.dump(self.metadata, f)
            
        print(f"Saved FAISS index and metadata to {index_dir}")

    def load(self, index_dir: str) -> FaissIndexer:
        """Loads FAISS index and metadata from disk."""
        in_dir = Path(index_dir)
        index_file = in_dir / "index.faiss"
        meta_file = in_dir / "metadata.pkl"
        
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"FAISS artifacts not found in {index_dir}")
            
        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self.metadata = pickle.load(f)
            
        print(f"Loaded FAISS index ({self.index.ntotal} vectors) from {index_dir}")
        return self
