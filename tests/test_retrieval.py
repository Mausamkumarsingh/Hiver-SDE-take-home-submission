import pytest
import pandas as pd
from pathlib import Path
from src.retrieval.indexer import FaissIndexer
from src.retrieval.retriever import FaissRetriever

def test_faiss_indexer_and_retriever(tmp_path):
    df = pd.DataFrame([
        {
            "customer_tweet_id": 101,
            "brand_tweet_id": 201,
            "customer_text": "Where is my package? It is two days late.",
            "brand_text": "You can track your package here: https://amazon.com/track",
            "links": "https://amazon.com/track"
        },
        {
            "customer_tweet_id": 102,
            "brand_tweet_id": 202,
            "customer_text": "How do I return my shoes?",
            "brand_text": "Visit our returns portal here: https://amazon.com/returns",
            "links": "https://amazon.com/returns"
        }
    ])
    
    indexer = FaissIndexer()
    indexer.build_from_dataframe(df)
    
    index_dir = str(tmp_path / "test_faiss")
    indexer.save(index_dir)
    
    assert (Path(index_dir) / "index.faiss").exists()
    assert (Path(index_dir) / "metadata.pkl").exists()
    
    # Test Retriever
    retriever = FaissRetriever(index_dir=index_dir, similarity_threshold=0.3)
    results = retriever.retrieve("Where is my delivery tracking?", top_k=1)
    
    assert len(results) == 1
    assert "track your package" in results[0].resolution
    assert results[0].similarity > 0.4
