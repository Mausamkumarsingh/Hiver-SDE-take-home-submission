from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
from src.config import config
from src.retrieval.indexer import FaissIndexer
from src.intent.classifier import EmbeddingIntentClassifier
from src.intent.taxonomy import IntentName, TAXONOMY

def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector database and train intent probe")
    parser.add_argument("--corpus-path", default=config.data.processed_corpus_path, help="Path to processed conversations CSV")
    parser.add_argument("--index-dir", default=config.data.faiss_index_dir, help="Directory to save FAISS artifacts")
    parser.add_argument("--max-records", type=int, default=12000, help="Max records to index")
    args = parser.parse_args()

    print(f"Loading conversation corpus from {args.corpus_path}...")
    df = pd.read_csv(args.corpus_path)
    
    if len(df) > args.max_records:
        df = df.sample(args.max_records, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.max_records} records for FAISS index.")
    else:
        print(f"Using all {len(df)} records for FAISS index.")

    # 1. Build FAISS Index
    indexer = FaissIndexer(
        model_name=config.model.embedding_model,
        dimension=config.model.embedding_dim
    )
    indexer.build_from_dataframe(df)
    indexer.save(args.index_dir)

    print("\nFAISS vector indexing successfully completed!")

if __name__ == "__main__":
    main()
