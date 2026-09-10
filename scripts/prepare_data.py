from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import pandas as pd

from src.config import config
from src.data.preprocessor import process_raw_dataset
from src.data.brand_analysis import analyze_candidate_brands
from src.data.golden_set_builder import build_golden_evaluation_set

def create_unannotated_template(
    processed_df: pd.DataFrame,
    golden_df: pd.DataFrame,
    num_samples: int = 50,
    output_path: str = "data/annotation_template.csv"
):
    """
    Creates an annotation template for human reviewers with real customer inquiries,
    strictly marking unreviewed fields as REQUIRES HUMAN ANNOTATION.
    """
    golden_ids = set(golden_df["customer_tweet_id"].tolist())
    candidate_pool = processed_df[~processed_df["customer_tweet_id"].isin(golden_ids)]
    
    samples = candidate_pool.sample(min(num_samples, len(candidate_pool)), random_state=42)
    
    template_rows = []
    for idx, row in samples.reset_index().iterrows():
        template_rows.append({
            "task_id": f"human_task_{idx+1:03d}",
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "historical_brand_reply": row["brand_text"],
            "assigned_intent": "REQUIRES HUMAN ANNOTATION",
            "assigned_action": "REQUIRES HUMAN ANNOTATION",
            "escalation_reason": "REQUIRES HUMAN ANNOTATION",
            "annotator_id": "PENDING_ASSIGNMENT",
            "annotation_status": "REQUIRES HUMAN ANNOTATION"
        })
        
    template_df = pd.DataFrame(template_rows)
    template_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Created human annotation template with {len(template_df)} rows at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Data preparation pipeline for Hiver Support Agent")
    parser.add_argument("--raw-csv", default=config.data.raw_twcs_path, help="Path to raw twcs.csv")
    parser.add_argument("--max-pairs", type=int, default=config.data.max_training_pairs, help="Max pairs to extract")
    parser.add_argument("--golden-size-per-intent", type=int, default=20, help="Number of golden samples per intent")
    parser.add_argument("--skip-brand-analysis", action="store_true", help="Skip candidate brand analysis report")
    args = parser.parse_args()

    print("=== Step 1: Candidate Brand Selection Analysis ===")
    if not args.skip_brand_analysis:
        analyze_candidate_brands(args.raw_csv, nrows=500000)
    else:
        print("Skipping brand selection analysis.")

    print("\n=== Step 2: Processing Target Brand (AmazonHelp) Conversations ===")
    processed_df = process_raw_dataset(
        csv_path=args.raw_csv,
        brand=config.brand.selected,
        max_pairs=args.max_pairs
    )
    
    Path(config.data.processed_corpus_path).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(config.data.processed_corpus_path, index=False, encoding="utf-8")
    print(f"Saved processed conversation corpus ({len(processed_df)} pairs) to {config.data.processed_corpus_path}")

    print("\n=== Step 3: Generating Stratified Golden Evaluation Set ===")
    golden_df = build_golden_evaluation_set(
        conversations_df=processed_df,
        samples_per_intent=args.golden_size_per_intent,
        output_csv=config.data.golden_set_csv,
        output_jsonl=config.data.golden_set_jsonl
    )

    print("\n=== Step 4: Generating Human Annotation Template ===")
    create_unannotated_template(processed_df, golden_df, num_samples=50)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
