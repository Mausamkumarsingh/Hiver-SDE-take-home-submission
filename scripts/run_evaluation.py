from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.evaluation.harness import EvaluationHarness

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark on golden set")
    parser.add_argument("--golden-csv", default="data/golden_eval_set.csv", help="Path to golden set CSV")
    parser.add_argument("--training-csv", default="data/processed/amazon_conversations.csv", help="Path to processed training CSV")
    parser.add_argument("--max-train", type=int, default=3000, help="Max training samples for baselines")
    args = parser.parse_args()

    harness = EvaluationHarness(
        golden_set_path=args.golden_csv,
        training_corpus_path=args.training_csv
    )
    
    print("=== Training Evaluation Baselines ===")
    harness.train_baselines(max_train_samples=args.max_train)
    
    print("\n=== Executing Benchmark on Golden Evaluation Set ===")
    results = harness.run()
    
    print("\nBenchmark successfully executed! Review reports/benchmark_summary.md")

if __name__ == "__main__":
    main()
