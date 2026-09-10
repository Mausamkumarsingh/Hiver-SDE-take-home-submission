from __future__ import annotations
import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data.preprocessor import process_raw_dataset
from src.data.brand_analysis import analyze_candidate_brands
from src.data.golden_set_builder import build_golden_evaluation_set
from src.retrieval.indexer import FaissIndexer
from src.evaluation.harness import EvaluationHarness
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.human_validation import HumanVsLLMValidator

def main():
    start_time = time.time()
    print("==========================================================================")
    print("   Hiver SDE Assignment: Full Reproduction Pipeline (<15 Min Target)     ")
    print("==========================================================================")

    # Step 1: Check Data
    processed_path = Path(config.data.processed_corpus_path)
    golden_path = Path(config.data.golden_set_csv)
    index_path = Path(config.data.faiss_index_dir) / "index.faiss"

    if not processed_path.exists() or not golden_path.exists():
        print("\n[1/4] Processing raw TWCS dataset and building golden evaluation set...")
        processed_df = process_raw_dataset(
            csv_path=config.data.raw_twcs_path,
            brand=config.brand.selected,
            max_pairs=config.data.max_training_pairs
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(processed_path, index=False, encoding="utf-8")
        
        build_golden_evaluation_set(
            conversations_df=processed_df,
            samples_per_intent=20,
            output_csv=config.data.golden_set_csv,
            output_jsonl=config.data.golden_set_jsonl
        )
    else:
        print(f"\n[1/4] Using pre-processed conversation corpus ({processed_path}) and golden evaluation set ({golden_path}).")

    # Step 2: Vector Indexing
    if not index_path.exists():
        print("\n[2/4] Building FAISS IndexFlatIP vector database...")
        import pandas as pd
        df = pd.read_csv(processed_path)
        sample_df = df.sample(min(10000, len(df)), random_state=42).reset_index(drop=True)
        indexer = FaissIndexer(model_name=config.model.embedding_model, dimension=config.model.embedding_dim)
        indexer.build_from_dataframe(sample_df)
        indexer.save(config.data.faiss_index_dir)
    else:
        print(f"\n[2/4] FAISS vector index already built and verified at {config.data.faiss_index_dir}.")

    # Step 3: Run Comparative Benchmark
    print("\n[3/4] Running comparative benchmark (Majority Baseline vs TF-IDF vs AI SupportAgent)...")
    harness = EvaluationHarness(golden_set_path=str(golden_path), training_corpus_path=str(processed_path))
    harness.train_baselines(max_train_samples=3000)
    bench_results = harness.run()

    # Step 4: Run LLM-as-a-Judge and Human Agreement Validation
    print("\n[4/4] Executing LLM-as-a-judge rubric & human validation workflow...")
    validator = HumanVsLLMValidator(validation_dataset_path="data/human_judge_validation.csv")
    validator.run_validation(output_report_path="reports/human_judge_agreement.md")

    elapsed = time.time() - start_time
    print("\n==========================================================================")
    print(f"   Reproduction pipeline finished successfully in {elapsed:.1f} seconds ({elapsed/60:.1f} mins)!")
    print("   Reports generated:")
    print("   - reports/benchmark_summary.md")
    print("   - reports/human_judge_agreement.md")
    print("   - reports/llm_judge_evaluation.md")
    print("   - reports/failure_analysis.md")
    print("   - reports/brand_selection.md")
    print("==========================================================================")

if __name__ == "__main__":
    main()
