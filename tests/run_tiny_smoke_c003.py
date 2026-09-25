import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.business_entity_resolution.features.feature_dataset import build_c003_feature_dataset
import json

def make_tiny_smoke():
    out_dir = Path("tests/fixtures/c003")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. C002 Shards
    c002_dir = out_dir / "c002_shards"
    c002_dir.mkdir(parents=True, exist_ok=True)
    
    df1 = pd.DataFrame({
        "entity_id_s1": ["S1-1", "S1-1", "S1-2"],
        "candidate_source": ["S2", "S3", "S2"],
        "entity_id_cand": ["S2-A", "S3-B", "S2-C"],
        "country": ["US", "US", "India"],
        "fold": pd.Series([1, 1, 2], dtype="int8"),
        "label": pd.Series([1, 0, 1], dtype="int8"),
        "retrieved_exact": [1, 0, 1],
        "name_word_score": [1.5, 0.5, 2.0]
    })
    df1.to_parquet(c002_dir / "shard_1.parquet")
    
    # 2. Normalized Sources
    norm_dir = out_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "name_norm_clean": ["apple inc", "banana corp"],
        "addr_norm_clean": ["123 main st", ""],
        "numeric_tokens": [["123"], []]
    })
    s1_df.to_parquet(norm_dir / "train_source1.parquet")
    
    s2_df = pd.DataFrame({
        "entity_id": ["S2-A", "S2-C"],
        "name_norm_clean": ["apple incorporated", "banana corp"],
        "addr_norm_clean": ["123 main street", ""],
        "numeric_tokens": [["123"], []]
    })
    s2_df.to_parquet(norm_dir / "train_source2.parquet")
    
    s3_df = pd.DataFrame({
        "entity_id": ["S3-B"],
        "name_norm_clean": ["aple inc"],
        "addr_norm_clean": ["456 broadway"],
        "numeric_tokens": [["456"]]
    })
    s3_df.to_parquet(norm_dir / "train_source3.parquet")
    
    # 3. Config
    cfg = {
        "experiment_id": "C003_SMOKE",
        "parent_experiment": "C002",
        "c002_pair_dataset_dir": str(c002_dir),
        "processed_s1_path": str(norm_dir / "train_source1.parquet"),
        "processed_s2_path": str(norm_dir / "train_source2.parquet"),
        "processed_s3_path": str(norm_dir / "train_source3.parquet"),
        "output_dir": str(out_dir / "output")
    }
    
    build_c003_feature_dataset(cfg)
    
    # 4. Read results
    out_pkg = pd.read_parquet(out_dir / "output/package/c003_features_v1.parquet")
    diag = json.load(open(out_dir / "output/diagnostics/c003_diagnostics.json"))
    
    print("input_candidate_rows =", diag["total_input_rows"])
    print("output_pair_rows =", diag["total_output_rows"])
    print("positive_rows =", diag["total_positive_rows"])
    print("feature_count =", diag["feature_count"])
    print("unique_s1 =", len(out_pkg["entity_id_s1"].unique()))
    print("S2_rows =", len(out_pkg[out_pkg["candidate_source"] == "S2"]))
    print("S3_rows =", len(out_pkg[out_pkg["candidate_source"] == "S3"]))
    
    from src.business_entity_resolution.features.schema import MODEL_FEATURE_COLUMNS, EXCLUDED_NON_FEATURE_COLUMNS
    print("Leakage Check: 'label' in MODEL_FEATURE_COLUMNS =", "label" in MODEL_FEATURE_COLUMNS)
    print("Leakage Check: 'fold' in MODEL_FEATURE_COLUMNS =", "fold" in MODEL_FEATURE_COLUMNS)
    
    print("\nSample Relatives for S1-1:")
    s1_1 = out_pkg[out_pkg["entity_id_s1"] == "S1-1"]
    print(s1_1[["candidate_source", "name_char_ratio", "name_similarity_rank_within_s1", "name_gap_from_best"]])

if __name__ == "__main__":
    make_tiny_smoke()
