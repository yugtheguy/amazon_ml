import os
import pandas as pd
from pathlib import Path
from src.business_entity_resolution.pairs.pair_dataset import build_c002_pair_dataset
import yaml
import json

def make_tiny_smoke():
    out_dir = Path("tests/fixtures/c002")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Candidate shards
    cand_dir = out_dir / "candidate_pool/final"
    cand_dir.mkdir(parents=True, exist_ok=True)
    
    df1 = pd.DataFrame({
        "entity_id_s1": ["S1-1", "S1-1", "S1-2"],
        "candidate_source": ["S2", "S3", "S2"],
        "entity_id_cand": ["S2-A", "S3-B", "S2-C"],
        "country": ["US", "US", "India"],
        "retrieved_exact": [1, 0, 1]
    })
    df1.to_parquet(cand_dir / "shard_1.parquet")
    
    # 2. GT
    gt_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-A,S3-B", "S2-D", ""]
    })
    gt_path = out_dir / "train_ground_truth.tsv"
    gt_df.to_csv(gt_path, sep="\t", index=False)
    
    # 3. Folds
    folds_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "fold_id": [1, 1, 2]
    })
    folds_path = out_dir / "folds_v1.parquet"
    folds_df.to_parquet(folds_path)
    
    import hashlib
    def sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    
    fold_sha = sha256_file(folds_path)
    
    # 4. Config
    cfg = {
        "experiment_id": "C002_SMOKE",
        "parent_experiment": "C001",
        "candidate_pool_dir": str(cand_dir),
        "ground_truth_path": str(gt_path),
        "fold_manifest_path": str(folds_path),
        "fold_manifest_sha256": fold_sha,
        "output_dir": str(out_dir / "output"),
        "include_diagnostics": True
    }
    
    build_c002_pair_dataset(cfg)
    
    # 5. Read results
    out_pkg = pd.read_parquet(out_dir / "output/package/c002_pairs_v1.parquet")
    diag = json.load(open(out_dir / "output/diagnostics/c002_diagnostics.json"))
    
    print("input_candidate_rows =", diag["input_candidate_rows"])
    print("output_pair_rows =", diag["output_pair_rows"])
    print("positive_rows =", diag["positive_rows"])
    print("negative_rows =", diag["negative_rows"])
    print("unique_s1 =", len(out_pkg["entity_id_s1"].unique()))
    print("S2_rows =", len(out_pkg[out_pkg["candidate_source"] == "S2"]))
    print("S3_rows =", len(out_pkg[out_pkg["candidate_source"] == "S3"]))
    print("duplicate_pair_keys =", out_pkg.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"]).sum())
    print("missing_folds =", out_pkg["fold"].isna().sum())
    
    valid_sources = {"S2", "S3"}
    invalid = set(out_pkg["candidate_source"].unique()) - valid_sources
    print("invalid_sources =", len(invalid))
    print("retrieved_gt_pairs =", diag["s1_retrieved_gt_pairs"])
    print("missing_gt_pairs =", diag["s1_missing_gt_pairs"])
    print("rows_dropped =", diag["input_candidate_rows"] - diag["output_pair_rows"])

if __name__ == "__main__":
    make_tiny_smoke()
