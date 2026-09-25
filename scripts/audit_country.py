import json
import pandas as pd
from pathlib import Path
import time

def run_audit():
    print("Loading datasets...")
    t0 = time.time()
    s1 = pd.read_parquet("data/processed/v001/train_source1.parquet", columns=["entity_id", "country"])
    s2 = pd.read_parquet("data/processed/v001/train_source2.parquet", columns=["entity_id", "country"])
    s3 = pd.read_parquet("data/processed/v001/train_source3.parquet", columns=["entity_id", "country"])
    
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype={'matched_entity_ids': str})
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    print(f"Loaded in {time.time() - t0:.1f}s")
    
    t0 = time.time()
    print("Exploding ground truth...")
    gt = gt[gt['matched_entity_ids'] != '']
    gt['cand_id'] = gt['matched_entity_ids'].str.split(',')
    gt_exp = gt.explode('cand_id')[['source1_entity_id', 'cand_id']]
    
    # Merge S1 country
    gt_exp = gt_exp.merge(s1.rename(columns={'entity_id': 'source1_entity_id', 'country': 's1_country'}), on='source1_entity_id', how='left')
    
    # Merge cand country (concatenate s2 and s3)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    gt_exp = gt_exp.merge(s2s3.rename(columns={'entity_id': 'cand_id', 'country': 'cand_country'}), on='cand_id', how='left')
    
    gt_exp['country_match'] = gt_exp['s1_country'] == gt_exp['cand_country']
    
    total_pairs = len(gt_exp)
    country_equal = gt_exp['country_match'].sum()
    country_mismatch = total_pairs - country_equal
    
    mismatches = gt_exp[~gt_exp['country_match']]
    
    mismatches_by_source = {
        "S2": int(mismatches['cand_id'].str.startswith('S2-').sum()),
        "S3": int(mismatches['cand_id'].str.startswith('S3-').sum())
    }
    
    mismatch_examples = []
    if len(mismatches) > 0:
        mismatch_examples = mismatches.head(10).to_dict('records')
        
    results = {
        "total_gt_pairs": int(total_pairs),
        "country_equal_gt_pairs": int(country_equal),
        "country_mismatch_gt_pairs": int(country_mismatch),
        "mismatches_by_source": mismatches_by_source,
        "mismatch_examples": mismatch_examples,
        "generic_country_equality_valid": bool(country_mismatch == 0)
    }
    
    print(f"Computed in {time.time() - t0:.1f}s")
    
    out_dir = Path("artifacts/retrieval/E000")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "country_consistency_audit.json"
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Done. Wrote to {out_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_audit()
