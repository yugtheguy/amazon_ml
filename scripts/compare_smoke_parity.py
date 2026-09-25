import os
import pandas as pd
import json

def compare_dataframes(name, local_df, kaggle_df, sort_keys, compare_cols=None):
    if len(local_df) != len(kaggle_df):
        print(f"[{name}] LENGTH MISMATCH: Local={len(local_df)}, Kaggle={len(kaggle_df)}")
        return False
        
    local_df = local_df.sort_values(sort_keys).reset_index(drop=True)
    kaggle_df = kaggle_df.sort_values(sort_keys).reset_index(drop=True)
    
    if compare_cols is None:
        compare_cols = local_df.columns
        
    mismatches = 0
    for col in compare_cols:
        if col not in kaggle_df.columns:
            print(f"[{name}] Missing column in Kaggle: {col}")
            mismatches += 1
            continue
            
        if pd.api.types.is_numeric_dtype(local_df[col]):
            diff = (local_df[col].fillna(0) - kaggle_df[col].fillna(0)).abs().max()
            if diff > 1e-4:
                print(f"[{name}] Numeric mismatch in {col}. Max diff: {diff}")
                mismatches += 1
        else:
            if not local_df[col].equals(kaggle_df[col]):
                print(f"[{name}] Categorical mismatch in {col}")
                mismatches += 1
                
    if mismatches == 0:
        print(f"[{name}] SUCCESS. Exact parity verified across {len(compare_cols)} columns.")
        return True
    return False

def main():
    local_dir = "artifacts/candidate_pool/R001/smoke_1k"
    kaggle_dir = "artifacts/candidate_pool/R001/kaggle_smoke_1k" # User should unzip here
    
    if not os.path.exists(kaggle_dir):
        print(f"Kaggle artifacts not found at {kaggle_dir}. Please unzip R001_smoke_artifacts.zip there.")
        return
        
    # Compare internal candidates
    local_int = pd.read_parquet(os.path.join(local_dir, "local_internal_candidates.parquet"))
    kaggle_int = pd.read_parquet(os.path.join(kaggle_dir, "internal_candidates.parquet"))
    
    print("=== INTERNAL CANDIDATES PARITY ===")
    compare_dataframes(
        "INTERNAL", 
        local_int, 
        kaggle_int, 
        sort_keys=['entity_id_s1', 'entity_id_cand'],
        compare_cols=[
            'entity_id_s1', 'entity_id_cand', 'candidate_source', 
            'is_exact', 'retrieval_channel_count', 'both_name_address', 
            'best_lexical_rank', 'rare_token_overlap_count', 'shared_numeric_count',
            'name_word_score', 'address_word_score', 'candidate_rank'
        ]
    )
    
    # Compare final candidates
    local_fin = pd.read_parquet(os.path.join(local_dir, "local_final_candidates.parquet"))
    kaggle_fin = pd.read_parquet(os.path.join(kaggle_dir, "final_candidates.parquet"))
    
    print("\n=== FINAL CANDIDATES PARITY (BUDGET 15) ===")
    compare_dataframes(
        "FINAL", 
        local_fin, 
        kaggle_fin, 
        sort_keys=['entity_id_s1', 'entity_id_cand'],
        compare_cols=[
            'entity_id_s1', 'entity_id_cand', 'candidate_source', 'candidate_rank'
        ]
    )
    
    # Compare metrics
    with open(os.path.join(kaggle_dir, "retrieval_metrics.json")) as f:
        kaggle_metrics = json.load(f)['internal']
        
    print("\n=== METRICS PARITY ===")
    print(f"Kaggle Pair Recall: {kaggle_metrics['pair_recall']:.4f}")
    print(f"Kaggle Full Coverage: {kaggle_metrics['full_coverage']:.4f}")
    
if __name__ == "__main__":
    main()
