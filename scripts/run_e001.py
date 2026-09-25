import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import psutil
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from src.business_entity_resolution.retrieval.sparse_topk import sparse_top_k

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_retrieval(s1, sX, view_col, k, chunk_size, source_name):
    # sX is s2 or s3
    retrieved_pairs = []
    countries = s1['country'].unique()
    
    vec_kwargs = {
        'analyzer': 'char_wb',
        'ngram_range': (3, 4),
        'sublinear_tf': True,
        'dtype': np.float32,
        'min_df': 5,
        'max_df': 0.01,
        'max_features': 100000
    }
    
    t0 = time.time()
    for country in countries:
        c_s1 = s1[s1['country'] == country]
        c_sX = sX[sX['country'] == country]
        
        # Filter empty names for queries and targets
        valid_s1 = c_s1[c_s1[view_col].str.len() > 0]
        valid_sX = c_sX[c_sX[view_col].str.len() > 0]
        
        if len(valid_s1) == 0 or len(valid_sX) == 0:
            continue
            
        vec = TfidfVectorizer(**vec_kwargs)
        X_target = vec.fit_transform(valid_sX[view_col])
        X_query = vec.transform(valid_s1[view_col])
        
        top_indices, top_scores = sparse_top_k(X_query, X_target.T, k=k, chunk_size=200)
        
        sX_ids = valid_sX['entity_id'].values
        s1_ids = valid_s1['entity_id'].values
        
        for i, s1_id in enumerate(s1_ids):
            valid_idx = top_indices[i] >= 0
            if not valid_idx.any():
                continue
                
            match_ids = sX_ids[top_indices[i][valid_idx]]
            scores = top_scores[i][valid_idx]
            
            for rank, (m_id, score) in enumerate(zip(match_ids, scores)):
                retrieved_pairs.append({
                    'entity_id_s1': s1_id,
                    'entity_id_cand': m_id,
                    'candidate_source': source_name,
                    f'{view_col}_score': float(score),
                    f'{view_col}_rank': int(rank)
                })
                
    print(f"Retrieval {source_name} complete in {time.time() - t0:.1f}s. Found {len(retrieved_pairs)} pairs.")
    return pd.DataFrame(retrieved_pairs)

def main():
    print("=== E001: NAME CHARACTER TF-IDF RETRIEVAL ===")
    
    out_dir = Path("artifacts/retrieval/E001")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading data...")
    s1 = pd.read_parquet("data/processed/v001/train_source1.parquet").fillna("")
    s2 = pd.read_parquet("data/processed/v001/train_source2.parquet").fillna("")
    s3 = pd.read_parquet("data/processed/v001/train_source3.parquet").fillna("")
    
    # We will use `name_norm_clean` as primary based on typical ablation results, but we will configure this below
    # Let's say K=10 is our operating point based on probe (we will adjust this after seeing probe results)
    K = 10 
    
    print("Retrieving S2...")
    df_s2_clean = run_retrieval(s1, s2, 'name_norm_clean', K, 2000, 'S2')
    df_s2_accent = run_retrieval(s1, s2, 'name_norm_accent_fold', K, 2000, 'S2')
    
    print("Retrieving S3...")
    df_s3_clean = run_retrieval(s1, s3, 'name_norm_clean', K, 2000, 'S3')
    df_s3_accent = run_retrieval(s1, s3, 'name_norm_accent_fold', K, 2000, 'S3')
    
    # Merge clean and accent results
    df_s2 = pd.merge(df_s2_clean, df_s2_accent, on=['entity_id_s1', 'entity_id_cand', 'candidate_source'], how='outer')
    df_s3 = pd.merge(df_s3_clean, df_s3_accent, on=['entity_id_s1', 'entity_id_cand', 'candidate_source'], how='outer')
    
    e001_cands = pd.concat([df_s2, df_s3], ignore_index=True)
    e001_cands['retrieved_name_char_clean'] = e001_cands['name_norm_clean_score'].notna().astype(int)
    e001_cands['retrieved_name_char_accent'] = e001_cands['name_norm_accent_fold_score'].notna().astype(int)
    
    # Load E000 candidates
    print("Loading E000 candidates...")
    e000_cands = pd.read_parquet("artifacts/retrieval/E000/train_candidates.parquet")
    e000_cands['retrieved_e000'] = 1
    
    # Union E000 and E001
    print("Unioning E000 and E001 candidates...")
    # Merge on s1, cand, source
    all_cands = pd.merge(e000_cands, e001_cands, on=['entity_id_s1', 'entity_id_cand', 'candidate_source'], how='outer')
    
    # Fill NAs for retrieved flags
    all_cands['retrieved_e000'] = all_cands['retrieved_e000'].fillna(0).astype(int)
    all_cands['retrieved_name_char_clean'] = all_cands['retrieved_name_char_clean'].fillna(0).astype(int)
    all_cands['retrieved_name_char_accent'] = all_cands['retrieved_name_char_accent'].fillna(0).astype(int)
    
    # Preserve existing channel flags from E000, fill NAs for new E001-only cands
    e000_flags = ['exact_name_address_clean', 'exact_name_address_accent', 'exact_name_address_punct', 'unique_exact_name']
    for col in e000_flags:
        all_cands[col] = all_cands[col].fillna(0).astype(int)
        
    # Update retrieval channel count
    all_cands['retrieval_channel_count'] = all_cands[e000_flags + ['retrieved_name_char_clean', 'retrieved_name_char_accent']].sum(axis=1)
    
    # Save train candidates
    print("Saving E001 candidate pool...")
    all_cands.to_parquet(out_dir / "train_candidates.parquet", index=False)
    
    # GT Analysis
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_pairs = set()
    for row in gt.itertuples():
        s = row.source1_entity_id
        m = row.matched_entity_ids
        if m:
            for match in m.split(','):
                gt_pairs.add((s, match))
                
    e000_pairs_set = set(zip(e000_cands['entity_id_s1'], e000_cands['entity_id_cand']))
    e001_only_pairs_set = set(zip(e001_cands['entity_id_s1'], e001_cands['entity_id_cand']))
    all_pairs_set = set(zip(all_cands['entity_id_s1'], all_cands['entity_id_cand']))
    
    e000_gt = gt_pairs.intersection(e000_pairs_set)
    all_gt = gt_pairs.intersection(all_pairs_set)
    
    unique_gt_rescued = len(all_gt) - len(e000_gt)
    extra_cands = len(all_pairs_set) - len(e000_pairs_set)
    
    print(f"E000 Pair Recall: {len(e000_gt) / len(gt_pairs):.4f}")
    print(f"E000 ∪ E001 Pair Recall: {len(all_gt) / len(gt_pairs):.4f}")
    print(f"Unique GT Rescued by E001: {unique_gt_rescued}")
    print(f"Extra Candidates Introduced: {extra_cands}")
    if extra_cands > 0:
        print(f"Rescue Efficiency: {unique_gt_rescued / extra_cands:.6f}")
        
    # Missed GT pairs
    missed = gt_pairs - all_pairs_set
    missed_list = [{"source1_entity_id": s, "candidate_entity_id": c} for s, c in missed]
    pd.DataFrame(missed_list).to_parquet(out_dir / "missed_gt_pairs.parquet", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
