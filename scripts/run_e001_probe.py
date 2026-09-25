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

def sample_s1(folds_df, n_samples=50000, seed=42):
    # Stratify by match_bucket
    sampled = folds_df.groupby('match_bucket', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), int(n_samples * len(x)/len(folds_df))), random_state=seed)
    )
    return sampled['source1_entity_id'].values

def evaluate_retrieval(retrieved_pairs, gt_dict, s1_sample, k_vals):
    # retrieved_pairs is dict: (s1_id) -> list of candidates (sorted by score)
    # gt_dict is dict: (s1_id) -> set of GT matches
    
    results = {}
    for k in k_vals:
        gt_found = 0
        total_gt = 0
        full_cov_count = 0
        s1_with_gt = 0
        cands_generated = 0
        
        for s1_id in s1_sample:
            gt_matches = gt_dict.get(s1_id, set())
            cands = retrieved_pairs.get(s1_id, [])[:k]
            
            cands_generated += len(cands)
            
            if len(gt_matches) > 0:
                total_gt += len(gt_matches)
                s1_with_gt += 1
                
                found = len(gt_matches.intersection(set(cands)))
                gt_found += found
                if found == len(gt_matches):
                    full_cov_count += 1
                    
        pair_recall = gt_found / total_gt if total_gt > 0 else 0
        full_cov = full_cov_count / s1_with_gt if s1_with_gt > 0 else 0
        mean_cands = cands_generated / len(s1_sample)
        
        results[k] = {
            "pair_recall": pair_recall,
            "full_coverage": full_cov,
            "mean_candidates": mean_cands
        }
    return results

def run_probe(view_col):
    print(f"=== E001-PROBE: {view_col} ===")
    start_time = time.time()
    
    # 1. Load Data
    s1 = pd.read_parquet("data/processed/v001/train_source1.parquet")
    s2 = pd.read_parquet("data/processed/v001/train_source2.parquet")
    s3 = pd.read_parquet("data/processed/v001/train_source3.parquet")
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    folds = pd.read_parquet("artifacts/folds/folds_v1.parquet")
    
    # 2. Sample S1
    s1_sample_ids = sample_s1(folds, n_samples=50000)
    s1_sample = s1[s1['entity_id'].isin(s1_sample_ids)].copy()
    
    print(f"Sampled S1: {len(s1_sample)} entities")
    
    # 3. Ground Truth pre-processing for S2 and S3 separately
    gt_s2 = {}
    gt_s3 = {}
    s1_sample_ids_set = set(s1_sample_ids)
    for row in gt.itertuples():
        s = row.source1_entity_id
        if s not in s1_sample_ids_set:
            continue
        matches = row.matched_entity_ids
        if matches:
            for match in matches.split(','):
                if match.startswith('S2'):
                    gt_s2.setdefault(s, set()).add(match)
                elif match.startswith('S3'):
                    gt_s3.setdefault(s, set()).add(match)
                    
    # 4. Filter empty names
    s1_sample['is_empty'] = s1_sample[view_col].isna() | (s1_sample[view_col] == "")
    valid_s1 = s1_sample[~s1_sample['is_empty']]
    
    print(f"Valid S1 queries: {len(valid_s1)} ({s1_sample['is_empty'].sum()} empty)")
    
    # 5. TF-IDF Config
    vectorizer_kwargs = {
        'analyzer': 'char_wb',
        'ngram_range': (3, 4),
        'sublinear_tf': True,
        'dtype': np.float32,
        'min_df': 5,
        'max_df': 0.01,
        'max_features': 100000
    }
    
    def retrieve(source_df, source_name):
        retrieved_dict = {}
        countries = valid_s1['country'].unique()
        
        t0 = time.time()
        for country in countries:
            s_s1 = valid_s1[valid_s1['country'] == country]
            s_sX = source_df[source_df['country'] == country]
            
            # Filter empty targets
            s_sX = s_sX[~s_sX[view_col].isna() & (s_sX[view_col] != "")]
            if len(s_s1) == 0 or len(s_sX) == 0:
                continue
                
            vec = TfidfVectorizer(**vectorizer_kwargs)
            # Fit on Target Corpus!
            X_target = vec.fit_transform(s_sX[view_col])
            X_query = vec.transform(s_s1[view_col])
            
            top_indices, top_scores = sparse_top_k(X_query, X_target.T, k=20, chunk_size=200)
            
            s_sX_ids = s_sX['entity_id'].values
            s_s1_ids = s_s1['entity_id'].values
            
            for i, s1_id in enumerate(s_s1_ids):
                valid_idx = top_indices[i] >= 0
                match_ids = s_sX_ids[top_indices[i][valid_idx]]
                retrieved_dict[s1_id] = list(match_ids)
                
        print(f"Retrieval {source_name} time: {time.time() - t0:.1f}s")
        return retrieved_dict
        
    s2_retrieved = retrieve(s2, "S2")
    s3_retrieved = retrieve(s3, "S3")
    
    # Evaluate
    k_vals = [1, 3, 5, 10, 20]
    print("--- S2 Results ---")
    s2_eval = evaluate_retrieval(s2_retrieved, gt_s2, set(s1_sample_ids), k_vals)
    for k in k_vals:
        print(f"K={k}: Recall={s2_eval[k]['pair_recall']:.4f}, Cov={s2_eval[k]['full_coverage']:.4f}, Cands/S1={s2_eval[k]['mean_candidates']:.2f}")
        
    print("--- S3 Results ---")
    s3_eval = evaluate_retrieval(s3_retrieved, gt_s3, set(s1_sample_ids), k_vals)
    for k in k_vals:
        print(f"K={k}: Recall={s3_eval[k]['pair_recall']:.4f}, Cov={s3_eval[k]['full_coverage']:.4f}, Cands/S1={s3_eval[k]['mean_candidates']:.2f}")

    print(f"Memory Usage: {memory_usage():.1f} MB")
    print(f"Total Time: {time.time() - start_time:.1f}s")
    
    return s2_eval, s3_eval
    
if __name__ == "__main__":
    run_probe("name_norm_clean")
    run_probe("name_norm_accent_fold")
