import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import psutil
import yaml
from collections import defaultdict

from src.business_entity_resolution.retrieval.gpu_char_tfidf import sp_matmul_topn_cupy

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def safe_sample_s1(folds_df, n_samples=50000, seed=42):
    sampled = folds_df.groupby('match_bucket').sample(
        frac=n_samples/len(folds_df), random_state=seed
    )
    return sampled['source1_entity_id'].values

def load_e000_candidates(data_dir):
    p = "artifacts/retrieval/E000/train_candidates.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.DataFrame()

def evaluate_retrieval(retrieved_pairs, gt_dict, s1_sample, k_vals, e000_cands_dict=None):
    results = {}
    for k in k_vals:
        gt_found = 0
        total_gt = 0
        full_cov_count = 0
        s1_with_gt = 0
        cands_generated = 0
        
        unique_rescued = 0
        extra_cands = 0
        
        for s1_id in s1_sample:
            gt_matches = gt_dict.get(s1_id, set())
            cands = retrieved_pairs.get(s1_id, [])[:k]
            
            cands_generated += len(cands)
            
            e000_matches = e000_cands_dict.get(s1_id, set()) if e000_cands_dict else set()
            
            if len(gt_matches) > 0:
                total_gt += len(gt_matches)
                s1_with_gt += 1
                
                found_by_e002 = set(cands).intersection(gt_matches)
                gt_found += len(found_by_e002)
                
                if len(found_by_e002) == len(gt_matches):
                    full_cov_count += 1
                    
                found_by_e000 = e000_matches.intersection(gt_matches)
                newly_found = found_by_e002 - found_by_e000
                unique_rescued += len(newly_found)
                
            extra_cands += max(0, len(cands) - len(e000_matches))
            
        pair_recall = gt_found / total_gt if total_gt > 0 else 0
        full_cov = full_cov_count / s1_with_gt if s1_with_gt > 0 else 0
        mean_cands = cands_generated / len(s1_sample)
        rescue_eff = unique_rescued / extra_cands if extra_cands > 0 else 0
        
        results[k] = {
            "pair_recall": pair_recall,
            "full_coverage": full_cov,
            "mean_candidates": mean_cands,
            "unique_gt_rescued": unique_rescued,
            "extra_candidates": extra_cands,
            "rescue_efficiency": rescue_eff
        }
    return results

def run_probe_gpu(view_col, config_path, data_dir, artifact_dir):
    print(f"=== E002-A GPU WORD TF-IDF PROBE: {view_col} ===")
    start_time = time.time()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)['retrieval']
        
    print(f"Loaded config: {config}")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "config.yaml"), "w") as f:
        yaml.dump({"retrieval": config}, f)
        
    s1 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source1.parquet"))
    s2 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source2.parquet"))
    s3 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source3.parquet"))
    gt = pd.read_csv(os.path.join(data_dir, "raw", "train", "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    folds = pd.read_parquet("artifacts/folds/folds_v1.parquet")
    
    e000_df = load_e000_candidates(data_dir)
    e000_dict_s2 = defaultdict(set)
    e000_dict_s3 = defaultdict(set)
    if not e000_df.empty:
        for row in e000_df.itertuples():
            if row.candidate_source == 'S2':
                e000_dict_s2[row.source1_entity_id].add(row.candidate_entity_id)
            else:
                e000_dict_s3[row.source1_entity_id].add(row.candidate_entity_id)

    total_s1 = len(s1)
    
    n_samples = config.get('sample_size', 50000)
    s1_sample_ids = safe_sample_s1(folds, n_samples=n_samples)
    s1_sample = s1[s1['entity_id'].isin(s1_sample_ids)].copy()
    
    print(f"Total S1 entities (for projection): {total_s1}")
    print(f"Sampled S1: {len(s1_sample)} entities")
    
    pd.DataFrame({'entity_id': s1_sample_ids}).to_parquet(os.path.join(artifact_dir, "probe_ids.parquet"))
    
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
                    
    s1_sample['is_empty'] = s1_sample[view_col].isna() | (s1_sample[view_col] == "")
    valid_s1 = s1_sample[~s1_sample['is_empty']]
    
    try:
        from cuml.feature_extraction.text import TfidfVectorizer
        TFIDF = TfidfVectorizer
        print("Using cuML TfidfVectorizer")
    except ImportError:
        raise RuntimeError("cuML is required for E002-A GPU Word TF-IDF. CPU fallback is strictly disabled.")

    vectorizer_kwargs = {
        'analyzer': config['analyzer'],
        'ngram_range': (config['ngram_min'], config['ngram_max']),
        'sublinear_tf': config.get('sublinear_tf', True),
        'dtype': np.float32 if config['dtype'] == 'float32' else np.float64,
        'min_df': config['min_df'],
        'max_df': config['max_df'],
        'max_features': config['max_features']
    }
    
    def retrieve(source_df, source_name):
        print(f"\n--- Processing {source_name} ---")
        retrieved_dict = {}
        countries = valid_s1['country'].unique()
        
        total_queries = 0
        total_docs = 0
        
        t0 = time.time()
        for country in countries:
            s_s1 = valid_s1[valid_s1['country'] == country]
            s_sX = source_df[source_df['country'] == country]
            s_sX = s_sX[~s_sX[view_col].isna() & (s_sX[view_col] != "")]
            
            if len(s_s1) == 0 or len(s_sX) == 0:
                continue
                
            total_queries += len(s_s1)
            total_docs += len(s_sX)
                
            vec = TFIDF(**vectorizer_kwargs)
            
            try:
                s_sX_reset = s_sX.reset_index(drop=True)
                X_target = vec.fit_transform(s_sX_reset[view_col])
            except Exception as e:
                raise RuntimeError(f"GPU vectorization failed on {source_name} {country}: {e}")
                
            s_s1_reset = s_s1.reset_index(drop=True)
            X_query = vec.transform(s_s1_reset[view_col])
            
            if hasattr(X_target, 'get'):
                X_target = X_target.get()
            if hasattr(X_query, 'get'):
                X_query = X_query.get()
                
            if country == 'US' or country == 'India':
                print(f"[{country}] Target Matrix shape: {X_target.shape}, NNZ: {X_target.nnz}, Vocab: {len(vec.vocabulary_)}")
                print(f"[{country}] Query Matrix shape: {X_query.shape}, NNZ: {X_query.nnz}")
                
            top_sparse = sp_matmul_topn_cupy(
                X_query, X_target.T, top_k=config['top_k'], batch_size=config.get('batch_size', 1000)
            )
            
            s_sX_ids = s_sX['entity_id'].values
            s_s1_ids = s_s1['entity_id'].values
            
            for i, s1_id in enumerate(s_s1_ids):
                row_start = top_sparse.indptr[i]
                row_end = top_sparse.indptr[i+1]
                match_idxs = top_sparse.indices[row_start:row_end]
                match_ids = s_sX_ids[match_idxs]
                retrieved_dict[s1_id] = list(match_ids)
                
        t_total = time.time() - t0
        qps = total_queries / t_total if t_total > 0 else 0
        
        print(f"{source_name} Summary: {total_queries} Queries, {total_docs} Targets in {t_total:.2f}s ({qps:.1f} Q/sec)")
        return retrieved_dict
        
    s2_retrieved = retrieve(s2, "S2")
    s3_retrieved = retrieve(s3, "S3")
    
    k_vals = [1, 3, 5, 10, 20]
    s2_eval = evaluate_retrieval(s2_retrieved, gt_s2, set(s1_sample_ids), k_vals, e000_dict_s2)
    s3_eval = evaluate_retrieval(s3_retrieved, gt_s3, set(s1_sample_ids), k_vals, e000_dict_s3)
    
    with open(os.path.join(artifact_dir, "probe_metrics.json"), "w") as f:
        json.dump({"S2": s2_eval, "S3": s3_eval}, f, indent=4)
        
    # Write k_frontier.csv
    frontier_rows = []
    for k in k_vals:
        frontier_rows.append({
            "K": k,
            "S2_Recall": s2_eval[k]["pair_recall"],
            "S3_Recall": s3_eval[k]["pair_recall"],
            "S2_UniqueRescue": s2_eval[k]["unique_gt_rescued"],
            "S3_UniqueRescue": s3_eval[k]["unique_gt_rescued"]
        })
    pd.DataFrame(frontier_rows).to_csv(os.path.join(artifact_dir, "k_frontier.csv"), index=False)
    
    print(f"Total E002-A Probe Time: {time.time() - start_time:.1f}s")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/e002_word_gpu.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='artifacts/retrieval/E002-A')
    args = parser.parse_args()
    
    data_dir = os.environ.get("KAGGLE_DATA_ROOT", args.data_dir)
    out_dir = os.environ.get("KAGGLE_ARTIFACT_DIR", args.out_dir)
    
    run_probe_gpu("name_norm_clean", args.config, data_dir, out_dir)
