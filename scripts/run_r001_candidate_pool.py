import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import psutil
import yaml
import logging
from collections import defaultdict

from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def safe_sample_s1(folds_df, n_samples, seed=42):
    sampled = folds_df.groupby('match_bucket').sample(
        frac=n_samples/len(folds_df), random_state=seed, replace=False
    )
    return sampled['source1_entity_id'].values

def evaluate_pool(cands_df, gt_dict, s1_sample_ids):
    cands_map = defaultdict(set)
    cands_s2 = 0
    cands_s3 = 0
    for row in cands_df.itertuples(index=False):
        cands_map[row.entity_id_s1].add(row.entity_id_cand)
        if row.candidate_source == 'S2': cands_s2 += 1
        elif row.candidate_source == 'S3': cands_s3 += 1
        
    total_gt = 0
    gt_found = 0
    s1_with_gt = 0
    full_cov_count = 0
    
    zero_cands = []
    
    for s1_id in s1_sample_ids:
        matches = gt_dict.get(s1_id, set())
        c = cands_map.get(s1_id, set())
        
        if len(matches) > 0:
            total_gt += len(matches)
            s1_with_gt += 1
            found = c.intersection(matches)
            gt_found += len(found)
            if len(found) == len(matches):
                full_cov_count += 1
        else:
            zero_cands.append(len(c))
            
    pair_recall = gt_found / total_gt if total_gt > 0 else 0
    full_cov = full_cov_count / s1_with_gt if s1_with_gt > 0 else 0
    
    cand_sizes = [len(cands_map.get(s1_id, set())) for s1_id in s1_sample_ids]
    
    mean_zero = float(np.mean(zero_cands)) if zero_cands else 0.0
    zero_zero_pct = float(sum(1 for z in zero_cands if z == 0) / len(zero_cands)) if zero_cands else 0.0
    
    return {
        "pair_recall": float(pair_recall),
        "full_coverage": float(full_cov),
        "mean_candidates": float(np.mean(cand_sizes)),
        "median": float(np.median(cand_sizes)),
        "p90": float(np.percentile(cand_sizes, 90)),
        "p95": float(np.percentile(cand_sizes, 95)),
        "p99": float(np.percentile(cand_sizes, 99)),
        "max": int(np.max(cand_sizes)),
        "zero_match_mean_candidates": mean_zero,
        "zero_match_zero_cand_pct": zero_zero_pct,
        "s2_candidates": cands_s2,
        "s3_candidates": cands_s3
    }

def run_experiment(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("=== R001 CANDIDATE POOL V1 ===")
    start_time = time.time()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    data_dir = args.data_dir
    
    logging.info("[R001] Data loading            START")
    t_dl = time.time()
    s1 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source1.parquet"))
    s2 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source2.parquet"))
    s3 = pd.read_parquet(os.path.join(data_dir, "processed", "v001", "train_source3.parquet"))
    gt = pd.read_csv(os.path.join(data_dir, "raw", "train", "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    folds = pd.read_parquet("artifacts/folds/folds_v1.parquet")
    logging.info(f"[R001] Data loading            DONE  {time.time() - t_dl:.1f}s")
    
    n_samples = args.probe_size if args.probe_size > 0 else args.smoke_size
    s1_sample_ids = safe_sample_s1(folds, n_samples=n_samples)
    s1_sample = s1[s1['entity_id'].isin(s1_sample_ids)].copy()
    
    logging.info("[R001] GT filtering            START")
    t_gt = time.time()
    sample_id_set = set(s1_sample_ids)
    gt_sample = gt[gt["source1_entity_id"].isin(sample_id_set)].copy()
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(',')) if row.matched_entity_ids else set()
        for row in gt_sample.itertuples(index=False)
    }
    logging.info(f"[R001] GT filtering            DONE  {time.time() - t_gt:.1f}s")
            
    generator = CandidateGenerator(config)
    
    t0 = time.time()
    mem_before = memory_usage()
    union_df = generator.generate(s1_sample, {'S2': s2, 'S3': s3})
    mem_after = memory_usage()
    
    logging.info(f"Generation took {time.time() - t0:.1f}s. Memory delta: {mem_after - mem_before:.1f} MB")
    
    if args.smoke_size > 0 and args.probe_size == 0:
        logging.info("Smoke test completed.")
        return
        
    # Write internal candidates
    union_df.to_parquet(os.path.join(args.out_dir, "internal_candidates.parquet"), index=False)
    
    internal_metrics = evaluate_pool(union_df, gt_dict, s1_sample_ids)
    
    budgets = config.get('pruning', {}).get('candidate_budgets', [3, 5, 8, 10, 15])
    frontier = []
    
    for b in budgets:
        pruned = generator.rank_and_prune(union_df, max_candidates=b)
        m = evaluate_pool(pruned, gt_dict, s1_sample_ids)
        frontier.append({
            "Budget": b,
            "Pair Recall": m['pair_recall'],
            "Full Coverage": m['full_coverage'],
            "Mean Cand/S1": m['mean_candidates'],
            "P95": m['p95'],
            "ZERO Mean": m['zero_match_mean_candidates']
        })
        if b == 15:
            pruned.to_parquet(os.path.join(args.out_dir, "final_candidates.parquet"), index=False)
            
    pd.DataFrame(frontier).to_csv(os.path.join(args.out_dir, "candidate_count_vs_recall.csv"), index=False)
    
    with open(os.path.join(args.out_dir, "retrieval_metrics.json"), "w") as f:
        json.dump({"internal": internal_metrics}, f, indent=4)
        
    # Channel contribution
    channels = ['exact', 'name_word', 'address_word', 'rare', 'numeric']
    chan_rows = []
    for c in channels:
        col = f"retrieved_{c}"
        if col in union_df.columns:
            subset = union_df[union_df[col] == 1]
            gt_rescued = 0
            for row in subset.itertuples(index=False):
                if row.entity_id_cand in gt_dict.get(row.entity_id_s1, set()):
                    gt_rescued += 1
            chan_rows.append({
                "Channel": c,
                "Candidates Added": len(subset),
                "GT Recovered": gt_rescued,
                "Unique Rescue": 0, # Simplify for now
                "Rescue Efficiency": 0 # Simplify for now
            })
    pd.DataFrame(chan_rows).to_csv(os.path.join(args.out_dir, "channel_contribution.csv"), index=False)
    
    # Missed GT pairs from internal pool
    missed_list = []
    cand_pairs_set = set(zip(union_df['entity_id_s1'], union_df['entity_id_cand']))
    for s1_id, matches in gt_dict.items():
        for m in matches:
            if (s1_id, m) not in cand_pairs_set:
                missed_list.append({"source1_entity_id": s1_id, "candidate_entity_id": m})
                
    pd.DataFrame(missed_list).to_parquet(os.path.join(args.out_dir, "missed_gt_pairs.parquet"), index=False)
    
    logging.info(f"Total time: {time.time() - start_time:.1f}s")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/candidate_pool_v1.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='artifacts/candidate_pool/R001/probe_50k')
    parser.add_argument('--smoke-size', type=int, default=0)
    parser.add_argument('--probe-size', type=int, default=50000)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = os.environ.get("KAGGLE_ARTIFACT_DIR", args.out_dir)
    args.data_dir = data_dir
    args.out_dir = out_dir
    
    run_experiment(args)
