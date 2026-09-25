import os
import sys
import json
import time
import argparse
import hashlib
import logging
import subprocess
import tarfile
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
import yaml
import psutil
import gc

from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def check_gpu():
    try:
        import cuml
        import cupy as cp
        return "cuML/CuPy Available"
    except ImportError:
        return "CPU Only"

def compute_config_hash(config):
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit
    except Exception:
        return "unknown"

def check_shard_done(final_dir, shard_name):
    done_file = os.path.join(final_dir, f"{shard_name}.done")
    return os.path.exists(done_file)

def mark_shard_done(final_dir, shard_name, metadata):
    metadata_file = os.path.join(final_dir, f"{shard_name}.metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    done_file = os.path.join(final_dir, f"{shard_name}.done")
    with open(done_file, "w") as f:
        f.write("DONE")

def atomic_write_parquet(df, filepath):
    tmp_path = filepath + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.rename(tmp_path, filepath)

def validate_shard(s1_chunk, final_df, gt_dict, union_df, max_candidates=15):
    # Invariant A, B, C conceptually covered by joins, but let's check basic things
    if not final_df.empty:
        # Candidate count per S1 <= 15
        counts = final_df.groupby('entity_id_s1').size()
        if (counts > max_candidates).any():
            raise ValueError("Candidate count exceeds max_candidates")
        
        # Final subset of internal
        final_pairs = set(zip(final_df['entity_id_s1'], final_df['entity_id_cand']))
        internal_pairs = set(zip(union_df['entity_id_s1'], union_df['entity_id_cand']))
        if not final_pairs.issubset(internal_pairs):
            raise ValueError("Final candidates are not a subset of internal candidates")
            
        # No duplicate relationship
        dups = final_df.duplicated(subset=['entity_id_s1', 'candidate_source', 'entity_id_cand'])
        if dups.any():
            raise ValueError("Duplicate relationships found in final candidates")

def evaluate_pool(cands_df, gt_dict, s1_sample_ids):
    cands_map = defaultdict(set)
    for row in cands_df.itertuples(index=False):
        cands_map[row.entity_id_s1].add(row.entity_id_cand)
        
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
        "zero_match_zero_cand_pct": zero_zero_pct
    }

def run_experiment(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("=== C001 FULL TRAIN CANDIDATE POOL V1 ===")
    
    if os.environ.get("ALLOW_CPU_TFIDF") != "1":
        gpu_status = check_gpu()
        if "CPU Only" in gpu_status:
            raise RuntimeError("GPU_REQUIRED=TRUE. Stopping loudly. cuML/CuPy is not available.")
            
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    config_hash = compute_config_hash(config)
    git_commit = get_git_commit()
    
    run_dir = os.path.join(args.out_dir, "C001", "candidate_pool_v1")
    manifest_dir = os.path.join(run_dir, "manifest")
    internal_dir = os.path.join(run_dir, "internal")
    final_dir = os.path.join(run_dir, "final")
    metrics_dir = os.path.join(run_dir, "metrics")
    logs_dir = os.path.join(run_dir, "logs")
    package_dir = os.path.join(run_dir, "package")
    
    for d in [manifest_dir, internal_dir, final_dir, metrics_dir, logs_dir, package_dir]:
        os.makedirs(d, exist_ok=True)
        
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)
        
    data_dir = args.data_dir
    processed_dir = args.processed_dir
    
    logging.info("[C001] Loading Processed Data START")
    t_dl = time.time()
    s1 = pd.read_parquet(os.path.join(processed_dir, "train_source1.parquet"))
    s2 = pd.read_parquet(os.path.join(processed_dir, "train_source2.parquet"))
    s3 = pd.read_parquet(os.path.join(processed_dir, "train_source3.parquet"))
    gt = pd.read_csv(os.path.join(data_dir, "raw", "train", "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    
    if args.smoke_size > 0:
        s1 = s1.head(args.smoke_size)
    logging.info(f"[C001] Data Loaded in {time.time()-t_dl:.1f}s. S1 rows: {len(s1)}")
    
    # Ground truth mapping
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(',')) if row.matched_entity_ids else set()
        for row in gt.itertuples(index=False)
    }
    
    generator = CandidateGenerator(config)
    max_cands = config.get('pruning', {}).get('candidate_budgets', [15])[-1]
    
    # Partition logic: Country + Shard
    chunk_size = args.chunk_size
    s1_shards = []
    for country in s1['country'].unique():
        country_s1 = s1[s1['country'] == country].sort_values('entity_id').reset_index(drop=True)
        num_chunks = (len(country_s1) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            chunk = country_s1.iloc[i*chunk_size:(i+1)*chunk_size]
            shard_name = f"{country}_shard_{i:03d}"
            s1_shards.append((shard_name, chunk))
            
    logging.info(f"Total partitions: {len(s1_shards)}")
    
    all_final_dfs = []
    
    for i, (shard_name, s1_chunk) in enumerate(s1_shards):
        pct = (i / len(s1_shards)) * 100
        logging.info(f"--- Processing {shard_name} | Shard {i+1}/{len(s1_shards)} ({pct:.1f}%) | Rows: {len(s1_chunk)} ---")
        shard_misses_file = os.path.join(metrics_dir, f"{shard_name}_misses.parquet")
        
        if check_shard_done(final_dir, shard_name):
            logging.info(f"Shard {shard_name} already DONE. Skipping.")
            final_df = pd.read_parquet(os.path.join(final_dir, f"{shard_name}.parquet"))
            all_final_dfs.append(final_df)
            continue
            
        t0 = time.time()
        union_df = generator.generate(s1_chunk, {'S2': s2, 'S3': s3})
        atomic_write_parquet(union_df, os.path.join(internal_dir, f"{shard_name}.parquet"))
        
        final_df = generator.rank_and_prune(union_df, max_candidates=max_cands)
        
        validate_shard(s1_chunk, final_df, gt_dict, union_df, max_cands)
        
        atomic_write_parquet(final_df, os.path.join(final_dir, f"{shard_name}.parquet"))
        
        # Compute misses for this shard
        chunk_missed_list = []
        final_pairs = set(zip(final_df['entity_id_s1'], final_df['entity_id_cand']))
        internal_pairs = set(zip(union_df['entity_id_s1'], union_df['entity_id_cand']))
        
        for s1_id in s1_chunk['entity_id'].values:
            matches = gt_dict.get(s1_id, set())
            for m in matches:
                if (s1_id, m) not in final_pairs:
                    is_internal = (s1_id, m) in internal_pairs
                    miss_type = "pruning_miss" if is_internal else "retrieval_miss"
                    chunk_missed_list.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": m,
                        "miss_type": miss_type
                    })
                    
        misses_df = pd.DataFrame(chunk_missed_list)
        if chunk_missed_list:
            atomic_write_parquet(misses_df, shard_misses_file)
        else:
            # write empty df with right schema
            empty_miss = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "miss_type"])
            atomic_write_parquet(empty_miss, shard_misses_file)
            
        meta = {
            "shard": shard_name,
            "s1_rows": len(s1_chunk),
            "internal_cands": len(union_df),
            "final_cands": len(final_df),
            "config_hash": config_hash,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.time() - t0
        }
        mark_shard_done(final_dir, shard_name, meta)
        all_final_dfs.append(final_df)
        
        # Free memory
        del union_df
        del internal_pairs
        del final_pairs
        
        logging.info(f"Shard {shard_name} completed in {time.time()-t0:.1f}s. Memory: {memory_usage():.1f} MB")
        gc.collect()

    # Combine final
    logging.info("Combining all final shards...")
    if len(all_final_dfs) > 0:
        final_candidates_df = pd.concat(all_final_dfs, ignore_index=True)
    else:
        final_candidates_df = pd.DataFrame()
        
    final_candidates_df.to_parquet(os.path.join(package_dir, "train_candidates_v1.parquet"), index=False)
    
    # Entity Manifest
    cand_counts = final_candidates_df.groupby('entity_id_s1').size().reset_index(name='candidate_count')
    
    # Strictly join FOLDS_V1
    fold_path = os.path.join(args.out_dir, "..", "folds", "fold_manifest_v1.parquet")
    fold_path = os.path.normpath(fold_path)
    if not os.path.exists(fold_path):
        raise FileNotFoundError(f"CRITICAL: Frozen folds_v1.parquet not found at {fold_path}")
        
    folds_df = pd.read_parquet(fold_path)
    if 'fold_id' not in folds_df.columns or 'entity_id' not in folds_df.columns:
        raise ValueError("Frozen folds missing entity_id or fold_id")
        
    folds_df = folds_df[['entity_id', 'fold_id']]
    # Verify no duplicates
    if folds_df['entity_id'].duplicated().any():
        raise ValueError("Frozen folds contain duplicate entity_ids!")
        
    manifest_df = s1[['entity_id', 'country']].rename(columns={'entity_id': 'entity_id_s1'})
    manifest_df = manifest_df.merge(folds_df.rename(columns={'entity_id': 'entity_id_s1', 'fold_id': 'fold'}), on='entity_id_s1', how='left')
    
    if manifest_df['fold'].isna().any():
        raise ValueError("Some S1 entities are missing from the frozen folds_v1.parquet!")
        
    manifest_df = manifest_df.merge(cand_counts, on='entity_id_s1', how='left')
    manifest_df['candidate_count'] = manifest_df['candidate_count'].fillna(0).astype(int)
    manifest_df['has_candidates'] = (manifest_df['candidate_count'] > 0).astype(int)
    manifest_df.to_parquet(os.path.join(package_dir, "train_candidate_entity_manifest_v1.parquet"), index=False)
    
    # GT Miss Analysis
    logging.info("Combining misses...")
    all_misses = []
    for shard_name, _ in s1_shards:
        shard_misses_file = os.path.join(metrics_dir, f"{shard_name}_misses.parquet")
        if os.path.exists(shard_misses_file):
            all_misses.append(pd.read_parquet(shard_misses_file))
            
    if all_misses:
        misses_concat = pd.concat(all_misses, ignore_index=True)
        misses_concat.to_parquet(os.path.join(metrics_dir, "missed_gt_pairs_v1.parquet"), index=False)
    
    # Metrics Evaluation
    eval_res = evaluate_pool(final_candidates_df, gt_dict, s1['entity_id'].values)
    with open(os.path.join(metrics_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_res, f, indent=4)
        
    # Run Manifest
    run_manifest = {
        "experiment_id": "C001",
        "architecture": "R001",
        "candidate_budget": max_cands,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_s1_rows": len(s1),
        "output_final_cands": len(final_candidates_df),
        "validation_status": "PASSED"
    }
    with open(os.path.join(manifest_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)
        
    # Artifact Registry
    registry = {
        "logical_name": "CANDIDATE_POOL_V1",
        "version": "1.0",
        "path": "artifacts/candidate_pool/C001/candidate_pool_v1/package/train_candidates_v1.parquet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "status": "FROZEN"
    }
    os.makedirs(os.path.join(args.out_dir, "registry"), exist_ok=True)
    with open(os.path.join(args.out_dir, "registry", "artifact_registry.json"), "w") as f:
        json.dump(registry, f, indent=4)
        
    with open(os.path.join(run_dir, "candidate_pool_v1_FREEZE.json"), "w") as f:
        json.dump(registry, f, indent=4)
        
    # Pack Bundle
    logging.info("Packaging final bundle...")
    bundle_name = os.path.join(run_dir, "candidate_pool_v1_bundle.tar.gz")
    with tarfile.open(bundle_name, "w:gz") as tar:
        tar.add(package_dir, arcname="package")
        tar.add(manifest_dir, arcname="manifest")
        tar.add(metrics_dir, arcname="metrics")
        tar.add(os.path.join(run_dir, "config_snapshot.yaml"), arcname="config_snapshot.yaml")
        tar.add(os.path.join(run_dir, "candidate_pool_v1_FREEZE.json"), arcname="candidate_pool_v1_FREEZE.json")
        
    logging.info("DONE. FROZEN.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/c001_full_train.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--processed-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='artifacts/candidate_pool')
    parser.add_argument('--smoke-size', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=50000)
    args = parser.parse_args()
    
    if args.processed_dir is None:
        args.processed_dir = os.path.join(args.data_dir, "processed", "v001")
        
    run_experiment(args)
