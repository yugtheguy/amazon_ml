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
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
import psutil
import gc

from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator

def memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def gpu_usage_gb():
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / (1024**3)
    except Exception:
        return 0.0

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
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception:
        return "unknown"

def check_shard_done(final_dir, shard_name):
    return os.path.exists(os.path.join(final_dir, f"{shard_name}.done"))

def mark_shard_done(final_dir, shard_name, metadata):
    with open(os.path.join(final_dir, f"{shard_name}.metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(final_dir, f"{shard_name}.done"), "w") as f:
        f.write("DONE")

def atomic_write_parquet(df, filepath):
    tmp_path = filepath + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.rename(tmp_path, filepath)

def validate_shard(s1_chunk, final_df, gt_dict, union_df, max_candidates=15):
    if not final_df.empty:
        counts = final_df.groupby('entity_id_s1').size()
        if (counts > max_candidates).any():
            raise ValueError("Candidate count exceeds max_candidates")
        
        final_pairs = set(zip(final_df['entity_id_s1'], final_df['entity_id_cand']))
        internal_pairs = set(zip(union_df['entity_id_s1'], union_df['entity_id_cand']))
        if not final_pairs.issubset(internal_pairs):
            raise ValueError("Final candidates are not a subset of internal candidates")
            
        dups = final_df.duplicated(subset=['entity_id_s1', 'candidate_source', 'entity_id_cand'])
        if dups.any():
            raise ValueError("Duplicate relationships found in final candidates")

def evaluate_pool(cands_map, gt_dict, s1_sample_ids):
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

def run_worker(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    country = args.country
    logging.info(f"=== WORKER START: {country} ===")
    
    # 2. COLUMN PROJECTION
    required_cols = [
        'entity_id', 'country', 'name_norm_clean', 'addr_norm_clean',
        'name_norm_accent_fold', 'addr_norm_accent_fold', 'name_norm_punct',
        'addr_norm_punct', 'addr_numeric_tokens'
    ]
    
    processed_dir = args.processed_dir
    data_dir = args.data_dir
    
    # 3. FILTER AT READ TIME
    s1 = pd.read_parquet(os.path.join(processed_dir, "train_source1.parquet"), columns=required_cols, filters=[('country', '==', country)])
    s2 = pd.read_parquet(os.path.join(processed_dir, "train_source2.parquet"), columns=required_cols, filters=[('country', '==', country)])
    s3 = pd.read_parquet(os.path.join(processed_dir, "train_source3.parquet"), columns=required_cols, filters=[('country', '==', country)])
    
    if args.smoke_size > 0:
        # Just cut s1
        s1 = s1.head(args.smoke_size)
    
    logging.info(f"[{country}] S1:{len(s1)} S2:{len(s2)} S3:{len(s3)}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_hash = compute_config_hash(config)
    
    generator = CandidateGenerator(config)
    max_cands = config.get('pruning', {}).get('candidate_budgets', [15])[-1]
    
    run_dir = os.path.join(args.out_dir, "C001", "candidate_pool_v1")
    internal_dir = os.path.join(run_dir, "internal")
    final_dir = os.path.join(run_dir, "final")
    metrics_dir = os.path.join(run_dir, "metrics")
    
    # 4. DO NOT KEEP GT/FOLDS DURING RETRIEVAL
    # Load only GT for the active S1
    gt = pd.read_csv(os.path.join(data_dir, "raw", "train", "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    active_s1_ids = set(s1['entity_id'].values)
    gt = gt[gt['source1_entity_id'].isin(active_s1_ids)]
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(',')) if row.matched_entity_ids else set()
        for row in gt.itertuples(index=False)
    }
    del gt
    gc.collect()
    
    chunk_size = args.chunk_size
    s1_sorted = s1.sort_values('entity_id').reset_index(drop=True)
    num_chunks = (len(s1_sorted) + chunk_size - 1) // chunk_size
    
    # RAM WATCHDOG prep
    rss_history = []
    MIN_AVAILABLE_RAM_GB = 4.0
    
    for i in range(num_chunks):
        shard_name = f"{country}_shard_{i:03d}"
        
        # 8. RAM WATCHDOG
        avail_gb = psutil.virtual_memory().available / (1024**3)
        rss_gb = memory_usage_gb()
        gpu_gb = gpu_usage_gb()
        logging.info(f"--- Processing {shard_name} | Shard {i+1}/{num_chunks} ---")
        logging.info(f"WATCHDOG: RSS {rss_gb:.2f}GB | Avail {avail_gb:.2f}GB | GPU {gpu_gb:.2f}GB")
        
        if avail_gb < MIN_AVAILABLE_RAM_GB:
            logging.error(f"CRITICAL: Available RAM ({avail_gb:.2f}GB) is below safety floor ({MIN_AVAILABLE_RAM_GB}GB). Gracefully exiting.")
            sys.exit(2)
            
        # 9. MEMORY-GROWTH DETECTOR
        rss_history.append(rss_gb)
        if len(rss_history) >= 3:
            growth = rss_history[-1] - rss_history[-3]
            if growth > 3.0: # Material growth of >3GB over 3 shards
                logging.error(f"CRITICAL: Monotonic memory growth detected ({growth:.2f}GB). Halting to prevent crash.")
                sys.exit(3)
        
        if check_shard_done(final_dir, shard_name):
            logging.info(f"Shard {shard_name} already DONE. Skipping.")
            continue
            
        s1_chunk = s1_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        t0 = time.time()
        
        union_df = generator.generate(s1_chunk, {'S2': s2, 'S3': s3})
        atomic_write_parquet(union_df, os.path.join(internal_dir, f"{shard_name}.parquet"))
        
        final_df = generator.rank_and_prune(union_df, max_candidates=max_cands)
        validate_shard(s1_chunk, final_df, gt_dict, union_df, max_cands)
        atomic_write_parquet(final_df, os.path.join(final_dir, f"{shard_name}.parquet"))
        
        # Misses
        chunk_missed_list = []
        final_pairs = set(zip(final_df['entity_id_s1'], final_df['entity_id_cand']))
        internal_pairs = set(zip(union_df['entity_id_s1'], union_df['entity_id_cand']))
        for s1_id in s1_chunk['entity_id'].values:
            matches = gt_dict.get(s1_id, set())
            for m in matches:
                if (s1_id, m) not in final_pairs:
                    is_internal = (s1_id, m) in internal_pairs
                    chunk_missed_list.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": m,
                        "miss_type": "pruning_miss" if is_internal else "retrieval_miss"
                    })
        misses_df = pd.DataFrame(chunk_missed_list) if chunk_missed_list else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "miss_type"])
        atomic_write_parquet(misses_df, os.path.join(metrics_dir, f"{shard_name}_misses.parquet"))
        
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
        
        # 6. PER-SHARD MEMORY LIFECYCLE
        del s1_chunk
        del union_df
        del final_df
        del final_pairs
        del internal_pairs
        del misses_df
        if 'cp' in sys.modules:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        gc.collect()
        logging.info(f"Shard {shard_name} completed in {time.time()-t0:.1f}s.")
        
    # 7. COUNTRY-BOUNDARY CLEANUP
    # Release variables cleanly before exit
    generator._tfidf_cache.clear()
    del generator
    del s1
    del s2
    del s3
    gc.collect()
    if 'cp' in sys.modules:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    logging.info(f"=== WORKER END: {country} ===")

def run_orchestrator(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("=== C001 FULL TRAIN ORCHESTRATOR ===")
    
    if os.environ.get("ALLOW_CPU_TFIDF") != "1":
        gpu_status = check_gpu()
        if "CPU Only" in gpu_status:
            raise RuntimeError("GPU_REQUIRED=TRUE. Stopping loudly. cuML/CuPy is not available.")
            
    processed_dir = args.processed_dir
    
    # Fast unique country fetch
    s1_meta = pq.read_table(os.path.join(processed_dir, "train_source1.parquet"), columns=['country'])
    countries = sorted(pd.Series(s1_meta['country']).unique())
    del s1_meta
    gc.collect()
    
    logging.info(f"Found {len(countries)} countries: {countries}")
    
    for country in countries:
        logging.info(f"\n>>> Spawning worker for country: {country}")
        cmd = [
            sys.executable, "-u", __file__,
            "--country", country,
            "--config", args.config,
            "--data-dir", args.data_dir,
            "--processed-dir", args.processed_dir,
            "--out-dir", args.out_dir,
            "--smoke-size", str(args.smoke_size),
            "--chunk-size", str(args.chunk_size)
        ]
        
        env = os.environ.copy()
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            logging.error(f"CRITICAL: Worker for {country} failed with code {res.returncode}. Halting orchestrator.")
            sys.exit(res.returncode)
            
    logging.info("\n>>> ALL WORKERS FINISHED. Finalizing datasets...")
    
    run_dir = os.path.join(args.out_dir, "C001", "candidate_pool_v1")
    final_dir = os.path.join(run_dir, "final")
    metrics_dir = os.path.join(run_dir, "metrics")
    manifest_dir = os.path.join(run_dir, "manifest")
    package_dir = os.path.join(run_dir, "package")
    
    for d in [manifest_dir, package_dir]:
        os.makedirs(d, exist_ok=True)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_hash = compute_config_hash(config)
    git_commit = get_git_commit()
    max_cands = config.get('pruning', {}).get('candidate_budgets', [15])[-1]
    
    # 5. NO GLOBAL FINAL CONCAT
    # Streaming candidate write
    final_files = [os.path.join(final_dir, f) for f in os.listdir(final_dir) if f.endswith(".parquet") and "misses" not in f]
    final_out_path = os.path.join(package_dir, "train_candidates_v1.parquet")
    
    cands_map = defaultdict(set)
    writer = None
    output_final_cands = 0
    
    for f in final_files:
        df = pd.read_parquet(f)
        if not df.empty:
            output_final_cands += len(df)
            for row in df.itertuples(index=False):
                cands_map[row.entity_id_s1].add(row.entity_id_cand)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(final_out_path, table.schema)
            writer.write_table(table)
        del df
        gc.collect()
        
    if writer:
        writer.close()
    else:
        # write empty
        pd.DataFrame().to_parquet(final_out_path)
        
    # Process manifest stream
    cand_counts = pd.DataFrame([
        {'entity_id_s1': k, 'candidate_count': len(v)}
        for k, v in cands_map.items()
    ])
    
    fold_path = os.path.normpath(os.path.join(args.out_dir, "..", "folds", "fold_manifest_v1.parquet"))
    if not os.path.exists(fold_path):
        raise FileNotFoundError(f"CRITICAL: Frozen folds_v1.parquet not found at {fold_path}")
    folds_df = pd.read_parquet(fold_path)[['entity_id', 'fold_id']]
    if folds_df['entity_id'].duplicated().any():
        raise ValueError("Frozen folds contain duplicate entity_ids!")
        
    s1_all = pd.read_parquet(os.path.join(processed_dir, "train_source1.parquet"), columns=['entity_id', 'country'])
    if args.smoke_size > 0:
        s1_all = s1_all.head(args.smoke_size)
    
    manifest_df = s1_all.rename(columns={'entity_id': 'entity_id_s1'})
    manifest_df = manifest_df.merge(folds_df.rename(columns={'entity_id': 'entity_id_s1', 'fold_id': 'fold'}), on='entity_id_s1', how='left')
    
    if manifest_df['fold'].isna().any():
        raise ValueError("Some S1 entities are missing from the frozen folds_v1.parquet!")
        
    manifest_df = manifest_df.merge(cand_counts, on='entity_id_s1', how='left')
    manifest_df['candidate_count'] = manifest_df['candidate_count'].fillna(0).astype(int)
    manifest_df['has_candidates'] = (manifest_df['candidate_count'] > 0).astype(int)
    manifest_df.to_parquet(os.path.join(package_dir, "train_candidate_entity_manifest_v1.parquet"), index=False)
    
    # GT Miss Analysis stream
    miss_files = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith("_misses.parquet")]
    m_writer = None
    for mf in miss_files:
        df = pd.read_parquet(mf)
        if not df.empty:
            table = pa.Table.from_pandas(df)
            if m_writer is None:
                m_writer = pq.ParquetWriter(os.path.join(metrics_dir, "missed_gt_pairs_v1.parquet"), table.schema)
            m_writer.write_table(table)
        del df
        gc.collect()
    if m_writer:
        m_writer.close()
    else:
        pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "miss_type"]).to_parquet(os.path.join(metrics_dir, "missed_gt_pairs_v1.parquet"))
        
    # Evaluate
    gt = pd.read_csv(os.path.join(args.data_dir, "raw", "train", "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(',')) if row.matched_entity_ids else set()
        for row in gt.itertuples(index=False)
    }
    del gt
    gc.collect()
    
    eval_res = evaluate_pool(cands_map, gt_dict, s1_all['entity_id'].values)
    with open(os.path.join(metrics_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_res, f, indent=4)
        
    run_manifest = {
        "experiment_id": "C001",
        "architecture": "R001",
        "candidate_budget": max_cands,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_s1_rows": len(s1_all),
        "output_final_cands": output_final_cands,
        "validation_status": "PASSED"
    }
    with open(os.path.join(manifest_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)
        
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
        
    logging.info("Packaging final bundle...")
    bundle_name = os.path.join(run_dir, "candidate_pool_v1_bundle.tar.gz")
    with tarfile.open(bundle_name, "w:gz") as tar:
        tar.add(package_dir, arcname="package")
        tar.add(manifest_dir, arcname="manifest")
        tar.add(metrics_dir, arcname="metrics")
        
    logging.info("DONE. FROZEN.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/c001_full_train.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--processed-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='artifacts/candidate_pool')
    parser.add_argument('--smoke-size', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=50000)
    parser.add_argument('--country', type=str, default=None)
    args = parser.parse_args()
    
    if args.processed_dir is None:
        args.processed_dir = os.path.join(args.data_dir, "processed", "v001")
        
    if args.country:
        run_worker(args)
    else:
        run_orchestrator(args)
