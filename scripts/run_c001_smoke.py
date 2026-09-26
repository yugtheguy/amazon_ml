#!/usr/bin/env python3
import os
import sys
import gc
import json
import time
import yaml
import traceback
import psutil
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def get_host_ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def get_gpu_vram_gb():
    try:
        import cupy as cp
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        used_b = total_b - free_b
        return used_b / (1024 ** 3)
    except Exception:
        return None

def fmt(value, digits=2):
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def validate_shard(df, expected_start, expected_end):
    if df.empty:
        return True, "Empty output"
        
    required_cols = {"entity_id_s1", "candidate_source", "entity_id_cand"}
    if not required_cols.issubset(df.columns):
        return False, "Missing required columns"
        
    dups = df.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"]).sum()
    if dups > 0:
        return False, f"Duplicate pairs found: {dups}"
        
    sources = set(df["candidate_source"].unique())
    if not sources.issubset({"S2", "S3"}):
        return False, f"Invalid sources: {sources}"
        
    max_cands = df.groupby("entity_id_s1").size().max()
    if max_cands > 15:
        return False, f"Max candidates > 15: {max_cands}"
        
    # S1 identity (we could check if indices are in range if entity_id_s1 is sequential, but we just check presence)
    # The actual constraint: we shouldn't have candidates outside the requested range
    # In smoke test, it's just 0-5000 and 5000-10000 indices in the dataframe context
    
    return True, "Valid"

def process_shard(shard_id, s1_chunk, ctx, gen, out_dir):
    final_path = os.path.join(out_dir, f"shard_{shard_id}.parquet")
    meta_path = os.path.join(out_dir, f"shard_{shard_id}.metadata.json")
    done_path = os.path.join(out_dir, f"shard_{shard_id}.done")
    
    if os.path.exists(final_path) and os.path.exists(meta_path) and os.path.exists(done_path):
        print(f"Shard {shard_id} already complete. Skipping.")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return pd.read_parquet(final_path), meta
        
    print(f"Computing Shard {shard_id}...")
    t0 = time.time()
    union_df = gen.generate_with_context(s1_chunk, ctx)
    final_df = gen.rank_and_prune(union_df, max_candidates=15)
    elapsed = time.time() - t0
    
    tmp_path = final_path + ".tmp"
    final_df.to_parquet(tmp_path, index=False)
    
    # Validation
    is_valid, msg = validate_shard(final_df, None, None)
    if not is_valid:
        raise ValueError(f"Shard {shard_id} validation failed: {msg}")
        
    os.replace(tmp_path, final_path)
    
    meta = {
        "shard_id": shard_id,
        "country": "US",
        "s1_row_count": len(s1_chunk),
        "candidate_row_count": len(final_df),
        "unique_s1_count": final_df["entity_id_s1"].nunique() if not final_df.empty else 0,
        "duplicate_pair_count": int(final_df.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"]).sum()),
        "max_candidates_per_s1": int(final_df.groupby("entity_id_s1").size().max()) if not final_df.empty else 0,
        "elapsed_seconds": elapsed,
        "created_at": datetime.now().isoformat(),
        "status": "COMPLETE"
    }
    
    atomic_write_json(meta_path, meta)
    
    # Done marker last
    with open(done_path, "w") as f:
        f.write("")
        
    return final_df, meta

def print_smoke_report(metrics, passed):
    print("\n============================================================")
    print(f"C001_SMOKE_V2 = {'PASS' if passed else 'FAIL'}")
    print("============================================================")
    print(f"TARGET_ROWS_S2 = {fmt(metrics.get('target_rows_s2'), 0)}")
    print(f"TARGET_ROWS_S3 = {fmt(metrics.get('target_rows_s3'), 0)}")
    print(f"QUERY_ROWS = {fmt(metrics.get('query_rows'), 0)}\n")
    print(f"TARGET_CONTEXT_BUILD_SECONDS = {fmt(metrics.get('context_build_seconds'), 1)}")
    print(f"SHARD_A_SECONDS = {fmt(metrics.get('shard_a_seconds'), 1)}")
    print(f"SHARD_B_SECONDS = {fmt(metrics.get('shard_b_seconds'), 1)}")
    print(f"TOTAL_SECONDS = {fmt(metrics.get('total_seconds'), 1)}\n")
    print(f"HOST_RAM_AFTER_CONTEXT_GB = {fmt(metrics.get('host_ram_after_context_gb'), 2)}")
    print(f"HOST_RAM_AFTER_SHARD_A_GB = {fmt(metrics.get('host_ram_after_shard_a_gb'), 2)}")
    print(f"HOST_RAM_AFTER_SHARD_B_GB = {fmt(metrics.get('host_ram_after_shard_b_gb'), 2)}\n")

    try:
        import cupy as cp
        free_gb, total_gb = cp.cuda.runtime.memGetInfo()
        free_gb /= (1024**3)
        total_gb /= (1024**3)
        print(f"GPU_TOTAL_GB = {total_gb:.2f}")
        print(f"GPU_FREE_GB = {free_gb:.2f}")
        print(f"GPU_USED_GB = {total_gb - free_gb:.2f}\n")
    except Exception:
        pass

    print(f"GPU_AFTER_CONTEXT_GB = {fmt(metrics.get('gpu_after_context_gb'), 2)}")
    print(f"GPU_AFTER_SHARD_A_GB = {fmt(metrics.get('gpu_after_shard_a_gb'), 2)}")
    print(f"GPU_AFTER_SHARD_B_GB = {fmt(metrics.get('gpu_after_shard_b_gb'), 2)}\n")
    
    rebuild = metrics.get('shard_b_target_rebuild')
    reupload = metrics.get('shard_b_gpu_reupload')
    print(f"CACHE_OBJECT_IDENTITY_STABLE = {'YES' if rebuild is False else ('NO' if rebuild is True else 'UNKNOWN')}")
    print(f"SHARD_B_TARGET_REBUILD = {'YES' if rebuild is True else ('NO' if rebuild is False else 'UNKNOWN')}")
    print(f"SHARD_B_GPU_REUPLOAD = {'YES' if reupload is True else ('NO' if reupload is False else 'UNKNOWN')}\n")
    print(f"OUTPUT_ROWS = {fmt(metrics.get('output_rows'), 0)}")
    print(f"DUPLICATE_PAIR_KEYS = {fmt(metrics.get('duplicate_pair_keys'), 0)}")
    print(f"MAX_CANDIDATES_PER_S1 = {fmt(metrics.get('max_candidates_per_s1'), 0)}\n")
    
    print(f"PRODUCTION_METHOD = {fmt(metrics.get('production_method'), 0)}")
    print(f"NAME_TARGET_FIT_COUNT: {fmt(metrics.get('name_target_fit_count'), 0)}")
    print(f"ADDRESS_TARGET_FIT_COUNT: {fmt(metrics.get('address_target_fit_count'), 0)}")
    print(f"EXACT_BUILD_COUNT: {fmt(metrics.get('exact_build_count'), 0)}")
    print(f"RARE_BUILD_COUNT: {fmt(metrics.get('rare_build_count'), 0)}")
    print(f"NUMERIC_BUILD_COUNT: {fmt(metrics.get('numeric_build_count'), 0)}")
    print(f"GPU_TARGET_UPLOAD_COUNT: {fmt(metrics.get('gpu_target_upload_count'), 0)}\n")
    print(f"READY_FOR_FULL_C001 = {'YES' if passed else 'NO'}")
    print("============================================================")

def run_report_only(out_dir):
    report_path = os.path.join(out_dir, "smoke_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        print_smoke_report(report["metrics"], report["status"] == "PASS")
    else:
        print("ERROR: smoke_report.json not found.")
        sys.exit(1)

def run_validate_only(out_dir):
    print("Validating shards...")
    for shard_id in ["000", "001"]:
        final_path = os.path.join(out_dir, f"shard_{shard_id}.parquet")
        meta_path = os.path.join(out_dir, f"shard_{shard_id}.metadata.json")
        done_path = os.path.join(out_dir, f"shard_{shard_id}.done")
        
        if os.path.exists(final_path) and os.path.exists(meta_path) and os.path.exists(done_path):
            try:
                df = pd.read_parquet(final_path)
                is_valid, msg = validate_shard(df, None, None)
                if is_valid:
                    print(f"Shard {shard_id}: COMPLETE and VALID")
                else:
                    print(f"Shard {shard_id}: CORRUPT ({msg})")
            except Exception as e:
                print(f"Shard {shard_id}: ERROR reading ({e})")
        else:
            print(f"Shard {shard_id}: INCOMPLETE")
    sys.exit(0)

def main():
    print("SMOKE_HARNESS_VERSION = v3_metrics_dict")
    t_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed/v001")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR = args.processed_dir
    OUT_DIR = "artifacts/candidate_generation/C001_SMOKE_V3/US"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if args.report_only:
        run_report_only(OUT_DIR)
        return
        
    if args.validate_only:
        run_validate_only(OUT_DIR)
        return

    print("============================================================")
    print("C001 V3 SMOKE TEST")
    print("============================================================")
    
    s1_path = os.path.join(DATA_DIR, "train_source1.parquet")
    s2_path = os.path.join(DATA_DIR, "train_source2.parquet")
    s3_path = os.path.join(DATA_DIR, "train_source3.parquet")
    config_path = "configs/c001_full_train.yaml"
    
    for p in [s1_path, s2_path, s3_path, config_path]:
        if not os.path.exists(p):
            print(f"ERROR: Required path does not exist: {p}")
            sys.exit(1)
            
    if os.environ.get("ALLOW_CPU_TFIDF") == "1":
        print("ERROR: ALLOW_CPU_TFIDF is set to 1. Production GPU only.")
        sys.exit(1)
        
    try:
        import cuml
        import cupy as cp
        print("cuML and CuPy found.")
    except ImportError:
        print("C001_SMOKE_V3 = FAIL")
        print("REASON: cuML or CuPy not available. Production CPU fallback disabled.")
        sys.exit(1)
        
    from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator
    from src.business_entity_resolution.retrieval.target_context import TargetContext
            
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"\nLoading full US target data from {DATA_DIR}...")
    s2 = pd.read_parquet(s2_path, filters=[("country", "==", "US")])
    s3 = pd.read_parquet(s3_path, filters=[("country", "==", "US")])
    
    print("Loading 10k US query data...")
    s1 = pd.read_parquet(s1_path, filters=[("country", "==", "US")]).head(10000)
    
    metrics = {
        "target_rows_s2": len(s2),
        "target_rows_s3": len(s3),
        "query_rows": 10000,
        "context_build_seconds": None,
        "shard_a_seconds": None,
        "shard_b_seconds": None,
        "total_seconds": None,
        "host_ram_after_context_gb": None,
        "host_ram_after_shard_a_gb": None,
        "host_ram_after_shard_b_gb": None,
        "gpu_after_context_gb": None,
        "gpu_after_shard_a_gb": None,
        "gpu_after_shard_b_gb": None,
        "shard_b_target_rebuild": None,
        "shard_b_gpu_reupload": None,
        "output_rows": None,
        "duplicate_pair_keys": None,
        "max_candidates_per_s1": None,
        "production_method": "generate_with_context",
        "name_target_fit_count": None,
        "address_target_fit_count": None,
        "exact_build_count": None,
        "rare_build_count": None,
        "numeric_build_count": None,
        "gpu_target_upload_count": None,
    }
    
    print("\nBuilding US TargetContext...")
    t0 = time.time()
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, config, countries=["US"])
    metrics["context_build_seconds"] = time.time() - t0
    
    metrics["host_ram_after_context_gb"] = get_host_ram_gb()
    metrics["gpu_after_context_gb"] = get_gpu_vram_gb()
    
    gen = CandidateGenerator(config)
    
    ctx_us_s2 = ctx.get("S2", "US")
    def get_identities(c):
        return {
            "name_vec": id(c.name_cache.vec) if c.name_cache else None,
            "name_gpu": id(c.name_cache.X_target_gpu) if c.name_cache and c.name_cache.X_target_gpu is not None else None,
            "exact": id(c.target_unique_name_df)
        }
    id_before_A = get_identities(ctx_us_s2)

    # SHARD A (000)
    s1_a = s1.iloc[0:5000]
    final_a, meta_a = process_shard("000", s1_a, ctx, gen, OUT_DIR)
    metrics["shard_a_seconds"] = meta_a["elapsed_seconds"]
    
    del s1_a
    gc.collect()
    if os.environ.get("ALLOW_CPU_TFIDF") != "1":
        cp.get_default_memory_pool().free_all_blocks()
    
    metrics["host_ram_after_shard_a_gb"] = get_host_ram_gb()
    metrics["gpu_after_shard_a_gb"] = get_gpu_vram_gb()
    
    id_before_B = get_identities(ctx_us_s2)
    
    # SHARD B (001)
    s1_b = s1.iloc[5000:10000]
    final_b, meta_b = process_shard("001", s1_b, ctx, gen, OUT_DIR)
    metrics["shard_b_seconds"] = meta_b["elapsed_seconds"]
    
    del s1_b
    gc.collect()
    if os.environ.get("ALLOW_CPU_TFIDF") != "1":
        cp.get_default_memory_pool().free_all_blocks()
    
    metrics["host_ram_after_shard_b_gb"] = get_host_ram_gb()
    metrics["gpu_after_shard_b_gb"] = get_gpu_vram_gb()
    
    id_after_B = get_identities(ctx_us_s2)
    
    # Combine outputs for overall stats
    output_df = pd.concat([final_a, final_b])
    metrics["output_rows"] = len(output_df)
    
    dups = output_df.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"]).sum()
    metrics["duplicate_pair_keys"] = int(dups)
    metrics["max_candidates_per_s1"] = int(output_df.groupby("entity_id_s1").size().max())
    
    rebuild_target = (id_before_A != id_before_B) or (id_before_B != id_after_B)
    reupload_gpu = (id_before_A["name_gpu"] != id_before_B["name_gpu"]) or (id_before_B["name_gpu"] != id_after_B["name_gpu"])
    metrics["shard_b_target_rebuild"] = rebuild_target
    metrics["shard_b_gpu_reupload"] = reupload_gpu
    
    metrics["name_target_fit_count"] = gen._NAME_TARGET_FIT_COUNT
    metrics["address_target_fit_count"] = gen._ADDRESS_TARGET_FIT_COUNT
    metrics["exact_build_count"] = gen._EXACT_BUILD_COUNT
    metrics["rare_build_count"] = gen._RARE_BUILD_COUNT
    metrics["numeric_build_count"] = gen._NUMERIC_BUILD_COUNT
    metrics["gpu_target_upload_count"] = gen._TARGET_GPU_UPLOAD_COUNT
    
    metrics["total_seconds"] = time.time() - t_start

    passed = (
        metrics["max_candidates_per_s1"] <= 15 and 
        metrics["duplicate_pair_keys"] == 0 and 
        not rebuild_target and not reupload_gpu
    )
    
    report = {
        "status": "PASS" if passed else "FAIL",
        "metrics": metrics,
        "implementation_version": "v3_metrics_dict",
        "timestamp": datetime.now().isoformat()
    }
    
    atomic_write_json(os.path.join(OUT_DIR, "smoke_report.json"), report)
    
    manifest = {
        "architecture_fingerprint": "C001_V2_US_SMOKE",
        "implementation_version": "v3",
        "completed_shards": ["000", "001"],
        "status": report["status"]
    }
    atomic_write_json(os.path.join(OUT_DIR, "run_manifest.json"), manifest)
    
    try:
        print_smoke_report(metrics, passed)
    except Exception as e:
        print(f"REPORTING_WARNING: {e}")
    
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("C001_SMOKE_V3 = FAIL")
        print("FAILED STAGE / EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)
