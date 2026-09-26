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

def get_host_ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def get_gpu_vram_gb():
    try:
        import cupy as cp
        return cp.get_default_memory_pool().used_bytes() / (1024 ** 3)
    except Exception:
        return 0.0

def main():
    t_start = time.time()
    print("============================================================")
    print("C001 V2 SMOKE TEST")
    print("============================================================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed/v001")
    args = parser.parse_args()

    DATA_DIR = args.processed_dir
    OUT_DIR = "artifacts/candidate_generation/C001_SMOKE_V2"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    s1_path = os.path.join(DATA_DIR, "train_source1.parquet")
    s2_path = os.path.join(DATA_DIR, "train_source2.parquet")
    s3_path = os.path.join(DATA_DIR, "train_source3.parquet")
    config_path = "configs/c001_full_train.yaml"
    
    print("\n============================================================")
    print("RESOLVED_PATHS:")
    print(f"S1 = {s1_path}")
    print(f"S2 = {s2_path}")
    print(f"S3 = {s3_path}")
    print(f"CONFIG = {config_path}")
    print(f"OUTPUT_ROOT = {OUT_DIR}")
    print("============================================================\n")
    
    for p in [s1_path, s2_path, s3_path, config_path]:
        if not os.path.exists(p):
            print(f"ERROR: Required path does not exist: {p}")
            sys.exit(1)
            
    # 1. PRODUCTION GPU ONLY
    if os.environ.get("ALLOW_CPU_TFIDF") == "1":
        print("ERROR: ALLOW_CPU_TFIDF is set to 1. Production GPU only.")
        sys.exit(1)
        
    try:
        import cuml
        import cupy as cp
        print("cuML and CuPy found.")
    except ImportError:
        print("C001_SMOKE_V2 = FAIL")
        print("REASON: cuML or CuPy not available. Production CPU fallback disabled.")
        sys.exit(1)
        
    # Import internals here after GPU check
    from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator
    from src.business_entity_resolution.retrieval.target_context import TargetContext
            
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"\nLoading full US target data from {DATA_DIR}...")
    s2 = pd.read_parquet(s2_path, filters=[("country", "==", "US")])
    s3 = pd.read_parquet(s3_path, filters=[("country", "==", "US")])
    
    print("Loading 10k US query data...")
    s1 = pd.read_parquet(s1_path, filters=[("country", "==", "US")]).head(10000)
    
    target_rows_s2 = len(s2)
    target_rows_s3 = len(s3)
    
    print("\nBuilding US TargetContext...")
    t0 = time.time()
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, config, countries=["US"])
    ctx_build_s = time.time() - t0
    
    ram_after_ctx = get_host_ram_gb()
    gpu_after_ctx = get_gpu_vram_gb()
    
    gen = CandidateGenerator(config)
    
    # SHARD A
    print("\nRunning Shard A (0:5000)...")
    s1_a = s1.iloc[0:5000]
    t0 = time.time()
    union_a = gen.generate_with_context(s1_a, ctx)
    final_a = gen.rank_and_prune(union_a, max_candidates=15)
    shard_a_s = time.time() - t0
    
    final_a.to_parquet(os.path.join(OUT_DIR, "shard_a.parquet"), index=False)
    
    del s1_a, union_a
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    ram_after_a = get_host_ram_gb()
    gpu_after_a = get_gpu_vram_gb()

    # SHARD B
    print("\nRunning Shard B (5000:10000)...")
    s1_b = s1.iloc[5000:10000]
    t0 = time.time()
    union_b = gen.generate_with_context(s1_b, ctx)
    final_b = gen.rank_and_prune(union_b, max_candidates=15)
    shard_b_s = time.time() - t0
    
    final_b.to_parquet(os.path.join(OUT_DIR, "shard_b.parquet"), index=False)
    
    del s1_b, union_b
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    ram_after_b = get_host_ram_gb()
    gpu_after_b = get_gpu_vram_gb()
    
    # 6. Basic validation
    output_df = pd.concat([final_a, final_b])
    output_rows = len(output_df)
    
    dups = output_df.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"]).sum()
    max_cands = output_df.groupby("entity_id_s1").size().max()
    sources = set(output_df["candidate_source"].unique())
    
    valid_sources = sources.issubset({"S2", "S3"})
    valid_count = max_cands <= 15
    valid_dups = dups == 0
    
    channels = [
        "retrieved_exact", "retrieved_name_word", "retrieved_addr_word", 
        "retrieved_rare", "retrieved_numeric"
    ]
    channels_executed = all(c in output_df.columns and output_df[c].sum() > 0 for c in channels)
    
    # 7. Verify reuse (implicitly checked by Shard B time << Context Build Time, and memory stability)
    rebuild_target = shard_b_s > (ctx_build_s * 0.5)
    reupload_gpu = shard_b_s > 30.0 # Strict bound for pure retrieval + rank
    
    passed = (
        valid_sources and valid_count and valid_dups and channels_executed and 
        not rebuild_target and not reupload_gpu
    )
    
    total_s = time.time() - t_start
    
    print("\n============================================================")
    print(f"C001_SMOKE_V2 = {'PASS' if passed else 'FAIL'}")
    print("============================================================")
    print(f"TARGET_ROWS_S2 = {target_rows_s2}")
    print(f"TARGET_ROWS_S3 = {target_rows_s3}")
    print(f"QUERY_ROWS = 10000\n")
    print(f"TARGET_CONTEXT_BUILD_SECONDS = {ctx_build_s:.1f}")
    print(f"SHARD_A_SECONDS = {shard_a_s:.1f}")
    print(f"SHARD_B_SECONDS = {shard_b_s:.1f}")
    print(f"TOTAL_SECONDS = {total_s:.1f}\n")
    print(f"HOST_RAM_AFTER_CONTEXT_GB = {ram_after_ctx:.2f}")
    print(f"HOST_RAM_AFTER_SHARD_A_GB = {ram_after_a:.2f}")
    print(f"HOST_RAM_AFTER_SHARD_B_GB = {ram_after_b:.2f}\n")
    print(f"GPU_AFTER_CONTEXT_GB = {gpu_after_ctx:.2f}")
    print(f"GPU_AFTER_SHARD_A_GB = {gpu_after_a:.2f}")
    print(f"GPU_AFTER_SHARD_B_GB = {gpu_after_b:.2f}\n")
    print(f"SHARD_B_TARGET_REBUILD = {'YES' if rebuild_target else 'NO'}")
    print(f"SHARD_B_GPU_REUPLOAD = {'YES' if reupload_gpu else 'NO'}\n")
    print(f"OUTPUT_ROWS = {output_rows}")
    print(f"DUPLICATE_PAIR_KEYS = {dups}")
    print(f"MAX_CANDIDATES_PER_S1 = {max_cands}\n")
    print(f"READY_FOR_FULL_C001 = {'YES' if passed else 'NO'}")
    print("============================================================")
    
    if not passed:
        sys.exit(1)
        
    # Write explicit machine-readable success marker
    with open(os.path.join(OUT_DIR, "C001_SMOKE_V2_PASS.json"), "w") as f:
        json.dump({"status": "PASS"}, f)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("C001_SMOKE_V2 = FAIL")
        print("FAILED STAGE / EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)
