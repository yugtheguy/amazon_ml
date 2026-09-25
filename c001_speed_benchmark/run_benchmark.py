import sys
import os
from pathlib import Path
import time
import pandas as pd
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator
import src.business_entity_resolution.retrieval.gpu_char_tfidf as gpu_char_tfidf
from c001_speed_benchmark.optimized_gpu import sp_matmul_topn_cupy_optimized

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def get_vram_usage():
    if not HAS_CUPY:
        return 0
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes() / (1024 ** 3)

def hash_candidates(df):
    df_sorted = df.sort_values(["entity_id_s1", "candidate_source", "entity_id_cand"]).reset_index(drop=True)
    csv_str = df_sorted.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(csv_str).hexdigest()

def run_benchmark():
    if not HAS_CUPY:
        print("ERROR: CuPy is not installed. Benchmark must be run on GPU.")
        return

    country = "India"
    n_queries = 2000
    
    print("Loading data...")
    s1 = pd.read_parquet("artifacts/datasets/train_source1.parquet")
    s2 = pd.read_parquet("artifacts/datasets/train_source2.parquet")
    s3 = pd.read_parquet("artifacts/datasets/train_source3.parquet")
    
    s1_ind = s1[s1["country"] == country].head(n_queries)
    s2_ind = s2[s2["country"] == country]
    s3_ind = s3[s3["country"] == country]
    
    print(f"Data loaded: S1={len(s1_ind)}, S2={len(s2_ind)}, S3={len(s3_ind)}")
    
    config = {
        "name_word": {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "max_df": 0.9, "top_k_per_source": 5, "batch_size": 100},
        "address_word": {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "max_df": 0.9, "top_k_per_source": 5, "batch_size": 100},
    }
    
    cg = CandidateGenerator(s2_ind, s3_ind, tfidf_config=config, device="gpu")
    
    # 1. Warmup and Baseline (Unoptimized Production Code)
    print("\n========================================================")
    print("Running Baseline (B100, Unoptimized CPU top-k)...")
    print("========================================================")
    t0 = time.perf_counter()
    base_cands = cg.generate_candidates(s1_ind, k=15, country=country)
    base_time = time.perf_counter() - t0
    base_hash = hash_candidates(base_cands)
    print(f"Baseline (B100) Runtime: {base_time:.2f}s, Hash: {base_hash}")
    
    # Patch the candidate generator to use our OPTIMIZED GPU top-k kernel
    print("\nPatching production code with GPU-only top-K kernel...")
    gpu_char_tfidf.sp_matmul_topn_cupy = sp_matmul_topn_cupy_optimized
    
    batch_sizes = [100, 256, 512, 1024]
    results = []
    
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} (Optimized) ---")
        cg.tfidf_config["name_word"]["batch_size"] = bs
        cg.tfidf_config["address_word"]["batch_size"] = bs
        
        # Clear VRAM if possible
        cp.get_default_memory_pool().free_all_blocks()
        
        t0 = time.perf_counter()
        cands = cg.generate_candidates(s1_ind, k=15, country=country)
        runtime = time.perf_counter() - t0
        
        cands_hash = hash_candidates(cands)
        equivalent = (cands_hash == base_hash)
        
        vram = get_vram_usage()
        qps = n_queries / runtime
        speedup = base_time / runtime
        
        print(f"Runtime: {runtime:.2f}s | QPS: {qps:.1f} | Speedup: {speedup:.2f}x | VRAM: {vram:.2f}GB | Match: {equivalent}")
        results.append({
            "Batch": bs,
            "Runtime": runtime,
            "QPS": qps,
            "Speedup": speedup,
            "Peak VRAM": vram,
            "Equivalent": equivalent,
            "Hash": cands_hash
        })
        
    print("\n--- Summary ---")
    print(f"{'Batch':<10} | {'Runtime (s)':<12} | {'QPS':<10} | {'Speedup':<10} | {'Peak VRAM':<10} | {'Output Equivalent'}")
    print("-" * 80)
    for r in results:
        print(f"{r['Batch']:<10} | {r['Runtime']:<12.2f} | {r['QPS']:<10.1f} | {r['Speedup']:<9.2f}x | {r['Peak VRAM']:<10.2f} | {str(r['Equivalent'])}")

if __name__ == "__main__":
    run_benchmark()
