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
from src.business_entity_resolution.retrieval.target_context import (
    TargetContext,
    IMPLEMENTATION_VERSION,
)

# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINT SEPARATION
# ARCHITECTURE_FINGERPRINT = hash of retrieval config (normalization, params, K, ranking)
# IMPLEMENTATION_VERSION   = "v2_target_context" (cache/execution changes only)
# Engineering-only changes do NOT invalidate completed shards if arch fingerprint matches.
# ─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE_KEYS = [
    "name_word", "address_word", "rare_token", "numeric", "pruning"
]


def memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def available_ram_gb():
    return psutil.virtual_memory().available / (1024 ** 3)


def disk_free_gb(path="/kaggle/working"):
    try:
        usage = psutil.disk_usage(path if os.path.exists(path) else ".")
        return usage.free / (1024 ** 3)
    except Exception:
        return -1.0


def gpu_usage_gb():
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / (1024 ** 3)
    except Exception:
        return 0.0


def gpu_free_gb():
    try:
        import cupy as cp
        mem = cp.cuda.Device().mem_info
        return mem[0] / (1024 ** 3)
    except Exception:
        return -1.0


def check_gpu():
    try:
        import cuml
        import cupy as cp
        return "cuML/CuPy Available"
    except ImportError:
        return "CPU Only"


def compute_config_hash(config):
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def compute_architecture_fingerprint(config):
    """Hash only the semantic retrieval keys. Excludes implementation fields."""
    arch = {k: config[k] for k in ARCHITECTURE_KEYS if k in config}
    s = json.dumps(arch, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
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
    os.replace(tmp_path, filepath)


def validate_shard(s1_chunk, final_df, gt_dict, union_df, max_candidates=15):
    if not final_df.empty:
        counts = final_df.groupby("entity_id_s1").size()
        if (counts > max_candidates).any():
            raise ValueError("Candidate count exceeds max_candidates")
        final_pairs    = set(zip(final_df["entity_id_s1"], final_df["entity_id_cand"]))
        internal_pairs = set(zip(union_df["entity_id_s1"], union_df["entity_id_cand"]))
        if not final_pairs.issubset(internal_pairs):
            raise ValueError("Final candidates are not a subset of internal candidates")
        dups = final_df.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"])
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
    full_cov    = full_cov_count / s1_with_gt if s1_with_gt > 0 else 0
    cand_sizes  = [len(cands_map.get(s1_id, set())) for s1_id in s1_sample_ids]
    mean_zero   = float(np.mean(zero_cands)) if zero_cands else 0.0
    zero_zero_pct = float(sum(1 for z in zero_cands if z == 0) / len(zero_cands)) if zero_cands else 0.0
    return {
        "pair_recall":   float(pair_recall),
        "full_coverage": float(full_cov),
        "mean_candidates": float(np.mean(cand_sizes)),
        "median": float(np.median(cand_sizes)),
        "p90": float(np.percentile(cand_sizes, 90)),
        "p95": float(np.percentile(cand_sizes, 95)),
        "p99": float(np.percentile(cand_sizes, 99)),
        "max": int(np.max(cand_sizes)),
        "zero_match_mean_candidates": mean_zero,
        "zero_match_zero_cand_pct":   zero_zero_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — one per country, exits after all shards done
# ─────────────────────────────────────────────────────────────────────────────

def run_worker(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    country = args.country
    logging.info("=== WORKER START: %s (impl=%s) ===", country, IMPLEMENTATION_VERSION)

    required_cols = [
        "entity_id", "country", "name_norm_clean", "addr_norm_clean",
        "name_norm_accent_fold", "addr_norm_accent_fold",
        "name_norm_punct", "addr_norm_punct", "addr_numeric_tokens",
    ]
    processed_dir = args.processed_dir

    # ── 1. Load country-filtered sources (projection + filter at read time) ─
    logging.info("[%s] Loading S1/S2/S3 (country filter at read)...", country)
    t_load = time.time()
    s1 = pd.read_parquet(
        os.path.join(processed_dir, "train_source1.parquet"),
        columns=required_cols, filters=[("country", "==", country)],
    )
    s2 = pd.read_parquet(
        os.path.join(processed_dir, "train_source2.parquet"),
        columns=required_cols, filters=[("country", "==", country)],
    )
    s3 = pd.read_parquet(
        os.path.join(processed_dir, "train_source3.parquet"),
        columns=required_cols, filters=[("country", "==", country)],
    )
    if args.smoke_size > 0:
        s1 = s1.head(args.smoke_size)
    logging.info(
        "[%s] Load done in %.1fs | S1:%d S2:%d S3:%d",
        country, time.time() - t_load, len(s1), len(s2), len(s3),
    )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_hash              = compute_config_hash(config)
    architecture_fingerprint = compute_architecture_fingerprint(config)
    generator  = CandidateGenerator(config)
    max_cands  = config.get("pruning", {}).get("candidate_budgets", [15])[-1]

    run_dir      = os.path.join(args.out_dir, "C001", "candidate_pool_v1")
    internal_dir = os.path.join(run_dir, "internal")
    final_dir    = os.path.join(run_dir, "final")
    metrics_dir  = os.path.join(run_dir, "metrics")
    for d in [internal_dir, final_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)

    # GT (for diagnostics only — not used during retrieval)
    gt = pd.read_csv(
        os.path.join(args.data_dir, "raw", "train", "train_ground_truth.tsv"),
        sep="\t", dtype=str,
    ).fillna("")
    active_s1_ids = set(s1["entity_id"].values)
    gt = gt[gt["source1_entity_id"].isin(active_s1_ids)]
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        for row in gt.itertuples(index=False)
    }
    del gt
    gc.collect()

    # ── 2. BUILD TARGET CONTEXT ONCE ────────────────────────────────────────
    logging.info("[%s] Building TargetContext (ONCE for all shards)...", country)
    t_ctx = time.time()
    target_ctx = TargetContext()
    target_ctx.build(
        source_dfs={"S2": s2, "S3": s3},
        config=config,
        countries=[country],
    )
    ctx_seconds = time.time() - t_ctx
    logging.info(
        "[%s] TargetContext built in %.1fs. Summary: %s",
        country, ctx_seconds, json.dumps(target_ctx.summary()),
    )

    # Memory after context build
    rss_after_ctx = memory_usage_gb()
    logging.info(
        "[%s] Memory after TargetContext: RSS=%.2fGB Avail=%.2fGB GPU=%.2fGB",
        country, rss_after_ctx, available_ram_gb(), gpu_usage_gb(),
    )

    chunk_size  = args.chunk_size
    s1_sorted   = s1.sort_values("entity_id").reset_index(drop=True)
    num_chunks  = (len(s1_sorted) + chunk_size - 1) // chunk_size

    MIN_AVAILABLE_RAM_GB  = 4.0
    MIN_DISK_FREE_GB      = 5.0
    rss_history: list     = []

    # ── 3. SHARD LOOP ───────────────────────────────────────────────────────
    for i in range(num_chunks):
        shard_name = f"{country}_shard_{i:03d}"

        # ── WATCHDOG ──────────────────────────────────────────────────────
        avail_gb = available_ram_gb()
        rss_gb   = memory_usage_gb()
        gpu_gb   = gpu_usage_gb()
        disk_gb  = disk_free_gb()
        logging.info(
            "--- Shard %s | %d/%d | RSS=%.2fGB Avail=%.2fGB GPU=%.2fGB Disk=%.1fGB ---",
            shard_name, i + 1, num_chunks, rss_gb, avail_gb, gpu_gb, disk_gb,
        )

        if avail_gb < MIN_AVAILABLE_RAM_GB:
            logging.error(
                "CRITICAL: Available RAM %.2fGB < floor %.2fGB. RESUMABLE_LOW_MEMORY_EXIT.",
                avail_gb, MIN_AVAILABLE_RAM_GB,
            )
            sys.exit(2)

        if disk_gb >= 0 and disk_gb < MIN_DISK_FREE_GB:
            logging.error(
                "CRITICAL: Disk free %.2fGB < floor %.2fGB. DISK_FULL_EXIT.",
                disk_gb, MIN_DISK_FREE_GB,
            )
            sys.exit(4)

        rss_history.append(rss_gb)
        if len(rss_history) >= 3:
            growth = rss_history[-1] - rss_history[-3]
            if growth > 3.0:
                logging.error(
                    "CRITICAL: Monotonic RSS growth %.2fGB over 3 shards. MEMORY_GROWTH_EXIT.",
                    growth,
                )
                sys.exit(3)

        # ── RESUME: skip valid completed shards ───────────────────────────
        if check_shard_done(final_dir, shard_name):
            logging.info("Shard %s already DONE — SKIP.", shard_name)
            continue

        s1_chunk = s1_sorted.iloc[i * chunk_size:(i + 1) * chunk_size]
        t0 = time.time()

        # ── RETRIEVAL (uses pre-built target context) ─────────────────────
        t_retrieval = time.time()
        union_df = generator.generate_with_context(s1_chunk, target_ctx)
        retrieval_s = time.time() - t_retrieval

        # ── ATOMIC WRITE: internal ────────────────────────────────────────
        atomic_write_parquet(union_df, os.path.join(internal_dir, f"{shard_name}.parquet"))

        # ── RANKING + PRUNING ─────────────────────────────────────────────
        t_rank = time.time()
        final_df = generator.rank_and_prune(union_df, max_candidates=max_cands)
        rank_s = time.time() - t_rank

        # ── VALIDATE ─────────────────────────────────────────────────────
        validate_shard(s1_chunk, final_df, gt_dict, union_df, max_cands)

        # ── ATOMIC WRITE: final ───────────────────────────────────────────
        atomic_write_parquet(final_df, os.path.join(final_dir, f"{shard_name}.parquet"))

        # ── GT DIAGNOSTICS (after write — not in retrieval hot path) ─────
        t_diag = time.time()
        chunk_missed_list = []
        final_pairs    = set(zip(final_df["entity_id_s1"], final_df["entity_id_cand"]))
        internal_pairs = set(zip(union_df["entity_id_s1"], union_df["entity_id_cand"]))
        for s1_id in s1_chunk["entity_id"].values:
            matches = gt_dict.get(s1_id, set())
            for m in matches:
                if (s1_id, m) not in final_pairs:
                    is_internal = (s1_id, m) in internal_pairs
                    chunk_missed_list.append({
                        "source1_entity_id":    s1_id,
                        "candidate_entity_id":  m,
                        "miss_type": "pruning_miss" if is_internal else "retrieval_miss",
                    })
        misses_df = (
            pd.DataFrame(chunk_missed_list)
            if chunk_missed_list
            else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "miss_type"])
        )
        atomic_write_parquet(misses_df, os.path.join(metrics_dir, f"{shard_name}_misses.parquet"))
        diag_s = time.time() - t_diag

        elapsed = time.time() - t0
        meta = {
            "shard":                    shard_name,
            "s1_rows":                  len(s1_chunk),
            "internal_cands":           len(union_df),
            "final_cands":              len(final_df),
            "config_hash":              config_hash,
            "architecture_fingerprint": architecture_fingerprint,
            "implementation_version":   IMPLEMENTATION_VERSION,
            "completed_at":             datetime.now(timezone.utc).isoformat(),
            "elapsed_s":                elapsed,
            "retrieval_s":              retrieval_s,
            "rank_s":                   rank_s,
            "diag_s":                   diag_s,
        }
        mark_shard_done(final_dir, shard_name, meta)

        logging.info(
            "Shard %s DONE in %.1fs | retrieval=%.1fs rank=%.1fs diag=%.1fs | "
            "internal=%d final=%d",
            shard_name, elapsed, retrieval_s, rank_s, diag_s,
            len(union_df), len(final_df),
        )

        # ── PER-SHARD CLEANUP (do NOT release target context) ────────────
        del s1_chunk, union_df, final_df, final_pairs, internal_pairs, misses_df
        del chunk_missed_list
        if "cp" in sys.modules:
            try:
                import cupy as cp
                # Only free blocks NOT in the persistent target context
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        gc.collect()

        rss_post = memory_usage_gb()
        logging.info(
            "Post-cleanup RSS=%.2fGB Avail=%.2fGB",
            rss_post, available_ram_gb(),
        )

    # ── 4. COUNTRY EXIT: release target context ──────────────────────────
    logging.info("[%s] All shards done. Releasing TargetContext...", country)
    target_ctx.release()
    del generator, s1, s2, s3, gt_dict
    gc.collect()
    if "cp" in sys.modules:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    logging.info("=== WORKER END: %s ===", country)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_orchestrator(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("=== C001 FULL TRAIN ORCHESTRATOR (impl=%s) ===", IMPLEMENTATION_VERSION)

    if os.environ.get("ALLOW_CPU_TFIDF") != "1":
        gpu_status = check_gpu()
        if "CPU Only" in gpu_status:
            raise RuntimeError("GPU_REQUIRED=TRUE. Stopping loudly. cuML/CuPy not available.")

    processed_dir = args.processed_dir
    s1_meta   = pq.read_table(os.path.join(processed_dir, "train_source1.parquet"), columns=["country"])
    countries = sorted(pd.Series(s1_meta["country"]).unique())
    del s1_meta
    gc.collect()
    logging.info("Found %d countries: %s", len(countries), countries)

    for country in countries:
        logging.info("\n>>> Spawning worker for country: %s", country)
        cmd = [
            sys.executable, "-u", __file__,
            "--country", country,
            "--config", args.config,
            "--data-dir", args.data_dir,
            "--processed-dir", args.processed_dir,
            "--out-dir", args.out_dir,
            "--smoke-size", str(args.smoke_size),
            "--chunk-size", str(args.chunk_size),
        ]
        if args.fold_manifest:
            cmd += ["--fold-manifest", args.fold_manifest]
        env = os.environ.copy()
        res = subprocess.run(cmd, env=env)
        if res.returncode not in (0,):
            logging.error(
                "CRITICAL: Worker for %s exited with code %d. Halting.", country, res.returncode
            )
            sys.exit(res.returncode)

    logging.info("\n>>> ALL WORKERS FINISHED. Finalizing...")

    run_dir     = os.path.join(args.out_dir, "C001", "candidate_pool_v1")
    final_dir   = os.path.join(run_dir, "final")
    metrics_dir = os.path.join(run_dir, "metrics")
    manifest_dir = os.path.join(run_dir, "manifest")
    package_dir  = os.path.join(run_dir, "package")
    for d in [manifest_dir, package_dir]:
        os.makedirs(d, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_hash              = compute_config_hash(config)
    architecture_fingerprint = compute_architecture_fingerprint(config)
    git_commit = get_git_commit()
    max_cands  = config.get("pruning", {}).get("candidate_budgets", [15])[-1]

    # Streaming final candidates (no global concat)
    final_files = [
        os.path.join(final_dir, fn)
        for fn in sorted(os.listdir(final_dir))
        if fn.endswith(".parquet") and "misses" not in fn
    ]
    final_out_path = os.path.join(package_dir, "train_candidates_v1.parquet")
    cands_map = defaultdict(set)
    writer    = None
    output_final_cands = 0
    for fpath in final_files:
        df = pd.read_parquet(fpath)
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
        pd.DataFrame().to_parquet(final_out_path)

    cand_counts = pd.DataFrame([
        {"entity_id_s1": k, "candidate_count": len(v)}
        for k, v in cands_map.items()
    ])

    # Fold manifest
    if args.fold_manifest:
        fold_path = os.path.abspath(args.fold_manifest)
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fold_path = os.path.normpath(os.path.join(repo_root, "artifacts", "folds", "folds_v1.parquet"))
    if not os.path.exists(fold_path):
        raise FileNotFoundError(
            f"CRITICAL: Frozen fold manifest not found at {fold_path}\n"
            "Run: python scripts/build_folds.py"
        )
    folds_df = pd.read_parquet(fold_path)[["source1_entity_id", "fold_id"]]
    if folds_df["source1_entity_id"].duplicated().any():
        raise ValueError("Frozen fold manifest contains duplicate source1_entity_id entries!")
    if folds_df["fold_id"].isnull().any():
        raise ValueError("Frozen fold manifest contains null fold_id values!")
    fold_sha256 = hashlib.sha256(open(fold_path, "rb").read()).hexdigest()
    n_folds     = folds_df["fold_id"].nunique()
    logging.info("FOLD MANIFEST: %s", fold_path)
    logging.info("FOLD SHA256:   %s", fold_sha256)
    logging.info("FOLD ROWS:     %d entities, %d folds", len(folds_df), n_folds)

    s1_all = pd.read_parquet(
        os.path.join(processed_dir, "train_source1.parquet"), columns=["entity_id", "country"]
    )
    if args.smoke_size > 0:
        s1_all = s1_all.head(args.smoke_size)

    manifest_df = s1_all.rename(columns={"entity_id": "entity_id_s1"})
    manifest_df = manifest_df.merge(
        folds_df.rename(columns={"source1_entity_id": "entity_id_s1", "fold_id": "fold"}),
        on="entity_id_s1", how="left",
    )
    missing_folds = manifest_df["fold"].isna().sum()
    if missing_folds > 0:
        raise ValueError(
            f"CRITICAL: {missing_folds} S1 entities missing from fold manifest at {fold_path}!"
        )
    manifest_df = manifest_df.merge(cand_counts, on="entity_id_s1", how="left")
    manifest_df["candidate_count"] = manifest_df["candidate_count"].fillna(0).astype(int)
    manifest_df["has_candidates"]  = (manifest_df["candidate_count"] > 0).astype(int)
    manifest_df.to_parquet(
        os.path.join(package_dir, "train_candidate_entity_manifest_v1.parquet"), index=False
    )

    fold_provenance = {
        "fold_manifest_path":   fold_path,
        "fold_manifest_sha256": fold_sha256,
        "fold_manifest_rows":   len(folds_df),
        "n_folds":              n_folds,
    }

    # Streaming missed-gt merge
    miss_files = [
        os.path.join(metrics_dir, fn)
        for fn in os.listdir(metrics_dir)
        if fn.endswith("_misses.parquet")
    ]
    m_writer = None
    for mf in miss_files:
        df = pd.read_parquet(mf)
        if not df.empty:
            table = pa.Table.from_pandas(df)
            if m_writer is None:
                m_writer = pq.ParquetWriter(
                    os.path.join(metrics_dir, "missed_gt_pairs_v1.parquet"), table.schema
                )
            m_writer.write_table(table)
        del df
        gc.collect()
    if m_writer:
        m_writer.close()
    else:
        pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "miss_type"]).to_parquet(
            os.path.join(metrics_dir, "missed_gt_pairs_v1.parquet")
        )

    # Evaluation
    gt = pd.read_csv(
        os.path.join(args.data_dir, "raw", "train", "train_ground_truth.tsv"),
        sep="\t", dtype=str,
    ).fillna("")
    gt_dict = {
        row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        for row in gt.itertuples(index=False)
    }
    del gt
    gc.collect()
    eval_res = evaluate_pool(cands_map, gt_dict, s1_all["entity_id"].values)
    with open(os.path.join(metrics_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_res, f, indent=4)

    run_manifest = {
        "experiment_id":            "C001",
        "architecture":             "R001",
        "candidate_budget":         max_cands,
        "config_hash":              config_hash,
        "architecture_fingerprint": architecture_fingerprint,
        "implementation_version":   IMPLEMENTATION_VERSION,
        "git_commit":               git_commit,
        "completed_at":             datetime.now(timezone.utc).isoformat(),
        "input_s1_rows":            len(s1_all),
        "output_final_cands":       output_final_cands,
        "validation_status":        "PASSED",
        "fold_provenance":          fold_provenance,
    }
    with open(os.path.join(manifest_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)

    registry = {
        "logical_name":             "CANDIDATE_POOL_V1",
        "version":                  "1.0",
        "path":                     "artifacts/candidate_pool/C001/candidate_pool_v1/package/train_candidates_v1.parquet",
        "created_at":               datetime.now(timezone.utc).isoformat(),
        "config_hash":              config_hash,
        "architecture_fingerprint": architecture_fingerprint,
        "implementation_version":   IMPLEMENTATION_VERSION,
        "git_commit":               git_commit,
        "status":                   "FROZEN",
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
        tar.add(metrics_dir,  arcname="metrics")
    logging.info("DONE. FROZEN.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, default="configs/c001_full_train.yaml")
    parser.add_argument("--data-dir",       type=str, default="data")
    parser.add_argument("--processed-dir",  type=str, default=None)
    parser.add_argument("--out-dir",        type=str, default="artifacts/candidate_pool")
    parser.add_argument("--fold-manifest",  type=str, default=None,
                        help="Path to frozen fold manifest parquet (folds_v1.parquet).")
    parser.add_argument("--smoke-size",     type=int, default=0)
    parser.add_argument("--chunk-size",     type=int, default=50000)
    parser.add_argument("--country",        type=str, default=None)
    args = parser.parse_args()

    if args.processed_dir is None:
        args.processed_dir = os.path.join(args.data_dir, "processed", "v001")

    s1_path = os.path.join(args.processed_dir, "train_source1.parquet")
    s2_path = os.path.join(args.processed_dir, "train_source2.parquet")
    s3_path = os.path.join(args.processed_dir, "train_source3.parquet")
    config_path = args.config
    
    # Optional files depending on context, but require at least these 4 always
    print("\n============================================================")
    print("RESOLVED_PATHS:")
    print(f"S1 = {s1_path}")
    print(f"S2 = {s2_path}")
    print(f"S3 = {s3_path}")
    print(f"CONFIG = {config_path}")
    print(f"OUTPUT_ROOT = {args.out_dir}")
    print("============================================================\n")

    for p in [s1_path, s2_path, s3_path, config_path]:
        if not os.path.exists(p):
            print(f"ERROR: Required path does not exist: {p}")
            sys.exit(1)

    if args.country:
        run_worker(args)
    else:
        run_orchestrator(args)
