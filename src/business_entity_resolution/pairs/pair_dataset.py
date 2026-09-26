"""
C002 Pair Dataset — Core Builder

Converts C001 candidate pool shards into canonical labeled pair shards.

DESIGN:
- One shard at a time — never loads all candidates simultaneously.
- Labels derived ONLY from official GT (train_ground_truth.tsv).
- Fold inherited from frozen folds_v1.parquet via entity_id_s1.
- Retrieval misses are NOT injected.
- All C001 candidates are preserved (subject only to schema validation).
- Leakage firewall enforced via schema categories.

PUBLIC INTERFACE:
  build_c002_pair_dataset(cfg)  — main entry point
  load_c002_shard(path)         — stable reader for C003 integration
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.business_entity_resolution.pairs.schema import (
    C002_PAIR_DATASET_SCHEMA_VERSION,
    COMPOSITE_KEY,
    IDENTITY_COLUMNS,
    CORE_METADATA_COLUMNS,
    LABEL_COLUMN,
    OPTIONAL_PROVENANCE_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    VALID_CANDIDATE_SOURCES,
    DIAGNOSTIC_MARKER,
    validate_input_schema,
    validate_output_schema,
    write_schema_contract,
)
from src.business_entity_resolution.pairs.validation import (
    sha256_file,
    sha256_string,
    check_shard_resumable,
    write_shard_done,
    validate_pair_shard,
    shard_parquet_path,
    shard_done_path,
    shard_meta_path,
)
from src.business_entity_resolution.data.schema import classify_match_bucket, classify_source_pattern
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)

EXPECTED_FOLD_SHA256 = "c31fe0adea7cd85e703b6535627e8ad155e0579097321771b10ee2569571937d"


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str | Path) -> Dict[str, set]:
    """Load official GT into a dict {source1_entity_id: set(matched_entity_ids)}.

    Only reads official train_ground_truth.tsv.
    No heuristics.
    """
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt_dict: Dict[str, set] = {}
    for row in gt_df.itertuples(index=False):
        if row.matched_entity_ids:
            gt_dict[row.source1_entity_id] = set(row.matched_entity_ids.split(","))
        else:
            gt_dict[row.source1_entity_id] = set()
    logger.info(f"[C002] Loaded GT for {len(gt_dict):,} S1 entities from {gt_path}")
    return gt_dict


# ---------------------------------------------------------------------------
# Fold loading and validation
# ---------------------------------------------------------------------------

def load_and_validate_folds(
    fold_path: str | Path,
    expected_sha256: Optional[str] = None,
) -> Dict[str, int]:
    """Load frozen fold manifest and return {source1_entity_id: fold_id}.

    Validates SHA256 if expected_sha256 is provided.
    Fails loudly on duplicates or null fold IDs.
    """
    fold_path = Path(fold_path)
    actual_sha = sha256_file(fold_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(
            f"[C002] CRITICAL: Fold manifest SHA256 mismatch!\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual:   {actual_sha}\n"
            f"  Path:     {fold_path}\n"
            f"Do NOT regenerate folds. Investigate."
        )
    logger.info(f"[C002] Fold SHA256 verified: {actual_sha}")

    folds_df = pd.read_parquet(fold_path, columns=["source1_entity_id", "fold_id"])
    if folds_df["source1_entity_id"].duplicated().any():
        raise ValueError("[C002] Frozen fold manifest contains duplicate source1_entity_id entries!")
    if folds_df["fold_id"].isnull().any():
        raise ValueError("[C002] Frozen fold manifest contains null fold_id values!")

    fold_map = folds_df.set_index("source1_entity_id")["fold_id"].to_dict()
    logger.info(f"[C002] Loaded {len(fold_map):,} fold assignments, {folds_df['fold_id'].nunique()} folds.")
    return fold_map


# ---------------------------------------------------------------------------
# Label logic — the core contract
# ---------------------------------------------------------------------------

def assign_labels(
    df: pd.DataFrame,
    gt_dict: Dict[str, set],
) -> pd.Series:
    """Assign binary labels to candidate rows.

    label = 1 iff entity_id_cand ∈ gt_dict[entity_id_s1]
    label = 0 otherwise

    RULE: Only official GT is used. No heuristics. No fuzzy matching.
    RULE: Retrieval misses are NOT injected.

    Args:
        df: Candidate DataFrame with entity_id_s1 and entity_id_cand columns.
        gt_dict: {s1_id: set(gt_cand_ids)} from official train_ground_truth.tsv

    Returns:
        pd.Series of int8 labels aligned with df index.
    """
    gt_sets = df["entity_id_s1"].map(gt_dict)
    labels = pd.array(
        [
            1 if (cand_id in gt_sets_row) else 0
            for cand_id, gt_sets_row in zip(df["entity_id_cand"], gt_sets)
        ],
        dtype="Int8",
    ).astype("int8")
    return pd.Series(labels, index=df.index, name=LABEL_COLUMN)


# ---------------------------------------------------------------------------
# Diagnostic attachment (TRAINING-DIAGNOSTIC-ONLY)
# ---------------------------------------------------------------------------

def attach_s1_diagnostics(
    df: pd.DataFrame,
    gt_dict: Dict[str, set],
) -> pd.DataFrame:
    """Attach GT-derived diagnostic columns to the pair DataFrame.

    These are marked TRAINING_DIAGNOSTIC_ONLY and must NEVER become features.

    Columns added:
        gt_match_count            — total GT matches for this S1
        retrieved_positive_count  — GT positives in candidate pool
        missing_positive_count    — retrieval misses (not injected)
    """
    # Build per-S1 retrieved positive count
    pos_per_s1 = (
        df.groupby("entity_id_s1")[LABEL_COLUMN].sum().rename("retrieved_positive_count")
    )

    s1_ids = df["entity_id_s1"].unique()
    gt_count = pd.Series(
        {sid: len(gt_dict.get(sid, set())) for sid in s1_ids},
        name="gt_match_count",
    )

    diag = pd.DataFrame({"gt_match_count": gt_count, "retrieved_positive_count": pos_per_s1})
    diag["missing_positive_count"] = (
        diag["gt_match_count"] - diag["retrieved_positive_count"]
    ).clip(lower=0)
    diag = diag.astype("int16")

    df = df.merge(diag.reset_index().rename(columns={"index": "entity_id_s1"}), on="entity_id_s1", how="left")
    return df


# ---------------------------------------------------------------------------
# Single-shard processing
# ---------------------------------------------------------------------------

def process_shard(
    shard_path: str | Path,
    shard_name: str,
    gt_dict: Dict[str, set],
    fold_map: Dict[str, int],
    s1_country_map: Optional[Dict[str, str]] = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """Process one C001 candidate shard into a labeled pair shard.

    RULES:
    - No retrieval-miss injection.
    - No negative downsampling.
    - No class balancing.
    - All candidates preserved (minus schema validation rejections).
    - Label derived from GT only.
    - Fold inherited from frozen map.

    Args:
        shard_path: Path to C001 final shard parquet.
        shard_name: Human-readable shard name for logging.
        gt_dict: Official GT mapping.
        fold_map: Frozen fold mapping {source1_entity_id: fold_id}.
        s1_country_map: Optional {entity_id_s1: country} for country attachment.
        include_diagnostics: Whether to attach Category C diagnostic columns.

    Returns:
        Labeled pair DataFrame.
    """
    logger.info(f"[C002] Processing shard {shard_name} ...")
    df = pd.read_parquet(shard_path)
    n_input = len(df)
    logger.info(f"[C002] Shard {shard_name}: {n_input:,} input candidate rows.")

    # --- Schema validation (input) ---
    validate_input_schema(list(df.columns), context=shard_name)

    # --- Validate candidate sources ---
    invalid_sources = set(df["candidate_source"].unique()) - VALID_CANDIDATE_SOURCES
    if invalid_sources:
        raise ValueError(
            f"[C002] Shard {shard_name}: Unknown candidate_source values: {invalid_sources}. "
            f"Valid: {VALID_CANDIDATE_SOURCES}"
        )

    # --- Validate no null IDs ---
    for col in IDENTITY_COLUMNS:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"[C002] Shard {shard_name}: Null values in required column {col}.")

    # --- Validate no duplicate composite key ---
    if df.duplicated(subset=COMPOSITE_KEY).any():
        n_dups = df.duplicated(subset=COMPOSITE_KEY).sum()
        raise ValueError(
            f"[C002] Shard {shard_name}: {n_dups} duplicate composite keys. "
            f"C001 should have deduped. Investigate upstream."
        )

    # --- Attach fold (from frozen folds_v1.parquet) ---
    df["fold"] = df["entity_id_s1"].map(fold_map)
    missing_fold_ids = df[df["fold"].isna()]["entity_id_s1"].unique()
    if len(missing_fold_ids) > 0:
        raise ValueError(
            f"[C002] Shard {shard_name}: {len(missing_fold_ids)} S1 IDs missing from fold manifest: "
            f"{list(missing_fold_ids[:5])} ... (total {len(missing_fold_ids)})"
        )
    df["fold"] = df["fold"].astype("int8")

    # --- Validate one fold per S1 ---
    fold_nunique = df.groupby("entity_id_s1")["fold"].nunique()
    if (fold_nunique > 1).any():
        bad_ids = fold_nunique[fold_nunique > 1].index.tolist()
        raise ValueError(
            f"[C002] Shard {shard_name}: S1 IDs mapped to multiple folds (fold contamination): {bad_ids[:5]}"
        )

    # --- Attach country (from s1_country_map if provided) ---
    if s1_country_map is not None:
        df["country"] = df["entity_id_s1"].map(s1_country_map).astype("category")
    elif "country" not in df.columns:
        warnings.warn(
            f"[C002] Shard {shard_name}: 'country' not in C001 shard and no s1_country_map provided. "
            f"'country' will be null.",
            UserWarning,
        )
        df["country"] = pd.NA

    # --- Label assignment (GT only) ---
    df[LABEL_COLUMN] = assign_labels(df, gt_dict)

    # --- Attach diagnostics (Category C) ---
    if include_diagnostics:
        df = attach_s1_diagnostics(df, gt_dict)
        for col in DIAGNOSTIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("int16")

    # --- Preserve available optional provenance columns ---
    available_provenance = [c for c in OPTIONAL_PROVENANCE_COLUMNS if c in df.columns]

    # --- Build ordered output ---
    output_cols = (
        IDENTITY_COLUMNS
        + ["country", "fold", LABEL_COLUMN]
        + available_provenance
        + ([c for c in DIAGNOSTIC_COLUMNS if c in df.columns] if include_diagnostics else [])
    )
    # Deduplicate while preserving order
    seen = set()
    output_cols_dedup = [c for c in output_cols if not (c in seen or seen.add(c))]
    df = df[[c for c in output_cols_dedup if c in df.columns]]

    validate_output_schema(list(df.columns))
    logger.info(
        f"[C002] Shard {shard_name}: {len(df):,} output rows | "
        f"pos={df[LABEL_COLUMN].sum():,} neg={(df[LABEL_COLUMN]==0).sum():,}"
    )
    return df


# ---------------------------------------------------------------------------
# Entity manifest builder
# ---------------------------------------------------------------------------

def build_entity_manifest(
    pair_shards_dir: str | Path,
    s1_country_map: Optional[Dict[str, str]],
    fold_map: Dict[str, int],
    gt_dict: Dict[str, set],
) -> pd.DataFrame:
    """Build S1-level entity manifest by streaming all output shards.

    Never materializes all pair rows simultaneously.
    """
    logger.info("[C002] Building entity manifest (streaming shards)...")
    records = {}

    for fpath in sorted(Path(pair_shards_dir).glob("*.parquet")):
        df = pd.read_parquet(
            fpath,
            columns=["entity_id_s1", "candidate_source", LABEL_COLUMN]
            + (["country"] if True else []),
        )
        for s1_id, grp in df.groupby("entity_id_s1"):
            if s1_id not in records:
                gt_matches = gt_dict.get(s1_id, set())
                n_gt = len(gt_matches)
                records[s1_id] = {
                    "entity_id_s1": s1_id,
                    "country": s1_country_map.get(s1_id, None) if s1_country_map else None,
                    "fold": fold_map.get(s1_id, -1),
                    "candidate_count": 0,
                    "positive_candidate_count": 0,
                    "retrieved_s2_count": 0,
                    "retrieved_s3_count": 0,
                    "has_retrieved_positive": False,
                    "gt_match_count": n_gt,
                    "gt_s2_count": sum(1 for x in gt_matches if x.startswith("S2")),
                    "gt_s3_count": sum(1 for x in gt_matches if x.startswith("S3")),
                    "match_bucket": classify_match_bucket(n_gt),
                }
            records[s1_id]["candidate_count"] += len(grp)
            
            pos_mask = grp[LABEL_COLUMN] == 1
            records[s1_id]["positive_candidate_count"] += int(pos_mask.sum())
            records[s1_id]["retrieved_s2_count"] += int((pos_mask & (grp["candidate_source"] == "S2")).sum())
            records[s1_id]["retrieved_s3_count"] += int((pos_mask & (grp["candidate_source"] == "S3")).sum())

        del df
        gc.collect()

    manifest_df = pd.DataFrame(list(records.values()))
    manifest_df["has_retrieved_positive"] = manifest_df["positive_candidate_count"] > 0
    manifest_df["missing_positive_count"] = (
        manifest_df["gt_match_count"] - manifest_df["positive_candidate_count"]
    ).clip(lower=0)
    manifest_df["fold"] = manifest_df["fold"].astype("int8")
    manifest_df["has_retrieved_positive"] = manifest_df["has_retrieved_positive"].astype(bool)
    return manifest_df.sort_values("entity_id_s1").reset_index(drop=True)

def compute_retrieval_audit(manifest_df: pd.DataFrame) -> dict:
    """Compute detailed retrieval audit metrics per C002 requirements."""
    def get_metrics(df_slice):
        slice_has_gt = df_slice[df_slice["gt_match_count"] > 0]
        if len(slice_has_gt) == 0:
            return {"pair_recall": 0.0, "any_gt": 0.0, "full_gt": 0.0}
            
        total_gt = slice_has_gt["gt_match_count"].sum()
        total_retrieved = slice_has_gt["positive_candidate_count"].sum()
        pair_recall = total_retrieved / total_gt if total_gt > 0 else 0.0
        
        any_gt = (slice_has_gt["positive_candidate_count"] > 0).mean()
        full_gt = (slice_has_gt["positive_candidate_count"] == slice_has_gt["gt_match_count"]).mean()
        return {"pair_recall": float(pair_recall), "any_gt": float(any_gt), "full_gt": float(full_gt)}

    overall = get_metrics(manifest_df)
    india = get_metrics(manifest_df[manifest_df["country"] == "India"])
    us = get_metrics(manifest_df[manifest_df["country"] == "US"])
    
    single = get_metrics(manifest_df[manifest_df["match_bucket"] == "SINGLE_MATCH"])
    multi = get_metrics(manifest_df[manifest_df["match_bucket"] == "MULTI_MATCH"])
    
    total_s2_gt = manifest_df["gt_s2_count"].sum()
    retrieved_s2 = manifest_df["retrieved_s2_count"].sum()
    s2_recall = retrieved_s2 / total_s2_gt if total_s2_gt > 0 else 0.0
    
    total_s3_gt = manifest_df["gt_s3_count"].sum()
    retrieved_s3 = manifest_df["retrieved_s3_count"].sum()
    s3_recall = retrieved_s3 / total_s3_gt if total_s3_gt > 0 else 0.0

    missed_gt_pairs = manifest_df["missing_positive_count"].sum()
    s1_with_gt_zero_retrieved = int((manifest_df["gt_match_count"] > 0) & (manifest_df["positive_candidate_count"] == 0)).sum()

    zero_match_df = manifest_df[manifest_df["match_bucket"] == "ZERO_MATCH"]
    
    return {
        "PAIR_RECALL": overall["pair_recall"],
        "ANY_GT_COVERAGE": overall["any_gt"],
        "FULL_GT_COVERAGE": overall["full_gt"],
        "INDIA_PAIR_RECALL": india["pair_recall"],
        "INDIA_ANY_GT": india["any_gt"],
        "INDIA_FULL_GT": india["full_gt"],
        "US_PAIR_RECALL": us["pair_recall"],
        "US_ANY_GT": us["any_gt"],
        "US_FULL_GT": us["full_gt"],
        "SINGLE_PAIR_RECALL": single["pair_recall"],
        "SINGLE_FULL_GT": single["full_gt"],
        "MULTI_PAIR_RECALL": multi["pair_recall"],
        "MULTI_ANY_GT": multi["any_gt"],
        "MULTI_FULL_GT": multi["full_gt"],
        "S2_PAIR_RECALL": float(s2_recall),
        "S3_PAIR_RECALL": float(s3_recall),
        "MISSED_GT_PAIRS": int(missed_gt_pairs),
        "S1_WITH_GT_BUT_ZERO_RETRIEVED": int(s1_with_gt_zero_retrieved),
        "ZERO_MATCH_ENTITY_COUNT": len(zero_match_df),
        "ZERO_MATCH_MEAN_CANDIDATES": float(zero_match_df["candidate_count"].mean()) if len(zero_match_df) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_c002_pair_dataset(cfg: dict) -> None:
    """Main C002 pipeline entry point.

    Processes C001 candidate shards into labeled pair shards.
    Restart-safe via fingerprint-validated .done markers.

    Args:
        cfg: Configuration dict. Keys documented in c002_pair_dataset_v1.yaml.
    """
    t_start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("C002 PAIR DATASET BUILDER — START")
    logger.info("=" * 60)

    # --- Resolve paths ---
    candidate_dir = Path(cfg["candidate_pool_dir"])
    gt_path = Path(cfg["ground_truth_path"])
    fold_path = Path(cfg["fold_manifest_path"])
    processed_s1_path = Path(cfg.get("processed_s1_path", "")) if cfg.get("processed_s1_path") else None
    out_dir = Path(cfg["output_dir"])
    schema_dir = out_dir / "schema"
    shard_dir = out_dir / "shards"
    manifest_dir = out_dir / "manifest"
    metrics_dir = out_dir / "metrics"
    package_dir = out_dir / "package"

    for d in [schema_dir, shard_dir, manifest_dir, metrics_dir, package_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Config hash ---
    config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    # --- Git commit ---
    git_commit = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    # --- Write schema contract ---
    write_schema_contract(schema_dir / "c002_pair_schema_v1.json")
    logger.info(f"[C002] Schema contract written to {schema_dir / 'c002_pair_schema_v1.json'}")

    # --- Load and validate folds ---
    fold_sha = sha256_file(fold_path)
    expected_sha = cfg.get("fold_manifest_sha256", EXPECTED_FOLD_SHA256)
    fold_map = load_and_validate_folds(fold_path, expected_sha256=expected_sha)
    logger.info(f"[C002] Fold SHA256: {fold_sha}")

    # --- Load GT ---
    gt_dict = load_ground_truth(gt_path)
    gt_fingerprint = sha256_file(gt_path)
    logger.info(f"[C002] GT fingerprint: {gt_fingerprint}")

    # --- Load S1 country map ---
    s1_country_map = None
    if processed_s1_path and processed_s1_path.exists():
        s1_df = pd.read_parquet(processed_s1_path, columns=["entity_id", "country"])
        s1_country_map = s1_df.set_index("entity_id")["country"].to_dict()
        del s1_df
        gc.collect()
        logger.info(f"[C002] Loaded {len(s1_country_map):,} S1 country mappings.")

    # --- Discover candidate shards ---
    shard_files = sorted(candidate_dir.glob("*.parquet"))
    if not shard_files:
        raise FileNotFoundError(
            f"[C002] No candidate parquet files found in {candidate_dir}. "
            f"Is C001 complete?"
        )
    logger.info(f"[C002] Found {len(shard_files)} candidate shards in {candidate_dir}.")

    # --- Per-shard loop ---
    total_input_rows = 0
    total_output_rows = 0
    total_positives = 0
    dropped_rows = 0

    for shard_file in shard_files:
        shard_name = shard_file.stem
        input_fingerprint = sha256_file(shard_file)

        # Resume check
        resumable = check_shard_resumable(
            shard_dir=shard_dir,
            shard_name=shard_name,
            config_hash=config_hash,
            input_fingerprint=input_fingerprint,
            gt_fingerprint=gt_fingerprint,
            fold_sha256=fold_sha,
            schema_version=C002_PAIR_DATASET_SCHEMA_VERSION,
        )
        if resumable:
            logger.info(f"[C002] Shard {shard_name} already DONE. Skipping.")
            # Still accumulate counts from metadata
            meta_path = shard_meta_path(shard_dir, shard_name)
            with open(meta_path) as f:
                meta = json.load(f)
            total_input_rows += meta.get("input_rows", 0)
            total_output_rows += meta.get("output_rows", 0)
            total_positives += meta.get("positive_rows", 0)
            continue

        t0 = time.time()
        try:
            df = process_shard(
                shard_path=shard_file,
                shard_name=shard_name,
                gt_dict=gt_dict,
                fold_map=fold_map,
                s1_country_map=s1_country_map,
                include_diagnostics=cfg.get("include_diagnostics", True),
            )
        except ValueError as e:
            logger.error(f"[C002] Shard {shard_name} FAILED schema/validation: {e}")
            raise

        n_in = pd.read_parquet(shard_file, columns=["entity_id_s1"]).shape[0]
        n_out = len(df)
        n_pos = int(df[LABEL_COLUMN].sum())
        n_neg = n_out - n_pos

        # Full shard invariant check
        validate_pair_shard(df, shard_name=shard_name, gt_dict=gt_dict)

        # Atomic write
        tmp_path = shard_dir / f"_{shard_name}.tmp.parquet"
        out_path = shard_parquet_path(shard_dir, shard_name)
        df.to_parquet(tmp_path, index=False, compression="snappy")
        os.replace(tmp_path, out_path)

        # Done marker
        elapsed = time.time() - t0
        meta = {
            "shard": shard_name,
            "schema_version": C002_PAIR_DATASET_SCHEMA_VERSION,
            "config_hash": config_hash,
            "input_fingerprint": input_fingerprint,
            "gt_fingerprint": gt_fingerprint,
            "fold_sha256": fold_sha,
            "input_rows": n_in,
            "output_rows": n_out,
            "positive_rows": n_pos,
            "negative_rows": n_neg,
            "elapsed_s": round(elapsed, 2),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_shard_done(shard_dir, shard_name, meta)

        total_input_rows += n_in
        total_output_rows += n_out
        total_positives += n_pos

        logger.info(
            f"[C002] Shard {shard_name} DONE in {elapsed:.1f}s | "
            f"in={n_in:,} out={n_out:,} pos={n_pos:,} neg={n_neg:,}"
        )

        # Release memory
        del df
        gc.collect()

    total_negatives = total_output_rows - total_positives

    # --- Build entity manifest ---
    manifest_df = build_entity_manifest(
        pair_shards_dir=shard_dir,
        s1_country_map=s1_country_map,
        fold_map=fold_map,
        gt_dict=gt_dict,
    )
    manifest_path = package_dir / "c002_entity_manifest_v1.parquet"
    manifest_df.to_parquet(manifest_path, index=False)
    logger.info(f"[C002] Entity manifest written: {len(manifest_df):,} S1 entities.")

    # --- Package: stream all shards into consolidated parquet ---
    package_path = package_dir / "c002_pairs_v1.parquet"
    writer = None
    for shard_file in sorted(shard_dir.glob("*.parquet")):
        if shard_file.stem.startswith("_"):
            continue
        df = pd.read_parquet(shard_file)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(package_path, table.schema, compression="snappy")
        writer.write_table(table)
        del df, table
        gc.collect()
    if writer:
        writer.close()
    logger.info(f"[C002] Consolidated package written: {package_path}")

    # --- Diagnostics & Retrieval Audit ---
    diag = {
        "total_s1_entities": len(manifest_df),
        "input_candidate_rows": total_input_rows,
        "output_pair_rows": total_output_rows,
        "positive_rows": total_positives,
        "negative_rows": total_negatives,
        "positive_rate": round(total_positives / total_output_rows, 6) if total_output_rows else 0,
        "match_bucket_distribution": manifest_df["match_bucket"].value_counts().to_dict(),
        "fold_distribution": manifest_df["fold"].value_counts().sort_index().to_dict(),
        "country_distribution": manifest_df["country"].value_counts().to_dict() if "country" in manifest_df.columns else {},
        "s1_with_retrieved_positive": int(manifest_df["has_retrieved_positive"].sum()),
        "s1_total_gt_pairs": int(manifest_df["gt_match_count"].sum()),
        "s1_retrieved_gt_pairs": int(manifest_df["positive_candidate_count"].sum()),
        "s1_missing_gt_pairs": int(manifest_df["missing_positive_count"].sum()),
        "retrieval_miss_rate": (
            round(manifest_df["missing_positive_count"].sum() / manifest_df["gt_match_count"].sum(), 6)
            if manifest_df["gt_match_count"].sum() > 0 else 0
        ),
    }
    
    label_dist_path = metrics_dir / "label_distribution.json"
    with open(label_dist_path, "w") as f:
        json.dump(diag, f, indent=2)
        
    audit_metrics = compute_retrieval_audit(manifest_df)
    audit_path = metrics_dir / "retrieval_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_metrics, f, indent=2)
        
    logger.info(f"[C002] Diagnostics and retrieval audit written to {metrics_dir}")

    # --- Run manifest ---
    t_end = datetime.now(timezone.utc)
    run_manifest = {
        "experiment_id": "C002",
        "parent_experiment": "C001",
        "schema_version": C002_PAIR_DATASET_SCHEMA_VERSION,
        "candidate_input_version": cfg.get("candidate_pool_version", "C001_R001_v1"),
        "candidate_input_path": str(candidate_dir),
        "candidate_input_fingerprint": "per_shard",
        "gt_path": str(gt_path),
        "gt_fingerprint": gt_fingerprint,
        "fold_path": str(fold_path),
        "fold_sha256": fold_sha,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "started_at": t_start.isoformat(),
        "completed_at": t_end.isoformat(),
        "elapsed_s": (t_end - t_start).total_seconds(),
        "input_candidate_rows": total_input_rows,
        "output_pair_rows": total_output_rows,
        "positive_rows": total_positives,
        "negative_rows": total_negatives,
        "output_artifacts": {
            "pair_shards": str(shard_dir),
            "consolidated_pairs": str(package_path),
            "entity_manifest": str(manifest_path),
            "schema_contract": str(schema_dir / "c002_pair_schema_v1.json"),
            "label_distribution": str(label_dist_path),
            "retrieval_audit": str(audit_path),
            "run_manifest": str(manifest_dir / "run_manifest.json"),
        },
        "validation_status": "PASSED",
    }
    with open(manifest_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("C002 PAIR DATASET BUILDER — COMPLETE")
    logger.info(f"  Output pairs:    {total_output_rows:,}")
    logger.info(f"  Positives:       {total_positives:,}")
    logger.info(f"  Negatives:       {total_negatives:,}")
    logger.info(f"  Elapsed:         {(t_end - t_start).total_seconds():.1f}s")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# C003 integration interface — stable public reader
# ---------------------------------------------------------------------------

def load_c002_shard(path: str | Path) -> pd.DataFrame:
    """Load a single C002 pair shard.

    Stable public interface for C003 consumption.
    Validates output schema on load.
    C003 must exclude DIAGNOSTIC_COLUMNS from any model feature matrix.

    Returns:
        DataFrame with identity, metadata, label, and retrieval provenance.
    """
    df = pd.read_parquet(path)
    validate_output_schema(list(df.columns))
    return df
