"""
Permanent fold manifest generation for Source-1 entities.

Produces a versioned, immutable fold assignment that all downstream experiments
(LightGBM, cross-encoder OOF, calibration, hard-negative mining, threshold
evaluation) reuse unchanged.

The fold assignment is at Source-1 entity level: every entity appears in exactly
one fold. The assignment attempts stratified grouping by (country, match_bucket)
when possible, falling back to unstratified splitting if the stratification key
has categories too small for the requested number of folds.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.business_entity_resolution.data.load import load_source_tsv, load_ground_truth
from src.business_entity_resolution.data.schema import classify_match_bucket, classify_source_pattern
from src.business_entity_resolution.utils.seed import seed_everything
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)


def _compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file for provenance tracking."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fold_manifest(
    source1_path: str,
    ground_truth_path: str,
    n_folds: int = 5,
    seed: int = 42,
    version: str = "v1",
    output_dir: str = "artifacts/folds",
) -> pd.DataFrame:
    """Build and persist a permanent Source-1-level fold manifest.

    Args:
        source1_path: Path to the training source1 TSV.
        ground_truth_path: Path to the training ground truth TSV.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.
        version: Version label for the fold artifact.
        output_dir: Directory to write the fold parquet and metadata.

    Returns:
        DataFrame with columns: source1_entity_id, country, n_true_matches,
        match_bucket, source_pattern, fold_id.
    """
    seed_everything(seed)

    # Load data
    s1 = load_source_tsv(source1_path)
    gt = load_ground_truth(ground_truth_path)

    # Validate alignment
    assert set(s1["entity_id"]) == set(gt["source1_entity_id"]), (
        "Source 1 entity IDs and ground truth source1_entity_id do not match."
    )

    # Build manifest DataFrame
    manifest = s1[["entity_id", "country"]].copy()
    manifest = manifest.rename(columns={"entity_id": "source1_entity_id"})

    # Merge ground truth
    manifest = manifest.merge(gt, on="source1_entity_id", how="left")

    # Compute n_true_matches
    manifest["n_true_matches"] = manifest["matched_entity_ids"].apply(
        lambda x: len(x.split(",")) if x else 0
    )

    # Compute match_bucket
    manifest["match_bucket"] = manifest["n_true_matches"].apply(classify_match_bucket)

    # Compute source_pattern
    manifest["source_pattern"] = manifest["matched_entity_ids"].apply(classify_source_pattern)

    # Drop matched_entity_ids — fold manifest should not carry ground truth labels
    manifest = manifest.drop(columns=["matched_entity_ids"])

    # Build stratification key
    manifest["_strat_key"] = manifest["country"] + "_" + manifest["match_bucket"]

    # Check that every stratum has at least n_folds members
    strat_counts = manifest["_strat_key"].value_counts()
    min_count = strat_counts.min()
    if min_count < n_folds:
        logger.warning(
            f"Smallest stratification group has {min_count} members "
            f"(< n_folds={n_folds}). Falling back to country-only stratification."
        )
        manifest["_strat_key"] = manifest["country"]
        strat_counts = manifest["_strat_key"].value_counts()
        min_count = strat_counts.min()
        if min_count < n_folds:
            logger.warning(
                f"Even country-only stratification has groups too small "
                f"({min_count} < {n_folds}). Using unstratified splitting."
            )
            manifest["_strat_key"] = "ALL"

    # Perform stratified k-fold splitting
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    manifest["fold_id"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(manifest, manifest["_strat_key"])):
        manifest.iloc[val_idx, manifest.columns.get_loc("fold_id")] = fold_idx

    manifest = manifest.drop(columns=["_strat_key"])

    # ---- Invariant checks ----
    _validate_fold_manifest(manifest, n_folds)

    # ---- Persist ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / f"folds_{version}.parquet"
    manifest.to_parquet(parquet_path, index=False)
    logger.info(f"Fold manifest written to {parquet_path}")

    # ---- Metadata ----
    metadata = _build_metadata(
        manifest=manifest,
        version=version,
        n_folds=n_folds,
        seed=seed,
        source1_path=source1_path,
        ground_truth_path=ground_truth_path,
        parquet_path=str(parquet_path),
    )

    meta_path = output_path / f"folds_{version}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Fold metadata written to {meta_path}")

    return manifest


def _validate_fold_manifest(manifest: pd.DataFrame, n_folds: int) -> None:
    """Programmatic invariant checks on the fold manifest."""
    # 1. Every S1 ID occurs exactly once
    assert manifest["source1_entity_id"].is_unique, "Duplicate source1_entity_id in fold manifest"

    # 2. No null fold IDs
    assert manifest["fold_id"].notnull().all(), "Null fold_id found in manifest"

    # 3. No S1 ID has multiple fold IDs (redundant with uniqueness, but explicit)
    assert manifest.groupby("source1_entity_id")["fold_id"].nunique().max() == 1, (
        "A source1_entity_id is assigned to multiple folds"
    )

    # 4. Every fold has at least one entity
    actual_folds = set(manifest["fold_id"].unique())
    expected_folds = set(range(n_folds))
    assert actual_folds == expected_folds, (
        f"Expected folds {expected_folds}, got {actual_folds}"
    )

    # 5. Fold IDs are valid integers [0, n_folds)
    assert manifest["fold_id"].isin(range(n_folds)).all(), (
        f"fold_id values outside [0, {n_folds})"
    )

    logger.info(f"Fold manifest invariant checks passed ({len(manifest)} entities, {n_folds} folds)")


def _build_metadata(
    manifest: pd.DataFrame,
    version: str,
    n_folds: int,
    seed: int,
    source1_path: str,
    ground_truth_path: str,
    parquet_path: str,
) -> dict:
    """Build machine-readable metadata for the fold artifact."""
    per_fold = {}
    for fold_id in range(n_folds):
        fold_df = manifest[manifest["fold_id"] == fold_id]
        per_fold[str(fold_id)] = {
            "n_entities": int(len(fold_df)),
            "country_distribution": fold_df["country"].value_counts().to_dict(),
            "match_bucket_distribution": fold_df["match_bucket"].value_counts().to_dict(),
            "source_pattern_distribution": fold_df["source_pattern"].value_counts().to_dict(),
        }

    # Compute input file hashes
    s1_hash = _compute_file_hash(source1_path)
    gt_hash = _compute_file_hash(ground_truth_path)

    metadata = {
        "fold_version": version,
        "n_folds": n_folds,
        "generation_seed": seed,
        "total_source1_entities": int(len(manifest)),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "source1_path": source1_path,
            "source1_sha256": s1_hash,
            "ground_truth_path": ground_truth_path,
            "ground_truth_sha256": gt_hash,
        },
        "overall_country_distribution": manifest["country"].value_counts().to_dict(),
        "overall_match_bucket_distribution": manifest["match_bucket"].value_counts().to_dict(),
        "overall_source_pattern_distribution": manifest["source_pattern"].value_counts().to_dict(),
        "per_fold": per_fold,
        "output_parquet_path": parquet_path,
    }
    return metadata


def load_fold_manifest(fold_path: str) -> pd.DataFrame:
    """Load a previously generated fold manifest from parquet.

    Args:
        fold_path: Path to the fold parquet file.

    Returns:
        DataFrame with fold assignments.
    """
    manifest = pd.read_parquet(fold_path)
    logger.info(f"Loaded fold manifest from {fold_path}: {len(manifest)} entities")
    return manifest
