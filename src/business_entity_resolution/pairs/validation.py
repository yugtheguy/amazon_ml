"""
C002 Pair Dataset — Validation Utilities

Shard-level and dataset-level invariant checks.
Resume / fingerprint logic.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.business_entity_resolution.pairs.schema import (
    COMPOSITE_KEY,
    VALID_CANDIDATE_SOURCES,
    LABEL_COLUMN,
    IDENTITY_COLUMNS,
    CORE_METADATA_COLUMNS,
    C002_PAIR_DATASET_SCHEMA_VERSION,
    validate_output_schema,
)
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File fingerprinting
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Shard resume / done-marker logic
# ---------------------------------------------------------------------------

def shard_done_path(shard_dir: str | Path, shard_name: str) -> Path:
    return Path(shard_dir) / f"{shard_name}.done"


def shard_meta_path(shard_dir: str | Path, shard_name: str) -> Path:
    return Path(shard_dir) / f"{shard_name}.metadata.json"


def shard_parquet_path(shard_dir: str | Path, shard_name: str) -> Path:
    return Path(shard_dir) / f"{shard_name}.parquet"


def check_shard_resumable(
    shard_dir: str | Path,
    shard_name: str,
    config_hash: str,
    input_fingerprint: str,
    gt_fingerprint: str,
    fold_sha256: str,
    schema_version: str,
) -> bool:
    """Return True only if a previously written shard passes all resume checks.

    File existence alone is NOT sufficient.
    All fingerprints and schema version must match.
    """
    parquet_path = shard_parquet_path(shard_dir, shard_name)
    meta_path = shard_meta_path(shard_dir, shard_name)
    done_path = shard_done_path(shard_dir, shard_name)

    # All three files must exist
    if not (parquet_path.exists() and meta_path.exists() and done_path.exists()):
        return False

    # Read metadata
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        logger.warning(f"[C002] Corrupt metadata for {shard_name}, will recompute.")
        return False

    # Validate all fingerprints
    checks = {
        "schema_version": schema_version,
        "config_hash": config_hash,
        "input_fingerprint": input_fingerprint,
        "gt_fingerprint": gt_fingerprint,
        "fold_sha256": fold_sha256,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if actual != expected:
            logger.warning(
                f"[C002] Resume check FAILED for {shard_name}: "
                f"{key}={actual!r} (expected {expected!r}). Will recompute."
            )
            return False

    # Validate the parquet itself is readable and passes schema
    try:
        df = pd.read_parquet(parquet_path, columns=IDENTITY_COLUMNS + CORE_METADATA_COLUMNS + [LABEL_COLUMN])
        validate_output_schema(list(df.columns))
        del df
        gc.collect()
    except Exception as e:
        logger.warning(f"[C002] Parquet validation failed for {shard_name}: {e}. Will recompute.")
        return False

    return True


def write_shard_done(
    shard_dir: str | Path,
    shard_name: str,
    metadata: dict,
) -> None:
    """Write metadata JSON and .done marker atomically (metadata first, then done)."""
    meta_path = shard_meta_path(shard_dir, shard_name)
    done_path = shard_done_path(shard_dir, shard_name)

    # Write metadata first
    meta_path.write_text(json.dumps(metadata, indent=2))
    # .done only after metadata is safely written
    done_path.write_text("DONE")


# ---------------------------------------------------------------------------
# Shard invariant checks
# ---------------------------------------------------------------------------

def validate_pair_shard(
    df: pd.DataFrame,
    shard_name: str,
    gt_dict: Optional[dict] = None,
) -> None:
    """Run all invariant checks on a labeled pair shard.

    Raises ValueError if any invariant is violated.

    Args:
        df: The labeled pair DataFrame.
        shard_name: Used in error messages.
        gt_dict: Optional dict of {s1_id: set(gt_cand_ids)} for label consistency check.
    """
    errors = []

    # 1. Required output columns present
    try:
        validate_output_schema(list(df.columns))
    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError(f"[C002 Validation] {shard_name}: " + "; ".join(errors))

    # 2. No null identity columns
    for col in IDENTITY_COLUMNS:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Null values in {col}")

    # 3. Duplicate composite key
    if df.duplicated(subset=COMPOSITE_KEY).any():
        n_dups = df.duplicated(subset=COMPOSITE_KEY).sum()
        errors.append(f"{n_dups} duplicate composite keys (entity_id_s1, candidate_source, entity_id_cand)")

    # 4. Valid candidate sources
    if "candidate_source" in df.columns:
        invalid = set(df["candidate_source"].unique()) - VALID_CANDIDATE_SOURCES
        if invalid:
            errors.append(f"Invalid candidate_source values: {invalid}")

    # 5. Label in {0, 1}
    if LABEL_COLUMN in df.columns:
        invalid_labels = set(df[LABEL_COLUMN].unique()) - {0, 1}
        if invalid_labels:
            errors.append(f"Invalid label values: {invalid_labels}")

    # 6. Fold present and valid
    if "fold" in df.columns:
        if df["fold"].isna().any():
            errors.append("Null fold values — fold join failed for some S1 IDs")
        if (df.groupby("entity_id_s1")["fold"].nunique() > 1).any():
            errors.append("Some entity_id_s1 mapped to multiple folds — fold contamination")

    # 7. GT consistency check (optional, only when gt_dict provided)
    if gt_dict is not None and LABEL_COLUMN in df.columns:
        for (s1_id, src), grp in df.groupby(["entity_id_s1", "candidate_source"]):
            for row in grp[["entity_id_cand", LABEL_COLUMN]].itertuples(index=False):
                gt_matches = gt_dict.get(s1_id, set())
                expected_label = 1 if row.entity_id_cand in gt_matches else 0
                if row.label != expected_label:
                    errors.append(
                        f"Label mismatch: S1={s1_id} src={src} cand={row.entity_id_cand} "
                        f"label={row.label} expected={expected_label}"
                    )
                    break  # Report first mismatch only

    if errors:
        raise ValueError(f"[C002 Validation] {shard_name}: " + "; ".join(errors))

    logger.info(f"[C002] Shard {shard_name}: {len(df):,} rows passed all invariant checks.")
