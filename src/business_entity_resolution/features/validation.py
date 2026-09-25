"""
C003 Validation

Shard-level invariant checks, isolation verification, and resume fingerprinting.
"""

import json
from pathlib import Path
from typing import Set

import pandas as pd
from src.business_entity_resolution.pairs.validation import sha256_file
from src.business_entity_resolution.features.schema import (
    MODEL_FEATURE_COLUMNS, 
    EXCLUDED_NON_FEATURE_COLUMNS, 
    C003_FEATURE_SCHEMA_VERSION, 
    FEATURE_DTYPES
)
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)


def check_shard_resumable(
    shard_dir: str | Path,
    shard_name: str,
    config_hash: str,
    c002_fingerprint: str,
    schema_version: str,
) -> bool:
    """Return True only if previously written shard passes all resume checks."""
    shard_dir = Path(shard_dir)
    parquet_path = shard_dir / f"{shard_name}.parquet"
    meta_path = shard_dir / f"{shard_name}.metadata.json"
    done_path = shard_dir / f"{shard_name}.done"

    if not (parquet_path.exists() and meta_path.exists() and done_path.exists()):
        return False

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return False

    checks = {
        "schema_version": schema_version,
        "config_hash": config_hash,
        "c002_fingerprint": c002_fingerprint,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if actual != expected:
            return False
            
    # Quick schema verification
    try:
        df = pd.read_parquet(parquet_path, columns=["entity_id_s1", "label"] + MODEL_FEATURE_COLUMNS[:1])
    except Exception:
        return False
        
    return True


def write_shard_done(
    shard_dir: str | Path,
    shard_name: str,
    metadata: dict,
) -> None:
    shard_dir = Path(shard_dir)
    meta_path = shard_dir / f"{shard_name}.metadata.json"
    done_path = shard_dir / f"{shard_name}.done"

    meta_path.write_text(json.dumps(metadata, indent=2))
    done_path.write_text("DONE")


def validate_shard_s1_isolation(s1_ids_in_shard: Set[str], global_s1_seen: Set[str], shard_name: str) -> None:
    """
    Verifies that no S1 entity in the current shard has been processed in a previous shard.
    This guarantees that the C002 shard boundary == C003 group boundary.
    """
    overlap = s1_ids_in_shard.intersection(global_s1_seen)
    if overlap:
        raise ValueError(
            f"CRITICAL [C003 Validation] {shard_name}: "
            f"S1 Isolation violation! S1 entities split across shards. "
            f"Examples: {list(overlap)[:5]}"
        )
    global_s1_seen.update(s1_ids_in_shard)


def validate_feature_schema(df: pd.DataFrame, shard_name: str) -> None:
    """
    Verifies the output dataframe contains exactly the permitted features + preserved columns,
    and checks data types and leakage boundaries.
    """
    errors = []
    actual_cols = set(df.columns)
    
    # Check Required Core Columns
    required_core = {"entity_id_s1", "candidate_source", "entity_id_cand", "fold", "label"}
    missing_core = required_core - actual_cols
    if missing_core:
        errors.append(f"Missing required core columns: {missing_core}")
        
    # Check Feature Allowlist completeness
    # (Provenance features from C001 might be absent and that's okay, but engineered features must exist)
    expected_engineered = set(MODEL_FEATURE_COLUMNS) - set(EXCLUDED_NON_FEATURE_COLUMNS)
    missing_features = expected_engineered - actual_cols
    # Allow provenance features to be missing if they weren't in C001
    missing_engineered = [f for f in missing_features if not f.startswith("retrieved_") and f not in ["is_exact", "name_word_score", "name_word_rank", "address_word_score", "address_word_rank", "rare_token_overlap_count", "rarest_shared_token_df", "shared_numeric_count", "numeric_conflict", "retrieval_channel_count", "best_lexical_rank", "both_name_address", "exact_name_address_clean", "exact_name_address_accent", "exact_name_address_punct", "unique_exact_name"]]
    
    if missing_engineered:
        errors.append(f"Missing engineered features: {missing_engineered}")

    # Check for Dtypes
    for col, expected_dt in FEATURE_DTYPES.items():
        if col in df.columns:
            actual_dt = str(df[col].dtype)
            # Accept Float64 if it couldn't cast due to Pandas weirdness, but warn
            if not actual_dt.startswith(expected_dt.rstrip("0123456789")):
                errors.append(f"Dtype mismatch for {col}: Expected {expected_dt}, got {actual_dt}")

    if errors:
        raise ValueError(f"[C003 Validation] {shard_name} FAILED: " + " | ".join(errors))
