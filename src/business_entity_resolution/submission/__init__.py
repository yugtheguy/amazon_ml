"""
Submission building and validation infrastructure.

Official submission format (verified from competition README):
- Tab-separated .tsv files
- matching_results.tsv columns: source1_entity_id, matched_entity_ids
- candidate_pairs.tsv columns: source1_entity_id, candidate_entity_ids
- matched_entity_ids: comma-separated S2-/S3- IDs, empty for ZERO_MATCH
- candidate_entity_ids: comma-separated S2-/S3- IDs, empty when blocking found nothing
- One row per test Source 1 entity
- No duplicate source1_entity_id rows
- No duplicate IDs within any ID list
- Only S2-/S3- prefixed IDs (no S1- self-matches)
- Final matches must be a subset of candidates

Internal representation:
  predictions: dict mapping source1_entity_id -> set of predicted entity IDs
  candidates: dict mapping source1_entity_id -> set of candidate entity IDs
"""

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, Optional, List

import pandas as pd

from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Internal Prediction Representation
# ============================================================================

class EntityPredictions:
    """Internal representation of entity-level predictions.

    Supports zero, one, or many predicted matches per Source-1 entity.
    """

    def __init__(self):
        self._predictions: Dict[str, Set[str]] = {}

    def set_prediction(self, source1_id: str, matched_ids: Set[str]) -> None:
        """Set the predicted matches for a Source-1 entity."""
        self._predictions[source1_id] = set(matched_ids)

    def get_prediction(self, source1_id: str) -> Set[str]:
        """Get the predicted matches for a Source-1 entity."""
        return self._predictions.get(source1_id, set())

    def get_all(self) -> Dict[str, Set[str]]:
        """Return the full predictions mapping."""
        return dict(self._predictions)

    @property
    def n_entities(self) -> int:
        return len(self._predictions)

    @property
    def n_zero_match(self) -> int:
        return sum(1 for v in self._predictions.values() if len(v) == 0)

    @property
    def n_single_match(self) -> int:
        return sum(1 for v in self._predictions.values() if len(v) == 1)

    @property
    def n_multi_match(self) -> int:
        return sum(1 for v in self._predictions.values() if len(v) >= 2)


# ============================================================================
# Submission Serialization
# ============================================================================

def build_matching_results_df(predictions: EntityPredictions) -> pd.DataFrame:
    """Serialize predictions to the official matching_results.tsv format.

    Args:
        predictions: EntityPredictions instance.

    Returns:
        DataFrame with columns: source1_entity_id, matched_entity_ids.
    """
    rows = []
    for s1_id, matched_ids in predictions.get_all().items():
        if matched_ids:
            ids_str = ",".join(sorted(matched_ids))
        else:
            ids_str = ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ids_str})

    return pd.DataFrame(rows)


def build_candidate_pairs_df(candidates: EntityPredictions) -> pd.DataFrame:
    """Serialize candidate pairs to the official candidate_pairs.tsv format.

    Args:
        candidates: EntityPredictions instance with candidate sets.

    Returns:
        DataFrame with columns: source1_entity_id, candidate_entity_ids.
    """
    rows = []
    for s1_id, cand_ids in candidates.get_all().items():
        if cand_ids:
            ids_str = ",".join(sorted(cand_ids))
        else:
            ids_str = ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ids_str})

    return pd.DataFrame(rows)


def write_submission_tsv(df: pd.DataFrame, output_path: str) -> None:
    """Write a submission DataFrame to TSV with deterministic encoding.

    Args:
        df: DataFrame with the required columns.
        output_path: Output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Submission written to {output_path}: {len(df)} rows")


# ============================================================================
# Submission Validation
# ============================================================================

def validate_submission(
    matching_df: pd.DataFrame,
    candidate_df: Optional[pd.DataFrame],
    test_s1_ids: Set[str],
    valid_s2_s3_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Validate submission files against official competition rules.

    Args:
        matching_df: DataFrame of matching_results.tsv.
        candidate_df: DataFrame of candidate_pairs.tsv (optional).
        test_s1_ids: Set of all test Source-1 entity IDs.
        valid_s2_s3_ids: Optional set of valid S2/S3 IDs for existence check.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors = []

    # --- matching_results.tsv checks ---

    # 1. Required columns
    if "source1_entity_id" not in matching_df.columns:
        errors.append("matching_results: missing column 'source1_entity_id'")
        return errors
    if "matched_entity_ids" not in matching_df.columns:
        errors.append("matching_results: missing column 'matched_entity_ids'")
        return errors

    # 2. Exactly one row per test S1 entity
    sub_s1_ids = set(matching_df["source1_entity_id"])
    missing = test_s1_ids - sub_s1_ids
    if missing:
        errors.append(f"matching_results: {len(missing)} test S1 entities missing")

    extra = sub_s1_ids - test_s1_ids
    if extra:
        errors.append(f"matching_results: {len(extra)} extra S1 entities not in test set")

    # 3. No duplicate S1 rows
    dup_count = matching_df["source1_entity_id"].duplicated().sum()
    if dup_count > 0:
        errors.append(f"matching_results: {dup_count} duplicate source1_entity_id rows")

    # 4. Per-row ID checks
    for idx, row in matching_df.iterrows():
        s1_id = row["source1_entity_id"]
        ids_str = row["matched_entity_ids"]

        if pd.isna(ids_str) or ids_str == "":
            continue  # Valid empty prediction

        ids = ids_str.split(",")

        # No duplicate IDs within a list
        if len(ids) != len(set(ids)):
            errors.append(f"matching_results: duplicate IDs in list for {s1_id}")

        for mid in ids:
            # Only S2-/S3- prefixes
            if mid.startswith("S1-"):
                errors.append(f"matching_results: self-match S1 ID '{mid}' for {s1_id}")
            elif not mid.startswith(("S2-", "S3-")):
                errors.append(f"matching_results: invalid ID prefix '{mid}' for {s1_id}")

            # Existence check if valid_ids provided
            if valid_s2_s3_ids is not None and mid not in valid_s2_s3_ids:
                errors.append(f"matching_results: unknown ID '{mid}' for {s1_id}")

    # --- candidate_pairs.tsv checks ---
    if candidate_df is not None:
        if "source1_entity_id" not in candidate_df.columns:
            errors.append("candidate_pairs: missing column 'source1_entity_id'")
        elif "candidate_entity_ids" not in candidate_df.columns:
            errors.append("candidate_pairs: missing column 'candidate_entity_ids'")
        else:
            cand_s1_ids = set(candidate_df["source1_entity_id"])
            missing_cand = test_s1_ids - cand_s1_ids
            if missing_cand:
                errors.append(f"candidate_pairs: {len(missing_cand)} test S1 entities missing")

            extra_cand = cand_s1_ids - test_s1_ids
            if extra_cand:
                errors.append(f"candidate_pairs: {len(extra_cand)} extra S1 entities not in test set")

            dup_cand = candidate_df["source1_entity_id"].duplicated().sum()
            if dup_cand > 0:
                errors.append(f"candidate_pairs: {dup_cand} duplicate source1_entity_id rows")

            # Subset check: final matches must be subset of candidates
            cand_map = {}
            for _, crow in candidate_df.iterrows():
                cid = crow["source1_entity_id"]
                cids_str = crow["candidate_entity_ids"]
                if pd.isna(cids_str) or cids_str == "":
                    cand_map[cid] = set()
                else:
                    cand_ids = set(cids_str.split(","))
                    cand_map[cid] = cand_ids
                    
                    if valid_s2_s3_ids is not None:
                        for cand_id in cand_ids:
                            if cand_id not in valid_s2_s3_ids:
                                errors.append(f"candidate_pairs: unknown ID '{cand_id}' for {cid}")

            for _, mrow in matching_df.iterrows():
                s1_id = mrow["source1_entity_id"]
                mids_str = mrow["matched_entity_ids"]
                if pd.isna(mids_str) or mids_str == "":
                    continue
                matched = set(mids_str.split(","))
                candidates = cand_map.get(s1_id, set())
                not_in_cands = matched - candidates
                if not_in_cands:
                    errors.append(
                        f"matching_results: {len(not_in_cands)} matched ID(s) for "
                        f"{s1_id} not in candidate_pairs"
                    )

    return errors


def validate_round_trip(matching_df: pd.DataFrame, output_path: str) -> List[str]:
    """Validate that a written TSV can be reloaded and passes validation.

    Args:
        matching_df: Original DataFrame.
        output_path: Path where the TSV was written.

    Returns:
        List of errors from the round-trip check.
    """
    errors = []
    try:
        reloaded = pd.read_csv(output_path, sep="\t", dtype=str)
        reloaded = reloaded.fillna("")

        if list(reloaded.columns) != list(matching_df.columns):
            errors.append(
                f"Round-trip column mismatch: {list(reloaded.columns)} vs {list(matching_df.columns)}"
            )
        if len(reloaded) != len(matching_df):
            errors.append(
                f"Round-trip row count mismatch: {len(reloaded)} vs {len(matching_df)}"
            )
    except Exception as e:
        errors.append(f"Round-trip reload failed: {e}")

    return errors


# ============================================================================
# Submission Metadata
# ============================================================================

def build_submission_metadata(
    predictions: EntityPredictions,
    experiment_id: str = "UNKNOWN",
    fold_version: str = "UNKNOWN",
    normalization_version: str = "UNKNOWN",
    candidate_pool_version: str = "UNKNOWN",
    matching_path: str = "",
    candidate_path: str = "",
) -> dict:
    """Build metadata for a submission package.

    Args:
        predictions: The final predictions.
        experiment_id: Experiment identifier.
        fold_version: Fold version used.
        normalization_version: Normalization version used.
        candidate_pool_version: Candidate pool version.
        matching_path: Path to matching_results.tsv.
        candidate_path: Path to candidate_pairs.tsv.

    Returns:
        Metadata dict.
    """
    meta = {
        "experiment_id": experiment_id,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "fold_version": fold_version,
        "normalization_version": normalization_version,
        "candidate_pool_version": candidate_pool_version,
        "row_count": predictions.n_entities,
        "zero_match_count": predictions.n_zero_match,
        "single_match_count": predictions.n_single_match,
        "multi_match_count": predictions.n_multi_match,
    }

    if matching_path and os.path.isfile(matching_path):
        h = hashlib.sha256()
        with open(matching_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        meta["matching_file_sha256"] = h.hexdigest()

    if candidate_path and os.path.isfile(candidate_path):
        h = hashlib.sha256()
        with open(candidate_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        meta["candidate_file_sha256"] = h.hexdigest()

    return meta
