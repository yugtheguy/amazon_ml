"""
C002 Pair Dataset — Schema Definition (Version v1)

Single source of truth for column names, categories, dtypes, and leakage firewall.

The schema is derived from the ACTUAL C001 output observed from:
  - src/business_entity_resolution/retrieval/candidate_generator.py
    (generate() + rank_and_prune() output columns)
  - Verified via audit_c001_smoke.py run on smoke artifacts.

LEAKAGE FIREWALL:
  Category A — MODEL-SAFE / INFERENCE-AVAILABLE
  Category B — TRAINING TARGET (label)
  Category C — TRAINING-DIAGNOSTIC-ONLY (GT-derived, never used as model features)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
C002_PAIR_DATASET_SCHEMA_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# CATEGORY A — IDENTITY (always required, inference-available)
# ---------------------------------------------------------------------------
IDENTITY_COLUMNS: List[str] = [
    "entity_id_s1",        # Source-1 entity ID
    "candidate_source",    # "S2" or "S3"
    "entity_id_cand",      # Candidate entity ID (source-relative)
]

# Composite unique key — source-aware
COMPOSITE_KEY: List[str] = IDENTITY_COLUMNS

VALID_CANDIDATE_SOURCES: frozenset = frozenset({"S2", "S3"})

# ---------------------------------------------------------------------------
# CATEGORY A — CORE METADATA (required, inference-available)
# ---------------------------------------------------------------------------
CORE_METADATA_COLUMNS: List[str] = [
    "country",   # Inherited from S1 (not from candidate)
    "fold",      # Inherited from frozen folds_v1.parquet via entity_id_s1
]

# ---------------------------------------------------------------------------
# CATEGORY A — C001 RETRIEVAL PROVENANCE (optional, inference-available)
# These columns come from CandidateGenerator.generate() → rank_and_prune().
# They are model-safe because they are computed from retrieval signals only,
# with NO access to ground-truth labels.
# ---------------------------------------------------------------------------
PROVENANCE_COLUMNS: List[str] = [
    "retrieved_exact",
    "retrieved_name_word",
    "retrieved_address_word",
    "retrieved_rare",
    "retrieved_numeric",
    "exact_name_address_clean",
    "exact_name_address_accent",
    "exact_name_address_punct",
    "unique_exact_name",
    "name_word_score",
    "name_word_rank",
    "address_word_score",
    "address_word_rank",
    "rare_token_overlap_count",
    "rarest_shared_token_df",
    "shared_numeric_count",
    "numeric_conflict",
    "retrieval_channel_count",
    # rank_and_prune computed columns
    "is_exact",
    "best_lexical_rank",
    "both_name_address",
]

# Provenance columns that may be absent in a given C001 version
# C002 will preserve them if present; will not fabricate if absent.
OPTIONAL_PROVENANCE_COLUMNS: List[str] = PROVENANCE_COLUMNS.copy()

# ---------------------------------------------------------------------------
# CATEGORY B — TRAINING TARGET
# ---------------------------------------------------------------------------
LABEL_COLUMN: str = "label"
# label = 1 iff (candidate_source, entity_id_cand) ∈ official GT for entity_id_s1
# label = 0 otherwise
# Source: train_ground_truth.tsv ONLY. No heuristics.

# ---------------------------------------------------------------------------
# CATEGORY C — TRAINING-DIAGNOSTIC-ONLY
# These are GT-derived fields. Must NEVER become model features.
# C003 must exclude these from the feature matrix.
# ---------------------------------------------------------------------------
DIAGNOSTIC_COLUMNS: List[str] = [
    "gt_match_count",            # Total GT matches for this S1 (from GT, not candidates)
    "retrieved_positive_count",  # GT positives present in candidate pool for this S1
    "missing_positive_count",    # GT positives NOT retrieved (retrieval misses)
]

DIAGNOSTIC_MARKER: str = "TRAINING_DIAGNOSTIC_ONLY"

# ---------------------------------------------------------------------------
# DTYPE PREFERENCES
# ---------------------------------------------------------------------------
DTYPE_MAP: Dict[str, str] = {
    # Identity
    "entity_id_s1": "str",
    "candidate_source": "category",
    "entity_id_cand": "str",
    # Metadata
    "country": "category",
    "fold": "int8",
    # Target
    "label": "int8",
    # Provenance — flags (0/1)
    "retrieved_exact": "int8",
    "retrieved_name_word": "int8",
    "retrieved_address_word": "int8",
    "retrieved_rare": "int8",
    "retrieved_numeric": "int8",
    "exact_name_address_clean": "int8",
    "exact_name_address_accent": "int8",
    "exact_name_address_punct": "int8",
    "unique_exact_name": "int8",
    "is_exact": "int8",
    "both_name_address": "int8",
    "numeric_conflict": "int8",
    # Provenance — scores/ranks
    "name_word_score": "float32",
    "name_word_rank": "int16",
    "address_word_score": "float32",
    "address_word_rank": "int16",
    "rare_token_overlap_count": "int16",
    "rarest_shared_token_df": "int32",
    "shared_numeric_count": "int16",
    "retrieval_channel_count": "int8",
    "best_lexical_rank": "int16",
    # Diagnostics
    "gt_match_count": "int16",
    "retrieved_positive_count": "int16",
    "missing_positive_count": "int16",
}

# ---------------------------------------------------------------------------
# ALL OUTPUT COLUMNS (ordered)
# ---------------------------------------------------------------------------
ALL_OUTPUT_COLUMNS: List[str] = (
    IDENTITY_COLUMNS
    + CORE_METADATA_COLUMNS
    + [LABEL_COLUMN]
    + OPTIONAL_PROVENANCE_COLUMNS
    + DIAGNOSTIC_COLUMNS
)

# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

def validate_input_schema(df_columns: List[str], context: str = "") -> None:
    """Assert that required identity columns are present in a candidate DataFrame.

    Raises ValueError loudly if any required identity column is missing.
    Optional provenance columns are reported but not raised.

    Args:
        df_columns: Actual columns present in the DataFrame.
        context: String used in error messages for traceability.
    """
    missing_required = [c for c in IDENTITY_COLUMNS if c not in df_columns]
    if missing_required:
        raise ValueError(
            f"[C002 Schema] CRITICAL: Required identity columns missing "
            f"in {context}: {missing_required}. "
            f"Cannot proceed — fix upstream C001 output."
        )

    missing_provenance = [c for c in OPTIONAL_PROVENANCE_COLUMNS if c not in df_columns]
    if missing_provenance:
        import warnings
        warnings.warn(
            f"[C002 Schema] Optional provenance columns absent in {context}: "
            f"{missing_provenance}. They will be omitted from output (not fabricated).",
            UserWarning,
            stacklevel=2,
        )


def validate_output_schema(df_columns: List[str]) -> None:
    """Assert that required output columns are present in final pair dataset."""
    required = IDENTITY_COLUMNS + CORE_METADATA_COLUMNS + [LABEL_COLUMN]
    missing = [c for c in required if c not in df_columns]
    if missing:
        raise ValueError(
            f"[C002 Schema] CRITICAL: Output DataFrame missing required columns: {missing}"
        )


# ---------------------------------------------------------------------------
# SCHEMA CONTRACT (machine-readable JSON)
# ---------------------------------------------------------------------------

def build_schema_contract() -> dict:
    """Build the machine-readable C002 schema contract dict."""
    return {
        "schema_version": C002_PAIR_DATASET_SCHEMA_VERSION,
        "description": "C002 Canonical Labeled Candidate-Pair Dataset",
        "composite_key": COMPOSITE_KEY,
        "source_aware_identity": True,
        "valid_candidate_sources": sorted(VALID_CANDIDATE_SOURCES),
        "label_definition": (
            "label=1 iff (candidate_source, entity_id_cand) is in official GT for entity_id_s1. "
            "Source: train_ground_truth.tsv ONLY. No heuristics."
        ),
        "fold_semantics": (
            "Fold is inherited from folds_v1.parquet via entity_id_s1. "
            "One S1 entity → exactly one fold. Never inferred from pairs."
        ),
        "column_categories": {
            "A_identity": IDENTITY_COLUMNS,
            "A_core_metadata": CORE_METADATA_COLUMNS,
            "A_provenance_optional": OPTIONAL_PROVENANCE_COLUMNS,
            "B_target": [LABEL_COLUMN],
            "C_diagnostic_training_only": {
                "columns": DIAGNOSTIC_COLUMNS,
                "marker": DIAGNOSTIC_MARKER,
                "warning": "These columns are GT-derived. Never use as model features.",
            },
        },
        "dtypes": DTYPE_MAP,
        "nullable_columns": OPTIONAL_PROVENANCE_COLUMNS + DIAGNOSTIC_COLUMNS,
        "invariants": [
            "No duplicate composite key (entity_id_s1, candidate_source, entity_id_cand)",
            "No null entity_id_s1 or entity_id_cand",
            "candidate_source in {S2, S3}",
            "label in {0, 1}",
            "Every entity_id_s1 has exactly one fold",
            "No retrieval-miss injection — candidates only from C001",
            "Candidate count unchanged from C001 final after valid integrity filtering",
        ],
        "input_candidate_schema_version": "C001_R001_v1",
    }


def write_schema_contract(output_path: str | Path) -> None:
    """Write schema contract JSON to disk."""
    contract = build_schema_contract()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
