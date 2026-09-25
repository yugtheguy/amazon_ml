"""
C003 Explicit Feature Dataset — Schema & Firewalls

Defines the exact inference-available model features for C004 LightGBM training.
Enforces the leakage firewall by guaranteeing no GT-derived or raw identity
columns enter MODEL_FEATURE_COLUMNS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from src.business_entity_resolution.pairs.schema import (
    IDENTITY_COLUMNS,
    CORE_METADATA_COLUMNS,
    LABEL_COLUMN,
    OPTIONAL_PROVENANCE_COLUMNS,
    DIAGNOSTIC_COLUMNS,
)

C003_FEATURE_SCHEMA_VERSION = "v1"

# ---------------------------------------------------------------------------
# C003A PAIR-LOCAL FEATURES
# ---------------------------------------------------------------------------

# Retrieval Provenance (passed through from C001/C002)
# Excludes 'final_candidate_rank' as it is absent
RETRIEVAL_FEATURES = [
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
    "is_exact",
    "best_lexical_rank",
    "both_name_address",
]

# Name Features
NAME_FEATURES = [
    "name_exact_clean",               # 1/0
    "name_char_ratio",                # RapidFuzz ratio (0-1)
    "name_token_set_ratio",           # RapidFuzz token_set_ratio (0-1)
    "name_token_sort_ratio",          # RapidFuzz token_sort_ratio (0-1)
    "name_token_jaccard",             # (0-1)
    "name_token_containment_s1_in_cand", # (0-1)
    "name_token_containment_cand_in_s1", # (0-1)
    "name_shared_token_count",        # count
    "name_s1_token_count",            # count
    "name_cand_token_count",          # count
    "name_unmatched_token_count",     # count
    "name_length_ratio",              # min(len)/max(len) (0-1)
    "name_char_length_diff",          # abs diff
]

# Address Features
ADDRESS_FEATURES = [
    "address_exact_clean",            # 1/0
    "address_char_ratio",             # RapidFuzz ratio (0-1)
    "address_token_set_ratio",        # RapidFuzz token_set_ratio (0-1)
    "address_token_jaccard",          # (0-1)
    "address_shared_token_count",     # count
    "address_length_ratio",           # (0-1)
    "address_char_length_diff",       # abs diff
]

# Numeric Features
NUMERIC_FEATURES = [
    "numeric_s1_count",               # count
    "numeric_cand_count",             # count
    "shared_numeric_count_pair",      # count of exact intersecting tokens
    "numeric_jaccard",                # (0-1)
    "numeric_overlap_ratio_s1",       # shared / s1_count (0-1)
    "numeric_overlap_ratio_cand",     # shared / cand_count (0-1)
    "numeric_exact_set_match",        # 1/0
    "numeric_agreement_flag",         # 1/0 (shared > 0)
    "numeric_conflict_flag",          # 1/0 (s1>0, cand>0, shared==0)
]

# Missingness Features
MISSINGNESS_FEATURES = [
    "name_s1_missing",
    "name_cand_missing",
    "name_both_missing",
    "address_s1_missing",
    "address_cand_missing",
    "address_both_missing",
    "numeric_s1_missing",
    "numeric_cand_missing",
    "numeric_both_missing",
]

# ---------------------------------------------------------------------------
# C003B RELATIVE FEATURES
# ---------------------------------------------------------------------------
RELATIVE_FEATURES = [
    "candidate_count",                            # total candidates for this S1
    "name_similarity_rank_within_s1",             # rank of name_char_ratio (1 is best)
    "address_similarity_rank_within_s1",          # rank of address_char_ratio
    "name_gap_from_best",                         # best - current (0-1)
    "address_gap_from_best",                      # best - current (0-1)
    "name_ratio_to_best",                         # current / best (0-1)
    "address_ratio_to_best",                      # current / best (0-1)
    "count_candidates_with_higher_name_sim",      # integer
    "count_candidates_with_higher_address_sim",   # integer
    "count_exact_name_candidates",                # integer
]


# ---------------------------------------------------------------------------
# MODEL FEATURE ALLOWLIST
# ---------------------------------------------------------------------------
# This is the EXACT and EXCLUSIVE list of columns that C004 LightGBM is
# permitted to use as features.
MODEL_FEATURE_COLUMNS = (
    RETRIEVAL_FEATURES +
    NAME_FEATURES +
    ADDRESS_FEATURES +
    NUMERIC_FEATURES +
    MISSINGNESS_FEATURES +
    RELATIVE_FEATURES
)

# ---------------------------------------------------------------------------
# NON-FEATURE COLUMNS (Preserved in artifact, EXCLUDED from model training)
# ---------------------------------------------------------------------------
# We explicitly document what is NOT a feature to prevent leakage.
EXCLUDED_NON_FEATURE_COLUMNS = (
    IDENTITY_COLUMNS + 
    CORE_METADATA_COLUMNS + 
    [LABEL_COLUMN] + 
    DIAGNOSTIC_COLUMNS
)

# ---------------------------------------------------------------------------
# DTYPES & MISSING VALUE POLICY
# ---------------------------------------------------------------------------
# Similarity ranges: 0.0 to 1.0 (float32).
# Missing behavior:
# - If S1 or Cand text is missing, similarity = NaN (Not 0). 
#   LightGBM handles NaN natively. Missingness features explicitly flag this.
# - Ranks: 1-indexed. If all NaN, rank is NaN.
# - Numeric conflict: 0 if not conflicting, 1 if conflicting. (Missing != Conflict).

FEATURE_DTYPES: Dict[str, str] = {
    # Ratios (0-1)
    "name_char_ratio": "float32",
    "name_token_set_ratio": "float32",
    "name_token_sort_ratio": "float32",
    "name_token_jaccard": "float32",
    "name_token_containment_s1_in_cand": "float32",
    "name_token_containment_cand_in_s1": "float32",
    "name_length_ratio": "float32",
    "address_char_ratio": "float32",
    "address_token_set_ratio": "float32",
    "address_token_jaccard": "float32",
    "address_length_ratio": "float32",
    "numeric_jaccard": "float32",
    "numeric_overlap_ratio_s1": "float32",
    "numeric_overlap_ratio_cand": "float32",
    
    # Flags (0/1)
    "name_exact_clean": "int8",
    "address_exact_clean": "int8",
    "numeric_exact_set_match": "int8",
    "numeric_agreement_flag": "int8",
    "numeric_conflict_flag": "int8",
    
    # Missingness Flags (0/1)
    "name_s1_missing": "int8",
    "name_cand_missing": "int8",
    "name_both_missing": "int8",
    "address_s1_missing": "int8",
    "address_cand_missing": "int8",
    "address_both_missing": "int8",
    "numeric_s1_missing": "int8",
    "numeric_cand_missing": "int8",
    "numeric_both_missing": "int8",
    
    # Counts / Lengths / Ranks
    "name_shared_token_count": "int16",
    "name_s1_token_count": "int16",
    "name_cand_token_count": "int16",
    "name_unmatched_token_count": "int16",
    "name_char_length_diff": "float32", # float32 due to possible NaN
    "address_shared_token_count": "int16",
    "address_char_length_diff": "float32",
    "numeric_s1_count": "int16",
    "numeric_cand_count": "int16",
    "shared_numeric_count_pair": "int16",
    
    # Relatives
    "candidate_count": "int16",
    "name_similarity_rank_within_s1": "float32", # rank with NA
    "address_similarity_rank_within_s1": "float32",
    "name_gap_from_best": "float32",
    "address_gap_from_best": "float32",
    "name_ratio_to_best": "float32",
    "address_ratio_to_best": "float32",
    "count_candidates_with_higher_name_sim": "int16",
    "count_candidates_with_higher_address_sim": "int16",
    "count_exact_name_candidates": "int16",
}

# ---------------------------------------------------------------------------
# FIREWALL VALIDATION
# ---------------------------------------------------------------------------

def validate_leakage_firewall():
    """Asserts that no forbidden columns are in MODEL_FEATURE_COLUMNS."""
    model_features_set = set(MODEL_FEATURE_COLUMNS)
    forbidden_set = set(EXCLUDED_NON_FEATURE_COLUMNS)
    
    leakage = model_features_set.intersection(forbidden_set)
    if leakage:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {leakage} found in MODEL_FEATURE_COLUMNS!")
        
    if "country" in model_features_set:
        raise ValueError("CRITICAL LEAKAGE: 'country' must not be a direct model feature in V1!")

# Run immediately upon import to guarantee security
validate_leakage_firewall()


def build_c003_schema_contract() -> dict:
    return {
        "schema_version": C003_FEATURE_SCHEMA_VERSION,
        "description": "C003 Explicit Feature Dataset",
        "model_features": MODEL_FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_NON_FEATURE_COLUMNS,
        "dtypes": FEATURE_DTYPES,
        "feature_groups": {
            "retrieval": RETRIEVAL_FEATURES,
            "name": NAME_FEATURES,
            "address": ADDRESS_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "missingness": MISSINGNESS_FEATURES,
            "relative": RELATIVE_FEATURES
        },
        "missing_value_policy": "NaN for similarities if text missing. LightGBM handles natively.",
        "numeric_conflict_policy": "0 if not conflicting (missing!=conflict)."
    }

def write_c003_schema_contract(output_path: str | Path) -> None:
    contract = build_c003_schema_contract()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
