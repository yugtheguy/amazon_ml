import pytest
from src.business_entity_resolution.features.schema import (
    validate_leakage_firewall,
    MODEL_FEATURE_COLUMNS,
    EXCLUDED_NON_FEATURE_COLUMNS
)

def test_leakage_firewall():
    # If the firewall is correctly configured, this should not raise an error
    validate_leakage_firewall()
    
    # Assert specific forbidden columns are not in features
    forbidden = ["label", "entity_id_s1", "entity_id_cand", "fold", "country", "gt_match_count"]
    for f in forbidden:
        assert f not in MODEL_FEATURE_COLUMNS
        assert f in EXCLUDED_NON_FEATURE_COLUMNS

def test_missingness_features_exist():
    expected_missing = [
        "name_s1_missing", "name_cand_missing", "name_both_missing",
        "address_s1_missing", "address_cand_missing", "address_both_missing",
        "numeric_s1_missing", "numeric_cand_missing", "numeric_both_missing"
    ]
    for feat in expected_missing:
        assert feat in MODEL_FEATURE_COLUMNS
