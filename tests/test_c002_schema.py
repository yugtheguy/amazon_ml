import pytest
import pandas as pd
from src.business_entity_resolution.pairs.schema import (
    validate_input_schema,
    validate_output_schema,
    build_schema_contract,
    C002_PAIR_DATASET_SCHEMA_VERSION,
    IDENTITY_COLUMNS,
    CORE_METADATA_COLUMNS,
    LABEL_COLUMN
)

def test_validate_input_schema_missing_identity():
    # Missing candidate_source
    cols = ["entity_id_s1", "entity_id_cand", "name_word_score"]
    with pytest.raises(ValueError, match="Required identity columns missing"):
        validate_input_schema(cols, context="test")

def test_validate_input_schema_success(recwarn):
    cols = ["entity_id_s1", "candidate_source", "entity_id_cand", "retrieved_exact"]
    validate_input_schema(cols, context="test")
    # Missing optional provenance raises warning, not error
    assert len(recwarn) == 1
    assert "Optional provenance columns absent" in str(recwarn[0].message)

def test_validate_output_schema_missing():
    # Missing fold and label
    cols = ["entity_id_s1", "candidate_source", "entity_id_cand", "country"]
    with pytest.raises(ValueError, match="missing required columns"):
        validate_output_schema(cols)

def test_validate_output_schema_success():
    cols = ["entity_id_s1", "candidate_source", "entity_id_cand", "country", "fold", "label", "retrieved_exact"]
    validate_output_schema(cols)

def test_build_schema_contract():
    contract = build_schema_contract()
    assert contract["schema_version"] == C002_PAIR_DATASET_SCHEMA_VERSION
    assert "A_identity" in contract["column_categories"]
    assert "B_target" in contract["column_categories"]
    assert contract["column_categories"]["B_target"] == ["label"]
    
    # Check diagnostic training only marker
    assert "C_diagnostic_training_only" in contract["column_categories"]
    assert contract["column_categories"]["C_diagnostic_training_only"]["marker"] == "TRAINING_DIAGNOSTIC_ONLY"
