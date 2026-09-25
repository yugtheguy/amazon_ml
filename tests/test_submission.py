"""Tests for submission building and validation infrastructure."""

import pytest
import pandas as pd
import os

from src.business_entity_resolution.submission import (
    EntityPredictions,
    build_matching_results_df,
    build_candidate_pairs_df,
    write_submission_tsv,
    validate_submission,
    validate_round_trip,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def test_s1_ids():
    return {"S1-1", "S1-2", "S1-3"}


@pytest.fixture
def valid_s2_s3_ids():
    return {"S2-100", "S2-200", "S3-300", "S3-400"}


# ===========================================================================
# EntityPredictions
# ===========================================================================

class TestEntityPredictions:
    def test_zero_match(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", set())
        assert preds.get_prediction("S1-1") == set()
        assert preds.n_zero_match == 1

    def test_single_match(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        assert preds.n_single_match == 1

    def test_multi_match(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100", "S3-300"})
        assert preds.n_multi_match == 1


# ===========================================================================
# Serialization
# ===========================================================================

class TestSerialization:
    def test_empty_prediction_serialized(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", set())
        df = build_matching_results_df(preds)
        assert df.iloc[0]["matched_entity_ids"] == ""

    def test_single_prediction_serialized(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        df = build_matching_results_df(preds)
        assert df.iloc[0]["matched_entity_ids"] == "S2-100"

    def test_multi_prediction_serialized(self):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100", "S3-300"})
        df = build_matching_results_df(preds)
        ids = df.iloc[0]["matched_entity_ids"]
        assert "S2-100" in ids
        assert "S3-300" in ids

    def test_candidate_pairs_serialized(self):
        cands = EntityPredictions()
        cands.set_prediction("S1-1", {"S2-100", "S3-300", "S3-400"})
        df = build_candidate_pairs_df(cands)
        assert "candidate_entity_ids" in df.columns

    def test_write_and_reload_tsv(self, tmp_path):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        preds.set_prediction("S1-2", set())
        df = build_matching_results_df(preds)
        path = str(tmp_path / "matching_results.tsv")
        write_submission_tsv(df, path)

        reloaded = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        assert len(reloaded) == 2
        assert set(reloaded.columns) == {"source1_entity_id", "matched_entity_ids"}


# ===========================================================================
# Validation
# ===========================================================================

class TestValidation:
    def test_valid_submission(self, test_s1_ids, valid_s2_s3_ids):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        preds.set_prediction("S1-2", set())
        preds.set_prediction("S1-3", {"S2-200", "S3-300"})
        df = build_matching_results_df(preds)
        errors = validate_submission(df, None, test_s1_ids, valid_s2_s3_ids)
        assert errors == []

    def test_missing_s1_rejected(self, test_s1_ids, valid_s2_s3_ids):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        preds.set_prediction("S1-2", set())
        # Missing S1-3
        df = build_matching_results_df(preds)
        errors = validate_submission(df, None, test_s1_ids, valid_s2_s3_ids)
        assert any("missing" in e for e in errors)

    def test_extra_s1_rejected(self, test_s1_ids, valid_s2_s3_ids):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        preds.set_prediction("S1-2", set())
        preds.set_prediction("S1-3", set())
        preds.set_prediction("S1-999", {"S2-100"})  # Extra
        df = build_matching_results_df(preds)
        errors = validate_submission(df, None, test_s1_ids, valid_s2_s3_ids)
        assert any("extra" in e.lower() for e in errors)

    def test_duplicate_s1_rejected(self, test_s1_ids, valid_s2_s3_ids):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-100", "S2-200", "", ""],
        })
        errors = validate_submission(df, None, test_s1_ids, valid_s2_s3_ids)
        assert any("duplicate" in e.lower() for e in errors)

    def test_duplicate_candidate_in_list_rejected(self, test_s1_ids, valid_s2_s3_ids):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-100,S2-100", "", ""],
        })
        errors = validate_submission(df, None, test_s1_ids, valid_s2_s3_ids)
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_id_prefix_rejected(self, test_s1_ids):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["INVALID-123", "", ""],
        })
        errors = validate_submission(df, None, test_s1_ids, None)
        assert any("prefix" in e.lower() for e in errors)

    def test_self_match_rejected(self, test_s1_ids):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S1-1", "", ""],
        })
        errors = validate_submission(df, None, test_s1_ids, None)
        assert any("self-match" in e.lower() for e in errors)

    def test_matches_must_be_subset_of_candidates(self, test_s1_ids, valid_s2_s3_ids):
        matching_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-100", "", "S3-400"],
        })
        candidate_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_ids": ["S2-200", "", "S3-400"],  # S2-100 not in candidates!
        })
        errors = validate_submission(matching_df, candidate_df, test_s1_ids, valid_s2_s3_ids)
        assert any("not in candidate_pairs" in e for e in errors)

    def test_invalid_candidate_id_rejected(self, test_s1_ids, valid_s2_s3_ids):
        matching_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-100", "", "S3-400"],
        })
        candidate_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_ids": ["S2-100", "", "S3-999"],  # S3-999 is NOT in valid_s2_s3_ids!
        })
        errors = validate_submission(matching_df, candidate_df, test_s1_ids, valid_s2_s3_ids)
        assert any("unknown ID 'S3-999'" in e for e in errors)

    def test_round_trip(self, tmp_path, test_s1_ids):
        preds = EntityPredictions()
        preds.set_prediction("S1-1", {"S2-100"})
        preds.set_prediction("S1-2", set())
        preds.set_prediction("S1-3", {"S2-200", "S3-300"})
        df = build_matching_results_df(preds)
        path = str(tmp_path / "matching_results.tsv")
        write_submission_tsv(df, path)
        errors = validate_round_trip(df, path)
        assert errors == []
