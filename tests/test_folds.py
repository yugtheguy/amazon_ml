"""Tests for fold manifest generation and invariants."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.business_entity_resolution.data.schema import classify_match_bucket, classify_source_pattern
from src.business_entity_resolution.data.folds import build_fold_manifest, _validate_fold_manifest


# ===========================================================================
# Match bucket tests
# ===========================================================================

class TestMatchBucket:
    def test_zero_match(self):
        assert classify_match_bucket(0) == "ZERO_MATCH"

    def test_single_match(self):
        assert classify_match_bucket(1) == "SINGLE_MATCH"

    def test_multi_match_two(self):
        assert classify_match_bucket(2) == "MULTI_MATCH"

    def test_multi_match_many(self):
        assert classify_match_bucket(10) == "MULTI_MATCH"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            classify_match_bucket(-1)


# ===========================================================================
# Source pattern tests
# ===========================================================================

class TestSourcePattern:
    def test_none_empty_string(self):
        assert classify_source_pattern("") == "NONE"

    def test_s2_only(self):
        assert classify_source_pattern("S2-123,S2-456") == "S2_ONLY"

    def test_s3_only(self):
        assert classify_source_pattern("S3-001") == "S3_ONLY"

    def test_both(self):
        assert classify_source_pattern("S2-100,S3-200") == "BOTH"

    def test_both_complex(self):
        assert classify_source_pattern("S2-1,S2-2,S3-3") == "BOTH"


# ===========================================================================
# Fold manifest integration tests (using fixture data)
# ===========================================================================

@pytest.fixture
def fixture_data_dir(tmp_path):
    """Create minimal fixture TSV files for fold testing."""
    # Source 1
    s1_data = pd.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(20)],
        "business_name": [f"Business {i}" for i in range(20)],
        "business_address": [f"Address {i}" for i in range(20)],
        "country": ["US"] * 10 + ["India"] * 10,
    })
    s1_path = tmp_path / "source1.tsv"
    s1_data.to_csv(s1_path, sep="\t", index=False)

    # Ground truth: mix of ZERO_MATCH, SINGLE_MATCH, MULTI_MATCH
    gt_rows = []
    for i in range(20):
        if i < 4:  # ZERO_MATCH
            gt_rows.append({"source1_entity_id": f"S1-{i}", "matched_entity_ids": ""})
        elif i < 12:  # SINGLE_MATCH
            gt_rows.append({"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{100+i}"})
        else:  # MULTI_MATCH
            gt_rows.append({"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{100+i},S3-{200+i}"})
    gt_data = pd.DataFrame(gt_rows)
    gt_path = tmp_path / "ground_truth.tsv"
    gt_data.to_csv(gt_path, sep="\t", index=False)

    return tmp_path, str(s1_path), str(gt_path)


class TestFoldManifest:
    def test_deterministic(self, fixture_data_dir):
        """Fold generation is deterministic with the same seed."""
        tmp_path, s1_path, gt_path = fixture_data_dir
        m1 = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                 output_dir=str(tmp_path / "folds1"))
        m2 = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                 output_dir=str(tmp_path / "folds2"))
        pd.testing.assert_frame_equal(m1, m2)

    def test_every_s1_assigned_exactly_once(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                output_dir=str(tmp_path / "folds"))
        assert m["source1_entity_id"].is_unique
        assert len(m) == 20

    def test_no_s1_in_multiple_folds(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                output_dir=str(tmp_path / "folds"))
        assert m.groupby("source1_entity_id")["fold_id"].nunique().max() == 1

    def test_no_null_fold_ids(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                output_dir=str(tmp_path / "folds"))
        assert m["fold_id"].notnull().all()

    def test_correct_match_buckets(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                output_dir=str(tmp_path / "folds"))
        zero = m[m["match_bucket"] == "ZERO_MATCH"]
        single = m[m["match_bucket"] == "SINGLE_MATCH"]
        multi = m[m["match_bucket"] == "MULTI_MATCH"]
        assert len(zero) == 4
        assert len(single) == 8
        assert len(multi) == 8

    def test_manifest_cardinality(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        s1 = pd.read_csv(s1_path, sep="\t")
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                output_dir=str(tmp_path / "folds"))
        assert len(m) == len(s1)

    def test_metadata_file_created(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        out = str(tmp_path / "folds")
        build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                            version="v1", output_dir=out)
        assert os.path.isfile(os.path.join(out, "folds_v1.parquet"))
        assert os.path.isfile(os.path.join(out, "folds_v1_metadata.json"))

    def test_parquet_round_trip(self, fixture_data_dir):
        tmp_path, s1_path, gt_path = fixture_data_dir
        out = str(tmp_path / "folds")
        m = build_fold_manifest(s1_path, gt_path, n_folds=3, seed=42,
                                version="v1", output_dir=out)
        reloaded = pd.read_parquet(os.path.join(out, "folds_v1.parquet"))
        pd.testing.assert_frame_equal(m, reloaded)
