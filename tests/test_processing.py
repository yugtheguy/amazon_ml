"""Tests for processed data generation."""

import pytest
import pandas as pd
import os
import json

from src.business_entity_resolution.data.processing import process_source_file


@pytest.fixture
def fixture_source_tsv(tmp_path):
    """Create a minimal source TSV for testing."""
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Alpha Corp", "Société Beta", None],
        "business_address": ["21 Park Road", None, "456 Oak Ave"],
        "country": ["US", "India", "France"],
    })
    tsv_path = tmp_path / "test_source.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    return str(tsv_path), tmp_path


class TestProcessedData:
    def test_row_count_preserved(self, fixture_source_tsv):
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        meta = process_source_file(raw_path, out_path, "test_source")
        assert meta["raw_row_count"] == meta["output_row_count"]
        assert meta["output_row_count"] == 3

    def test_entity_ids_preserved(self, fixture_source_tsv):
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        process_source_file(raw_path, out_path, "test_source")
        result = pd.read_parquet(out_path)
        assert set(result["entity_id"]) == {"S1-1", "S1-2", "S1-3"}

    def test_raw_fields_preserved(self, fixture_source_tsv):
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        process_source_file(raw_path, out_path, "test_source")
        result = pd.read_parquet(out_path)
        assert "business_name" in result.columns
        assert "business_address" in result.columns
        assert "country" in result.columns

    def test_normalized_columns_added(self, fixture_source_tsv):
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        process_source_file(raw_path, out_path, "test_source")
        result = pd.read_parquet(out_path)
        assert "name_norm_clean" in result.columns
        assert "addr_norm_clean" in result.columns
        assert "addr_numeric_tokens" in result.columns

    def test_missing_rows_not_dropped(self, fixture_source_tsv):
        """Records with missing name/address must NOT be dropped."""
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        process_source_file(raw_path, out_path, "test_source")
        result = pd.read_parquet(out_path)
        assert len(result) == 3  # All 3 rows preserved

    def test_parquet_round_trip(self, fixture_source_tsv):
        raw_path, tmp_path = fixture_source_tsv
        out_path = str(tmp_path / "output.parquet")
        process_source_file(raw_path, out_path, "test_source")
        result = pd.read_parquet(out_path)
        assert "entity_id" in result.columns
        assert len(result) == 3
