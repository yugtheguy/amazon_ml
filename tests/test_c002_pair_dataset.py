import pytest
import pandas as pd
import numpy as np

from src.business_entity_resolution.pairs.pair_dataset import (
    assign_labels,
    attach_s1_diagnostics,
    process_shard
)
from src.business_entity_resolution.pairs.validation import validate_pair_shard

def test_assign_labels():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A", "B", "B", "C"],
        "candidate_source": ["S2", "S3", "S2", "S3", "S2"],
        "entity_id_cand": ["X", "Y", "Z", "W", "V"]
    })
    
    gt_dict = {
        "A": {"X", "Y"},  # MULTI match, both retrieved
        "B": {"Z"},       # SINGLE match, one retrieved, one miss (W is just a negative)
        "C": {"U"},       # SINGLE match, missed (retrieved V instead)
        "D": set()        # ZERO match, no candidates retrieved
    }
    
    labels = assign_labels(df, gt_dict)
    assert list(labels) == [1, 1, 1, 0, 0]


def test_attach_s1_diagnostics():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A", "B", "C", "C"],
        "label": pd.Series([1, 1, 0, 1, 0], dtype="int8")
    })
    
    gt_dict = {
        "A": {"X", "Y", "Z"},  # 3 GT matches, 2 retrieved
        "B": set(),            # ZERO match
        "C": {"U"},            # 1 GT match, 1 retrieved
    }
    
    out = attach_s1_diagnostics(df, gt_dict)
    
    # A should have gt=3, retrieved=2, missing=1
    a_rows = out[out["entity_id_s1"] == "A"]
    assert (a_rows["gt_match_count"] == 3).all()
    assert (a_rows["retrieved_positive_count"] == 2).all()
    assert (a_rows["missing_positive_count"] == 1).all()

    # B should have gt=0, retrieved=0, missing=0
    b_rows = out[out["entity_id_s1"] == "B"]
    assert (b_rows["gt_match_count"] == 0).all()
    assert (b_rows["retrieved_positive_count"] == 0).all()
    assert (b_rows["missing_positive_count"] == 0).all()


def test_validate_pair_shard_success():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A"],
        "candidate_source": ["S2", "S3"],
        "entity_id_cand": ["X", "Y"],
        "country": ["US", "US"],
        "fold": pd.Series([1, 1], dtype="int8"),
        "label": pd.Series([1, 0], dtype="int8")
    })
    # Should not raise
    validate_pair_shard(df, "test_shard")


def test_validate_pair_shard_duplicate_key():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A"],
        "candidate_source": ["S2", "S2"], # DUPLICATE!
        "entity_id_cand": ["X", "X"],
        "country": ["US", "US"],
        "fold": pd.Series([1, 1], dtype="int8"),
        "label": pd.Series([1, 1], dtype="int8")
    })
    with pytest.raises(ValueError, match="duplicate composite keys"):
        validate_pair_shard(df, "test_shard")


def test_validate_pair_shard_fold_contamination():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A"],
        "candidate_source": ["S2", "S3"],
        "entity_id_cand": ["X", "Y"],
        "country": ["US", "US"],
        "fold": pd.Series([1, 2], dtype="int8"), # Same S1, multiple folds!
        "label": pd.Series([1, 0], dtype="int8")
    })
    with pytest.raises(ValueError, match="mapped to multiple folds"):
        validate_pair_shard(df, "test_shard")



def test_validate_pair_shard_gt_consistency():
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A"],
        "candidate_source": ["S2", "S3"],
        "entity_id_cand": ["X", "Y"],
        "country": ["US", "US"],
        "fold": pd.Series([1, 1], dtype="int8"),
        "label": pd.Series([1, 1], dtype="int8") # Label is 1 for Y, but GT says no
    })
    gt_dict = {"A": {"X"}} # Only X is GT
    with pytest.raises(ValueError, match="Label mismatch"):
        validate_pair_shard(df, "test_shard", gt_dict=gt_dict)


def test_process_shard_missing_fold_fails(tmp_path):
    shard_path = tmp_path / "test_shard.parquet"
    df = pd.DataFrame({
        "entity_id_s1": ["A", "B"],
        "candidate_source": ["S2", "S3"],
        "entity_id_cand": ["X", "Y"]
    })
    df.to_parquet(shard_path)
    
    gt_dict = {"A": {"X"}, "B": {"Z"}}
    fold_map = {"A": 1} # Missing B
    
    with pytest.raises(ValueError, match="S1 IDs missing from fold manifest"):
        process_shard(shard_path, "test_shard", gt_dict, fold_map)


def test_process_shard_retrieval_miss_not_injected(tmp_path):
    # GT says A matches X and Y.
    # Retrieval only found X.
    # We should NOT see Y in the output.
    shard_path = tmp_path / "test_shard.parquet"
    df = pd.DataFrame({
        "entity_id_s1": ["A"],
        "candidate_source": ["S2"],
        "entity_id_cand": ["X"]
    })
    df.to_parquet(shard_path)
    
    gt_dict = {"A": {"X", "Y"}}
    fold_map = {"A": 1}
    
    out_df = process_shard(shard_path, "test_shard", gt_dict, fold_map)
    
    assert len(out_df) == 1
    assert out_df.iloc[0]["entity_id_cand"] == "X"
    assert out_df.iloc[0]["label"] == 1
    assert out_df.iloc[0]["missing_positive_count"] == 1

