import pytest
import numpy as np
import pandas as pd
from src.business_entity_resolution.features.relative_features import compute_relative_features

def test_relative_features_ranking():
    # Setup dataframe with 2 groups (A and B)
    df = pd.DataFrame({
        "entity_id_s1": ["A", "A", "A", "B", "B"]
    })
    
    pair_feat = pd.DataFrame({
        "name_char_ratio": [0.9, 0.95, 0.8, 1.0, 0.5],
        "address_char_ratio": [0.5, 0.6, 0.7, np.nan, 0.8],
        "name_exact_clean": [0, 0, 0, 1, 0]
    })
    
    out = compute_relative_features(df, pair_feat)
    
    # A group
    # 0.95 is rank 1, 0.9 is rank 2, 0.8 is rank 3
    assert list(out["name_similarity_rank_within_s1"][:3]) == [2.0, 1.0, 3.0]
    # gap from best (0.95)
    np.testing.assert_array_almost_equal(out["name_gap_from_best"][:3], [0.05, 0.0, 0.15])
    assert list(out["candidate_count"][:3]) == [3, 3, 3]
    
    # B group
    # 1.0 is rank 1, 0.5 is rank 2
    assert list(out["name_similarity_rank_within_s1"][3:]) == [1.0, 2.0]
    assert list(out["count_exact_name_candidates"][3:]) == [1, 1]

def test_relative_features_tied():
    df = pd.DataFrame({"entity_id_s1": ["A", "A", "A"]})
    pair_feat = pd.DataFrame({
        "name_char_ratio": [0.9, 0.9, 0.8],
        "address_char_ratio": [1.0, 1.0, 1.0],
        "name_exact_clean": [0, 0, 0]
    })
    
    out = compute_relative_features(df, pair_feat)
    # Tied for 1st place -> rank 1 for both (using method='min')
    assert list(out["name_similarity_rank_within_s1"]) == [1.0, 1.0, 3.0]
    # 'min' method ensures identical rank assigned deterministically without arbitrary breaking
