import pytest
import numpy as np
import pandas as pd
from src.business_entity_resolution.features.text_features import compute_text_features
from src.business_entity_resolution.features.numeric_features import compute_numeric_features

def test_text_features_missingness():
    df = pd.DataFrame({
        "name_s1": ["Apple Inc", "", "Banana Corp"],
        "name_cand": ["", "Apple", "Banana"]
    })
    feat = compute_text_features(df, "name")
    
    assert list(feat["name_s1_missing"]) == [0, 1, 0]
    assert list(feat["name_cand_missing"]) == [1, 0, 0]
    assert list(feat["name_both_missing"]) == [0, 0, 0]
    
    # Check NaN handling
    assert np.isnan(feat["name_char_ratio"].iloc[0])
    assert np.isnan(feat["name_char_ratio"].iloc[1])
    assert not np.isnan(feat["name_char_ratio"].iloc[2])

def test_numeric_conflict_missing_not_conflict():
    df = pd.DataFrame({
        "numeric_tokens_s1": [["123"], ["456"], None, ["789"]],
        "numeric_tokens_cand": [["123"], ["999"], ["111"], None]
    })
    feat = compute_numeric_features(df)
    
    # 0: exact match -> agreement=1, conflict=0
    assert feat["numeric_agreement_flag"].iloc[0] == 1
    assert feat["numeric_conflict_flag"].iloc[0] == 0
    
    # 1: conflicting sets -> agreement=0, conflict=1
    assert feat["numeric_agreement_flag"].iloc[1] == 0
    assert feat["numeric_conflict_flag"].iloc[1] == 1
    
    # 2: s1 missing -> conflict=0 (missing!=conflict)
    assert feat["numeric_conflict_flag"].iloc[2] == 0
    
    # 3: cand missing -> conflict=0
    assert feat["numeric_conflict_flag"].iloc[3] == 0
