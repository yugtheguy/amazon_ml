import pandas as pd
import pytest
from scripts.run_e000 import run_retrieval, evaluate_decision

def test_retrieval_channels_and_deduplication():
    s1 = pd.DataFrame([
        {"entity_id": "S1-1", "name_norm_clean": "abc", "addr_norm_clean": "123", "name_norm_accent_fold": "abc", "addr_norm_accent_fold": "123", "name_norm_punct": "abc", "addr_norm_punct": "123", "country": "US"},
        {"entity_id": "S1-2", "name_norm_clean": "def", "addr_norm_clean": "456", "name_norm_accent_fold": "def", "addr_norm_accent_fold": "456", "name_norm_punct": "def", "addr_norm_punct": "456", "country": "FR"},
    ])
    cands = pd.DataFrame([
        # Cand 1 matches exactly on everything
        {"entity_id": "S2-1", "name_norm_clean": "abc", "addr_norm_clean": "123", "name_norm_accent_fold": "abc", "addr_norm_accent_fold": "123", "name_norm_punct": "abc", "addr_norm_punct": "123", "country": "US"},
        # Cand 2 matches only on accent
        {"entity_id": "S2-2", "name_norm_clean": "diff", "addr_norm_clean": "diff", "name_norm_accent_fold": "def", "addr_norm_accent_fold": "456", "name_norm_punct": "diff", "addr_norm_punct": "diff", "country": "FR"},
    ])
    
    res = run_retrieval(s1, cands, "S2", use_country=True)
    
    assert len(res) == 2
    
    # Check Cand 1
    c1_res = res[(res['entity_id_s1'] == 'S1-1') & (res['entity_id_cand'] == 'S2-1')].iloc[0]
    assert c1_res['exact_name_address_clean'] == 1
    assert c1_res['exact_name_address_accent'] == 1
    assert c1_res['exact_name_address_punct'] == 1
    # also it's unique name
    assert c1_res['unique_exact_name'] == 1
    assert c1_res['retrieval_channel_count'] == 4
    
    # Check Cand 2
    c2_res = res[(res['entity_id_s1'] == 'S1-2') & (res['entity_id_cand'] == 'S2-2')].iloc[0]
    assert c2_res['exact_name_address_clean'] == 0
    assert c2_res['exact_name_address_accent'] == 1
    assert c2_res['exact_name_address_punct'] == 0

def test_missing_key_explosion():
    s1 = pd.DataFrame([
        {"entity_id": "S1-1", "name_norm_clean": "", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": "", "country": "US"},
    ])
    cands = pd.DataFrame([
        {"entity_id": "S2-1", "name_norm_clean": "", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": "", "country": "US"},
        {"entity_id": "S2-2", "name_norm_clean": "", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": "", "country": "US"},
    ])
    
    res = run_retrieval(s1, cands, "S2", use_country=True)
    # Should not retrieve anything because keys are empty
    assert len(res) == 0

def test_unique_exact_name():
    s1 = pd.DataFrame([
        {"entity_id": "S1-1", "name_norm_clean": "multi", "country": "US", "addr_norm_clean": ""},
        {"entity_id": "S1-2", "name_norm_clean": "unique", "country": "US", "addr_norm_clean": ""},
    ])
    cands = pd.DataFrame([
        # multi occurs twice
        {"entity_id": "S2-1", "name_norm_clean": "multi", "country": "US", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": ""},
        {"entity_id": "S2-2", "name_norm_clean": "multi", "country": "US", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": ""},
        # unique occurs once
        {"entity_id": "S2-3", "name_norm_clean": "unique", "country": "US", "addr_norm_clean": "", "name_norm_accent_fold": "", "addr_norm_accent_fold": "", "name_norm_punct": "", "addr_norm_punct": ""},
    ])
    
    # We need all columns for run_retrieval even if empty
    for df in [s1, cands]:
        for c in ['name_norm_accent_fold', 'addr_norm_accent_fold', 'name_norm_punct', 'addr_norm_punct']:
            if c not in df.columns:
                df[c] = ""
                
    res = run_retrieval(s1, cands, "S2", use_country=True)
    # multi should not be rescued because it is not unique
    # unique should be rescued
    assert len(res) == 1
    assert res.iloc[0]['entity_id_s1'] == 'S1-2'
    assert res.iloc[0]['entity_id_cand'] == 'S2-3'
    assert res.iloc[0]['unique_exact_name'] == 1

def test_country_partition():
    s1 = pd.DataFrame([
        {"entity_id": "S1-1", "name_norm_clean": "test", "addr_norm_clean": "123", "country": "US"},
        {"entity_id": "S1-2", "name_norm_clean": "test", "addr_norm_clean": "123", "country": "FRANCE"}, # arbitrary
    ])
    cands = pd.DataFrame([
        {"entity_id": "S2-1", "name_norm_clean": "test", "addr_norm_clean": "123", "country": "US"},
        {"entity_id": "S2-2", "name_norm_clean": "test", "addr_norm_clean": "123", "country": "FRANCE"},
    ])
    
    for df in [s1, cands]:
        for c in ['name_norm_accent_fold', 'addr_norm_accent_fold', 'name_norm_punct', 'addr_norm_punct']:
            df[c] = ""
            
    res = run_retrieval(s1, cands, "S2", use_country=True)
    assert len(res) == 2
    # Check that US didn't match FRANCE
    c1 = res[(res['entity_id_s1'] == 'S1-1')]
    assert len(c1) == 1
    assert c1.iloc[0]['entity_id_cand'] == 'S2-1'

def test_decision_policies():
    cands_df = pd.DataFrame([
        # Retrieved by D0 (exact_name_address_clean)
        {"entity_id_s1": "S1-1", "entity_id_cand": "S2-1", "exact_name_address_clean": 1, "exact_name_address_accent": 0, "exact_name_address_punct": 0, "unique_exact_name": 1, "addr_norm_clean_s1": "123", "addr_norm_clean_cand": "123"},
        # Retrieved only by unique name, and missing address -> D1 adds it
        {"entity_id_s1": "S1-2", "entity_id_cand": "S2-2", "exact_name_address_clean": 0, "exact_name_address_accent": 0, "exact_name_address_punct": 0, "unique_exact_name": 1, "addr_norm_clean_s1": "", "addr_norm_clean_cand": ""},
        # Retrieved only by unique name, but has address -> D1 does NOT add it
        {"entity_id_s1": "S1-3", "entity_id_cand": "S2-3", "exact_name_address_clean": 0, "exact_name_address_accent": 0, "exact_name_address_punct": 0, "unique_exact_name": 1, "addr_norm_clean_s1": "123", "addr_norm_clean_cand": "123"},
    ])
    
    gt_dict = {}
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    
    preds_d0, _ = evaluate_decision(cands_df, gt_dict, s1_ids, 'D0')
    assert "S2-1" in preds_d0["S1-1"]
    assert "S2-2" not in preds_d0["S1-2"]
    assert "S2-3" not in preds_d0["S1-3"]
    
    preds_d1, _ = evaluate_decision(cands_df, gt_dict, s1_ids, 'D1')
    assert "S2-1" in preds_d1["S1-1"]
    assert "S2-2" in preds_d1["S1-2"]
    assert "S2-3" not in preds_d1["S1-3"]
