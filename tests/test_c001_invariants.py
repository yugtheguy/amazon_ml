import pytest
import pandas as pd
import numpy as np

from scripts.run_c001_full_train import validate_shard

def test_validate_shard_exceeds_budget():
    s1 = pd.DataFrame({'entity_id': ['S1_01']})
    # internal with 20 candidates
    union = pd.DataFrame({
        'entity_id_s1': ['S1_01'] * 20,
        'entity_id_cand': [f'C_{i}' for i in range(20)],
        'candidate_source': ['S2'] * 20
    })
    # final with 16 candidates
    final = pd.DataFrame({
        'entity_id_s1': ['S1_01'] * 16,
        'entity_id_cand': [f'C_{i}' for i in range(16)],
        'candidate_source': ['S2'] * 16
    })
    
    with pytest.raises(ValueError, match="Candidate count exceeds max_candidates"):
        validate_shard(s1, final, {}, union, max_candidates=15)

def test_validate_shard_not_subset():
    s1 = pd.DataFrame({'entity_id': ['S1_01']})
    # internal with 10 candidates
    union = pd.DataFrame({
        'entity_id_s1': ['S1_01'] * 10,
        'entity_id_cand': [f'C_{i}' for i in range(10)],
        'candidate_source': ['S2'] * 10
    })
    # final with a candidate NOT in internal
    final = pd.DataFrame({
        'entity_id_s1': ['S1_01'] * 2,
        'entity_id_cand': ['C_0', 'C_UNKNOWN'],
        'candidate_source': ['S2', 'S3']
    })
    
    with pytest.raises(ValueError, match="Final candidates are not a subset of internal candidates"):
        validate_shard(s1, final, {}, union, max_candidates=15)

def test_validate_shard_duplicates():
    s1 = pd.DataFrame({'entity_id': ['S1_01']})
    union = pd.DataFrame({
        'entity_id_s1': ['S1_01', 'S1_01'],
        'entity_id_cand': ['C_0', 'C_0'],
        'candidate_source': ['S2', 'S2']
    })
    final = union.copy()
    
    with pytest.raises(ValueError, match="Duplicate relationships found in final candidates"):
        validate_shard(s1, final, {}, union, max_candidates=15)

def test_validate_shard_success():
    s1 = pd.DataFrame({'entity_id': ['S1_01']})
    union = pd.DataFrame({
        'entity_id_s1': ['S1_01', 'S1_01'],
        'entity_id_cand': ['C_0', 'C_1'],
        'candidate_source': ['S2', 'S3']
    })
    final = pd.DataFrame({
        'entity_id_s1': ['S1_01'],
        'entity_id_cand': ['C_0'],
        'candidate_source': ['S2']
    })
    
    # Should not raise
    validate_shard(s1, final, {}, union, max_candidates=15)
