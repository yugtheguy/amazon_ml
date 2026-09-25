import pandas as pd
import numpy as np
import pytest
from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator

def test_candidate_generator():
    config = {
        'name_word': {'enabled': True, 'top_k_per_source': 2},
        'address_word': {'enabled': True, 'top_k_per_source': 2},
        'rare_token': {'enabled': True, 'max_df_threshold': 5, 'max_tokens_per_s1': 3},
        'numeric': {'enabled': True, 'max_df_threshold': 5}
    }
    
    gen = CandidateGenerator(config)
    
    s1 = pd.DataFrame({
        'entity_id': ['s1_1', 's1_2', 's1_3'],
        'country': ['US', 'US', 'US'],
        'name_norm_clean': ['amazon inc', 'google llc', 'empty'],
        'addr_norm_clean': ['123 main st', '456 market', ''],
        'name_norm_accent_fold': ['amazon inc', 'google llc', 'empty'],
        'addr_norm_accent_fold': ['123 main st', '456 market', ''],
        'name_norm_punct': ['amazon inc', 'google llc', 'empty'],
        'addr_norm_punct': ['123 main st', '456 market', ''],
        'addr_numeric_tokens': ['123', '456', '']
    })
    
    s2 = pd.DataFrame({
        'entity_id': ['s2_1', 's2_2', 's2_3', 's2_4'],
        'country': ['US', 'US', 'US', 'US'],
        'name_norm_clean': ['amazon incorporated', 'google llc', 'microsoft', 'amazon'],
        'addr_norm_clean': ['123 main street', '456 market', 'one microsoft way', ''],
        'name_norm_accent_fold': ['amazon incorporated', 'google llc', 'microsoft', 'amazon'],
        'addr_norm_accent_fold': ['123 main street', '456 market', 'one microsoft way', ''],
        'name_norm_punct': ['amazon incorporated', 'google llc', 'microsoft', 'amazon'],
        'addr_norm_punct': ['123 main street', '456 market', 'one microsoft way', ''],
        'addr_numeric_tokens': ['123', '456', '', '']
    })
    
    union_df = gen.generate(s1, {'S2': s2})
    
    assert len(union_df) > 0
    assert 'candidate_source' in union_df.columns
    assert 'retrieval_channel_count' in union_df.columns
    assert 'exact_name_address_clean' in union_df.columns
    
    pruned = gen.rank_and_prune(union_df, max_candidates=2)
    assert len(pruned) <= 6
    
