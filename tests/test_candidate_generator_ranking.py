import pandas as pd
import numpy as np
import pytest

from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator

def test_rank_and_prune_symmetry():
    generator = CandidateGenerator({})
    
    # Create dummy union_df with name and address symmetry
    data = [
        # cand 1: name rank 1, addr rank 9999
        {'entity_id_s1': 'S1', 'entity_id_cand': 'C1', 'exact_name_address_clean': 0, 'exact_name_address_accent': 0, 'exact_name_address_punct': 0,
         'retrieved_name_word': 1, 'retrieved_address_word': 0, 'name_word_rank': 1, 'address_word_rank': 9999, 'rare_token_overlap_count': 0, 'shared_numeric_count': 0, 'retrieval_channel_count': 1},
        
        # cand 2: addr rank 1, name rank 9999
        {'entity_id_s1': 'S1', 'entity_id_cand': 'C2', 'exact_name_address_clean': 0, 'exact_name_address_accent': 0, 'exact_name_address_punct': 0,
         'retrieved_name_word': 0, 'retrieved_address_word': 1, 'name_word_rank': 9999, 'address_word_rank': 1, 'rare_token_overlap_count': 0, 'shared_numeric_count': 0, 'retrieval_channel_count': 1},
        
        # cand 3: both rank 1
        {'entity_id_s1': 'S1', 'entity_id_cand': 'C3', 'exact_name_address_clean': 0, 'exact_name_address_accent': 0, 'exact_name_address_punct': 0,
         'retrieved_name_word': 1, 'retrieved_address_word': 1, 'name_word_rank': 1, 'address_word_rank': 1, 'rare_token_overlap_count': 0, 'shared_numeric_count': 0, 'retrieval_channel_count': 2},
         
        # cand 4: exact
        {'entity_id_s1': 'S1', 'entity_id_cand': 'C4', 'exact_name_address_clean': 1, 'exact_name_address_accent': 0, 'exact_name_address_punct': 0,
         'retrieved_name_word': 1, 'retrieved_address_word': 1, 'name_word_rank': 1, 'address_word_rank': 1, 'rare_token_overlap_count': 0, 'shared_numeric_count': 0, 'retrieval_channel_count': 3},
    ]
    
    df = pd.DataFrame(data)
    pruned = generator.rank_and_prune(df, max_candidates=10)
    
    # Order should be C4 (exact), C3 (both), then C1/C2 tie-broken deterministically (by cand_id)
    cand_order = pruned['entity_id_cand'].tolist()
    assert cand_order[0] == 'C4'
    assert cand_order[1] == 'C3'
    # C1 and C2 should be 3rd and 4th, order doesn't strictly matter as long as they are treated symmetrically
    assert set(cand_order[2:4]) == {'C1', 'C2'}
    
def test_nested_budgets():
    generator = CandidateGenerator({})
    data = []
    for i in range(20):
        data.append({
            'entity_id_s1': 'S1', 'entity_id_cand': f'C{i}', 'exact_name_address_clean': 0, 'exact_name_address_accent': 0, 'exact_name_address_punct': 0,
            'retrieved_name_word': 1, 'retrieved_address_word': 0, 'name_word_rank': i+1, 'address_word_rank': 9999, 'rare_token_overlap_count': 0, 'shared_numeric_count': 0, 'retrieval_channel_count': 1
        })
    df = pd.DataFrame(data)
    
    b3 = set(generator.rank_and_prune(df, 3)['entity_id_cand'])
    b5 = set(generator.rank_and_prune(df, 5)['entity_id_cand'])
    b10 = set(generator.rank_and_prune(df, 10)['entity_id_cand'])
    b15 = set(generator.rank_and_prune(df, 15)['entity_id_cand'])
    
    assert b3.issubset(b5)
    assert b5.issubset(b10)
    assert b10.issubset(b15)
