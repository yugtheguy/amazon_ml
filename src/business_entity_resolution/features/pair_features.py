"""
C003A Pair-Local Features

Orchestrator for text, numeric, and retrieval provenance features.
Runs row-wise independent of other candidates.
"""

import pandas as pd

from src.business_entity_resolution.features.schema import RETRIEVAL_FEATURES
from src.business_entity_resolution.features.text_features import compute_text_features
from src.business_entity_resolution.features.numeric_features import compute_numeric_features

def compute_pair_local_features(df: pd.DataFrame, c001_cols: list) -> pd.DataFrame:
    """
    Computes all C003A features.
    df must contain required joined text columns:
      - name_norm_clean_s1, name_norm_clean_cand
      - addr_norm_clean_s1, addr_norm_clean_cand
      - numeric_tokens_s1, numeric_tokens_cand
    """
    
    # 1. Text Features
    name_feat = compute_text_features(df, "name")
    addr_feat = compute_text_features(df, "address")
    
    # Rename 'name_..._cand' internally generated variables to match schema if any
    # (Not needed as schema handles exact naming prefix mappings)
    
    # 2. Numeric Features
    num_feat = compute_numeric_features(df)
    
    # 3. Provenance Features
    # Just extract the ones that actually exist in the dataframe
    # Missing optional ones are safely ignored and will not be fabricated
    available_retrieval = [c for c in RETRIEVAL_FEATURES if c in c001_cols]
    ret_feat = df[available_retrieval].copy()
    
    # Combine
    out_df = pd.concat([name_feat, addr_feat, num_feat, ret_feat], axis=1)
    
    return out_df
