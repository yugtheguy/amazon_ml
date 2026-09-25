"""
C003B Relative Features

Computes candidate-relative features (ranks, margins) grouped by entity_id_s1.
Ensures tied similarities are handled deterministically.
"""

import numpy as np
import pandas as pd

def compute_relative_features(df: pd.DataFrame, pair_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes C003B relative features.
    
    df: Base dataframe (must contain entity_id_s1)
    pair_features_df: Previously computed C003A features containing name/address similarities.
    """
    # 1. Combine necessary columns for groupby operations
    rel_base = df[["entity_id_s1"]].copy()
    rel_base["name_char_ratio"] = pair_features_df["name_char_ratio"]
    rel_base["address_char_ratio"] = pair_features_df["address_char_ratio"]
    rel_base["name_exact_clean"] = pair_features_df["name_exact_clean"]
    
    out = pd.DataFrame(index=df.index)
    
    # 2. Total candidate count per S1
    cand_counts = rel_base.groupby("entity_id_s1").size()
    out["candidate_count"] = rel_base["entity_id_s1"].map(cand_counts).astype("int16")
    
    # 3. Name relative features
    name_best = rel_base.groupby("entity_id_s1")["name_char_ratio"].transform("max")
    
    # Rank (min method handles ties deterministically, ascending=False means 1 is best)
    out["name_similarity_rank_within_s1"] = rel_base.groupby("entity_id_s1")["name_char_ratio"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("float32")
    
    out["name_gap_from_best"] = (name_best - rel_base["name_char_ratio"]).astype("float32")
    out["name_ratio_to_best"] = np.where(name_best > 0, rel_base["name_char_ratio"] / name_best, np.nan).astype("float32")
    
    out["count_candidates_with_higher_name_sim"] = (out["name_similarity_rank_within_s1"] - 1).clip(lower=0).fillna(0).astype("int16")
    
    exact_name_counts = rel_base.groupby("entity_id_s1")["name_exact_clean"].transform("sum")
    out["count_exact_name_candidates"] = exact_name_counts.fillna(0).astype("int16")

    # 4. Address relative features
    addr_best = rel_base.groupby("entity_id_s1")["address_char_ratio"].transform("max")
    
    out["address_similarity_rank_within_s1"] = rel_base.groupby("entity_id_s1")["address_char_ratio"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("float32")
    
    out["address_gap_from_best"] = (addr_best - rel_base["address_char_ratio"]).astype("float32")
    out["address_ratio_to_best"] = np.where(addr_best > 0, rel_base["address_char_ratio"] / addr_best, np.nan).astype("float32")
    
    out["count_candidates_with_higher_address_sim"] = (out["address_similarity_rank_within_s1"] - 1).clip(lower=0).fillna(0).astype("int16")
    
    return out
