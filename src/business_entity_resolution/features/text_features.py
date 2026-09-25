"""
C003 Text Features (Name & Address)

Implements rapidfuzz-based string similarity and token-based features.
Vectorized where possible using Pandas/NumPy.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

def safe_len(s):
    if pd.isna(s) or s == "":
        return 0
    return len(str(s))

def safe_token_count(s):
    if pd.isna(s) or s == "":
        return 0
    return len(str(s).split())

def safe_set(s):
    if pd.isna(s) or s == "":
        return set()
    return set(str(s).split())

def jaccard(s1_set, s2_set):
    if not s1_set and not s2_set:
        return np.nan
    if not s1_set or not s2_set:
        return 0.0
    return len(s1_set & s2_set) / len(s1_set | s2_set)

def containment(small_set, large_set):
    if not small_set:
        return np.nan
    return len(small_set & large_set) / len(small_set)


def compute_text_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Computes text features between S1 and Candidate.
    Expects df to have:
      - f"{prefix}_s1"
      - f"{prefix}_cand"
    """
    s1_col = f"{prefix}_s1"
    cand_col = f"{prefix}_cand"
    
    # 1. Missingness
    miss_s1 = df[s1_col].isna() | (df[s1_col] == "")
    miss_cand = df[cand_col].isna() | (df[cand_col] == "")
    
    feat = pd.DataFrame(index=df.index)
    feat[f"{prefix}_s1_missing"] = miss_s1.astype("int8")
    feat[f"{prefix}_cand_missing"] = miss_cand.astype("int8")
    feat[f"{prefix}_both_missing"] = (miss_s1 & miss_cand).astype("int8")
    
    # Pre-filter valid rows for similarity computation
    valid_mask = ~(miss_s1 | miss_cand)
    valid_df = df[valid_mask]
    
    # Default values for float features = NaN
    feat[f"{prefix}_char_ratio"] = np.nan
    feat[f"{prefix}_token_set_ratio"] = np.nan
    feat[f"{prefix}_token_jaccard"] = np.nan
    feat[f"{prefix}_length_ratio"] = np.nan
    feat[f"{prefix}_char_length_diff"] = np.nan
    feat[f"{prefix}_exact_clean"] = 0
    
    if prefix == "name":
        feat["name_token_sort_ratio"] = np.nan
        feat["name_token_containment_s1_in_cand"] = np.nan
        feat["name_token_containment_cand_in_s1"] = np.nan
    
    # Counts (0 default)
    feat[f"{prefix}_s1_token_count"] = 0
    feat[f"{prefix}_cand_token_count"] = 0
    feat[f"{prefix}_shared_token_count"] = 0
    if prefix == "name":
        feat["name_unmatched_token_count"] = 0
    
    if not valid_df.empty:
        s1_arr = valid_df[s1_col].astype(str).values
        cand_arr = valid_df[cand_col].astype(str).values
        
        # Exact
        exact = (s1_arr == cand_arr).astype("int8")
        
        # RapidFuzz (returns 0-100, we scale to 0-1)
        # Using list comprehension for speed over pandas apply
        ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_arr, cand_arr)]
        token_set = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_arr, cand_arr)]
        
        if prefix == "name":
            token_sort = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_arr, cand_arr)]
            
        # Sets
        s1_sets = [set(a.split()) for a in s1_arr]
        cand_sets = [set(b.split()) for b in cand_arr]
        
        jac = [jaccard(a, b) for a, b in zip(s1_sets, cand_sets)]
        shared = [len(a & b) for a, b in zip(s1_sets, cand_sets)]
        
        len_s1 = np.array([len(a) for a in s1_arr])
        len_cand = np.array([len(b) for b in cand_arr])
        
        len_min = np.minimum(len_s1, len_cand)
        len_max = np.maximum(len_s1, len_cand)
        len_ratio = np.where(len_max > 0, len_min / len_max, 0.0)
        len_diff = len_max - len_min
        
        # Assign back
        feat.loc[valid_mask, f"{prefix}_exact_clean"] = exact
        feat.loc[valid_mask, f"{prefix}_char_ratio"] = ratio
        feat.loc[valid_mask, f"{prefix}_token_set_ratio"] = token_set
        feat.loc[valid_mask, f"{prefix}_token_jaccard"] = jac
        
        feat.loc[valid_mask, f"{prefix}_s1_token_count"] = [len(a) for a in s1_sets]
        feat.loc[valid_mask, f"{prefix}_cand_token_count"] = [len(b) for b in cand_sets]
        feat.loc[valid_mask, f"{prefix}_shared_token_count"] = shared
        
        feat.loc[valid_mask, f"{prefix}_length_ratio"] = len_ratio
        feat.loc[valid_mask, f"{prefix}_char_length_diff"] = len_diff
        
        if prefix == "name":
            feat.loc[valid_mask, "name_token_sort_ratio"] = token_sort
            feat.loc[valid_mask, "name_token_containment_s1_in_cand"] = [containment(a, b) for a, b in zip(s1_sets, cand_sets)]
            feat.loc[valid_mask, "name_token_containment_cand_in_s1"] = [containment(b, a) for a, b in zip(s1_sets, cand_sets)]
            
            s1_counts = np.array([len(a) for a in s1_sets])
            cand_counts = np.array([len(b) for b in cand_sets])
            max_counts = np.maximum(s1_counts, cand_counts)
            feat.loc[valid_mask, "name_unmatched_token_count"] = max_counts - np.array(shared)

    # Cast Dtypes explicitly as per schema
    dtypes = {
        f"{prefix}_char_ratio": "float32",
        f"{prefix}_token_set_ratio": "float32",
        f"{prefix}_token_jaccard": "float32",
        f"{prefix}_length_ratio": "float32",
        f"{prefix}_char_length_diff": "float32",
        f"{prefix}_exact_clean": "int8",
        f"{prefix}_s1_token_count": "int16",
        f"{prefix}_cand_token_count": "int16",
        f"{prefix}_shared_token_count": "int16"
    }
    if prefix == "name":
        dtypes.update({
            "name_token_sort_ratio": "float32",
            "name_token_containment_s1_in_cand": "float32",
            "name_token_containment_cand_in_s1": "float32",
            "name_unmatched_token_count": "int16"
        })
        
    for col, dt in dtypes.items():
        feat[col] = feat[col].astype(dt)
        
    return feat
