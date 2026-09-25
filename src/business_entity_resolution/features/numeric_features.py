"""
C003 Numeric Features

Computes intersection features from normalized numeric_tokens lists.
Implements specific conflict logic: Missing != Conflict.
"""

import numpy as np
import pandas as pd

def compute_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes numeric features between S1 and Candidate.
    Expects df to have:
      - 'numeric_tokens_s1' (list of str, or numpy array of lists)
      - 'numeric_tokens_cand'
    """
    s1_col = "numeric_tokens_s1"
    cand_col = "numeric_tokens_cand"
    
    # Pre-process lists to sets
    # Note: numpy arrays containing lists or None
    def safe_set(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return set()
        # Handle numpy arrays or lists
        try:
            return set(val)
        except TypeError: # unhashable types
            return set()
            
    s1_sets = [safe_set(v) for v in df[s1_col]]
    cand_sets = [safe_set(v) for v in df[cand_col]]
    
    # 1. Missingness
    s1_counts = np.array([len(s) for s in s1_sets])
    cand_counts = np.array([len(s) for s in cand_sets])
    
    miss_s1 = s1_counts == 0
    miss_cand = cand_counts == 0
    
    feat = pd.DataFrame(index=df.index)
    feat["numeric_s1_missing"] = miss_s1.astype("int8")
    feat["numeric_cand_missing"] = miss_cand.astype("int8")
    feat["numeric_both_missing"] = (miss_s1 & miss_cand).astype("int8")
    
    feat["numeric_s1_count"] = s1_counts.astype("int16")
    feat["numeric_cand_count"] = cand_counts.astype("int16")
    
    # 2. Intersections
    shared_counts = np.array([len(s1 & s2) for s1, s2 in zip(s1_sets, cand_sets)])
    feat["shared_numeric_count_pair"] = shared_counts.astype("int16")
    
    # Jaccard (NaN if both missing)
    union_counts = np.array([len(s1 | s2) for s1, s2 in zip(s1_sets, cand_sets)])
    jac = np.where(union_counts > 0, shared_counts / union_counts, np.nan)
    feat["numeric_jaccard"] = jac.astype("float32")
    
    # Overlap Ratios (NaN if source is missing)
    feat["numeric_overlap_ratio_s1"] = np.where(s1_counts > 0, shared_counts / s1_counts, np.nan).astype("float32")
    feat["numeric_overlap_ratio_cand"] = np.where(cand_counts > 0, shared_counts / cand_counts, np.nan).astype("float32")
    
    # Exact Match (1 if exactly the same non-empty set)
    exact = (s1_counts > 0) & (s1_counts == cand_counts) & (shared_counts == s1_counts)
    feat["numeric_exact_set_match"] = exact.astype("int8")
    
    # 3. Conflict Logic
    # Agreement: at least one shared numeric token
    feat["numeric_agreement_flag"] = (shared_counts > 0).astype("int8")
    
    # Conflict: BOTH sides have numeric tokens, but ZERO are shared. (Missing != Conflict).
    conflict = (s1_counts > 0) & (cand_counts > 0) & (shared_counts == 0)
    feat["numeric_conflict_flag"] = conflict.astype("int8")
    
    return feat
