import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from src.business_entity_resolution.evaluation.f05 import calculate_macro_f05
from src.business_entity_resolution.submission import EntityPredictions, write_submission_tsv
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger("run_e000")

def load_data(path):
    logger.info(f"Loading {path}...")
    cols = ['entity_id', 'country', 'name_norm_clean', 'name_norm_accent_fold', 'name_norm_punct',
            'addr_norm_clean', 'addr_norm_accent_fold', 'addr_norm_punct']
    df = pd.read_parquet(path, columns=cols)
    return df

def get_collision_stats(cands, use_country=True):
    stats = {}
    for col in ['name_norm_clean', 'name_norm_accent_fold']:
        c = cands[cands[col] != '']
        if use_country:
            counts = c.groupby([col, 'country']).size()
        else:
            counts = c.groupby(col).size()
            
        counts = counts.values
        if len(counts) == 0:
            stats[col] = {}
            continue
            
        stats[col] = {
            "keys_mapping_to_1": int(np.sum(counts == 1)),
            "keys_mapping_to_2": int(np.sum(counts == 2)),
            "keys_mapping_to_3_5": int(np.sum((counts >= 3) & (counts <= 5))),
            "keys_mapping_to_6_10": int(np.sum((counts >= 6) & (counts <= 10))),
            "keys_mapping_to_gt_10": int(np.sum(counts > 10)),
            "p50_multiplicity": float(np.percentile(counts, 50)),
            "p95_multiplicity": float(np.percentile(counts, 95)),
            "p99_multiplicity": float(np.percentile(counts, 99)),
            "max_multiplicity": int(np.max(counts))
        }
    return stats

def run_retrieval(s1, cands, cand_source, use_country=True):
    logger.info(f"Retrieving from {cand_source}...")
    dfs = []
    
    def merge_channel(s1_df, c_df, merge_cols, flag_name):
        # filter empty
        for c in merge_cols:
            if c != 'country':
                s1_df = s1_df[s1_df[c] != '']
                c_df = c_df[c_df[c] != '']
        res = s1_df.merge(c_df, on=merge_cols, suffixes=('_s1', '_cand'))
        if len(res) > 0:
            res[flag_name] = 1
            if 'addr_norm_clean' in merge_cols:
                res['addr_norm_clean_s1'] = res['addr_norm_clean']
                res['addr_norm_clean_cand'] = res['addr_norm_clean']
            dfs.append(res[['entity_id_s1', 'entity_id_cand', 'addr_norm_clean_s1', 'addr_norm_clean_cand', flag_name]])
            
    # R1
    r1_cols = ['name_norm_clean', 'addr_norm_clean'] + (['country'] if use_country else [])
    merge_channel(s1, cands, r1_cols, 'exact_name_address_clean')
    
    # R2
    r2_cols = ['name_norm_accent_fold', 'addr_norm_accent_fold'] + (['country'] if use_country else [])
    merge_channel(s1, cands, r2_cols, 'exact_name_address_accent')
    
    # R3
    r3_cols = ['name_norm_punct', 'addr_norm_punct'] + (['country'] if use_country else [])
    merge_channel(s1, cands, r3_cols, 'exact_name_address_punct')
    
    # R4
    c_r4 = cands[cands['name_norm_clean'] != '']
    r4_cols = ['name_norm_clean'] + (['country'] if use_country else [])
    counts = c_r4.groupby(r4_cols).size().reset_index(name='count')
    unique_keys = counts[counts['count'] == 1].drop(columns=['count'])
    c_r4_unique = c_r4.merge(unique_keys, on=r4_cols)
    merge_channel(s1, c_r4_unique, r4_cols, 'unique_exact_name')
    
    if not dfs:
        return pd.DataFrame()
        
    union_df = pd.concat(dfs, ignore_index=True)
    union_df = union_df.fillna(0)
    
    flag_cols = ['exact_name_address_clean', 'exact_name_address_accent', 'exact_name_address_punct', 'unique_exact_name']
    for c in flag_cols:
        if c not in union_df.columns:
            union_df[c] = 0
            
    agg_funcs = {
        'addr_norm_clean_s1': 'first',
        'addr_norm_clean_cand': 'first',
        **{c: 'max' for c in flag_cols}
    }
    
    res = union_df.groupby(['entity_id_s1', 'entity_id_cand']).agg(agg_funcs).reset_index()
    res['candidate_source'] = cand_source
    res['retrieval_channel_count'] = res[flag_cols].sum(axis=1)
    return res

def evaluate_decision(cands_df, gt_dict, s1_ids, policy):
    logger.info(f"Evaluating policy {policy}...")
    
    if policy == 'D0':
        mask = (cands_df['exact_name_address_clean'] == 1) | \
               (cands_df['exact_name_address_accent'] == 1) | \
               (cands_df['exact_name_address_punct'] == 1)
        matches = cands_df[mask]
    elif policy == 'D1':
        mask_d0 = (cands_df['exact_name_address_clean'] == 1) | \
                  (cands_df['exact_name_address_accent'] == 1) | \
                  (cands_df['exact_name_address_punct'] == 1)
                  
        mask_d1_extra = (cands_df['unique_exact_name'] == 1) & \
                        ((cands_df['addr_norm_clean_s1'] == '') | (cands_df['addr_norm_clean_cand'] == ''))
        matches = cands_df[mask_d0 | mask_d1_extra]
    else:
        raise ValueError("Invalid policy")
        
    preds = {}
    for s1_id in s1_ids:
        preds[s1_id] = set()
        
    for _, row in matches.iterrows():
        preds[row['entity_id_s1']].add(row['entity_id_cand'])
        
    macro_f05 = calculate_macro_f05(gt_dict, preds)
    return preds, macro_f05

def get_gt_dict(gt_df, s1_ids):
    gt = {}
    for s1_id in s1_ids:
        gt[s1_id] = set()
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in gt:
            m = row['matched_entity_ids']
            if m and not pd.isna(m):
                gt[s1_id] = set(m.split(','))
    return gt

def main():
    t0 = time.time()
    
    # Create directories
    retrieval_dir = Path("artifacts/retrieval/E000")
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    subs_dir = Path("artifacts/submissions/E000")
    subs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load country consistency audit
    with open("artifacts/retrieval/E000/country_consistency_audit.json", "r") as f:
        audit = json.load(f)
    use_country = audit.get("generic_country_equality_valid", False)
    logger.info(f"Using generic country equality: {use_country}")
    
    s1 = load_data('data/processed/v001/train_source1.parquet')
    
    # Exact name collision diagnostics
    collision_stats = {}
    
    # Process S2
    s2 = load_data('data/processed/v001/train_source2.parquet')
    collision_stats["S2"] = get_collision_stats(s2, use_country)
    cands_s2 = run_retrieval(s1, s2, 'S2', use_country)
    del s2  # Free memory
    
    # Process S3
    s3 = load_data('data/processed/v001/train_source3.parquet')
    collision_stats["S3"] = get_collision_stats(s3, use_country)
    cands_s3 = run_retrieval(s1, s3, 'S3', use_country)
    del s3  # Free memory
    
    with open(retrieval_dir / "exact_name_collision_stats.json", "w") as f:
        json.dump(collision_stats, f, indent=2)
        
    cands = pd.concat([cands_s2, cands_s3], ignore_index=True)
    del cands_s2, cands_s3
    
    # Save training candidates for evaluation
    cands.to_parquet(retrieval_dir / "train_candidates.parquet", index=False)
    
    # Evaluate Retrieval
    gt_df = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_dict = get_gt_dict(gt_df, s1['entity_id'].values)
    
    # Calculate retrieval metrics
    # Pair candidate recall
    total_gt_pairs = sum(len(v) for v in gt_dict.values())
    
    cand_pairs_set = set(zip(cands['entity_id_s1'], cands['entity_id_cand']))
    gt_pairs = set()
    for s1_id, matches in gt_dict.items():
        for m in matches:
            gt_pairs.add((s1_id, m))
            
    retrieved_gt = len(gt_pairs.intersection(cand_pairs_set))
    pair_recall = retrieved_gt / total_gt_pairs if total_gt_pairs > 0 else 0
    
    # Full-GT entity coverage
    matched_s1 = [s1_id for s1_id, matches in gt_dict.items() if len(matches) > 0]
    full_gt_coverage = 0
    
    # group candidates by s1
    cand_dict = {}
    for _, row in cands.iterrows():
        cand_dict.setdefault(row['entity_id_s1'], set()).add(row['entity_id_cand'])
        
    for s1_id in matched_s1:
        c = cand_dict.get(s1_id, set())
        if gt_dict[s1_id].issubset(c):
            full_gt_coverage += 1
            
    full_gt_coverage_rate = full_gt_coverage / len(matched_s1) if matched_s1 else 0
    
    # Candidate volume
    cand_counts = cands.groupby('entity_id_s1').size().values
    if len(cand_counts) == 0: cand_counts = np.array([0])
    
    retrieval_metrics = {
        "total_candidates": len(cands),
        "pair_candidate_recall": pair_recall,
        "full_gt_entity_coverage": full_gt_coverage_rate,
        "mean_candidates_per_s1": float(np.mean(cand_counts)),
        "median_candidates": float(np.median(cand_counts)),
        "p95_candidates": float(np.percentile(cand_counts, 95)),
        "p99_candidates": float(np.percentile(cand_counts, 99)),
        "max_candidates": int(np.max(cand_counts)),
        "s2_candidates": int((cands['candidate_source'] == 'S2').sum()),
        "s3_candidates": int((cands['candidate_source'] == 'S3').sum())
    }
    
    with open(retrieval_dir / "retrieval_metrics.json", "w") as f:
        json.dump(retrieval_metrics, f, indent=2)
        
    # Missed GT pairs
    missed = gt_pairs - cand_pairs_set
    missed_list = [{"source1_entity_id": s, "candidate_entity_id": c} for s, c in missed]
    pd.DataFrame(missed_list).to_parquet(retrieval_dir / "missed_gt_pairs.parquet", index=False)
    
    # Evaluate D0 vs D1
    preds_D0, f05_D0 = evaluate_decision(cands, gt_dict, s1['entity_id'].values, 'D0')
    preds_D1, f05_D1 = evaluate_decision(cands, gt_dict, s1['entity_id'].values, 'D1')
    
    with open(retrieval_dir / "decision_metrics_D0.json", "w") as f:
        json.dump({"macro_f05": f05_D0}, f, indent=2)
    with open(retrieval_dir / "decision_metrics_D1.json", "w") as f:
        json.dump({"macro_f05": f05_D1}, f, indent=2)
        
    logger.info(f"Train D0 F0.5: {f05_D0:.4f}, D1 F0.5: {f05_D1:.4f}")
    
    # Choose best policy
    best_policy = 'D0' if f05_D0 >= f05_D1 else 'D1'
    logger.info(f"Selected policy: {best_policy}")
    
    # Now run test
    logger.info("Running on test set...")
    t_s1 = load_data('data/processed/v001/test_source1.parquet')
    
    t_s2 = load_data('data/processed/v001/test_source2.parquet')
    valid_s2_ids = set(t_s2['entity_id'])
    t_cands_s2 = run_retrieval(t_s1, t_s2, 'S2', use_country)
    del t_s2
    
    t_s3 = load_data('data/processed/v001/test_source3.parquet')
    valid_s3_ids = set(t_s3['entity_id'])
    t_cands_s3 = run_retrieval(t_s1, t_s3, 'S3', use_country)
    del t_s3
    
    t_cands = pd.concat([t_cands_s2, t_cands_s3], ignore_index=True)
    del t_cands_s2, t_cands_s3
    
    test_s1_ids = t_s1['entity_id'].values
    del t_s1
    
    # Note: evaluate_decision also just generates predictions
    # We pass an empty gt_dict for test
    empty_gt = {s1_id: set() for s1_id in test_s1_ids}
    test_preds, _ = evaluate_decision(t_cands, empty_gt, test_s1_ids, best_policy)
    
    # Write submissions
    # First, candidate pairs
    cand_map = {}
    for s1_id in test_s1_ids: cand_map[s1_id] = set()
    for _, row in t_cands.iterrows():
        cand_map[row['entity_id_s1']].add(row['entity_id_cand'])
        
    cand_df = pd.DataFrame([
        {"source1_entity_id": k, "candidate_entity_ids": ",".join(sorted(v))}
        for k, v in cand_map.items()
    ])
    
    match_df = pd.DataFrame([
        {"source1_entity_id": k, "matched_entity_ids": ",".join(sorted(v))}
        for k, v in test_preds.items()
    ])
    
    write_submission_tsv(cand_df, str(subs_dir / "candidate_pairs.tsv"))
    write_submission_tsv(match_df, str(subs_dir / "matching_results.tsv"))
    
    logger.info("Running internal validation...")
    valid_s2_s3_ids = valid_s2_ids.union(valid_s3_ids)
    
    errors = validate_submission(match_df, cand_df, set(test_s1_ids), valid_s2_s3_ids)
    if errors:
        logger.error(f"Internal Validation failed: {errors}")
    else:
        logger.info("Internal Validation passed successfully.")
        
    logger.info(f"E000 completed in {time.time() - t0:.1f}s")
    
if __name__ == "__main__":
    main()
