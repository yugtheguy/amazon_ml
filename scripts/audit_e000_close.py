import pandas as pd
import json
import numpy as np
from pathlib import Path

def main():
    print("Loading data...")
    cands = pd.read_parquet("artifacts/retrieval/E000/train_candidates.parquet")
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    folds = pd.read_parquet("artifacts/folds/folds_v1.parquet")
    s1 = pd.read_parquet("data/processed/v001/train_source1.parquet", columns=["entity_id", "country"])
    
    # Pre-computations
    s1_countries = dict(zip(s1['entity_id'], s1['country']))
    folds_dict = folds.set_index('source1_entity_id').to_dict('index')
    
    # 4. Actual Candidate-Pair Volume
    total_cand_pairs = len(cands)
    s2_cand_pairs = (cands['candidate_source'] == 'S2').sum()
    s3_cand_pairs = (cands['candidate_source'] == 'S3').sum()
    
    # 5. Candidates per S1 distribution
    cand_counts = cands.groupby('entity_id_s1').size()
    all_s1_ids = pd.Series(list(folds_dict.keys()))
    cand_counts = cand_counts.reindex(all_s1_ids).fillna(0)
    
    def get_dist(series):
        if len(series) == 0: return {}
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p90": float(series.quantile(0.90)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "frac_0": float((series == 0).mean()),
            "frac_1": float((series == 1).mean()),
            "frac_gt_1": float((series > 1).mean()),
            "frac_gt_10": float((series > 10).mean())
        }
        
    dist_overall = get_dist(cand_counts)
    
    zero_ids = [k for k,v in folds_dict.items() if v['match_bucket'] == 'ZERO_MATCH']
    single_ids = [k for k,v in folds_dict.items() if v['match_bucket'] == 'SINGLE_MATCH']
    multi_ids = [k for k,v in folds_dict.items() if v['match_bucket'] == 'MULTI_MATCH']
    
    dist_zero = get_dist(cand_counts.loc[zero_ids])
    dist_single = get_dist(cand_counts.loc[single_ids])
    dist_multi = get_dist(cand_counts.loc[multi_ids])
    
    # 6. Pair Candidate Recall
    gt_pairs = []
    gt_dict = {}
    for _, row in gt.iterrows():
        s = row['source1_entity_id']
        m = row['matched_entity_ids']
        if m:
            matches = m.split(',')
            gt_dict[s] = set(matches)
            for match in matches:
                gt_pairs.append((s, match))
        else:
            gt_dict[s] = set()
            
    gt_pairs_set = set(gt_pairs)
    cand_pairs_set = set(zip(cands['entity_id_s1'], cands['entity_id_cand']))
    
    def pair_recall(s1_subset=None, source_filter=None):
        subset_gt = gt_pairs_set
        if s1_subset is not None:
            subset_gt = {(s, c) for s, c in subset_gt if s in s1_subset}
        if source_filter is not None:
            subset_gt = {(s, c) for s, c in subset_gt if c.startswith(source_filter)}
            
        if len(subset_gt) == 0: return 0.0
        return len(subset_gt.intersection(cand_pairs_set)) / len(subset_gt)
        
    us_ids = {k for k, v in s1_countries.items() if v == 'US'}
    in_ids = {k for k, v in s1_countries.items() if v == 'India'}
    
    recalls = {
        "overall": pair_recall(),
        "S2": pair_recall(source_filter="S2"),
        "S3": pair_recall(source_filter="S3"),
        "US": pair_recall(s1_subset=us_ids),
        "India": pair_recall(s1_subset=in_ids),
        "SINGLE_MATCH": pair_recall(s1_subset=set(single_ids)),
        "MULTI_MATCH": pair_recall(s1_subset=set(multi_ids))
    }
    
    # 7. Full-GT Entity Coverage
    cand_dict = {}
    for _, row in cands.iterrows():
        cand_dict.setdefault(row['entity_id_s1'], set()).add(row['entity_id_cand'])
        
    def full_coverage(s1_subset=None, source_filter=None): 
        matched_s1 = [s for s, matches in gt_dict.items() if len(matches) > 0]
        if s1_subset is not None:
            matched_s1 = [s for s in matched_s1 if s in s1_subset]
            
        if source_filter == 'S2-only':
            matched_s1 = [s for s in matched_s1 if all(m.startswith('S2') for m in gt_dict[s])]
        elif source_filter == 'S3-only':
            matched_s1 = [s for s in matched_s1 if all(m.startswith('S3') for m in gt_dict[s])]
        elif source_filter == 'BOTH':
            matched_s1 = [s for s in matched_s1 if any(m.startswith('S2') for m in gt_dict[s]) and any(m.startswith('S3') for m in gt_dict[s])]
            
        if len(matched_s1) == 0: return 0.0
        
        covered = sum(1 for s in matched_s1 if gt_dict[s].issubset(cand_dict.get(s, set())))
        return covered / len(matched_s1)
        
    coverages = {
        "overall": full_coverage(),
        "US": full_coverage(s1_subset=us_ids),
        "India": full_coverage(s1_subset=in_ids),
        "SINGLE_MATCH": full_coverage(s1_subset=set(single_ids)),
        "MULTI_MATCH": full_coverage(s1_subset=set(multi_ids)),
        "S2-only": full_coverage(source_filter="S2-only"),
        "S3-only": full_coverage(source_filter="S3-only"),
        "BOTH": full_coverage(source_filter="BOTH"),
    }
    
    # 8. Zero-Match Retrieval Noise (already done in dist_zero)
    zero_noise = {
        "num_zero_cands": int((cand_counts.loc[zero_ids] == 0).sum()),
        "num_gt_0_cands": int((cand_counts.loc[zero_ids] > 0).sum()),
        "mean_cands": dist_zero['mean'],
        "p95_cands": dist_zero['p95'],
        "p99_cands": dist_zero['p99']
    }
    
    # 9. Channel Contribution
    channels = ['exact_name_address_clean', 'exact_name_address_accent', 'exact_name_address_punct', 'unique_exact_name']
    contrib = {}
    
    for ch in channels:
        ch_cands = cands[cands[ch] == 1]
        ch_pairs = set(zip(ch_cands['entity_id_s1'], ch_cands['entity_id_cand']))
        
        other_cols = [c for c in channels if c != ch]
        unique_mask = (cands[ch] == 1) & (cands[other_cols].sum(axis=1) == 0)
        unique_cands = cands[unique_mask]
        unique_pairs = set(zip(unique_cands['entity_id_s1'], unique_cands['entity_id_cand']))
        
        gt_recovered = len(gt_pairs_set.intersection(ch_pairs))
        gt_unique = len(gt_pairs_set.intersection(unique_pairs))
        
        contrib[ch] = {
            "cand_pairs": len(ch_pairs),
            "unique_cands": len(unique_pairs),
            "gt_recovered": gt_recovered,
            "unique_gt_rescued": gt_unique
        }
        
    # 10. Channel Overlap
    num_channels_match = cands[channels].sum(axis=1)
    overlap = {
        "exactly_1": int((num_channels_match == 1).sum()),
        "exactly_2": int((num_channels_match == 2).sum()),
        "exactly_3": int((num_channels_match == 3).sum()),
        "all_4": int((num_channels_match == 4).sum()),
    }
    
    # Test cardinality
    test_s1 = pd.read_parquet("data/processed/v001/test_source1.parquet", columns=["entity_id"])
    
    try:
        test_cand_df = pd.read_csv("artifacts/submissions/E000/candidate_pairs.tsv", sep="\t", usecols=["source1_entity_id"])
        cand_rows = len(test_cand_df)
    except:
        cand_rows = 0
        
    try:
        test_match_df = pd.read_csv("artifacts/submissions/E000/matching_results.tsv", sep="\t", usecols=["source1_entity_id"])
        match_rows = len(test_match_df)
    except:
        match_rows = 0
    
    cardinality = {
        "test_source1_entities": len(test_s1),
        "candidate_pairs_rows": cand_rows,
        "matching_results_rows": match_rows,
        "equality": len(test_s1) == cand_rows == match_rows
    }
    
    results = {
        "volume": {
            "total": int(total_cand_pairs),
            "S2": int(s2_cand_pairs),
            "S3": int(s3_cand_pairs)
        },
        "distributions": {
            "overall": dist_overall,
            "zero_match": dist_zero,
            "single_match": dist_single,
            "multi_match": dist_multi
        },
        "recall": recalls,
        "coverage": coverages,
        "zero_noise": zero_noise,
        "contributions": contrib,
        "overlap": overlap,
        "cardinality": cardinality
    }
    
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    with open("artifacts/reports/e000_close_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Metrics written to artifacts/reports/e000_close_metrics.json")

if __name__ == "__main__":
    main()
