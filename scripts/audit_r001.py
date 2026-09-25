import os
import pandas as pd
import numpy as np
import json

def get_list_overlap(list1, list2):
    if not isinstance(list1, (list, np.ndarray)) or not isinstance(list2, (list, np.ndarray)): return 0
    return len(set(list1).intersection(set(list2)))

def main():
    os.makedirs("artifacts/candidate_pool/R001/audit", exist_ok=True)
    
    # Load data
    s1 = pd.read_parquet("data/processed/v001/train_source1.parquet")
    internal = pd.read_parquet("artifacts/candidate_pool/R001/probe_50k/internal_candidates.parquet")
    
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep='\t')
    gt['match_entity_ids'] = gt['matched_entity_ids'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
    gt.rename(columns={'source1_entity_id': 'entity_id'}, inplace=True)
    
    # Filter S1 to just those in the probe
    probe_s1_ids = internal['entity_id_s1'].unique()
    s1_probe = s1[s1['entity_id'].isin(probe_s1_ids)].copy()
    s1_probe = s1_probe.merge(gt[['entity_id', 'match_entity_ids']], on='entity_id', how='left')
    
    # Reconstruct ranking on internal candidates independent of final_candidates.parquet
    internal['is_exact'] = (
        (internal['exact_name_address_clean'] == 1) | 
        (internal['exact_name_address_accent'] == 1) | 
        (internal['exact_name_address_punct'] == 1)
    ).astype(int)
    
    internal.sort_values(
        by=[
            'entity_id_s1',
            'is_exact', 
            'retrieval_channel_count', 
            'name_word_score', 
            'address_word_score',
            'rare_token_overlap_count',
            'shared_numeric_count',
            'name_word_rank'
        ],
        ascending=[True, False, False, False, False, False, False, True],
        inplace=True
    )
    
    # Create rank column
    internal['candidate_rank'] = internal.groupby('entity_id_s1').cumcount() + 1
    
    # Merge GT
    s1_gt = s1_probe[['entity_id', 'match_entity_ids', 'country']].rename(columns={'entity_id': 'entity_id_s1'})
    internal = internal.merge(s1_gt, on='entity_id_s1', how='left')
    
    # Flag hits
    def is_hit(row):
        return int(row['entity_id_cand'] in set(row['match_entity_ids']) if isinstance(row['match_entity_ids'], (list, np.ndarray)) else False)
    
    internal['is_gt'] = internal.apply(is_hit, axis=1)
    
    # Pre-calculate true pairs per S1
    s1_gt['total_gt'] = s1_gt['match_entity_ids'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    
    def calc_metrics(df_candidates, budget):
        if budget is not None:
            cands = df_candidates[df_candidates['candidate_rank'] <= budget]
        else:
            cands = df_candidates
            
        hits = cands[cands['is_gt'] == 1].groupby('entity_id_s1').size().reset_index(name='retrieved_gt')
        cand_counts = cands.groupby('entity_id_s1').size().reset_index(name='c_count')
        
        merged = s1_gt.merge(hits, on='entity_id_s1', how='left').merge(cand_counts, on='entity_id_s1', how='left')
        merged['retrieved_gt'] = merged['retrieved_gt'].fillna(0)
        merged['c_count'] = merged['c_count'].fillna(0)
        
        valid_s1 = merged[merged['total_gt'] > 0]
        total_gt_pairs = valid_s1['total_gt'].sum()
        retrieved_gt_pairs = valid_s1['retrieved_gt'].sum()
        pair_recall = retrieved_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0
        
        full_cov = (valid_s1['retrieved_gt'] == valid_s1['total_gt']).mean()
        
        zero_s1 = merged[merged['total_gt'] == 0]
        zero_mean = zero_s1['c_count'].mean() if len(zero_s1) > 0 else 0
        
        return {
            "budget": budget if budget is not None else "Internal",
            "pair_recall": pair_recall,
            "full_coverage": full_cov,
            "mean_candidates": merged['c_count'].mean(),
            "p50_candidates": merged['c_count'].median(),
            "p90_candidates": merged['c_count'].quantile(0.90),
            "p95_candidates": merged['c_count'].quantile(0.95),
            "p99_candidates": merged['c_count'].quantile(0.99),
            "max_candidates": merged['c_count'].max(),
            "zero_match_mean_candidates": zero_mean
        }
        
    budgets = [3, 5, 8, 10, 15, None]
    frontier = []
    for b in budgets:
        frontier.append(calc_metrics(internal, b))
        
    df_frontier = pd.DataFrame(frontier)
    df_frontier.to_csv("artifacts/candidate_pool/R001/audit/candidate_frontier_audited.csv", index=False)
    
    # 6. INVESTIGATE THE BUDGET 10 -> 15 JUMP
    positives = internal[internal['is_gt'] == 1]
    rank_counts = positives['candidate_rank'].value_counts().sort_index()
    
    hist_data = []
    cum_recall = 0
    total_pairs = s1_gt['total_gt'].sum()
    
    for r in range(1, 16):
        c = rank_counts.get(r, 0)
        cum_recall += c
        hist_data.append({
            "rank": str(r),
            "gt_pairs": c,
            "cumulative_recall": cum_recall / total_pairs
        })
    gt_over_15 = positives[positives['candidate_rank'] > 15].shape[0]
    cum_recall += gt_over_15
    hist_data.append({"rank": ">15", "gt_pairs": gt_over_15, "cumulative_recall": cum_recall / total_pairs})
    pd.DataFrame(hist_data).to_csv("artifacts/candidate_pool/R001/audit/positive_rank_histogram.csv", index=False)
    
    # 15. CHANNEL CONTRIBUTION (Unique Rescue)
    channels = {
        'Exact': 'retrieved_exact',
        'Name Word TF-IDF': 'retrieved_name_word',
        'Address Word TF-IDF': 'retrieved_address_word',
        'Rare Token': 'retrieved_rare',
        'Numeric': 'retrieved_numeric'
    }
    
    chan_stats = []
    all_hits_set = set(positives.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
    
    for c_name, c_col in channels.items():
        c_cands = internal[internal[c_col] == 1]
        c_hits = positives[positives[c_col] == 1]
        c_hits_set = set(c_hits.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
        
        # Unique rescue = pairs that were hit ONLY by this channel
        other_hits = positives[positives[c_col] == 0]
        other_hits_set = set(other_hits.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
        
        unique_rescue = len(c_hits_set - other_hits_set)
        
        chan_stats.append({
            "Channel": c_name,
            "Candidates Added": len(c_cands),
            "GT Recovered": len(c_hits_set),
            "Unique GT Rescue": unique_rescue,
            "Rescue Efficiency": unique_rescue / len(c_cands) if len(c_cands) > 0 else 0
        })
    pd.DataFrame(chan_stats).to_csv("artifacts/candidate_pool/R001/audit/channel_contribution_audited.csv", index=False)
    
    # Missing Analysis (Type A vs B)
    # Total GT pairs
    all_gt_pairs = []
    for _, row in s1_gt.iterrows():
        s1_id = row['entity_id_s1']
        if isinstance(row['match_entity_ids'], (list, np.ndarray)):
            for t_id in row['match_entity_ids']:
                all_gt_pairs.append((s1_id, t_id))
                
    all_gt_set = set(all_gt_pairs)
    internal_hit_set = set(positives.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
    
    pruned_positives = positives[positives['candidate_rank'] <= 15]
    final_hit_set = set(pruned_positives.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
    
    type_a_misses = all_gt_set - internal_hit_set
    type_b_misses = internal_hit_set - final_hit_set
    
    miss_stats = {
        "total_gt_pairs": len(all_gt_set),
        "type_a_retrieval_misses": len(type_a_misses),
        "type_b_pruning_misses": len(type_b_misses)
    }
    
    with open("artifacts/candidate_pool/R001/audit/audit_summary.json", "w") as f:
        json.dump(miss_stats, f, indent=2)

if __name__ == "__main__":
    main()
