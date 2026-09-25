import os
import pandas as pd
import numpy as np
import json
import csv

def main():
    os.makedirs("artifacts/candidate_pool/R001/ranking_audit", exist_ok=True)
    
    print("Loading data...")
    internal = pd.read_parquet("artifacts/candidate_pool/R001/probe_50k/internal_candidates.parquet")
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep='\t')
    gt['match_entity_ids'] = gt['matched_entity_ids'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
    gt.rename(columns={'source1_entity_id': 'entity_id_s1'}, inplace=True)
    
    # Load S1 meta
    s1_meta = pd.read_parquet("data/processed/v001/train_source1.parquet")
    probe_s1_ids = internal['entity_id_s1'].unique()
    s1_meta_probe = s1_meta[s1_meta['entity_id'].isin(probe_s1_ids)].rename(columns={'entity_id': 'entity_id_s1'})
    
    s1_gt = s1_meta_probe[['entity_id_s1', 'country']].merge(gt[['entity_id_s1', 'match_entity_ids']], on='entity_id_s1', how='left')
    s1_gt['match_entity_ids'] = s1_gt['match_entity_ids'].apply(lambda x: x if isinstance(x, list) else [])
    s1_gt['total_gt'] = s1_gt['match_entity_ids'].apply(len)
    
    print("Applying new ranking...")
    internal['is_exact'] = (
        (internal['exact_name_address_clean'] == 1) | 
        (internal['exact_name_address_accent'] == 1) | 
        (internal['exact_name_address_punct'] == 1)
    ).astype(int)
    
    internal['best_lexical_rank'] = internal[['name_word_rank', 'address_word_rank']].min(axis=1)
    internal['both_name_address'] = ((internal['retrieved_name_word'] == 1) & (internal['retrieved_address_word'] == 1)).astype(int)
    
    internal.sort_values(
        by=[
            'entity_id_s1',
            'is_exact', 
            'retrieval_channel_count', 
            'both_name_address',
            'best_lexical_rank', 
            'rare_token_overlap_count',
            'shared_numeric_count',
            'entity_id_cand'
        ],
        ascending=[True, False, False, False, True, False, False, True],
        inplace=True
    )
    
    internal['candidate_rank'] = internal.groupby('entity_id_s1').cumcount() + 1
    
    def is_hit(row):
        # We need a quick way to check hits. Let's merge GT first.
        return 0 # Placeholder
    
    # Merge GT to internal for faster hit checking
    print("Computing hits...")
    gt_exploded = s1_gt.explode('match_entity_ids').rename(columns={'match_entity_ids': 'entity_id_cand'})
    gt_exploded['is_gt'] = 1
    
    internal = internal.merge(gt_exploded[['entity_id_s1', 'entity_id_cand', 'is_gt']], on=['entity_id_s1', 'entity_id_cand'], how='left')
    internal['is_gt'] = internal['is_gt'].fillna(0).astype(int)
    
    # Function to calculate metrics
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
        zero_pct = (zero_s1['c_count'] == 0).mean() if len(zero_s1) > 0 else 0
        
        # S2/S3 recall
        s2_hits = cands[(cands['is_gt'] == 1) & (cands['candidate_source'] == 'S2')]['is_gt'].sum()
        s3_hits = cands[(cands['is_gt'] == 1) & (cands['candidate_source'] == 'S3')]['is_gt'].sum()
        
        s2_total = gt_exploded[gt_exploded['entity_id_cand'].str.startswith('S2', na=False)].shape[0]
        s3_total = gt_exploded[gt_exploded['entity_id_cand'].str.startswith('S3', na=False)].shape[0]
        
        # Single vs Multi
        single_s1 = valid_s1[valid_s1['total_gt'] == 1]
        multi_s1 = valid_s1[valid_s1['total_gt'] > 1]
        
        # Country
        us_s1 = valid_s1[valid_s1['country'] == 'US']
        in_s1 = valid_s1[valid_s1['country'] == 'India']
        
        return {
            "budget": budget if budget is not None else "Internal",
            "pair_recall": pair_recall,
            "full_coverage": full_cov,
            "mean_candidates": merged['c_count'].mean(),
            "p50_candidates": merged['c_count'].median(),
            "p90_candidates": merged['c_count'].quantile(0.90),
            "p95_candidates": merged['c_count'].quantile(0.95),
            "p99_candidates": merged['c_count'].quantile(0.99),
            "zero_match_mean_candidates": zero_mean,
            "zero_match_zero_pct": zero_pct,
            "single_coverage": (single_s1['retrieved_gt'] == single_s1['total_gt']).mean() if len(single_s1)>0 else 0,
            "multi_coverage": (multi_s1['retrieved_gt'] == multi_s1['total_gt']).mean() if len(multi_s1)>0 else 0,
            "s2_recall": s2_hits / s2_total if s2_total > 0 else 0,
            "s3_recall": s3_hits / s3_total if s3_total > 0 else 0,
            "us_recall": us_s1['retrieved_gt'].sum() / us_s1['total_gt'].sum() if us_s1['total_gt'].sum() > 0 else 0,
            "in_recall": in_s1['retrieved_gt'].sum() / in_s1['total_gt'].sum() if in_s1['total_gt'].sum() > 0 else 0
        }

    print("Generating frontier...")
    budgets = [3, 5, 8, 10, 12, 15, None]
    frontier = []
    for b in budgets:
        frontier.append(calc_metrics(internal, b))
        
    df_frontier = pd.DataFrame(frontier)
    df_frontier.to_csv("artifacts/candidate_pool/R001/ranking_audit/candidate_frontier_improved.csv", index=False)
    
    # Ranking comparison
    old_frontier = pd.read_csv("artifacts/candidate_pool/R001/audit/candidate_frontier_audited.csv")
    comp = []
    for b in budgets:
        n_row = df_frontier[df_frontier['budget'] == (str(b) if b is None else b)]
        o_row = old_frontier[old_frontier['budget'].astype(str) == str(b)] if str(b) in old_frontier['budget'].astype(str).values else None
        
        if len(n_row) > 0 and o_row is not None and len(o_row) > 0:
            n = n_row.iloc[0]
            o = o_row.iloc[0]
            comp.append({
                "Budget": b,
                "Old Pair Recall": o['pair_recall'],
                "New Pair Recall": n['pair_recall'],
                "Delta Pair Recall": n['pair_recall'] - o['pair_recall'],
                "Old Full Coverage": o['full_coverage'],
                "New Full Coverage": n['full_coverage'],
                "Delta Full Coverage": n['full_coverage'] - o['full_coverage']
            })
    pd.DataFrame(comp).to_csv("artifacts/candidate_pool/R001/ranking_audit/ranking_comparison.csv", index=False)
    
    # Histogram
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
    pd.DataFrame(hist_data).to_csv("artifacts/candidate_pool/R001/ranking_audit/positive_rank_histogram_improved.csv", index=False)
    
    # True Unique Rescue
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
        
        other_hits = positives[positives[c_col] == 0]
        other_hits_set = set(other_hits.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
        
        unique_rescue = len(c_hits_set - other_hits_set)
        
        chan_stats.append({
            "Channel": c_name,
            "Candidates Added": len(c_cands),
            "GT Recovered": len(c_hits_set),
            "GT Uniquely Recovered": unique_rescue,
            "Incremental Recall": unique_rescue / total_pairs if total_pairs > 0 else 0,
            "Rescue Efficiency": unique_rescue / len(c_cands) if len(c_cands) > 0 else 0
        })
    pd.DataFrame(chan_stats).to_csv("artifacts/candidate_pool/R001/ranking_audit/channel_contribution_final.csv", index=False)
    
    # Type A vs Type B Misses
    all_gt_set = set()
    for _, row in s1_gt.iterrows():
        s1_id = row['entity_id_s1']
        for t_id in row['match_entity_ids']:
            all_gt_set.add((s1_id, t_id))
            
    internal_hit_set = set(positives.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
    
    # Let's say Budget 8 is the new frontier choice (we will check the data)
    # But for type B, we usually mean pruned out of whatever budget we freeze. We will use budget 8 for now.
    pruned_positives = positives[positives['candidate_rank'] <= 8]
    final_hit_set = set(pruned_positives.apply(lambda x: (x['entity_id_s1'], x['entity_id_cand']), axis=1))
    
    type_a_misses = list(all_gt_set - internal_hit_set)
    type_b_misses = list(internal_hit_set - final_hit_set)
    
    print(f"Total GT: {len(all_gt_set)}")
    print(f"Type A: {len(type_a_misses)}")
    print(f"Type B (at budget 8): {len(type_b_misses)}")
    
    # Sample Type A for qualitative analysis
    sample_a = type_a_misses[:50]
    
    # Fetch actual names
    s1_raw = pd.read_csv("data/raw/train/train_source1.tsv", sep='\t', quoting=3, dtype=str)
    s2_raw = pd.read_csv("data/raw/train/train_source2.tsv", sep='\t', quoting=3, dtype=str)
    s3_raw = pd.read_csv("data/raw/train/train_source3.tsv", sep='\t', quoting=3, dtype=str)
    
    target_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)
    
    a_records = []
    for s1_id, t_id in sample_a:
        r1 = s1_raw[s1_raw['entity_id'] == s1_id]
        rt = target_raw[target_raw['entity_id'] == t_id]
        if not r1.empty and not rt.empty:
            a_records.append({
                "s1_id": s1_id,
                "t_id": t_id,
                "s1_name": r1.iloc[0].get('name', ''),
                "s1_addr": r1.iloc[0].get('address', ''),
                "t_name": rt.iloc[0].get('name', ''),
                "t_addr": rt.iloc[0].get('address', ''),
                "country": r1.iloc[0].get('country', ''),
            })
            
    pd.DataFrame(a_records).to_csv("artifacts/candidate_pool/R001/ranking_audit/miss_analysis_type_a.csv", index=False)
    
    print("Audit complete.")

if __name__ == "__main__":
    main()
