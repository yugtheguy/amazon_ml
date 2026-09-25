import pandas as pd
import json
from pathlib import Path
from src.business_entity_resolution.evaluation.f05 import calculate_macro_f05
from scripts.run_e000 import evaluate_decision, get_gt_dict

def main():
    print("Loading data for detailed evaluation...")
    folds_df = pd.read_parquet("artifacts/folds/folds_v1.parquet")
    gt_df = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    cands_df = pd.read_parquet("artifacts/retrieval/E000/train_candidates.parquet")
    s1_df = pd.read_parquet("data/processed/v001/train_source1.parquet", columns=["entity_id", "country"])
    s1_countries = dict(zip(s1_df['entity_id'], s1_df['country']))
    
    print("Pre-computing GT dictionaries...")
    all_s1_ids = set(folds_df['source1_entity_id'])
    
    def get_metrics_for_subset(s1_ids, preds):
        subset_preds = {k: v for k, v in preds.items() if k in s1_ids}
        subset_gt = get_gt_dict(gt_df[gt_df['source1_entity_id'].isin(s1_ids)], s1_ids)
        if len(s1_ids) == 0: return 0.0
        return calculate_macro_f05(subset_gt, subset_preds)
        
    for policy in ['D0', 'D1']:
        print(f"Evaluating {policy}...")
        preds, overall_f05 = evaluate_decision(cands_df, get_gt_dict(gt_df, all_s1_ids), all_s1_ids, policy)
        
        results = {"overall": overall_f05, "by_fold": {}, "by_country": {}, "by_match_type": {}}
        
        # By fold
        for fold in sorted(folds_df['fold_id'].unique()):
            fold_ids = set(folds_df[folds_df['fold_id'] == fold]['source1_entity_id'])
            results["by_fold"][int(fold)] = get_metrics_for_subset(fold_ids, preds)
            
        # By country
        us_ids = {i for i in all_s1_ids if s1_countries.get(i) == "US"}
        in_ids = {i for i in all_s1_ids if s1_countries.get(i) == "India"}
        results["by_country"]["US"] = get_metrics_for_subset(us_ids, preds)
        results["by_country"]["India"] = get_metrics_for_subset(in_ids, preds)
        
        # By match type
        match_types = folds_df.set_index('source1_entity_id')['match_bucket'].to_dict()
        zero_ids = {i for i in all_s1_ids if match_types.get(i) == "ZERO_MATCH"}
        single_ids = {i for i in all_s1_ids if match_types.get(i) == "SINGLE_MATCH"}
        multi_ids = {i for i in all_s1_ids if match_types.get(i) == "MULTI_MATCH"}
        
        results["by_match_type"]["ZERO_MATCH"] = get_metrics_for_subset(zero_ids, preds)
        results["by_match_type"]["SINGLE_MATCH"] = get_metrics_for_subset(single_ids, preds)
        results["by_match_type"]["MULTI_MATCH"] = get_metrics_for_subset(multi_ids, preds)
        
        out_path = f"artifacts/retrieval/E000/decision_metrics_{policy}_detailed.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
