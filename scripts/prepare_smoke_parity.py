import os
import pandas as pd

def main():
    print("Loading 50K internal candidates...")
    internal_50k = pd.read_parquet("artifacts/candidate_pool/R001/probe_50k/internal_candidates.parquet")
    
    # Get all unique S1 IDs in the 50K probe
    unique_s1 = internal_50k['entity_id_s1'].unique()
    
    # Take a deterministic 1000 subset
    unique_1k = sorted(list(unique_s1))[:1000]
    
    df_1k = pd.DataFrame({'entity_id_s1': unique_1k})
    os.makedirs("artifacts/candidate_pool/R001/smoke_1k", exist_ok=True)
    df_1k.to_csv("artifacts/candidate_pool/R001/smoke_1k/smoke_1k_ids.csv", index=False)
    
    # Slice the local candidates for baseline comparison
    internal_1k = internal_50k[internal_50k['entity_id_s1'].isin(set(unique_1k))]
    
    # Actually wait, in R001 we did rank_and_prune. We should load final_candidates.parquet and slice it too.
    # Wait, the user said "artifacts/candidate_pool/R001/probe_50k/final_candidates.parquet" didn't exist in my previous list because it was just candidate_count_vs_recall.csv?
    # No, it was created if budget == 15. Let's check if it exists.
    final_path = "artifacts/candidate_pool/R001/probe_50k/final_candidates.parquet"
    if os.path.exists(final_path):
        final_50k = pd.read_parquet(final_path)
        final_1k = final_50k[final_50k['entity_id_s1'].isin(set(unique_1k))]
        final_1k.to_parquet("artifacts/candidate_pool/R001/smoke_1k/local_final_candidates.parquet", index=False)
    
    internal_1k.to_parquet("artifacts/candidate_pool/R001/smoke_1k/local_internal_candidates.parquet", index=False)
    print("Done generating local smoke baselines.")

if __name__ == "__main__":
    main()
