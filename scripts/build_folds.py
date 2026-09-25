"""Build permanent Source-1 fold manifest from real training data."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.business_entity_resolution.data.folds import build_fold_manifest


def main():
    manifest = build_fold_manifest(
        source1_path="data/raw/train/train_source1.tsv",
        ground_truth_path="data/raw/train/train_ground_truth.tsv",
        n_folds=5,
        seed=42,
        version="v1",
        output_dir="artifacts/folds",
    )

    print(f"\n=== Fold Manifest Summary ===")
    print(f"Total entities: {len(manifest)}")
    print(f"\nPer-fold sizes:")
    for fold_id in sorted(manifest["fold_id"].unique()):
        fold_df = manifest[manifest["fold_id"] == fold_id]
        print(f"  Fold {fold_id}: {len(fold_df)} entities")

    print(f"\nCountry distribution:")
    print(manifest["country"].value_counts().to_string())

    print(f"\nMatch bucket distribution:")
    print(manifest["match_bucket"].value_counts().to_string())

    print(f"\nSource pattern distribution:")
    print(manifest["source_pattern"].value_counts().to_string())

    print(f"\nPer-fold country x match_bucket:")
    for fold_id in sorted(manifest["fold_id"].unique()):
        fold_df = manifest[manifest["fold_id"] == fold_id]
        print(f"\n  Fold {fold_id}:")
        ct = fold_df.groupby(["country", "match_bucket"]).size()
        for (country, bucket), count in ct.items():
            print(f"    {country} / {bucket}: {count}")


if __name__ == "__main__":
    main()
