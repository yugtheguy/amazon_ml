"""Build permanent Source-1 fold manifest from real training data."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.business_entity_resolution.data.folds import build_fold_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=str, default="data/raw/train/train_source1.tsv")
    parser.add_argument("--ground-truth", type=str, default="data/raw/train/train_ground_truth.tsv")
    parser.add_argument("--output-dir", type=str, default="artifacts/folds")
    args = parser.parse_args()

    manifest = build_fold_manifest(
        source1_path=args.source1,
        ground_truth_path=args.ground_truth,
        n_folds=5,
        seed=42,
        version="v1",
        output_dir=args.output_dir,
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
