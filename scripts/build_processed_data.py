"""Build processed dataset (Parquet) from raw TSVs with multi-view normalization."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.business_entity_resolution.data.processing import build_processed_dataset


def main():
    manifest = build_processed_dataset(
        raw_train_dir="data/raw/train",
        raw_test_dir="data/raw/test",
        output_dir="data/processed/v001",
        normalization_version="v001",
    )

    print(f"\n=== Processing Summary ===")
    print(f"Version: {manifest['processing_version']}")
    print(f"Total time: {manifest['total_processing_time_seconds']:.1f}s")

    for f in manifest["files"]:
        print(f"\n  {f['source_label']}:")
        print(f"    Raw rows: {f['raw_row_count']}")
        print(f"    Output rows: {f['output_row_count']}")
        print(f"    Columns: {len(f['output_columns'])}")


if __name__ == "__main__":
    main()
