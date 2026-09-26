"""
C002 Pair Dataset Builder CLI.

Usage:
  python scripts/build_c002_pair_dataset.py --config configs/c002_pair_dataset_v1.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.business_entity_resolution.pairs.pair_dataset import build_c002_pair_dataset
from src.business_entity_resolution.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C002 Canonical Pair Dataset Builder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/c002_pair_dataset_v1.yaml",
        help="Path to YAML config file"
    )
    # Allow overrides
    parser.add_argument("--candidate-pool-dir", type=str, help="Override candidate pool dir")
    parser.add_argument("--ground-truth-path", type=str, help="Override GT path")
    parser.add_argument("--fold-manifest", type=str, help="Override fold manifest path")
    parser.add_argument("--processed-s1-path", type=str, help="Override S1 parquet path")
    parser.add_argument("--out-dir", type=str, help="Override output dir")
    
    return parser.parse_args()


def main():
    logger = get_logger("C002_CLI")
    
    args = parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Apply CLI overrides
    if args.candidate_pool_dir:
        cfg["candidate_pool_dir"] = args.candidate_pool_dir
    if args.ground_truth_path:
        cfg["ground_truth_path"] = args.ground_truth_path
    if args.fold_manifest:
        cfg["fold_manifest_path"] = args.fold_manifest
    if args.processed_s1_path:
        cfg["processed_s1_path"] = args.processed_s1_path
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
        
    try:
        build_c002_pair_dataset(cfg)
        logger.info("C002 execution completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("CRITICAL: C002 execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
