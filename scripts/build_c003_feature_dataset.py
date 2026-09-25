"""
C003 Feature Dataset Builder CLI.

Usage:
  python scripts/build_c003_feature_dataset.py --config configs/c003_features_v1.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

from src.business_entity_resolution.features.feature_dataset import build_c003_feature_dataset
from src.business_entity_resolution.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C003 Feature Dataset Builder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/c003_features_v1.yaml",
        help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    logger = get_logger("C003_CLI")
    args = parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    try:
        build_c003_feature_dataset(cfg)
        logger.info("C003 execution completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("CRITICAL: C003 execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
