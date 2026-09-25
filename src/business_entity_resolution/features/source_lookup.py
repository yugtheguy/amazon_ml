"""
C003 Source Text Lookup

Handles scalable loading of normalized S1/S2/S3 text.
To avoid reloading tens of millions of rows per C002 shard, this module
maintains a memory-efficient Pandas-backed lookup dictionary for the required
text columns.

Scale analysis:
~ 2M S2, ~ 2M S3, ~ 2.2M S1. Total ~ 6.2M entities.
Columns needed: name_norm_clean, addr_norm_clean, numeric_tokens (list of str).
Memory for 6.2M entities * 3 string columns is ~ 500 MB - 1 GB.
This easily fits in bounded RAM and drastically out-performs per-shard repeated reads.

Missing value behavior:
If an entity ID is missing from the lookup, missing values (NaN/None) are returned.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)


class SourceTextLookup:
    """Memory-efficient text lookup registry."""

    def __init__(self):
        self.s1_df: Optional[pd.DataFrame] = None
        self.s2_df: Optional[pd.DataFrame] = None
        self.s3_df: Optional[pd.DataFrame] = None

    def load_sources(self, s1_path: str, s2_path: str, s3_path: str) -> None:
        """Load minimal required text columns from normalized parquet files."""
        cols = ["entity_id", "name_norm_clean", "addr_norm_clean", "numeric_tokens"]
        
        logger.info(f"[C003] Loading S1 text from {s1_path}")
        self.s1_df = pd.read_parquet(s1_path, columns=cols).set_index("entity_id")
        
        logger.info(f"[C003] Loading S2 text from {s2_path}")
        self.s2_df = pd.read_parquet(s2_path, columns=cols).set_index("entity_id")
        
        logger.info(f"[C003] Loading S3 text from {s3_path}")
        self.s3_df = pd.read_parquet(s3_path, columns=cols).set_index("entity_id")
        
        logger.info("[C003] Source text lookup ready.")

    def get_s1_text(self, s1_ids: pd.Series) -> pd.DataFrame:
        """Lookup text for an array of S1 IDs."""
        if self.s1_df is None:
            raise RuntimeError("S1 text not loaded.")
        # reindex guarantees alignment with the requested IDs. Missing IDs become NaN.
        return self.s1_df.reindex(s1_ids).reset_index(drop=True)

    def get_cand_text(self, cand_ids: pd.Series, sources: pd.Series) -> pd.DataFrame:
        """Lookup text for an array of candidate IDs, respecting their source."""
        if self.s2_df is None or self.s3_df is None:
            raise RuntimeError("Candidate text not loaded.")
            
        # Create an empty dataframe to hold results
        res = pd.DataFrame(index=cand_ids.index, columns=["name_norm_clean", "addr_norm_clean", "numeric_tokens"])
        
        # S2 mask
        s2_mask = sources == "S2"
        s2_ids = cand_ids[s2_mask]
        if not s2_ids.empty:
            s2_text = self.s2_df.reindex(s2_ids)
            res.loc[s2_mask, ["name_norm_clean", "addr_norm_clean", "numeric_tokens"]] = s2_text.values
            
        # S3 mask
        s3_mask = sources == "S3"
        s3_ids = cand_ids[s3_mask]
        if not s3_ids.empty:
            s3_text = self.s3_df.reindex(s3_ids)
            res.loc[s3_mask, ["name_norm_clean", "addr_norm_clean", "numeric_tokens"]] = s3_text.values
            
        return res

    def clear(self) -> None:
        """Release memory."""
        self.s1_df = None
        self.s2_df = None
        self.s3_df = None
        gc.collect()
