import pandas as pd
from typing import List, Tuple
from pathlib import Path
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)

def load_source_tsv(filepath: str) -> pd.DataFrame:
    """Loads a source TSV file."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, sep='\t')
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df

def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Loads ground truth TSV and ensures matched_entity_ids is treated as a string."""
    logger.info(f"Loading ground truth from {filepath}")
    # Read keeping matched_entity_ids as string so empty values become NaN which we can handle
    df = pd.read_csv(filepath, sep='\t', dtype={'matched_entity_ids': str})
    
    # Fill NaN with empty string
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')
    logger.info(f"Loaded {len(df)} ground truth rows from {filepath}")
    return df

def parse_matched_entity_ids(df: pd.DataFrame) -> dict:
    """
    Parses the ground truth DataFrame into a dictionary:
    { source1_entity_id : set([matched_id1, matched_id2, ...]) }
    """
    parsed = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if not matches_str:
            parsed[s1_id] = set()
        else:
            parsed[s1_id] = set(matches_str.split(','))
    return parsed
