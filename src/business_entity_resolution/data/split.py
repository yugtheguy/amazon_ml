import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)

def split_source1(df_s1: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the Source 1 entities into training and validation sets.
    This ensures validation is split at the Source 1 entity level.
    """
    logger.info(f"Splitting Source 1 data (N={len(df_s1)}) with test_size={test_size} and random_state={random_state}")
    
    # We can stratify by country to ensure representation
    if 'country' in df_s1.columns:
        train_s1, val_s1 = train_test_split(df_s1, test_size=test_size, random_state=random_state, stratify=df_s1['country'])
    else:
        train_s1, val_s1 = train_test_split(df_s1, test_size=test_size, random_state=random_state)
        
    logger.info(f"Train Source 1 size: {len(train_s1)}")
    logger.info(f"Val Source 1 size: {len(val_s1)}")
    
    return train_s1, val_s1
