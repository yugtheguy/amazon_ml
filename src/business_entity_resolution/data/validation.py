import pandas as pd
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)

def validate_schema(df: pd.DataFrame, expected_columns: list, name: str = "DataFrame"):
    """Validates that expected columns are present."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        msg = f"Schema validation failed for {name}. Missing columns: {missing}"
        logger.error(msg)
        raise ValueError(msg)
    
    logger.info(f"Schema validation passed for {name}.")

def validate_ids_unique(df: pd.DataFrame, id_column: str, name: str = "DataFrame"):
    """Validates that all IDs in the specified column are unique."""
    duplicates = df[id_column].duplicated().sum()
    if duplicates > 0:
        msg = f"ID uniqueness validation failed for {name}. {duplicates} duplicate IDs found in '{id_column}'."
        logger.error(msg)
        raise ValueError(msg)
    
    logger.info(f"ID uniqueness validation passed for {name}.")

def validate_test_submission_format(df: pd.DataFrame, test_s1_ids: set):
    """
    Validates that the output dataframe has exactly the expected test source 1 IDs,
    and has the correct columns.
    """
    validate_schema(df, ['source1_entity_id', 'matched_entity_ids'], "Submission DataFrame")
    validate_ids_unique(df, 'source1_entity_id', "Submission DataFrame")
    
    sub_ids = set(df['source1_entity_id'])
    
    missing_ids = test_s1_ids - sub_ids
    if missing_ids:
        msg = f"Submission validation failed. {len(missing_ids)} expected source1_entity_id(s) are missing."
        logger.error(msg)
        raise ValueError(msg)
        
    extra_ids = sub_ids - test_s1_ids
    if extra_ids:
        msg = f"Submission validation failed. {len(extra_ids)} unexpected source1_entity_id(s) found."
        logger.error(msg)
        raise ValueError(msg)
        
    logger.info("Submission validation passed: all test S1 IDs are accounted for exactly once.")
