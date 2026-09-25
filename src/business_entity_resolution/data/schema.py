"""
Match bucket classification — single source of truth.

ZERO_MATCH:  n_true_matches == 0
SINGLE_MATCH: n_true_matches == 1
MULTI_MATCH:  n_true_matches >= 2
"""


def classify_match_bucket(n_true_matches: int) -> str:
    """Classify a Source-1 entity into a match bucket based on its true match count.

    Args:
        n_true_matches: Number of ground-truth matches for the entity.

    Returns:
        One of 'ZERO_MATCH', 'SINGLE_MATCH', 'MULTI_MATCH'.

    Raises:
        ValueError: If n_true_matches is negative.
    """
    if n_true_matches < 0:
        raise ValueError(f"n_true_matches must be non-negative, got {n_true_matches}")
    if n_true_matches == 0:
        return "ZERO_MATCH"
    elif n_true_matches == 1:
        return "SINGLE_MATCH"
    else:
        return "MULTI_MATCH"


def classify_source_pattern(matched_ids_str: str) -> str:
    """Derive source pattern from comma-separated matched entity IDs.

    The source is determined by the ID prefix: S2- or S3-.

    Args:
        matched_ids_str: Comma-separated string of matched entity IDs (may be empty).

    Returns:
        One of 'NONE', 'S2_ONLY', 'S3_ONLY', 'BOTH'.
    """
    if not matched_ids_str:
        return "NONE"

    ids = matched_ids_str.split(",")
    has_s2 = any(mid.startswith("S2-") for mid in ids)
    has_s3 = any(mid.startswith("S3-") for mid in ids)

    if has_s2 and has_s3:
        return "BOTH"
    elif has_s2:
        return "S2_ONLY"
    elif has_s3:
        return "S3_ONLY"
    else:
        return "NONE"
