"""
Multi-view text normalization for business entity resolution.

Design principles:
- Never destroy information: raw values are always preserved.
- Generate secondary views (normalized, accent-folded, numeric tokens).
- Country-light: no US/India/France-specific heuristics.
- Unicode-safe: uses unicodedata for normalization and folding.
- Pure, deterministic, independently testable functions.
- Missing values handled explicitly — never silently converted to "nan"/"None".

Views generated for names:
  - raw (original)
  - name_norm_unicode:    NFC-normalized
  - name_norm_clean:      case-folded + whitespace-normalized
  - name_norm_punct:      punctuation/whitespace-normalized
  - name_norm_accent_fold: accent-folded (diacritics removed)

Views generated for addresses:
  - raw (original)
  - addr_norm_unicode:    NFC-normalized
  - addr_norm_clean:      case-folded + whitespace-normalized
  - addr_norm_punct:      punctuation/whitespace-normalized
  - addr_norm_accent_fold: accent-folded (diacritics removed)
  - addr_numeric_tokens:  extracted numeric digit sequences
"""

import re
import unicodedata
from typing import Optional, List

import pandas as pd
import numpy as np

# Sentinel for missing values in normalized text columns
MISSING_SENTINEL = ""


def _is_missing(value) -> bool:
    """Check if a value is missing (None, NaN, or empty-ish)."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization.

    NFC composes characters to their canonical composed form,
    e.g. combining accent + base letter -> single composed character.
    """
    return unicodedata.normalize("NFC", text)


def normalize_casefold(text: str) -> str:
    """Case-fold text using Python's casefold() for Unicode-aware lowering.

    casefold() is more aggressive than lower() and handles non-ASCII correctly,
    e.g. German ß -> ss.
    """
    return text.casefold()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation while preserving meaningful characters.

    - Replaces '&' with ' and '
    - Removes punctuation characters (Unicode category P) except hyphens within words
    - Normalizes whitespace after removal
    """
    # Replace & with 'and'
    result = text.replace("&", " and ")
    # Remove Unicode punctuation but preserve hyphens between word characters
    # First, protect intra-word hyphens
    result = re.sub(r"(?<=\w)-(?=\w)", "\x00", result)
    # Remove all Unicode punctuation
    result = "".join(
        ch if unicodedata.category(ch)[0] != "P" else " "
        for ch in result
    )
    # Restore protected hyphens
    result = result.replace("\x00", "-")
    return normalize_whitespace(result)


def fold_accents(text: str) -> str:
    """Remove diacritics/accents while preserving base characters.

    Uses Unicode NFD decomposition to separate base characters from combining
    marks, then strips combining marks.

    Example: 'Société' -> 'Societe'
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def extract_numeric_tokens(text: str) -> str:
    """Extract sequences of digits from text, space-separated.

    Preserves meaningful digit sequences (house numbers, postal codes, etc.).

    Example: '21 Park Road, PIN 400001' -> '21 400001'
    """
    nums = re.findall(r"\d+", text)
    return " ".join(nums)


def normalize_text_full(raw: Optional[str]) -> dict:
    """Apply all normalization views to a single text value.

    Args:
        raw: The raw text value. May be None/NaN.

    Returns:
        Dict with keys: norm_unicode, norm_clean, norm_punct, norm_accent_fold.
        All values are strings. Missing input produces MISSING_SENTINEL for all.
    """
    if _is_missing(raw):
        return {
            "norm_unicode": MISSING_SENTINEL,
            "norm_clean": MISSING_SENTINEL,
            "norm_punct": MISSING_SENTINEL,
            "norm_accent_fold": MISSING_SENTINEL,
        }

    raw_str = str(raw)
    unicode_normed = normalize_unicode(raw_str)
    casefolded = normalize_casefold(unicode_normed)
    ws_normed = normalize_whitespace(casefolded)
    punct_normed = normalize_punctuation(ws_normed)
    accent_folded = fold_accents(ws_normed)

    return {
        "norm_unicode": unicode_normed,
        "norm_clean": ws_normed,
        "norm_punct": punct_normed,
        "norm_accent_fold": accent_folded,
    }


def normalize_address_full(raw: Optional[str]) -> dict:
    """Apply all normalization views to an address value.

    Same as normalize_text_full plus a numeric_tokens view.

    Args:
        raw: The raw address value. May be None/NaN.

    Returns:
        Dict with keys: norm_unicode, norm_clean, norm_punct, norm_accent_fold,
        numeric_tokens. All values are strings.
    """
    base = normalize_text_full(raw)
    if _is_missing(raw):
        base["numeric_tokens"] = MISSING_SENTINEL
    else:
        base["numeric_tokens"] = extract_numeric_tokens(str(raw))
    return base


# ---------------------------------------------------------------------------
# Vectorized DataFrame application
# ---------------------------------------------------------------------------

def apply_name_normalization(df: pd.DataFrame, raw_col: str = "business_name", prefix: str = "name") -> pd.DataFrame:
    """Apply multi-view name normalization to a DataFrame column.

    Adds new columns with the given prefix. Does NOT remove the raw column.

    Args:
        df: Input DataFrame.
        raw_col: Column name containing raw business names.
        prefix: Prefix for generated column names.

    Returns:
        DataFrame with additional normalized columns.
    """
    results = df[raw_col].apply(normalize_text_full)
    expanded = pd.DataFrame(results.tolist(), index=df.index)
    expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
    return pd.concat([df, expanded], axis=1)


def apply_address_normalization(df: pd.DataFrame, raw_col: str = "business_address", prefix: str = "addr") -> pd.DataFrame:
    """Apply multi-view address normalization to a DataFrame column.

    Adds new columns with the given prefix plus numeric_tokens. Does NOT remove the raw column.

    Args:
        df: Input DataFrame.
        raw_col: Column name containing raw business addresses.
        prefix: Prefix for generated column names.

    Returns:
        DataFrame with additional normalized columns.
    """
    results = df[raw_col].apply(normalize_address_full)
    expanded = pd.DataFrame(results.tolist(), index=df.index)
    expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
    return pd.concat([df, expanded], axis=1)
