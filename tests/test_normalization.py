"""Tests for multi-view text normalization."""

import pytest
import pandas as pd
import numpy as np

from src.business_entity_resolution.normalization.text import (
    normalize_unicode,
    normalize_casefold,
    normalize_whitespace,
    normalize_punctuation,
    fold_accents,
    extract_numeric_tokens,
    normalize_text_full,
    normalize_address_full,
    apply_name_normalization,
    apply_address_normalization,
    MISSING_SENTINEL,
    _is_missing,
)


# ===========================================================================
# Case and whitespace
# ===========================================================================

class TestCaseAndWhitespace:
    def test_leading_trailing_inner_spaces(self):
        result = normalize_text_full("  Alpha   Engineering  ")
        # casefold("  Alpha   Engineering  ") -> "  alpha   engineering  "
        # then ws-norm -> "alpha engineering"
        assert result["norm_clean"] == "alpha engineering"

    def test_tabs_and_newlines(self):
        result = normalize_text_full("Alpha\tBeta\nGamma")
        assert result["norm_clean"] == "alpha beta gamma"


# ===========================================================================
# Punctuation
# ===========================================================================

class TestPunctuation:
    def test_ampersand_replacement(self):
        result = normalize_text_full("Alpha & Beta")
        assert "and" in result["norm_punct"]
        assert "&" not in result["norm_punct"]

    def test_pvt_ltd_dots(self):
        result = normalize_text_full("Alpha, Engineering Pvt. Ltd.")
        # Dots and comma should be removed/replaced
        assert "." not in result["norm_punct"]

    def test_hyphen_preserved_in_word(self):
        result = normalize_text_full("Coca-Cola")
        assert "coca-cola" in result["norm_punct"]


# ===========================================================================
# Unicode / accent preservation
# ===========================================================================

class TestUnicodeAccent:
    def test_accent_preserved_in_unicode_view(self):
        result = normalize_text_full("Société Alpha S.A.S.")
        # norm_unicode should preserve accented characters
        assert "é" in result["norm_unicode"]

    def test_accent_preserved_in_clean_view(self):
        result = normalize_text_full("Société Alpha S.A.S.")
        # norm_clean is case-folded but accents are preserved
        assert "é" in result["norm_clean"]

    def test_accent_folded_view(self):
        result = normalize_text_full("Société Alpha S.A.S.")
        # accent_fold should remove diacritics
        assert "é" not in result["norm_accent_fold"]
        assert "societe" in result["norm_accent_fold"].lower()


# ===========================================================================
# Numbers
# ===========================================================================

class TestNumbers:
    def test_numeric_extraction(self):
        assert extract_numeric_tokens("21 Park Road") == "21"

    def test_numeric_distinction(self):
        nums_21 = extract_numeric_tokens("21 Park Road")
        nums_61 = extract_numeric_tokens("61 Park Road")
        assert nums_21 != nums_61
        assert nums_21 == "21"
        assert nums_61 == "61"

    def test_multiple_numbers(self):
        result = extract_numeric_tokens("21 Park Road, PIN 400001")
        assert result == "21 400001"

    def test_no_numbers(self):
        assert extract_numeric_tokens("Main Street") == ""

    def test_address_numeric_tokens(self):
        result = normalize_address_full("21 Park Road, PIN 400001")
        assert result["numeric_tokens"] == "21 400001"


# ===========================================================================
# Null / missing input
# ===========================================================================

class TestMissingValues:
    def test_none_input(self):
        result = normalize_text_full(None)
        assert result["norm_unicode"] == MISSING_SENTINEL
        assert result["norm_clean"] == MISSING_SENTINEL
        assert result["norm_punct"] == MISSING_SENTINEL
        assert result["norm_accent_fold"] == MISSING_SENTINEL

    def test_nan_input(self):
        result = normalize_text_full(float("nan"))
        assert result["norm_unicode"] == MISSING_SENTINEL

    def test_none_does_not_become_nan_string(self):
        result = normalize_text_full(None)
        assert result["norm_clean"] != "nan"
        assert result["norm_clean"] != "None"
        assert result["norm_clean"] != "none"

    def test_nan_does_not_become_nan_string(self):
        result = normalize_text_full(float("nan"))
        assert result["norm_clean"] != "nan"

    def test_address_none(self):
        result = normalize_address_full(None)
        assert result["numeric_tokens"] == MISSING_SENTINEL


# ===========================================================================
# Empty string
# ===========================================================================

class TestEmptyString:
    def test_empty_string_input(self):
        result = normalize_text_full("")
        # Empty string is NOT missing — it's a legitimate value
        assert result["norm_unicode"] == ""
        assert result["norm_clean"] == ""

    def test_empty_address(self):
        result = normalize_address_full("")
        assert result["numeric_tokens"] == ""


# ===========================================================================
# Non-Latin Unicode
# ===========================================================================

class TestNonLatin:
    def test_devanagari_not_destroyed(self):
        """Hindi text should survive normalization without ASCII destruction."""
        hindi = "मुंबई बिजनेस सेंटर"
        result = normalize_text_full(hindi)
        # Should NOT be empty after normalization
        assert len(result["norm_unicode"]) > 0
        assert len(result["norm_clean"]) > 0
        # Accent fold should not destroy Devanagari base characters
        assert len(result["norm_accent_fold"]) > 0

    def test_mixed_script(self):
        """Mixed Latin + non-Latin should preserve both."""
        mixed = "ABC कंपनी Ltd."
        result = normalize_text_full(mixed)
        assert "abc" in result["norm_clean"]
        assert "कंपनी" in result["norm_clean"]


# ===========================================================================
# DataFrame application
# ===========================================================================

class TestDataFrameApplication:
    def test_name_normalization_adds_columns(self):
        df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Alpha Corp", "Beta Ltd"],
            "business_address": ["123 Main St", "456 Oak Ave"],
            "country": ["US", "India"],
        })
        result = apply_name_normalization(df)
        assert "business_name" in result.columns  # raw preserved
        assert "name_norm_unicode" in result.columns
        assert "name_norm_clean" in result.columns
        assert "name_norm_punct" in result.columns
        assert "name_norm_accent_fold" in result.columns
        assert len(result) == 2

    def test_address_normalization_adds_columns(self):
        df = pd.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["Alpha"],
            "business_address": ["21 Park Road"],
            "country": ["US"],
        })
        result = apply_address_normalization(df)
        assert "business_address" in result.columns  # raw preserved
        assert "addr_norm_unicode" in result.columns
        assert "addr_numeric_tokens" in result.columns
        assert result.iloc[0]["addr_numeric_tokens"] == "21"

    def test_missing_address_in_dataframe(self):
        df = pd.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["Alpha"],
            "business_address": [None],
            "country": ["US"],
        })
        result = apply_address_normalization(df)
        assert result.iloc[0]["addr_numeric_tokens"] == MISSING_SENTINEL
