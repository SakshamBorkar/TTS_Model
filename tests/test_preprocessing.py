"""Unit tests for src.preprocessing."""

import pytest

from src.preprocessing import (
    MAX_CHARS,
    TextPreprocessingError,
    normalize_unicode,
    normalize_whitespace,
    preprocess_text,
    preserve_punctuation,
    validate_input,
)


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_raises_on_empty_string(self):
        with pytest.raises(TextPreprocessingError, match="empty"):
            validate_input("")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(TextPreprocessingError, match="empty"):
            validate_input("   \t\n  ")

    def test_raises_on_non_string(self):
        with pytest.raises(TextPreprocessingError, match="str"):
            validate_input(123)  # type: ignore[arg-type]

    def test_raises_on_too_long(self):
        with pytest.raises(TextPreprocessingError, match="too long"):
            validate_input("a" * (MAX_CHARS + 1))

    def test_passes_on_valid_text(self):
        validate_input("Hello world!")  # should not raise


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------


class TestNormalizeUnicode:
    def test_nfc_normalization(self):
        # é as decomposed (e + combining accent) → composed form
        decomposed = "e\u0301"
        result = normalize_unicode(decomposed)
        assert result == "\xe9"

    def test_already_nfc_unchanged(self):
        text = "Hello world"
        assert normalize_unicode(text) == text


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("Hello    world") == "Hello world"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  Hello  ") == "Hello"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_whitespace("Hello\t\nworld") == "Hello world"

    def test_single_space_unchanged(self):
        assert normalize_whitespace("Hello world") == "Hello world"


# ---------------------------------------------------------------------------
# preserve_punctuation
# ---------------------------------------------------------------------------


class TestPreservePunctuation:
    def test_keeps_common_punctuation(self):
        text = "Hello, world! How are you?"
        result = preserve_punctuation(text)
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_removes_control_characters(self):
        text = "Hello\x00\x01world"
        result = preserve_punctuation(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result
        assert "world" in result


# ---------------------------------------------------------------------------
# preprocess_text (full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocessText:
    def test_basic_cleanup(self):
        assert preprocess_text("   Hello    world!   ") == "Hello world!"

    def test_raises_empty(self):
        with pytest.raises(TextPreprocessingError):
            preprocess_text("")

    def test_raises_whitespace_only(self):
        with pytest.raises(TextPreprocessingError):
            preprocess_text("   ")

    def test_preserves_punctuation(self):
        result = preprocess_text("Hello, how are you?")
        assert "," in result
        assert "?" in result

    def test_unicode_normalization(self):
        # Decomposed form should be normalized.
        result = preprocess_text("cafe\u0301")
        assert result == "caf\xe9"

    def test_numbers_preserved(self):
        result = preprocess_text("Your order number is 492817.")
        assert "492817" in result

    def test_currency_preserved(self):
        result = preprocess_text("Balance: $2,500.00")
        assert "$" in result
        assert "2,500.00" in result

    def test_long_sentence(self):
        sentence = (
            "We apologize for the delay and appreciate your patience "
            "while we process your request."
        )
        result = preprocess_text(sentence)
        assert result == sentence  # should be unchanged (already clean)

    def test_raises_when_too_long(self):
        with pytest.raises(TextPreprocessingError):
            preprocess_text("a" * (MAX_CHARS + 1))
