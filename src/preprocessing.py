"""Text preprocessing for TTS Baseline.

Converts raw user-supplied text into a clean, model-compatible string
without aggressively rewriting the content.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Maximum characters the SpeechT5 tokenizer reliably handles in one shot.
MAX_CHARS = 600


class TextPreprocessingError(ValueError):
    """Raised when the input text cannot be preprocessed into a valid string."""


def validate_input(text: str) -> None:
    """Raise :class:`TextPreprocessingError` for inputs that cannot be synthesised.

    Parameters
    ----------
    text:
        Raw input string.

    Raises
    ------
    TextPreprocessingError
        If *text* is empty, whitespace-only, or exceeds the character limit.
    """
    if not isinstance(text, str):
        raise TextPreprocessingError(
            f"Input must be a str, got {type(text).__name__!r}."
        )
    if not text.strip():
        raise TextPreprocessingError(
            "Input text is empty or whitespace-only. "
            "Please provide at least one word."
        )
    if len(text) > MAX_CHARS:
        raise TextPreprocessingError(
            f"Input text is too long ({len(text)} chars). "
            f"Maximum is {MAX_CHARS} characters."
        )


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form.

    Converts composed/decomposed characters to a canonical composed form so
    that downstream tokenisers see consistent byte sequences.

    Parameters
    ----------
    text:
        String that may contain decomposed Unicode characters.

    Returns
    -------
    str
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including tabs and newlines) to a single space.

    Leading and trailing whitespace is stripped.

    Parameters
    ----------
    text:
        Potentially multi-line or irregularly spaced string.

    Returns
    -------
    str
        Single-line string with normalized spacing.
    """
    return re.sub(r"\s+", " ", text).strip()


def preserve_punctuation(text: str) -> str:
    """Remove non-printable / control characters while keeping punctuation.

    Only ASCII control characters (except standard whitespace) and Unicode
    "Other" category characters are stripped.  Punctuation, digits, and
    letters are preserved.

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    str
        String with control characters removed.
    """
    cleaned_chars: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep letters (L*), numbers (N*), punctuation (P*), symbols (S*),
        # and the space separator (Zs).
        if cat.startswith(("L", "N", "P", "S", "Z")):
            cleaned_chars.append(ch)
        elif ch in (" ", "\t", "\n", "\r"):
            # Keep whitespace; it will be collapsed later.
            cleaned_chars.append(ch)
        # Drop control characters (Cc, Cf, …) and unassigned code points.
    return "".join(cleaned_chars)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline.

    Runs in the following order:

    1. Input validation
    2. Unicode normalization (NFC)
    3. Control-character removal (punctuation preserved)
    4. Whitespace normalization

    Parameters
    ----------
    text:
        Raw user-supplied string.

    Returns
    -------
    str
        Clean, model-compatible text suitable for the SpeechT5 tokenizer.

    Raises
    ------
    TextPreprocessingError
        If the input is empty, whitespace-only, or exceeds the character
        limit.

    Examples
    --------
    >>> preprocess_text("   Hello    world!   ")
    'Hello world!'
    """
    validate_input(text)

    text = normalize_unicode(text)
    text = preserve_punctuation(text)
    text = normalize_whitespace(text)

    # Re-validate after normalization (edge-case: input was all control chars).
    if not text:
        raise TextPreprocessingError(
            "Text became empty after preprocessing. "
            "Please provide printable characters."
        )

    logger.debug("Preprocessed text: %r", text)
    return text
