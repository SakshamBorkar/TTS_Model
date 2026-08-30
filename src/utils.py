"""General-purpose utilities for TTS Baseline."""

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """Configure root-level logging to stdout.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for :mod:`random`, :mod:`numpy`, and :mod:`torch` (if
    available).

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_text_file(path: str | Path) -> list[str]:
    """Read a plain-text file and return non-empty, non-comment lines.

    Lines starting with ``#`` are treated as comments and skipped.
    Blank lines are also skipped.

    Parameters
    ----------
    path:
        Path to the text file.

    Returns
    -------
    list[str]
        Stripped, non-empty lines.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    lines: list[str] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) if they do not already exist.

    Parameters
    ----------
    *paths:
        One or more directory paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
