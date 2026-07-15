"""
Reusable utility functions for the House Price Prediction project.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import OUTPUT_DIRECTORIES


# ============================================================
# LOGGING
# ============================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(message)s"
)


def configure_logging(
    level: int = logging.INFO
) -> None:
    """
    Configure application-wide logging.

    Existing handlers are preserved to prevent duplicate messages.
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a module."""

    configure_logging()

    return logging.getLogger(name)


# ============================================================
# DIRECTORY UTILITIES
# ============================================================

def create_output_directories() -> None:
    """Create all generated-output directories."""

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file_exists(
    file_path: Path,
    description: str
) -> None:
    """
    Raise a descriptive error when a required file is missing.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found.\n"
            f"Expected path: {file_path}"
        )


# ============================================================
# DATAFRAME OPERATIONS
# ============================================================

def load_csv(
    file_path: Path,
    description: str
) -> pd.DataFrame:
    """Load and validate a CSV dataset."""

    validate_file_exists(
        file_path,
        description
    )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"{description} is empty: {file_path}"
        )

    return dataframe


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    index: bool = False
) -> None:
    """Save a DataFrame and create its parent directory."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        output_path,
        index=index
    )


# ============================================================
# TEXT OUTPUT
# ============================================================

def save_text(
    content: str,
    output_path: Path
) -> None:
    """Save a UTF-8 text report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        content,
        encoding="utf-8"
    )


# ============================================================
# MODEL ARTIFACTS
# ============================================================

def save_joblib_artifact(
    artifact: Any,
    output_path: Path
) -> None:
    """Save a Python or scikit-learn artifact with Joblib."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        artifact,
        output_path
    )