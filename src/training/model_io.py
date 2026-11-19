"""Save and load the versioned model artifact.

Kept separate from train.py so the API can import `load_model` without
pulling in MLflow or the training code.

Layout of MODEL_DIR (default: artifacts/model):
    model.joblib              current model (sklearn Pipeline, predict_proba[:, 1] = P(phishing))
    model_metadata.json       metadata of the current model
    archive/v<N>/             previous versions, moved here on every new save
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODEL_DIR = "artifacts/model"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
ARCHIVE_DIRNAME = "archive"

REQUIRED_METADATA_KEYS = (
    "model_name",
    "version",
    "mlflow_run_id",
    "mlflow_experiment",
    "trained_at",
    "dataset",
    "train_rows",
    "test_rows",
    "features",
    "threshold",
    "metrics",
    "candidates",
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when MODEL_DIR has no model.joblib / model_metadata.json."""


def get_model_dir(model_dir: str | os.PathLike | None = None) -> Path:
    """Explicit argument > MODEL_DIR env var > artifacts/model."""
    if model_dir is not None:
        return Path(model_dir)
    return Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))


def read_metadata(model_dir: str | os.PathLike | None = None) -> dict[str, Any] | None:
    """Return the current metadata, or None if no model has been saved yet."""
    path = get_model_dir(model_dir) / METADATA_FILENAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _next_version(current: dict[str, Any] | None) -> str:
    if not current:
        return "1"
    try:
        return str(int(current.get("version", 0)) + 1)
    except (TypeError, ValueError):
        return "1"


def _archive_current(model_dir: Path, version: str) -> None:
    """Move the current model + metadata to archive/v<version>/."""
    target = model_dir / ARCHIVE_DIRNAME / f"v{version}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in (MODEL_FILENAME, METADATA_FILENAME):
        src = model_dir / name
        if src.exists():
            shutil.move(str(src), str(target / name))


def save_model(
    model: Any,
    metadata: dict[str, Any],
    model_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Save a model and its metadata as the new current version.

    The version is assigned here (previous version + 1, starting at "1");
    any `version` in `metadata` is ignored. The previous model, if any, is
    moved to archive/v<previous>/. Returns the metadata that was written.
    """
    directory = get_model_dir(model_dir)
    directory.mkdir(parents=True, exist_ok=True)

    current = read_metadata(directory)
    version = _next_version(current)
    if current is not None:
        _archive_current(directory, str(current.get("version", "unknown")))

    written = {**metadata, "version": version}
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in written]
    if missing:
        raise ValueError(f"metadata is missing required keys: {missing}")

    joblib.dump(model, directory / MODEL_FILENAME)
    with (directory / METADATA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(written, f, indent=2)
    return written


def load_model(model_dir: str | os.PathLike | None = None) -> tuple[Any, dict[str, Any]]:
    """Load (model, metadata) from MODEL_DIR.

    Raises ModelNotFoundError if either file is missing, so callers such as the
    API can turn it into a 503 instead of crashing.
    """
    directory = get_model_dir(model_dir)
    model_path = directory / MODEL_FILENAME
    meta_path = directory / METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotFoundError(
            f"No trained model in '{directory}'. Run `python -m src.training.train` first."
        )
    model = joblib.load(model_path)
    with meta_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata
