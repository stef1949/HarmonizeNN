"""Centralised project paths."""
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ASSETS_DIR = PROJECT_ROOT / "assets"


def project_relative(path: Union[Path, str]) -> Path:
    """Return an absolute path inside the project for the provided relative fragment."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "ARTIFACTS_DIR",
    "OUTPUTS_DIR",
    "CHECKPOINTS_DIR",
    "ASSETS_DIR",
    "project_relative",
]