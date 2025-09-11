"""Pytest configuration to ensure local package modules are importable.

Adds the repository root to sys.path so tests can import top-level modules
like `NN_batch_correct` and `vae_attention_model` without packaging.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path():
    here = Path(__file__).resolve()
    # Typically tests/ is one level below repo root
    candidates = [here.parent, here.parent.parent, here.parent.parent.parent]
    for p in candidates:
        if not p or not p.exists():
            continue
        if (p / "NN_batch_correct.py").exists() or (p / "vae_attention_model.py").exists():
            sys.path.insert(0, str(p))
            break


_ensure_repo_root_on_path()

