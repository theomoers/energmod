#!/usr/bin/env python3
"""Shared file-state guardrail helpers for iterative tuning scripts."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` if ``src`` exists; return whether a copy happened."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def restore_from_last_good(mutable_path: Path, last_good_path: Path) -> bool:
    """Restore mutable file from last-good snapshot if available."""
    return copy_if_exists(last_good_path, mutable_path)


def sync_last_good_from_mutable(mutable_path: Path, last_good_path: Path) -> bool:
    """Update last-good snapshot from the current mutable file if it exists."""
    return copy_if_exists(mutable_path, last_good_path)


def raise_if_simulated_failure(enabled: bool, label: str) -> None:
    """Inject a deterministic failure for rollback smoke tests."""
    if enabled:
        raise RuntimeError(f"Simulated post-write failure for {label}")
