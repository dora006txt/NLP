#!/usr/bin/env python3
"""Shared Colab utilities for path setup and Drive backup."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def find_project_root() -> str | None:
    """Find a project root containing `scripts/colab_setup.py`."""
    cwd = os.getcwd()
    direct_candidates = [
        cwd,
        "/content",
        "/content/DA NPL",
        "/content/DA_NPL",
        "/content/project",
        "/content/nlp-model-comparison",
    ]
    for path in direct_candidates:
        if os.path.exists(os.path.join(path, "scripts", "colab_setup.py")):
            return path

    if os.path.exists("/content"):
        for root, dirs, _ in os.walk("/content"):
            if "scripts" in dirs and os.path.exists(os.path.join(root, "scripts", "colab_setup.py")):
                return root
    return None


def setup_path() -> str:
    project_root = find_project_root()
    if project_root is None:
        raise FileNotFoundError("Could not find project root containing scripts/colab_setup.py")
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def mount_drive(drive_dir: str = "DA_NPL_Artifacts") -> str | None:
    existing_drive_root = "/content/drive/MyDrive"
    if os.path.exists(existing_drive_root):
        backup_root = os.path.join(existing_drive_root, drive_dir)
        os.makedirs(backup_root, exist_ok=True)
        return backup_root
    try:
        from google.colab import drive  # type: ignore
        from IPython import get_ipython
    except Exception:
        return None
    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        return None
    drive.mount("/content/drive", force_remount=False)
    backup_root = os.path.join(existing_drive_root, drive_dir)
    os.makedirs(backup_root, exist_ok=True)
    return backup_root


def sync_paths_to_drive(paths: Iterable[str], backup_root: str, project_root: str | None = None) -> None:
    if not backup_root:
        return
    base = project_root or setup_path()
    for src in paths:
        abs_src = os.path.abspath(src if os.path.isabs(src) else os.path.join(base, src))
        if not os.path.exists(abs_src):
            continue
        rel_path = os.path.relpath(abs_src, base)
        dst = os.path.join(backup_root, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(abs_src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(abs_src, dst)
        else:
            shutil.copy2(abs_src, dst)


def write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    root = setup_path()
    print(f"Project root: {root}")
