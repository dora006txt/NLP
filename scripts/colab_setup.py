#!/usr/bin/env python3
"""
Setup Google Colab for the project.

This script is optimized for Colab GPU T4 usage:
- keeps training artifacts on local Colab storage for speed
- mounts Google Drive for backup
- installs runtime dependencies only once
- installs KenLM binaries and Python bindings
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from colab_helper import write_json


def run(cmd: list[str], *, check: bool = True) -> None:
    subprocess.run(cmd, check=check)


def mount_drive(drive_dir: str) -> str | None:
    existing_drive_root = "/content/drive/MyDrive"
    if os.path.exists(existing_drive_root):
        backup_root = os.path.join(existing_drive_root, drive_dir)
        os.makedirs(backup_root, exist_ok=True)
        print(f"Google Drive already mounted. Backup root: {backup_root}")
        return backup_root

    try:
        from google.colab import drive  # type: ignore
        from IPython import get_ipython
    except Exception:
        print("Google Drive mount skipped: not running inside Google Colab.")
        return None

    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        print(
            "Google Drive mount skipped: script is running in a subprocess without Colab kernel access. "
            "Mount Drive in a notebook cell first, then rerun this command."
        )
        return None

    drive.mount("/content/drive", force_remount=False)
    backup_root = os.path.join(existing_drive_root, drive_dir)
    os.makedirs(backup_root, exist_ok=True)
    print(f"Google Drive mounted. Backup root: {backup_root}")
    return backup_root


def detect_device() -> dict[str, object]:
    import torch

    device_info: dict[str, object] = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_info.update(
            {
                "device": "cuda",
                "cuda_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(props.total_memory / 1e9, 2),
            }
        )
    return device_info


def install_runtime_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", os.path.join(PROJECT_ROOT, "requirements.txt")])


def install_kenlm(kenlm_root: str) -> str:
    kenlm_python_src = "https://github.com/kpu/kenlm/archive/master.zip"
    apt_packages = [
        "build-essential",
        "cmake",
        "libboost-program-options-dev",
        "libboost-system-dev",
        "libboost-thread-dev",
        "libboost-test-dev",
        "zlib1g-dev",
    ]
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", *apt_packages])

    existing_binary = os.path.join(kenlm_root, "build", "bin", "lmplz")
    if os.path.exists(existing_binary):
        run([sys.executable, "-m", "pip", "install", kenlm_python_src])
        os.environ["KENLM_ROOT"] = kenlm_root
        return kenlm_root

    if os.path.exists(kenlm_root):
        shutil.rmtree(kenlm_root)
    run(["git", "clone", "https://github.com/kpu/kenlm.git", kenlm_root])

    build_dir = os.path.join(kenlm_root, "build")
    os.makedirs(build_dir, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(build_dir)
        run(["cmake", ".."])
        run(["make", "-j4"])
    finally:
        os.chdir(cwd)

    run([sys.executable, "-m", "pip", "install", kenlm_python_src])
    os.environ["KENLM_ROOT"] = kenlm_root
    return kenlm_root


def write_environment_manifest(backup_root: str | None, kenlm_root: str) -> str:
    import torch
    import transformers

    artifact_dir = os.path.join(PROJECT_ROOT, "artifacts", "colab")
    os.makedirs(artifact_dir, exist_ok=True)
    manifest_path = os.path.join(artifact_dir, "environment.json")
    payload = {
        "project_root": PROJECT_ROOT,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": detect_device(),
        "drive_backup_root": backup_root,
        "kenlm_root": kenlm_root,
    }
    write_json(manifest_path, payload)
    return manifest_path


def setup_colab_environment(drive_dir: str = "DA_NPL_Artifacts") -> dict[str, object]:
    print("=" * 60)
    print("COLAB ENVIRONMENT SETUP")
    print("=" * 60)

    os.chdir(PROJECT_ROOT)
    backup_root = mount_drive(drive_dir)
    install_runtime_dependencies()
    kenlm_root = install_kenlm("/content/kenlm")
    env_manifest = write_environment_manifest(backup_root, kenlm_root)

    result = {
        "project_root": PROJECT_ROOT,
        "backup_root": backup_root,
        "kenlm_root": kenlm_root,
        "environment_manifest": env_manifest,
        "device": detect_device(),
    }

    print("\nSetup complete.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nNext steps:")
    print("1. python scripts/colab_prepare_data.py")
    print("2. python scripts/train_ngram.py")
    print("3. python scripts/colab_train_gpt2.py")
    print("4. python scripts/colab_train.py")
    print("5. python scripts/colab_evaluate.py")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", default="DA_NPL_Artifacts")
    args = parser.parse_args()
    setup_colab_environment(drive_dir=args.drive_dir)
