#!/usr/bin/env python3
"""
Data download pipeline for Wikitext-103 dataset.
Downloads and extracts the dataset to data/wikitext-103/
"""

import os
import sys
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Optional


WIKITEXT_URLS = {
    "train": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
}

DATA_DIR = Path(__file__).parent.parent / "data"
WIKITEXT_DIR = DATA_DIR / "wikitext-103"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download file with progress bar."""
    print(f"Downloading {url}...")
    print(f"Destination: {dest}")
    
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                    sys.stdout.flush()
    
    print("\nDownload complete!")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file."""
    import zipfile
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def verify_wikitext_structure() -> bool:
    """Verify that Wikitext-103 has correct structure."""
    required_files = [
        WIKITEXT_DIR / "wiki.train.tokens",
        WIKITEXT_DIR / "wiki.valid.tokens",
        WIKITEXT_DIR / "wiki.test.tokens",
    ]
    
    for f in required_files:
        if not f.exists():
            print(f"Missing required file: {f}")
            return False
    
    return True


def main() -> None:
    """Main download pipeline."""
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if already exists
    if verify_wikitext_structure():
        print("Wikitext-103 dataset already exists and is complete!")
        print(f"Location: {WIKITEXT_DIR}")
        return
    
    # Download
    zip_path = DATA_DIR / "wikitext-103-v1.zip"
    
    if not zip_path.exists():
        url = WIKITEXT_URLS["train"]
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"\nError downloading: {e}")
            if zip_path.exists():
                zip_path.unlink()
            sys.exit(1)
    else:
        print(f"Zip file already exists: {zip_path}")
    
    # Extract
    try:
        extract_zip(zip_path, DATA_DIR)
    except Exception as e:
        print(f"Error extracting: {e}")
        sys.exit(1)
    
    # Rename directory if needed
    extracted_dir = DATA_DIR / "wikitext-103"
    if not extracted_dir.exists():
        # Check for alternative naming
        alt_dirs = list(DATA_DIR.glob("wikitext*"))
        if alt_dirs:
            alt_dirs[0].rename(extracted_dir)
    
    # Verify
    if verify_wikitext_structure():
        print("\n✓ Wikitext-103 dataset successfully prepared!")
        print(f"  Train: {WIKITEXT_DIR / 'wiki.train.tokens'}")
        print(f"  Valid: {WIKITEXT_DIR / 'wiki.valid.tokens'}")
        print(f"  Test:  {WIKITEXT_DIR / 'wiki.test.tokens'}")
        
        # Cleanup zip file
        zip_path.unlink()
        print("\n✓ Cleanup complete (removed zip file)")
    else:
        print("\n✗ Dataset structure verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
