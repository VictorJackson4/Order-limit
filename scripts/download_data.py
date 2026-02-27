"""
scripts/download_data.py
========================
Downloads the FI-2010 Limit Order Book dataset from Kaggle.

Two methods supported:
  1. Kaggle API (automatic) — requires ~/.kaggle/kaggle.json
  2. Manual download fallback — prints clear instructions

Usage:
    python scripts/download_data.py

What this script does:
    - Checks if data already exists (skips re-download)
    - Attempts Kaggle API download first
    - Falls back to manual instructions if API is unavailable
    - Extracts the dataset into data/raw/

The FI-2010 dataset contains 10-level LOB snapshots from 5 Finnish stocks
traded on the Helsinki Exchange (NASDAQ Nordic) over 10 business days in 2010.
Each snapshot has 144 features (ask/bid prices + volumes × 10 levels × …).
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


# ── Configuration ───────────────────────────────────────────
DATASET_SLUG = "freemanone/fi2010"          # Kaggle dataset identifier
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # one level up from scripts/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_PLOTS = PROJECT_ROOT / "results" / "plots"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"
REPORT_FIGURES = PROJECT_ROOT / "report" / "figures"


def create_directories():
    """Create every directory required by the project."""
    dirs = [
        RAW_DIR,
        PROCESSED_DIR,
        RESULTS_PLOTS,
        RESULTS_TABLES,
        RESULTS_MODELS,
        REPORT_FIGURES,
        PROJECT_ROOT / "notebooks",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(PROJECT_ROOT)}/")

    # Add .gitkeep to empty tracked directories so Git preserves them
    for d in [RESULTS_PLOTS, RESULTS_TABLES, REPORT_FIGURES]:
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


def check_existing_data():
    """Return True if data/raw/ already contains files."""
    if RAW_DIR.exists():
        files = list(RAW_DIR.glob("*"))
        # filter out .gitkeep
        data_files = [f for f in files if f.name != ".gitkeep"]
        if data_files:
            return True
    return False


def download_via_kaggle_api():
    """
    Try to download using the official Kaggle CLI.
    Returns True on success, False on failure.
    """
    try:
        import kaggle  # noqa: F401 — just check if the package is installed
    except ImportError:
        print("  ⚠  'kaggle' package not installed. Install with: pip install kaggle")
        return False

    # Check for API credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_json_win = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"

    if not (kaggle_json.exists() or kaggle_json_win.exists()):
        env_user = os.environ.get("KAGGLE_USERNAME")
        env_key = os.environ.get("KAGGLE_KEY")
        if not (env_user and env_key):
            print("  ⚠  Kaggle API credentials not found.")
            print("     Expected at: ~/.kaggle/kaggle.json")
            print("     Or set KAGGLE_USERNAME + KAGGLE_KEY environment variables.")
            return False

    print("  ↓ Downloading via Kaggle API (this may take a few minutes)...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(RAW_DIR),
            unzip=True,
            quiet=False,
        )
        print("  ✓ Download complete!")
        return True
    except Exception as e:
        print(f"  ✗ Kaggle API download failed: {e}")
        return False


def manual_download_instructions():
    """Print step-by-step manual download instructions."""
    print()
    print("=" * 65)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print()
    print("  The Kaggle API method didn't work. No worries — follow these steps:")
    print()
    print("  1. Go to: https://www.kaggle.com/datasets/freemanone/fi2010")
    print("  2. Click the 'Download' button (you need a free Kaggle account)")
    print("  3. A ZIP file will download (≈ 600 MB)")
    print(f"  4. Extract all contents into: {RAW_DIR}")
    print("  5. Re-run this script to verify: python scripts/download_data.py")
    print()
    print("  ── Setting up Kaggle API for next time ──")
    print()
    print("  a. Go to https://www.kaggle.com/settings → 'Create New Token'")
    print("  b. This downloads 'kaggle.json'")
    print("  c. Move it to:")
    print("       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json")
    print("       Mac/Linux: ~/.kaggle/kaggle.json")
    print("  d. On Mac/Linux run: chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("=" * 65)


def verify_data():
    """Quick sanity check on downloaded files."""
    if not RAW_DIR.exists():
        return False

    files = list(RAW_DIR.rglob("*"))
    data_files = [f for f in files if f.is_file() and f.name != ".gitkeep"]

    if not data_files:
        return False

    total_size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
    print(f"\n  📊 Found {len(data_files)} file(s) in data/raw/ ({total_size_mb:.1f} MB)")
    for f in sorted(data_files)[:15]:  # show first 15 files
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"     • {f.relative_to(RAW_DIR)}  ({size_mb:.1f} MB)")
    if len(data_files) > 15:
        print(f"     … and {len(data_files) - 15} more files")

    return True


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FI-2010 LOB Dataset Downloader                           ║")
    print("║   Limit Order Book Dynamics Prediction Project              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Create all project directories
    print("📁 Creating project directories...")
    create_directories()
    print()

    # Step 2: Check if data already exists
    if check_existing_data():
        print("✅ Data already exists in data/raw/ — skipping download.")
        verify_data()
        print("\n🎉 You're all set! Proceed to the notebooks.")
        return

    # Step 3: Try Kaggle API
    print("🔑 Attempting Kaggle API download...")
    success = download_via_kaggle_api()

    if not success:
        manual_download_instructions()
        sys.exit(1)

    # Step 4: Verify
    if verify_data():
        print("\n🎉 Dataset downloaded and verified! You're ready to go.")
        print("   Next step: Open notebooks/01_eda.ipynb")
    else:
        print("\n⚠  Download completed but verification found no files.")
        print("   Please check data/raw/ manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
