"""
src/data_loader.py
==================
Functions to load and parse the FI-2010 Limit Order Book dataset.

The FI-2010 dataset is stored as .txt files with a specific structure:
    - 10 levels of ask/bid prices and volumes (40 columns per snapshot)
    - Features already extracted by the dataset authors (144 columns)
    - Labels for 5 prediction horizons

This module provides clean loading functions that return pandas DataFrames
ready for feature engineering and model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


# ── Column Names for the 10-Level LOB ──────────────────────
def get_lob_column_names(n_levels: int = 10) -> list:
    """
    Generate column names for an n-level LOB snapshot.

    Each level has 4 values: ask_price, ask_volume, bid_price, bid_volume.
    So 10 levels = 40 columns.

    Parameters
    ----------
    n_levels : int
        Number of price levels (default 10 for FI-2010).

    Returns
    -------
    list of str
        Column names like ['ask_price_1', 'ask_vol_1', 'bid_price_1', 'bid_vol_1', ...]
    """
    cols = []
    for i in range(1, n_levels + 1):
        cols.extend([
            f"ask_price_{i}",
            f"ask_vol_{i}",
            f"bid_price_{i}",
            f"bid_vol_{i}",
        ])
    return cols


def load_fi2010_raw(data_dir: str = "data/raw",
                    normalization: str = "zscore") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the FI-2010 dataset from .txt files.

    The dataset provides three normalization schemes:
        - 'zscore'    : z-score normalization (recommended)
        - 'minmax'    : min-max normalization
        - 'decpre'    : decimal precision normalization

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.
    normalization : str
        Which normalization to use: 'zscore', 'minmax', or 'decpre'.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 144)
        Feature matrix.
    y : np.ndarray, shape (5, n_samples)
        Labels for 5 horizons (k=1,2,3,5,10 corresponding to
        10, 20, 30, 50, 100 event intervals).

    Raises
    ------
    FileNotFoundError
        If data files are not found in data_dir.
    """
    data_path = Path(data_dir)

    # The FI-2010 dataset organizes files by normalization type
    # Common file patterns to search for
    search_patterns = [
        f"*{normalization}*",
        "*.txt",
        "*.csv",
        "*.npy",
    ]

    # Try to find the data files
    found_files = []
    for pattern in search_patterns:
        found_files.extend(list(data_path.rglob(pattern)))

    if not found_files:
        raise FileNotFoundError(
            f"No data files found in {data_path}. "
            f"Run 'python scripts/download_data.py' first."
        )

    print(f"Found {len(found_files)} files in {data_path}")
    for f in sorted(found_files)[:10]:
        print(f"  • {f.name}")

    # Load based on file format
    # The exact loading logic depends on how Kaggle packages the FI-2010 data
    # We'll handle the most common formats
    X = None
    y = None

    for f in sorted(found_files):
        if f.suffix == ".npy":
            arr = np.load(f)
            if "label" in f.name.lower() or "y" in f.name.lower():
                y = arr
            else:
                X = arr
        elif f.suffix in (".txt", ".csv"):
            arr = np.loadtxt(f) if f.suffix == ".txt" else np.genfromtxt(f, delimiter=",")
            if "label" in f.name.lower() or "y" in f.name.lower():
                y = arr
            elif X is None:
                X = arr

    if X is None:
        # If no structured files found, try loading everything as one matrix
        # and split features/labels
        all_files = sorted([f for f in found_files if f.suffix in (".txt", ".csv")])
        if all_files:
            data = np.loadtxt(all_files[0])
            # Convention: last 5 rows are labels for 5 horizons
            if data.shape[0] > data.shape[1]:
                # Data is (features, samples) — needs transpose
                X = data[:-5, :].T
                y = data[-5:, :]
            else:
                X = data[:, :-5]
                y = data[:, -5:].T

    print(f"\n✓ Loaded dataset:")
    print(f"  X shape: {X.shape if X is not None else 'None'}")
    print(f"  y shape: {y.shape if y is not None else 'None'}")

    return X, y


def prepare_labels(y_raw: np.ndarray,
                   horizon: int = 0) -> np.ndarray:
    """
    Extract labels for a specific prediction horizon.

    Parameters
    ----------
    y_raw : np.ndarray, shape (5, n_samples)
        Raw label matrix from FI-2010 (5 horizons).
    horizon : int
        Index of the horizon to use:
            0 → k=1  (10 events ≈ ~10 ms)
            1 → k=2  (20 events ≈ ~20 ms)
            2 → k=3  (30 events ≈ ~30 ms)
            3 → k=5  (50 events ≈ ~50 ms)
            4 → k=10 (100 events ≈ ~100 ms)

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Labels: 0 = down, 1 = stationary, 2 = up
        (shifted from original 1,2,3 to 0,1,2 for sklearn/PyTorch)
    """
    labels = y_raw[horizon, :].astype(int)

    # Original labels are 1, 2, 3 → shift to 0, 1, 2
    if labels.min() >= 1:
        labels = labels - 1

    print(f"Label distribution for horizon {horizon}:")
    for cls, name in [(0, "Down"), (1, "Stationary"), (2, "Up")]:
        count = (labels == cls).sum()
        pct = count / len(labels) * 100
        print(f"  {name} (class {cls}): {count:,} ({pct:.1f}%)")

    return labels


def create_sequences(X: np.ndarray,
                     y: np.ndarray,
                     seq_len: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for time-series models (LSTM/GRU/TFT).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Label vector.
    seq_len : int
        Length of each sequence (lookback window).

    Returns
    -------
    X_seq : np.ndarray, shape (n_sequences, seq_len, n_features)
        Sequence data ready for PyTorch.
    y_seq : np.ndarray, shape (n_sequences,)
        Corresponding labels (label of the last timestep in each window).
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - seq_len + 1

    X_seq = np.zeros((n_sequences, seq_len, n_features), dtype=np.float32)
    y_seq = np.zeros(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len - 1]  # label of the last timestep

    print(f"✓ Created {n_sequences:,} sequences of length {seq_len}")
    print(f"  X_seq shape: {X_seq.shape}")
    print(f"  y_seq shape: {y_seq.shape}")

    return X_seq, y_seq


def train_val_test_split(X: np.ndarray,
                         y: np.ndarray,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> dict:
    """
    Chronological train/val/test split (NO shuffling — time-series!).

    Parameters
    ----------
    X : np.ndarray
        Feature data.
    y : np.ndarray
        Labels.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
    }

    print(f"✓ Chronological split:")
    print(f"  Train: {splits['X_train'].shape[0]:,} samples")
    print(f"  Val:   {splits['X_val'].shape[0]:,} samples")
    print(f"  Test:  {splits['X_test'].shape[0]:,} samples")

    return splits
