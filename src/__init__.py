"""
src/__init__.py
===============
LOB Prediction Project — source package.

This package contains all reusable Python modules:
    - data_loader.py : Functions to load & parse the FI-2010 dataset
    - features.py    : Feature engineering pipeline
    - models.py      : Model definitions (LSTM, GRU, TFT, baselines)
    - backtest.py    : Backtesting engine with slippage & commission
    - utils.py       : Shared utilities (seeds, metrics, plotting helpers)
"""

__version__ = "1.0.0"
__author__ = "Victor"
