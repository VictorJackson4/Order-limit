"""
src/features.py
===============
Feature engineering pipeline for LOB data.

Implements all features described in the README:
    - Mid-price calculation
    - Bid-ask spread
    - Order Flow Imbalance (OFI)
    - Volume Delta
    - Aggressive trading pressure
    - Reconstructed OHLCV bars
    - Volatility regime indicators
    - Multi-level depth features

Every feature has its mathematical formula documented in comments
and in the project README.md.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_mid_price(ask_price_1: np.ndarray,
                      bid_price_1: np.ndarray) -> np.ndarray:
    """
    Mid-price = average of best ask and best bid.

    Formula:
        P_mid = (P_ask_1 + P_bid_1) / 2

    This is the "fair" price estimate at any moment. It's the most
    fundamental LOB feature — almost every other feature is derived from it.

    Parameters
    ----------
    ask_price_1 : np.ndarray
        Best (lowest) ask prices.
    bid_price_1 : np.ndarray
        Best (highest) bid prices.

    Returns
    -------
    np.ndarray
        Mid-price series.
    """
    return (ask_price_1 + bid_price_1) / 2.0


def compute_spread(ask_price_1: np.ndarray,
                   bid_price_1: np.ndarray) -> np.ndarray:
    """
    Bid-ask spread = difference between best ask and best bid.

    Formula:
        S = P_ask_1 - P_bid_1

    A wider spread indicates lower liquidity / higher uncertainty.
    Spread in basis points: S_bps = (S / P_mid) × 10000

    Parameters
    ----------
    ask_price_1, bid_price_1 : np.ndarray
        Best ask and bid prices.

    Returns
    -------
    np.ndarray
        Spread series.
    """
    return ask_price_1 - bid_price_1


def compute_order_flow_imbalance(ask_vol_1: np.ndarray,
                                  bid_vol_1: np.ndarray) -> np.ndarray:
    """
    Order Flow Imbalance (OFI) at the top of the book.

    Formula:
        OFI = (V_bid_1 - V_ask_1) / (V_bid_1 + V_ask_1)

    Range: [-1, +1]
        +1 = all volume on bid side → buying pressure → price likely goes UP
        -1 = all volume on ask side → selling pressure → price likely goes DOWN
         0 = balanced book

    This is one of the most powerful short-term predictive features in
    market microstructure research.

    Parameters
    ----------
    ask_vol_1, bid_vol_1 : np.ndarray
        Volumes at the best ask and bid.

    Returns
    -------
    np.ndarray
        OFI values in [-1, 1].
    """
    total = bid_vol_1 + ask_vol_1
    # Avoid division by zero
    ofi = np.where(total > 0,
                   (bid_vol_1 - ask_vol_1) / total,
                   0.0)
    return ofi


def compute_volume_delta(ask_vol: np.ndarray,
                         bid_vol: np.ndarray,
                         n_levels: int = 10) -> np.ndarray:
    """
    Volume Delta across multiple levels.

    Formula:
        ΔV = Σ_{i=1}^{n} V_bid_i  −  Σ_{i=1}^{n} V_ask_i

    Positive delta = more buying interest across the full depth of the book.
    Unlike OFI which only looks at the top level, this captures the full
    picture of supply vs demand.

    Parameters
    ----------
    ask_vol : np.ndarray, shape (n_samples, n_levels)
        Ask volumes for all levels.
    bid_vol : np.ndarray, shape (n_samples, n_levels)
        Bid volumes for all levels.
    n_levels : int
        Number of levels to aggregate.

    Returns
    -------
    np.ndarray
        Volume delta series.
    """
    total_bid = bid_vol[:, :n_levels].sum(axis=1)
    total_ask = ask_vol[:, :n_levels].sum(axis=1)
    return total_bid - total_ask


def compute_aggressive_pressure(ask_vol_1: np.ndarray,
                                 bid_vol_1: np.ndarray,
                                 window: int = 50) -> np.ndarray:
    """
    Aggressive trading pressure — rolling ratio of volume changes.

    When volume at the best bid decreases, it often means aggressive sellers
    are 'eating' the bid (market sell orders hitting limit buy orders).
    Vice versa for the ask side.

    Formula:
        AP_t = rolling_mean(ΔV_bid_consumed / ΔV_ask_consumed, window)

    Parameters
    ----------
    ask_vol_1, bid_vol_1 : np.ndarray
        Best ask and bid volumes.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Aggressive pressure indicator.
    """
    # Volume changes (negative = consumed by aggressive orders)
    d_bid = np.diff(bid_vol_1, prepend=bid_vol_1[0])
    d_ask = np.diff(ask_vol_1, prepend=ask_vol_1[0])

    # Consumption: we care about negative changes (volume being taken)
    bid_consumed = np.maximum(-d_bid, 0)  # aggressive sellers
    ask_consumed = np.maximum(-d_ask, 0)  # aggressive buyers

    # Rolling sum
    bid_roll = pd.Series(bid_consumed).rolling(window, min_periods=1).sum().values
    ask_roll = pd.Series(ask_consumed).rolling(window, min_periods=1).sum().values

    # Ratio: > 1 means more aggressive selling
    total = bid_roll + ask_roll
    pressure = np.where(total > 0,
                        (ask_roll - bid_roll) / total,
                        0.0)
    return pressure


def compute_weighted_mid_price(ask_price_1: np.ndarray,
                                bid_price_1: np.ndarray,
                                ask_vol_1: np.ndarray,
                                bid_vol_1: np.ndarray) -> np.ndarray:
    """
    Volume-weighted mid-price (VWMP).

    Formula:
        P_wmid = (P_ask × V_bid + P_bid × V_ask) / (V_bid + V_ask)

    Note the "cross" weighting: bid volume weights the ask price and vice
    versa. This gives a "fairer" estimate than the simple mid-price because
    the side with more volume will pull the mid-price towards it.

    Parameters
    ----------
    ask_price_1, bid_price_1 : np.ndarray
        Best ask and bid prices.
    ask_vol_1, bid_vol_1 : np.ndarray
        Volumes at the best ask and bid.

    Returns
    -------
    np.ndarray
        Volume-weighted mid-price.
    """
    total_vol = ask_vol_1 + bid_vol_1
    wmid = np.where(
        total_vol > 0,
        (ask_price_1 * bid_vol_1 + bid_price_1 * ask_vol_1) / total_vol,
        (ask_price_1 + bid_price_1) / 2.0,
    )
    return wmid


def compute_depth_imbalance(ask_vol: np.ndarray,
                             bid_vol: np.ndarray,
                             levels: list = None) -> np.ndarray:
    """
    Depth imbalance at specified levels.

    Formula:
        DI_l = (V_bid_l - V_ask_l) / (V_bid_l + V_ask_l)

    Parameters
    ----------
    ask_vol : np.ndarray, shape (n_samples, n_levels)
        Ask volumes.
    bid_vol : np.ndarray, shape (n_samples, n_levels)
        Bid volumes.
    levels : list of int
        Which levels to include (0-indexed). Default: all.

    Returns
    -------
    np.ndarray, shape (n_samples, len(levels))
        Depth imbalance per level.
    """
    if levels is None:
        levels = list(range(ask_vol.shape[1]))

    results = []
    for lvl in levels:
        total = ask_vol[:, lvl] + bid_vol[:, lvl]
        di = np.where(total > 0,
                      (bid_vol[:, lvl] - ask_vol[:, lvl]) / total,
                      0.0)
        results.append(di)
    return np.column_stack(results)


def compute_volatility(mid_price: np.ndarray,
                       window: int = 100) -> np.ndarray:
    """
    Realized volatility — rolling standard deviation of log returns.

    Formula:
        r_t = ln(P_t / P_{t-1})
        σ = std(r_{t-window+1}, ..., r_t)

    Parameters
    ----------
    mid_price : np.ndarray
        Mid-price series.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling volatility.
    """
    # Log returns
    log_ret = np.log(mid_price[1:] / mid_price[:-1])
    log_ret = np.concatenate([[0.0], log_ret])  # pad to match length

    vol = pd.Series(log_ret).rolling(window, min_periods=1).std().values
    return vol


def compute_volatility_regime(volatility: np.ndarray,
                               quantiles: tuple = (0.33, 0.67)) -> np.ndarray:
    """
    Classify volatility into regimes: low / medium / high.

    Parameters
    ----------
    volatility : np.ndarray
        Volatility series.
    quantiles : tuple
        Quantile thresholds for regime boundaries.

    Returns
    -------
    np.ndarray
        Regime labels: 0 = low, 1 = medium, 2 = high.
    """
    q_low = np.nanquantile(volatility, quantiles[0])
    q_high = np.nanquantile(volatility, quantiles[1])

    regime = np.zeros_like(volatility, dtype=int)
    regime[volatility > q_low] = 1
    regime[volatility > q_high] = 2

    return regime


def engineer_all_features(X: np.ndarray,
                          col_names: list = None,
                          n_levels: int = 10) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on raw LOB data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 40) or (n_samples, 144)
        Raw LOB data. If 40 columns: 10-level prices & volumes.
        If 144: full FI-2010 feature set (we extract the LOB columns).
    col_names : list, optional
        Column names. If None, auto-generated.
    n_levels : int
        Number of LOB levels.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features + original features.
    """
    # Extract price and volume columns
    # FI-2010 format: for each level: ask_price, ask_vol, bid_price, bid_vol
    n_cols_per_level = 4

    if X.shape[1] >= n_levels * n_cols_per_level:
        # Extract ask/bid prices and volumes
        ask_prices = X[:, 0::4][:, :n_levels]   # columns 0, 4, 8, ...
        ask_vols = X[:, 1::4][:, :n_levels]     # columns 1, 5, 9, ...
        bid_prices = X[:, 2::4][:, :n_levels]   # columns 2, 6, 10, ...
        bid_vols = X[:, 3::4][:, :n_levels]     # columns 3, 7, 11, ...
    else:
        raise ValueError(f"Expected at least {n_levels * 4} columns, got {X.shape[1]}")

    # Build feature dictionary
    features = {}

    # 1. Mid-price
    features["mid_price"] = compute_mid_price(ask_prices[:, 0], bid_prices[:, 0])

    # 2. Spread
    features["spread"] = compute_spread(ask_prices[:, 0], bid_prices[:, 0])
    features["spread_bps"] = (features["spread"] / features["mid_price"]) * 10000

    # 3. Order Flow Imbalance (top of book)
    features["ofi"] = compute_order_flow_imbalance(ask_vols[:, 0], bid_vols[:, 0])

    # 4. Volume Delta (across all levels)
    features["volume_delta"] = compute_volume_delta(ask_vols, bid_vols, n_levels)

    # 5. Aggressive trading pressure
    features["aggressive_pressure"] = compute_aggressive_pressure(
        ask_vols[:, 0], bid_vols[:, 0], window=50
    )

    # 6. Weighted mid-price
    features["weighted_mid"] = compute_weighted_mid_price(
        ask_prices[:, 0], bid_prices[:, 0],
        ask_vols[:, 0], bid_vols[:, 0]
    )

    # 7. Price-weighted mid deviation
    features["wmid_deviation"] = features["weighted_mid"] - features["mid_price"]

    # 8. Depth imbalance at each level
    depth_imb = compute_depth_imbalance(ask_vols, bid_vols)
    for i in range(n_levels):
        features[f"depth_imbalance_{i+1}"] = depth_imb[:, i]

    # 9. Volatility and regimes
    features["volatility"] = compute_volatility(features["mid_price"], window=100)
    features["vol_regime"] = compute_volatility_regime(features["volatility"])

    # 10. Log returns
    mp = features["mid_price"]
    features["log_return"] = np.concatenate([[0.0], np.log(mp[1:] / mp[:-1])])

    # 11. Total volume (both sides)
    features["total_ask_vol"] = ask_vols.sum(axis=1)
    features["total_bid_vol"] = bid_vols.sum(axis=1)
    features["total_volume"] = features["total_ask_vol"] + features["total_bid_vol"]

    df = pd.DataFrame(features)
    print(f"✓ Engineered {len(df.columns)} features from {X.shape[1]} raw columns")
    print(f"  Features: {list(df.columns)}")

    return df
