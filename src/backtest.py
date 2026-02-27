"""
src/backtest.py
===============
Backtesting engine for LOB prediction strategies.

Simulates realistic trading with slippage, commission, and market impact.
Computes Sharpe, Sortino, MaxDD, win rate, and per-regime performance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Sharpe Ratio = (E[R] - Rf) / std(R) * sqrt(N)
    """
    excess = returns - risk_free_rate
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)


def compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Sortino = (E[R] - Rf) / std(downside) * sqrt(N)
    Only penalizes downside volatility (negative returns).
    """
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve):
    """
    MaxDD = max peak-to-trough decline.
    Returns (max_dd, peak_idx, trough_idx).
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_dd = np.max(drawdowns)
    trough_idx = int(np.argmax(drawdowns))
    peak_idx = int(np.argmax(equity_curve[:trough_idx + 1]))
    return max_dd, peak_idx, trough_idx


def compute_win_rate(returns):
    """Win rate = fraction of positive-return periods."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def apply_transaction_costs(returns, positions, commission_bps=1.0, slippage_bps=0.5):
    """
    Apply commission + slippage whenever position changes.
    cost_per_trade = (commission + slippage) / 10000
    """
    cost = (commission_bps + slippage_bps) / 10000.0
    trades = np.abs(np.diff(positions, prepend=positions[0]))
    return returns - trades * cost


def backtest_strategy(predictions, mid_prices, commission_bps=1.0,
                      slippage_bps=0.5, initial_capital=100000.0):
    """
    Full backtest: predictions -> positions -> returns -> metrics.

    Trading rules:
        prediction=2 (Up)   -> Long  (+1)
        prediction=0 (Down) -> Short (-1)
        prediction=1 (Flat) -> Flat  (0)
    """
    pos_map = {0: -1, 1: 0, 2: 1}
    positions = np.array([pos_map.get(int(p), 0) for p in predictions])

    price_ret = np.diff(mid_prices) / mid_prices[:-1]
    pos_aligned = positions[:-1]
    gross_ret = pos_aligned * price_ret

    net_ret = apply_transaction_costs(
        gross_ret, pos_aligned, commission_bps, slippage_bps
    )

    equity = initial_capital * np.cumprod(1 + net_ret)
    equity = np.concatenate([[initial_capital], equity])

    sharpe = compute_sharpe_ratio(net_ret)
    sortino = compute_sortino_ratio(net_ret)
    max_dd, peak_idx, trough_idx = compute_max_drawdown(equity)
    wr = compute_win_rate(net_ret)
    total_ret = (equity[-1] - initial_capital) / initial_capital
    n_trades = int(np.sum(np.abs(np.diff(positions)) > 0))

    results = {
        "total_return": total_ret, "sharpe_ratio": sharpe,
        "sortino_ratio": sortino, "max_drawdown": max_dd,
        "win_rate": wr, "n_trades": n_trades,
        "equity_curve": equity, "net_returns": net_ret,
        "positions": positions,
    }

    print("=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return:   {total_ret:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.3f}")
    print(f"  Sortino Ratio:  {sortino:.3f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"  Win Rate:       {wr:.2%}")
    print(f"  # Trades:       {n_trades:,}")
    print("=" * 50)
    return results


def performance_by_regime(net_returns, regimes):
    """Break down performance by volatility regime (0=low, 1=med, 2=high)."""
    names = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
    rows = []
    for r in sorted(np.unique(regimes)):
        mask = regimes[:len(net_returns)] == r
        if mask.sum() == 0:
            continue
        rr = net_returns[mask]
        rows.append({
            "Regime": names.get(r, f"Regime {r}"),
            "N Periods": int(mask.sum()),
            "Mean Return": float(np.mean(rr)),
            "Sharpe": compute_sharpe_ratio(rr),
            "Sortino": compute_sortino_ratio(rr),
            "Win Rate": compute_win_rate(rr),
        })
    return pd.DataFrame(rows)
