<div align="center">

# Limit Order Book Dynamics Prediction & Trading Signal Generation

### Using Advanced Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: FI-2010](https://img.shields.io/badge/Data-FI--2010-orange.svg)](https://www.kaggle.com/datasets/freemanone/fi2010)

_Predicting mid-price direction from 10-level LOB snapshots with classical ML, LSTM/GRU, and Temporal Fusion Transformer — featuring realistic backtesting and SHAP explainability._

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Mathematical Concepts & Formulas](#mathematical-concepts--formulas)
- [Feature Dictionary](#feature-dictionary)
- [Setup & Installation](#setup--installation)
- [Data Download](#data-download)
- [How to Run](#how-to-run)
- [Results & Key Findings](#results--key-findings)
- [Folder Structure](#folder-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Project Overview

This project investigates **short-term price direction prediction** in Limit Order Books (LOBs) using the benchmark **FI-2010** dataset from the Helsinki Exchange. We systematically compare:

| Approach             | Models                                       |
| -------------------- | -------------------------------------------- |
| **Classical ML**     | Logistic Regression, LightGBM (Optuna-tuned) |
| **Sequential DL**    | LSTM, GRU                                    |
| **State-of-the-Art** | Temporal Fusion Transformer (CPU-optimized)  |

Each model is evaluated with both **statistical metrics** (accuracy, macro-F1) and **economic metrics** (Sharpe ratio, Sortino ratio, max drawdown) under realistic trading conditions with slippage and transaction costs.

**Key contributions:**

- Comprehensive feature engineering from raw 10-level LOB data
- Ablation study isolating feature group contributions
- SHAP-based model interpretability analysis
- Realistic backtesting with configurable transaction costs (0.5–2 bps)
- Volatility-regime-conditioned performance analysis

---

## Tech Stack

| Package             | Version  | Purpose                                  |
| ------------------- | -------- | ---------------------------------------- |
| `python`            | 3.11     | Runtime                                  |
| `numpy`             | 1.26.4   | Numerical computing — "math on arrays"   |
| `pandas`            | 2.2.1    | DataFrames — "Excel on steroids"         |
| `scipy`             | 1.13.0   | Scientific computing & statistics        |
| `scikit-learn`      | 1.4.2    | Classical ML models & metrics            |
| `lightgbm`          | 4.3.0    | Gradient-boosted trees (fast & accurate) |
| `optuna`            | 3.6.1    | Bayesian hyper-parameter tuning          |
| `torch`             | 2.3.0    | Deep learning framework (CPU-only)       |
| `pytorch-lightning` | 2.2.4    | Clean PyTorch training loops             |
| `shap`              | 0.45.1   | Model explainability (Shapley values)    |
| `matplotlib`        | 3.8.5    | Static plots                             |
| `seaborn`           | 0.13.2   | Statistical visualizations               |
| `plotly`            | 5.22.0   | Interactive plots                        |
| `vectorbt`          | 0.26.2   | Vectorized backtesting engine            |
| `pandas-ta`         | 0.3.14b1 | Technical analysis indicators            |
| `tqdm`              | 4.66.4   | Progress bars                            |
| `joblib`            | 1.4.2    | Model serialization & parallelism        |
| `jupyterlab`        | 4.2.1    | Interactive notebooks                    |
| `kaggle`            | 1.6.14   | Dataset download API                     |

> All packages are **CPU-only**. No GPU required. Runs on any laptop with 8–16 GB RAM.

---

## Mathematical Concepts & Formulas

### 1. Limit Order Book (LOB) Structure

A LOB is an electronic list of buy and sell orders organized by price level:

```
ASK side (sellers)     Price ↑     Volume
  Level 3              102.50       200
  Level 2              102.25       350
  Level 1 (best ask)   102.00       500    ← Lowest price someone will sell at
─────────────────── SPREAD ───────────────────
  Level 1 (best bid)   101.75       400    ← Highest price someone will buy at
  Level 2              101.50       300
  Level 3              101.25       150
BID side (buyers)      Price ↓     Volume
```

The **FI-2010 dataset** provides 10 levels on each side (10 ask + 10 bid = 40 raw columns per snapshot).

### 2. Mid-Price

$$P_{mid} = \frac{P_{ask,1} + P_{bid,1}}{2}$$

The mid-price is the "fair value" estimate — the average of the best ask and best bid prices. It is the fundamental reference price for all our features and labels.

### 3. Bid-Ask Spread

$$S = P_{ask,1} - P_{bid,1}$$

$$S_{bps} = \frac{S}{P_{mid}} \times 10{,}000$$

The spread measures the **cost of immediacy**. A wider spread means lower liquidity. $S_{bps}$ expresses the spread in basis points (1 bps = 0.01%).

### 4. Order Flow Imbalance (OFI)

$$\text{OFI} = \frac{V_{bid,1} - V_{ask,1}}{V_{bid,1} + V_{ask,1}} \in [-1, +1]$$

| Value | Meaning                                         |
| ----- | ----------------------------------------------- |
| +1    | All volume on bid → strong **buying** pressure  |
| 0     | Balanced book                                   |
| −1    | All volume on ask → strong **selling** pressure |

OFI at the top of the book is one of the most powerful short-term predictive signals in market microstructure research.

### 5. Volume Delta

$$\Delta V = \sum_{i=1}^{n} V_{bid,i} - \sum_{i=1}^{n} V_{ask,i}$$

Unlike OFI (which only uses Level 1), Volume Delta aggregates across all $n$ levels, capturing the **full depth** of supply vs. demand.

### 6. Aggressive Trading Pressure

When volume at the best bid **decreases**, aggressive sellers are consuming (hitting) resting buy orders. We measure this asymmetry:

$$\text{AP}_t = \text{RollingMean}\left(\frac{\Delta V_{ask,consumed} - \Delta V_{bid,consumed}}{\Delta V_{ask,consumed} + \Delta V_{bid,consumed}},\ w\right)$$

### 7. Volume-Weighted Mid-Price (VWMP)

$$P_{wmid} = \frac{P_{ask,1} \cdot V_{bid,1} + P_{bid,1} \cdot V_{ask,1}}{V_{bid,1} + V_{ask,1}}$$

Note the **cross-weighting**: the side with more volume pulls the estimate toward its price. This provides a fairer value estimate than the simple mid-price.

### 8. Realized Volatility

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \qquad \sigma_t = \text{std}(r_{t-w+1}, \ldots, r_t)$$

Rolling standard deviation of log returns. We classify volatility into **regimes**: Low / Medium / High using tercile quantiles.

### 9. Label Construction (3-Class Mid-Price Direction)

For a prediction horizon $k$:

$$\Delta P = \frac{P_{mid}(t+k) - P_{mid}(t)}{P_{mid}(t)}$$

$$y = \begin{cases} 0 & (\text{Down}) & \text{if } \Delta P < -\alpha \\ 1 & (\text{Stationary}) & \text{if } |\Delta P| \leq \alpha \\ 2 & (\text{Up}) & \text{if } \Delta P > +\alpha \end{cases}$$

where $\alpha$ is a threshold (set by the dataset authors in FI-2010).

### 10. LSTM Equations

The LSTM processes sequences through **gated memory cells**:

| Gate       | Equation                                              |
| ---------- | ----------------------------------------------------- |
| Forget     | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$        |
| Input      | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$        |
| Candidate  | $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ |
| Cell state | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$     |
| Output     | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$        |
| Hidden     | $h_t = o_t \odot \tanh(C_t)$                          |

### 11. GRU Equations

GRU uses 2 gates instead of 3, making it ~30% faster:

| Gate      | Equation                                                    |
| --------- | ----------------------------------------------------------- |
| Update    | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$              |
| Reset     | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$              |
| Candidate | $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$ |
| Hidden    | $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$     |

### 12. Temporal Fusion Transformer (TFT) Architecture

```
Input Features
    │
    ▼
┌─────────────────────┐
│ Variable Selection  │  ← GRN decides which features matter
│ (Gated Residual     │
│  Network per var)   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ LSTM Encoder        │  ← Temporal patterns
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Multi-Head Attention│  ← Long-range dependencies
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Output GRN +        │
│ Classification Head │  ← Down / Stationary / Up
└─────────────────────┘
```

**Gated Residual Network (GRN):**

$$\eta_1 = \text{ELU}(W_1 x + b_1), \quad \eta_2 = W_2 \eta_1 + b_2$$

$$\text{GLU}(\eta_2) = \sigma(W_g \eta_2 + b_g) \odot \eta_2$$

$$\text{GRN}(x) = \text{LayerNorm}(x + \text{GLU}(\eta_2))$$

### 13. Performance Metrics

| Metric            | Formula                                                                       | Interpretation                                  |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------- |
| **Sharpe Ratio**  | $SR = \frac{E[R] - R_f}{\sigma(R)} \times \sqrt{N}$                           | Risk-adjusted return; >1 decent, >2 very good   |
| **Sortino Ratio** | $\text{Sortino} = \frac{E[R] - R_f}{\sigma_{down}(R)} \times \sqrt{N}$        | Like Sharpe but only penalizes **downside** vol |
| **Max Drawdown**  | $\text{MaxDD} = \max_t \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t}$ | Worst peak-to-trough loss                       |
| **Win Rate**      | $\text{WR} = \frac{\\# \text{profitable trades}}{\\# \text{total trades}}$    | Fraction of winning trades                      |

---

## Feature Dictionary

| #    | Feature                 | Source           | Description                         |
| ---- | ----------------------- | ---------------- | ----------------------------------- |
| 1    | `mid_price`             | L1 prices        | $(P_{ask,1} + P_{bid,1}) / 2$       |
| 2    | `spread`                | L1 prices        | $P_{ask,1} - P_{bid,1}$             |
| 3    | `spread_bps`            | Derived          | Spread in basis points              |
| 4    | `ofi`                   | L1 volumes       | Order Flow Imbalance ∈ [-1, 1]      |
| 5    | `volume_delta`          | L1–L10 vols      | Total bid volume − total ask volume |
| 6    | `aggressive_pressure`   | L1 vol changes   | Rolling ratio of consumed volumes   |
| 7    | `weighted_mid`          | L1 prices + vols | Volume-weighted mid-price           |
| 8    | `wmid_deviation`        | Derived          | Weighted mid − simple mid           |
| 9–18 | `depth_imbalance_1..10` | All levels       | Per-level volume imbalance          |
| 19   | `volatility`            | Mid-price        | Rolling std of log returns          |
| 20   | `vol_regime`            | Volatility       | 0=Low, 1=Medium, 2=High             |
| 21   | `log_return`            | Mid-price        | $\ln(P_t / P_{t-1})$                |
| 22   | `total_ask_vol`         | All levels       | Sum of all ask volumes              |
| 23   | `total_bid_vol`         | All levels       | Sum of all bid volumes              |
| 24   | `total_volume`          | Derived          | Total volume both sides             |

---

## Setup & Installation

### Prerequisites

- **Python 3.10 or 3.11** (via [Anaconda](https://www.anaconda.com/download) or [python.org](https://python.org))
- **8+ GB RAM** (no GPU required)
- **Kaggle account** (free, for data download)

### Option A: pip (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/lob-prediction-project.git
cd lob-prediction-project

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate lob
```

---

## Data Download

```bash
python scripts/download_data.py
```

This script:

1. Creates all required directories
2. Tries the **Kaggle API** first (automatic)
3. Falls back to **manual download instructions** if API is unavailable

### Kaggle API Setup (one-time)

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New Token**
2. Move downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. Run: `chmod 600 ~/.kaggle/kaggle.json` (Mac/Linux only)

---

## How to Run

Run notebooks **in order** (each builds on the previous):

```bash
jupyter lab
```

| #   | Notebook                               | Purpose                                             |
| --- | -------------------------------------- | --------------------------------------------------- |
| 01  | `01_eda.ipynb`                         | Exploratory Data Analysis — visualize LOB structure |
| 02  | `02_feature_engineering.ipynb`         | Compute all 24+ features from raw LOB data          |
| 03  | `03_baseline_models.ipynb`             | Logistic Regression + LightGBM (Optuna tuning)      |
| 04  | `04_lstm_gru.ipynb`                    | LSTM & GRU sequential models                        |
| 05  | `05_temporal_fusion_transformer.ipynb` | Simplified TFT (CPU-friendly)                       |
| 06  | `06_backtesting.ipynb`                 | Realistic P&L with slippage & commissions           |
| 07  | `07_shap_explainability.ipynb`         | SHAP analysis + beautiful plots                     |

---

## Results & Key Findings

> _Results will be populated after running all notebooks._

### Classification Performance

| Model               | Accuracy | F1 (macro) | F1 (weighted) |
| ------------------- | -------- | ---------- | ------------- |
| Logistic Regression | —        | —          | —             |
| LightGBM (Optuna)   | —        | —          | —             |
| LSTM                | —        | —          | —             |
| GRU                 | —        | —          | —             |
| TFT                 | —        | —          | —             |

### Backtest Performance (1 bps commission + 0.5 bps slippage)

| Model    | Sharpe | Sortino | Max DD | Win Rate |
| -------- | ------ | ------- | ------ | -------- |
| LightGBM | —      | —       | —      | —        |
| LSTM     | —      | —       | —      | —        |
| GRU      | —      | —       | —      | —        |
| TFT      | —      | —       | —      | —        |

---

## Folder Structure

```
lob-prediction-project/
├── data/
│   ├── raw/                  # ← .gitignore'd (~600 MB)
│   └── processed/            # ← .gitignore'd
├── scripts/
│   └── download_data.py      # One-command data download
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_lstm_gru.ipynb
│   ├── 05_temporal_fusion_transformer.ipynb
│   ├── 06_backtesting.ipynb
│   └── 07_shap_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading & splitting
│   ├── features.py           # Feature engineering pipeline
│   ├── models.py             # LSTM, GRU, TFT, baselines
│   ├── backtest.py           # Backtesting engine
│   └── utils.py              # Seeding, metrics, plotting
├── results/
│   ├── plots/                # Generated figures
│   ├── tables/               # Performance tables (CSV)
│   └── models/               # ← .gitignore'd (model weights)
├── report/
│   ├── report.md             # 4–6 page PDF source
│   └── figures/              # Report figures
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── sop_paragraph.txt
└── LICENSE
```

---

## Reproducibility

- **Random seed**: `42` used everywhere (NumPy, PyTorch, scikit-learn)
- **Pinned dependencies**: All versions locked in `requirements.txt`
- **Chronological splits**: No future data leaks — train/val/test split by time
- **No shuffling**: Time-series data is never shuffled
- **CPU-only**: Results are identical regardless of hardware

---

## Citation

If you use this work, please cite:

```bibtex
@misc{lob_prediction_2026,
  author = {Victor},
  title  = {Limit Order Book Dynamics Prediction and Trading Signal Generation using Advanced Machine Learning},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/lob-prediction-project}
}
```

### Dataset Citation

```bibtex
@article{ntakaris2018fi2010,
  title     = {Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods},
  author    = {Ntakaris, Adamantios and Magris, Martin and Kanber, Juho and Kanniainen, Juho and Gabbouj, Moncef and Iosifidis, Alexandros},
  journal   = {Journal of Forecasting},
  volume    = {37},
  number    = {8},
  pages     = {852--866},
  year      = {2018}
}
```

---

<div align="center">

_Built for the University of Luxembourg Master's admission._

</div>
