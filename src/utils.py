"""
src/utils.py
============
Shared utilities: reproducibility, metrics, plotting helpers.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from pathlib import Path


# ── Reproducibility ─────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Random seed set to {seed}")


# ── Evaluation Metrics ──────────────────────────────────────

def evaluate_classifier(y_true, y_pred, class_names=None, title="Model"):
    """
    Print classification report + return metrics dict.
    """
    if class_names is None:
        class_names = ["Down", "Stationary", "Up"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"  {title} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  F1 (macro):      {f1_macro:.4f}")
    print(f"  F1 (weighted):   {f1_weighted:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ── Plotting Helpers ────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          title="Confusion Matrix", save_path=None):
    """Plot a pretty confusion matrix."""
    if class_names is None:
        class_names = ["Down", "Stationary", "Up"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    plt.show()
    return fig


def plot_equity_curve(equity, title="Equity Curve", save_path=None):
    """Plot the backtest equity curve."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.5, color="#2196F3")
    ax.fill_between(range(len(equity)), equity,
                    alpha=0.1, color="#2196F3")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    plt.show()
    return fig


def plot_feature_importance(feature_names, importances, top_n=20,
                            title="Feature Importance", save_path=None):
    """Bar chart of top-N feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx], color="#4CAF50")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# ── File I/O Helpers ────────────────────────────────────────

def get_project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dirs():
    """Create all required project directories."""
    root = get_project_root()
    dirs = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "results" / "plots",
        root / "results" / "tables",
        root / "results" / "models",
        root / "report" / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
