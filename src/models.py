"""
src/models.py
=============
Model definitions for the LOB prediction project.

Contains:
    - LSTM classifier (PyTorch)
    - GRU classifier (PyTorch)
    - Temporal Fusion Transformer — simplified CPU-friendly version
    - Helper functions to build baseline sklearn/LightGBM models

All deep learning models use PyTorch + PyTorch Lightning for clean
training loops and logging.
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════

class LOBDataset(Dataset):
    """
    PyTorch Dataset for LOB sequence data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, seq_len, n_features)
        Input sequences.
    y : np.ndarray, shape (n_samples,)
        Labels (0, 1, 2).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════
#  LSTM MODEL
# ══════════════════════════════════════════════════════════════

class LSTMClassifier(pl.LightningModule):
    """
    LSTM-based classifier for LOB mid-price direction prediction.

    Architecture:
        Input (seq_len, n_features)
        → LSTM (hidden_dim, num_layers, dropout)
        → Batch Norm
        → FC (hidden_dim → 64)
        → ReLU + Dropout
        → FC (64 → n_classes)

    LSTM Equations (computed internally by PyTorch):
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        ← forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        ← input gate
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)     ← candidate cell
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t            ← cell state
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        ← output gate
        h_t = o_t ⊙ tanh(C_t)                       ← hidden state

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_dim : int
        LSTM hidden dimension.
    num_layers : int
        Number of stacked LSTM layers.
    n_classes : int
        Number of output classes (3: down/stationary/up).
    dropout : float
        Dropout probability.
    lr : float
        Learning rate.
    """

    def __init__(self,
                 n_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.bn(last_hidden)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ══════════════════════════════════════════════════════════════
#  GRU MODEL
# ══════════════════════════════════════════════════════════════

class GRUClassifier(pl.LightningModule):
    """
    GRU-based classifier — similar to LSTM but with fewer gates.

    GRU Equations:
        z_t = σ(W_z · [h_{t-1}, x_t] + b_z)        ← update gate
        r_t = σ(W_r · [h_{t-1}, x_t] + b_r)        ← reset gate
        h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)  ← candidate
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t     ← hidden state

    GRU is ~30% faster than LSTM and often achieves similar accuracy
    on LOB data because the temporal dependencies are short.

    Parameters
    ----------
    Same as LSTMClassifier.
    """

    def __init__(self,
                 n_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        out = self.bn(last_hidden)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ══════════════════════════════════════════════════════════════
#  TEMPORAL FUSION TRANSFORMER (Simplified, CPU-Friendly)
# ══════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — core building block of TFT.

    GRN applies:
        η_1 = ELU(W_1 · x + b_1)
        η_2 = W_2 · η_1 + b_2
        GLU gate = σ(W_g · η_2 + b_g)
        output = LayerNorm(x + GLU(η_2))

    This allows the model to suppress irrelevant features by learning
    which inputs to zero out via the gating mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Gating layer
        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection (project if dimensions differ)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = self.sigmoid(self.gate(h))
        out = self.layer_norm(residual + gate * h)
        return out


class SimpleTFT(pl.LightningModule):
    """
    Simplified Temporal Fusion Transformer for CPU-based LOB prediction.

    Architecture overview:
        1. Variable Selection Network (GRN per feature)
        2. LSTM encoder for temporal processing
        3. Multi-head attention for long-range dependencies
        4. GRN-based output layer
        5. Classification head

    This is a simplified version optimized for CPU training while
    retaining the key TFT innovations: variable selection and
    interpretable attention.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dim : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_classes : int
        Number of output classes.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    """

    def __init__(self,
                 n_features: int,
                 hidden_dim: int = 32,
                 n_heads: int = 4,
                 n_classes: int = 3,
                 dropout: float = 0.2,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Variable selection
        self.var_selection = GatedResidualNetwork(n_features, hidden_dim, hidden_dim, dropout)

        # Temporal encoding via LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                            batch_first=True, dropout=0)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output GRN
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size, seq_len, _ = x.shape

        # 1. Variable selection
        selected = self.var_selection(x)  # (batch, seq_len, hidden_dim)

        # 2. Temporal encoding
        lstm_out, _ = self.lstm(selected)  # (batch, seq_len, hidden_dim)

        # 3. Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)  # residual

        # 4. Use last timestep
        last_step = attn_out[:, -1, :]  # (batch, hidden_dim)

        # 5. Output GRN + classifier
        out = self.output_grn(last_step)
        logits = self.classifier(out)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ══════════════════════════════════════════════════════════════
#  BASELINE MODELS (sklearn / LightGBM)
# ══════════════════════════════════════════════════════════════

def build_logistic_regression(C: float = 1.0,
                               max_iter: int = 1000,
                               seed: int = 42) -> LogisticRegression:
    """
    Build a Logistic Regression baseline.

    Logistic Regression is the simplest classifier. We use it as a
    "sanity check" — if our fancy models can't beat LR, something is wrong.

    Parameters
    ----------
    C : float
        Inverse regularization strength.
    max_iter : int
        Max iterations for solver.
    seed : int
        Random seed.

    Returns
    -------
    LogisticRegression
        Configured model (not yet fitted).
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
        n_jobs=-1,
    )


def build_lightgbm(params: dict = None,
                    seed: int = 42) -> lgb.LGBMClassifier:
    """
    Build a LightGBM classifier with sensible defaults.

    LightGBM is a gradient-boosted decision tree model. It's typically
    the strongest "classical" ML model for tabular data and serves as
    our primary baseline against the deep learning models.

    Parameters
    ----------
    params : dict, optional
        Override default parameters.
    seed : int
        Random seed.

    Returns
    -------
    lgb.LGBMClassifier
        Configured model (not yet fitted).
    """
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    return lgb.LGBMClassifier(**default_params)
