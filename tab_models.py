from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DROPOUT = 0.4


class TinyMLP(nn.Module):
    def __init__(self, n_in: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)


class SimpleMLP(nn.Module):
    def __init__(self, n_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(16, 2),
        )
    def forward(self, x): return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, n_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),    nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)


class ResMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, n_blocks: int = 3,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 2),
        )
    def forward(self, x): return self.head(self.blocks(self.proj(x)))


class _GLULayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.bn(self.fc(x))
        a, b = h.chunk(2, dim=-1)
        return self.drop(a * torch.sigmoid(b))


class GatedMLP(nn.Module):
    def __init__(self, n_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _GLULayer(n_in, 64, dropout),
            _GLULayer(64, 32, dropout),
            _GLULayer(32, 16, dropout * 0.5),
        )
        self.head = nn.Linear(16, 2)
    def forward(self, x): return self.head(self.layers(x))


class AttentionMLP(nn.Module):
    def __init__(self, n_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(n_in, n_in), nn.Sigmoid())
        self.mlp  = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),   nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(16, 2),
        )
    def forward(self, x): return self.mlp(x * self.gate(x))


class FTTransformer(nn.Module):
    def __init__(self, n_in: int, d_token: int = 16, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = DROPOUT) -> None:
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.W_num = nn.Parameter(torch.randn(n_in, d_token) * 0.02)
        self.b_num = nn.Parameter(torch.zeros(n_in, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_token, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = x.unsqueeze(-1) * self.W_num.unsqueeze(0) + self.b_num.unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0, :])


ARCHS: dict[str, type] = {
    "TinyMLP":       TinyMLP,
    "SimpleMLP":     SimpleMLP,
    "DeepMLP":       DeepMLP,
    "ResMLP":        ResMLP,
    "GatedMLP":      GatedMLP,
    "AttentionMLP":  AttentionMLP,
    "FTTransformer": FTTransformer,
}


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, arch: str = "DeepMLP", *, max_epochs: int = 300,
                 patience: int = 15, min_epochs: int = 25, smooth: int = 3,
                 lr: float = 1e-3, weight_decay: float = 1e-2, val_frac: float = 0.2,
                 dropout: float = DROPOUT, device: str | None = None, seed: int = 42):
        self.arch = arch
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_epochs = min_epochs
        self.smooth = smooth
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.dropout = dropout
        self.device = device
        self.seed = seed

    def _dev(self):
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self, n_in):
        return ARCHS[self.arch](n_in, dropout=self.dropout)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self._dev()

        if (X_val is not None and y_val is not None
                and len(set(np.asarray(y_val).astype(int).tolist())) > 1):
            Xtr, ytr = X, y
            Xv = np.asarray(X_val, dtype=np.float32); yv = np.asarray(y_val).astype(int)
        else:
            can_split = (len(self.classes_) > 1 and np.min(np.bincount(y)) >= 2
                         and len(y) >= 10)
            if can_split:
                Xtr, Xv, ytr, yv = train_test_split(
                    X, y, test_size=self.val_frac, stratify=y, random_state=self.seed)
            else:
                Xtr, ytr, Xv, yv = X, y, X, y

        model = self._build(X.shape[1]).to(dev)
        n_pos = int(ytr.sum()); n_neg = int(len(ytr) - n_pos)
        if n_pos > 0 and n_neg > 0:
            w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=dev)
            w = w / w.sum() * 2.0
        else:
            w = None
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                           patience=7, min_lr=1e-6)
        Xtr_t = torch.tensor(Xtr, device=dev)
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
        Xv_t = torch.tensor(Xv, device=dev)

        best_auc, best_state, no_imp, hist = -1.0, None, 0, []
        two_class_val = len(set(yv.tolist())) > 1
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            opt.zero_grad()
            out = model(Xtr_t)
            loss = crit(out, ytr_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                pv = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
            val_auc = roc_auc_score(yv, pv) if two_class_val else 0.5
            hist.append(float(val_auc))
            sm = float(np.mean(hist[-self.smooth:]))
            sched.step(sm)
            if sm > best_auc + 1e-4:
                best_auc = sm
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if epoch >= self.min_epochs and no_imp >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.device_ = dev
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            p = torch.softmax(self.model_(torch.tensor(X, device=self.device_)), dim=1)[:, 1]
            p = p.cpu().numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
