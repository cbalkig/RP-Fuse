import gc
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import logging
import random
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from data_augmentation import MedicalTransforms3D
from dataset import MedicalDataset, PRIMARY_STUDY_TOKEN, resolve_primary_study
from focal_loss import FocalLoss
from model import PredictiveResNet
from util import load_config, setup_logging, load_unified_split, read_table

_full_cfg = load_config()
_cfg = _full_cfg["TRAIN_DOSE"]
_data_cfg = _cfg["DATA"]
_train_cfg = _cfg["TRAINING"]
_model_cfg = _cfg["MODEL"]

_input_dir = _full_cfg["INPUTS_DIR"]
_output_dir = _full_cfg["OUTPUTS_DIR"]
_proc_dir = os.path.join(_full_cfg["INPUTS_DIR"], _full_cfg["INPUTS"]["PROCESSED"])

MANIFEST_FILE = os.path.join(_input_dir, _data_cfg["MANIFEST_FILE"])

import endpoint as _endpoint
_EP = _endpoint.EndpointConfig(_full_cfg)
def read_manifest():
    return _endpoint.read_manifest(MANIFEST_FILE, _input_dir, _EP, log=log)
RESULTS_DIR = os.path.join(_output_dir, _data_cfg["RESULTS_DIR"])
PRED_DIR = os.path.join(_output_dir, _data_cfg["PRED_DIR"])
TARGET_COL = _data_cfg.get("TARGET_COL", "rp_label")

DOSE_NORM_CGY = float(_data_cfg.get("DOSE_NORM_CGY", 6000.0))

SEED = _train_cfg["SEED"]
N_FOLDS = _full_cfg["SPLIT"]["N_FOLDS"]
BATCH_SIZE = _train_cfg["BATCH_SIZE"]
MAX_EPOCHS = _train_cfg["MAX_EPOCHS"]
LR = float(_train_cfg["LR"])
WEIGHT_DECAY = float(_train_cfg["WEIGHT_DECAY"])
DROPOUT = _train_cfg["DROPOUT"]
N_ENSEMBLE = _train_cfg["N_ENSEMBLE"]
N_MC_SAMPLES = _train_cfg["N_MC_SAMPLES"]
ACCUMULATION = _train_cfg["ACCUMULATION_STEPS"]
WARMUP_EPOCHS = _train_cfg["WARMUP_EPOCHS"]
EARLY_STOP_PATIENCE = int(_train_cfg.get("EARLY_STOP_PATIENCE", 15))
EARLY_STOP_MIN_EPOCHS = int(_train_cfg.get("EARLY_STOP_MIN_EPOCHS", 25))

_AMP_STR = str(_train_cfg.get("AMP_DTYPE", "bf16")).lower()
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "f16": torch.float16, "float16": torch.float16,
             "bfloat16": torch.bfloat16}.get(_AMP_STR, torch.bfloat16)


ARCH_BATCH_OVERRIDES: dict[str, dict] = {
    "MedicalNet3D_R18": {"batch_size": 4, "accumulation": 8},
    "MedicalNet3D_R50": {"batch_size": 2, "accumulation": 16},
}
NUM_WORKERS = _data_cfg.get("NUM_WORKERS", 2)
TARGET_SHAPE = tuple(_data_cfg.get("TARGET_SHAPE", [128, 128, 128]))
BACKBONE_SMALL = _model_cfg.get("BACKBONE_SMALL", "resnet18")
BACKBONE_LARGE = _model_cfg.get("BACKBONE_LARGE", "resnet50")
_MEDICALNET_R50_PATH: str = _model_cfg.get("MEDICALNET_PRETRAINED_R50", "") or ""
_MEDICALNET_R18_PATH: str = _model_cfg.get("MEDICALNET_PRETRAINED_R18", "") or ""

_aug_cfg = _cfg.get("AUGMENTATION", {})
MIXUP_ALPHA = float(_aug_cfg.get("MIXUP_ALPHA", 0.0))
P_MIXUP = float(_aug_cfg.get("P_MIXUP", 0.0))

_log_dir = os.path.join(_output_dir, _full_cfg["OUTPUTS"]["LOGS"])
log = setup_logging(_log_dir, "train_dose_data.log")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


DOSE_SUBMODALITIES: list[tuple[str, str]] = [
    ("dose_volume", "Has_Dose_Volume"),
]

CURRENT_SUBMODALITY: str = "dose_volume"
CURRENT_GEN_KEY: str = "Has_Dose_Volume"

DOSE_MODALITY_KEY: str = "dose"
DOSE_MODALITY_SUBDIR: str = (
    _full_cfg.get("PREPROCESS", {})
    .get("MODALITIES", {})
    .get(DOSE_MODALITY_KEY, {})
    .get("dest_subdir", "Dose")
)

_STUDY_LAYOUT: bool = bool(_full_cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))
DOSE_STUDY_SUBDIR: str = os.path.join("studies", "*_PRIMARY", "Dose")
DOSE_STUDY_FILE: str = "dose_regCT.npy"


def _active_file(fname: str) -> str:
    if _STUDY_LAYOUT:
        return os.path.join(DOSE_STUDY_SUBDIR, DOSE_STUDY_FILE)
    return os.path.join(DOSE_MODALITY_SUBDIR, fname)


def _gen_cfg_for_submodality(base_gen_cfg: dict, gen_key: str) -> dict | None:
    fname = base_gen_cfg.get(gen_key)
    if not fname:
        return None
    cfg = dict(base_gen_cfg)
    cfg["Active_Before_File"] = _active_file(fname)
    return cfg


def _resolved_exists(patient_dir: str, rel: str) -> bool:
    import glob as _glob
    parts = rel.split(os.sep)
    if PRIMARY_STUDY_TOKEN in parts:
        sdir = resolve_primary_study(patient_dir)
        if sdir is None:
            return False
        i = parts.index(PRIMARY_STUDY_TOKEN)
        return os.path.exists(os.path.join(sdir, *parts[i + 1:]))
    if "*" in rel or "?" in rel:
        return bool(_glob.glob(os.path.join(patient_dir, rel)))
    return os.path.exists(os.path.join(patient_dir, rel))


def _submodality_available(sub_name: str, gen_cfg: dict, processed_dir: str,
                           gen_key: str, min_patients: int = 10) -> bool:
    fname = gen_cfg.get(gen_key)
    if not fname:
        return False
    import glob as _glob
    rel = _active_file(fname)
    count = 0
    for patient_dir in _glob.glob(os.path.join(processed_dir, "*")):
        if os.path.isdir(patient_dir) and _resolved_exists(patient_dir, rel):
            count += 1
            if count >= min_patients:
                return True
    return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dose_split(manifest_df: pd.DataFrame) -> tuple[list[int], list[int], list]:
    import pandas as pd
    import os
    import train_rpfuse as dg
    from util import load_config

    _cfg_for_split = load_config()
    cohort_dir = os.path.join(_cfg_for_split["INPUTS_DIR"], _cfg_for_split["INPUTS"]["PROCESSED"])
    raw_split = load_unified_split(_input_dir)

    keep = []
    for _, r in raw_split.iterrows():
        ud = os.path.join(cohort_dir, str(r["uuid"]).strip())
        single = os.path.isdir(ud) and dg.n_ct_studies(ud) == 1
        ofld = r.get("OuterFold", -1)
        keep.append(bool(single) and pd.notna(ofld) and int(ofld) >= 0)

    df_split = raw_split[pd.Series(keep, index=raw_split.index)].reset_index(drop=True)
    dose_split = df_split.copy()

    uuid_to_mrow = {}
    for mrow, uuid in enumerate(manifest_df["UUID"].values):
        uuid_to_mrow[str(uuid).strip()] = mrow

    test_mrows, dev_mrows = [], []
    for _, row in dose_split.iterrows():
        uid = str(row["uuid"]).strip()
        if uid not in uuid_to_mrow:
            continue
        mrow = uuid_to_mrow[uid]
        if row["Split"] == "Test":
            test_mrows.append(mrow)
        else:
            dev_mrows.append(mrow)

    dev_mrows = sorted(dev_mrows)
    test_mrows = sorted(test_mrows)

    dev_fold_map = {uuid_to_mrow[str(r["uuid"]).strip()]: int(r["Fold"])
                    for _, r in dose_split[dose_split["Split"] == "Train_CV"].iterrows()
                    if str(r["uuid"]).strip() in uuid_to_mrow}
    dev_folds_sorted = [dev_fold_map.get(m, 0) for m in dev_mrows]

    folds = []
    for fi in sorted(set(dev_folds_sorted)):
        val_rel = np.array([i for i, f in enumerate(dev_folds_sorted) if f == fi])
        train_rel = np.array([i for i, f in enumerate(dev_folds_sorted) if f != fi])
        folds.append((train_rel, val_rel))

    log.info("Dose split: dev=%d  test=%d  folds=%d",
             len(dev_mrows), len(test_mrows), len(folds))
    return dev_mrows, test_mrows, folds


DOSE_FEATURE_NAMES: list[str] = [
    "DoseMean", "DoseStd", "DoseMin", "DoseMax",
    "P5", "P10", "P25", "DoseMedian", "P75", "P90", "P95",
    "Skewness", "Kurtosis",
    "Energy", "Entropy",
    "V5", "V10", "V20", "V30",
    "DynamicRange", "GradZ_Mean", "GradY_Mean",
]


def extract_dose_features(volume: np.ndarray) -> np.ndarray:
    v = volume.ravel().astype(np.float64)
    percs = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    n = len(v)
    safe_v = v + 1e-12
    entropy = float(-np.sum(safe_v * np.log(safe_v)) / n)

    v5  = float((v > 500  / DOSE_NORM_CGY).sum()) / (n + 1e-9)
    v10 = float((v > 1000 / DOSE_NORM_CGY).sum()) / (n + 1e-9)
    v20 = float((v > 2000 / DOSE_NORM_CGY).sum()) / (n + 1e-9)
    v30 = float((v > 3000 / DOSE_NORM_CGY).sum()) / (n + 1e-9)

    feat = [
        float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
        float(percs[0]), float(percs[1]), float(percs[2]), float(percs[3]),
        float(percs[4]), float(percs[5]), float(percs[6]),
        float(scipy.stats.skew(v)), float(scipy.stats.kurtosis(v)),
        float(np.sum(v ** 2) / n), entropy,
        v5, v10, v20, v30,
        float(v.max() - v.min()),
        float(np.mean(np.abs(np.diff(volume, axis=0)))),
        float(np.mean(np.abs(np.diff(volume, axis=1)))),
    ]
    return np.array(feat, dtype=np.float32)


def build_radiomics_matrix(dataset: MedicalDataset,
                           indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    orig_transform = dataset.transform
    dataset.transform = None
    feats, labels = [], []
    for i in indices:
        try:
            vol_t, lbl = dataset[i]
            vol = vol_t.numpy() if isinstance(vol_t, torch.Tensor) else vol_t
            if vol.ndim == 4:
                vol = vol[0]
            feats.append(extract_dose_features(vol))
            labels.append(int(lbl))
        except Exception as e:
            log.warning("Dose features failed for index %d: %s", i, e)
            feats.append(np.zeros(len(DOSE_FEATURE_NAMES), dtype=np.float32))
            labels.append(0)
    dataset.transform = orig_transform
    return np.vstack(feats), np.array(labels, dtype=np.int64)


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def scale_rad(X_tr, X_ev):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_ev), sc


class ResNet18_GAP(nn.Module):
    def __init__(self, dropout_rate=DROPOUT, in_chans=1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0,
            in_chans=in_chans, features_only=True)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, in_chans, 128, 128))[-1].shape[1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(256, 2))

    def pooled_feature(self, x):
        if x.dim() == 5:
            b, c, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
            return self.pool(self.backbone(x)[-1]).flatten(1).view(b, d, -1).mean(1)
        return self.pool(self.backbone(x)[-1]).flatten(1)

    def forward(self, x):
        return self.fc(self.pooled_feature(x))


def _make_resnet18_attn(): return PredictiveResNet(BACKBONE_SMALL, num_classes=2,
                                                    dropout_rate=DROPOUT, in_chans=1)
def _make_resnet50_attn(): return PredictiveResNet(BACKBONE_LARGE, num_classes=2,
                                                    dropout_rate=DROPOUT, in_chans=1)
def _make_resnet18_gap():  return ResNet18_GAP(dropout_rate=DROPOUT, in_chans=1)


class MedicalNet3D_R50(nn.Module):
    def __init__(self, pretrained_path="", dropout_rate=DROPOUT, in_chans=1):
        super().__init__()
        try:
            from monai.networks.nets.resnet import resnet50 as monai_resnet50
        except ImportError as exc:
            raise ImportError("monai required: pip install monai") from exc
        self.backbone = monai_resnet50(spatial_dims=3, n_input_channels=1,
                                       feed_forward=False, shortcut_type="B",
                                       bias_downsample=False, num_classes=2)
        self.input_adapter = (nn.Conv3d(in_chans, 1, 1, bias=False)
                              if in_chans != 1 else nn.Identity())
        if in_chans != 1:
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0 / in_chans)
        with torch.no_grad():
            p = self.backbone(torch.zeros(1, 1, 32, 32, 32))
            if isinstance(p, (list, tuple)): p = p[-1]
            if p.dim() > 2: p = p.mean(dim=tuple(range(2, p.dim())))
            feat_dim = p.shape[1]
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(feat_dim, 2))
        if pretrained_path and os.path.exists(pretrained_path):
            self._load(pretrained_path)
        elif pretrained_path:
            log.warning("MedicalNet3D_R50: weights not found at %s - random init.", pretrained_path)

    def _load(self, path):
        try: ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError: ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        m, u = self.backbone.load_state_dict(clean, strict=False)
        log.info("MedicalNet3D_R50: loaded %s (missing=%d unexpected=%d)",
                 os.path.basename(path), len(m), len(u))

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): m.eval()
        return self

    def forward(self, x):
        x = self.input_adapter(x)
        f = self.backbone(x)
        if isinstance(f, (list, tuple)): f = f[-1]
        if f.dim() > 2: f = f.mean(dim=tuple(range(2, f.dim())))
        return self.head(f)


class MedicalNet3D_R18(nn.Module):
    def __init__(self, pretrained_path="", dropout_rate=DROPOUT, in_chans=1):
        super().__init__()
        try:
            from monai.networks.nets.resnet import resnet18 as monai_resnet18
        except ImportError as exc:
            raise ImportError("monai required: pip install monai") from exc
        self.backbone = monai_resnet18(spatial_dims=3, n_input_channels=1,
                                       feed_forward=False, shortcut_type="A",
                                       bias_downsample=False, num_classes=2)
        self.input_adapter = (nn.Conv3d(in_chans, 1, 1, bias=False)
                              if in_chans != 1 else nn.Identity())
        if in_chans != 1:
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0 / in_chans)
        with torch.no_grad():
            p = self.backbone(torch.zeros(1, 1, 32, 32, 32))
            if isinstance(p, (list, tuple)): p = p[-1]
            if p.dim() > 2: p = p.mean(dim=tuple(range(2, p.dim())))
            feat_dim = p.shape[1]
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(feat_dim, 2))
        if pretrained_path and os.path.exists(pretrained_path):
            self._load(pretrained_path)
        else:
            log.warning("MedicalNet3D_R18: no R18 weights at %r - random init.", pretrained_path)

    def _load(self, path):
        try: ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError: ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        m, u = self.backbone.load_state_dict(clean, strict=False)
        log.info("MedicalNet3D_R18: loaded %s (missing=%d unexpected=%d)",
                 os.path.basename(path), len(m), len(u))

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): m.eval()
        return self

    def forward(self, x):
        x = self.input_adapter(x)
        f = self.backbone(x)
        if isinstance(f, (list, tuple)): f = f[-1]
        if f.dim() > 2: f = f.mean(dim=tuple(range(2, f.dim())))
        return self.head(f)


class Tiny3DCNN(nn.Module):
    def __init__(self, in_chans=1, dropout_rate=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_chans, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def _make_tiny3dcnn():        return Tiny3DCNN(in_chans=1, dropout_rate=DROPOUT)
def _make_medicalnet3d_r50(): return MedicalNet3D_R50(_MEDICALNET_R50_PATH, DROPOUT, 1)
def _make_medicalnet3d_r18(): return MedicalNet3D_R18(_MEDICALNET_R18_PATH, DROPOUT, 1)

CNN_ARCHITECTURES: dict[str, callable] = {
    "Tiny3DCNN":        _make_tiny3dcnn,
    "ResNet18_GAP":     _make_resnet18_gap,
    "ResNet18_Attn":    _make_resnet18_attn,
    "MedicalNet3D_R18": _make_medicalnet3d_r18,
    "MedicalNet3D_R50": _make_medicalnet3d_r50,
}

TREE_MODELS: dict[str, tuple] = {}


def optimal_f1_threshold(probs, labels):
    p, r, thr = precision_recall_curve(labels, probs)
    f1 = np.where((p + r) == 0, 0, 2 * p * r / (p + r + 1e-8))
    return float(thr[np.argmax(f1[:-1])])


def metrics_at_threshold(probs, labels, threshold):
    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, average_precision_score, brier_score_loss
    import numpy as np
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    probs = np.array(probs)
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        brier = brier_score_loss(labels, probs)
    else:
        auc, pr_auc, brier = 0.5, 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "f1": f1_score(labels, preds, zero_division=0),
        "f2": (5 * tp) / (5 * tp + 4 * fn + fp + 1e-8),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "pr_auc": pr_auc,
        "brier": brier,
        "acc": float((preds == labels).mean()),
    }


def fit_platt(probs, labels):
    import torch.nn as _nn
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 1.0, 0.0
    n_pos = int(labels.sum()); n_neg = int(len(labels) - n_pos)
    t = np.where(labels == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0)).astype(np.float32)
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    y = torch.tensor(t, dtype=torch.float32)
    a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100); bce = _nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); loss = bce(a * z + b, y); loss.backward(); return loss
    try:
        opt.step(closure)
        av, bv = float(a.detach().item()), float(b.detach().item())
        if not (np.isfinite(av) and np.isfinite(bv)) or not (0.05 < av < 20.0):
            return 1.0, 0.0
        return av, bv
    except Exception:
        return 1.0, 0.0


def apply_platt(probs, ab):
    a, b = ab
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    z = np.clip(a * np.log(p / (1 - p)) + b, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def youden_threshold(probs, labels):
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return t if np.isfinite(t) else 0.5


def _class_weights(labels, device):
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=device)
        return w / w.sum() * 2.0
    return torch.ones(2, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_cnn(model, loader, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, all_probs, all_labels = 0.0, [], []
    for vol_b, lbl in loader:
        vol_b, lbl = vol_b.to(device), lbl.to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
            out0 = model(vol_b)
            loss = criterion(out0, lbl)
            p_sum = (torch.softmax(out0, 1)[:, 1]
                     + torch.softmax(model(torch.flip(vol_b, [3])), 1)[:, 1]
                     + torch.softmax(model(torch.flip(vol_b, [4])), 1)[:, 1])
        total_loss += loss.item() * vol_b.size(0)
        all_probs.extend((p_sum / 3.0).cpu().numpy().tolist())
        all_labels.extend(lbl.cpu().numpy().tolist())
    return total_loss / max(len(loader.dataset), 1), np.array(all_probs), np.array(all_labels, dtype=int)


def _train_cnn(model_factory, train_loader, val_loader, device, train_labels,
               seed_offset=0, accumulation=ACCUMULATION):
    set_seed(SEED + seed_offset)
    model = model_factory().to(device)

    for n, p in model.named_parameters():
        if "backbone" in n: p.requires_grad = False

    head_params = [p for n, p in model.named_parameters()
                   if "backbone" not in n and p.requires_grad]
    if hasattr(model, "optimizer_param_groups"):
        optimizer = optim.AdamW(model.optimizer_param_groups(LR, WEIGHT_DECAY))
    else:
        optimizer = optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    for g in optimizer.param_groups: g["base_lr"] = g["lr"]

    weights = _class_weights(train_labels, device)
    criterion = FocalLoss(alpha=weights, gamma=2)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and AMP_DTYPE == torch.float16
    scaler = GradScaler("cuda") if use_scaler else None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-7)

    best_auc, best_state, best_thr, epoch_count, no_imp = -1.0, None, 0.5, 0, 0
    auc_hist: list[float] = []
    backbone_unfrozen = False

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1 and not backbone_unfrozen:
            backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
            for p in backbone_params: p.requires_grad = True
            optimizer.add_param_group(
                {"params": backbone_params, "lr": LR * 0.1, "base_lr": LR * 0.1})
            backbone_unfrozen = True

        model.train()
        optimizer.zero_grad()
        for step, (vol_b, lbl) in enumerate(train_loader, 1):
            vol_b, lbl = vol_b.to(device), lbl.to(device)
            use_mixup = (P_MIXUP > 0 and MIXUP_ALPHA > 0
                         and vol_b.size(0) > 1 and np.random.rand() < P_MIXUP)
            if use_mixup:
                vol_b, y_a, y_b, lam = mixup_batch(vol_b, lbl, MIXUP_ALPHA)
            else:
                y_a, y_b, lam = lbl, lbl, 1.0

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                out = model(vol_b)
                loss = (lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)) / accumulation

            if scaler:
                scaler.scale(loss).backward()
                if step % accumulation == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if step % accumulation == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()

        if len(train_loader) % accumulation != 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        _, vp, vl = evaluate_cnn(model, val_loader, criterion, device)
        val_auc = roc_auc_score(vl, vp) if len(set(np.asarray(vl).tolist())) > 1 else 0.5
        auc_hist.append(val_auc)
        sm_auc = float(np.mean(auc_hist[-3:]))
        scheduler.step(sm_auc)
        epoch_count = epoch
        thr = optimal_f1_threshold(vp, vl)
        if sm_auc > best_auc + 1e-4:
            best_auc, best_thr = sm_auc, thr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if epoch >= EARLY_STOP_MIN_EPOCHS and no_imp >= EARLY_STOP_PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_thr, epoch_count


FoldResult = dict


def _loader_kwargs(batch_size):
    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def _shutdown_loaders(*loaders):
    for ld in loaders:
        if ld is None: continue
        it = getattr(ld, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            try: it._shutdown_workers()
            except Exception: pass
            ld._iterator = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _make_loaders(ds_train, ds_val, ds_train_orig, batch_size=BATCH_SIZE):
    kw = _loader_kwargs(batch_size)
    return (DataLoader(ds_train, shuffle=True, **kw),
            DataLoader(ds_val, shuffle=False, **kw),
            DataLoader(ds_train_orig, shuffle=False, **kw))


EXPORT_EMB = bool(_data_cfg.get("EXPORT_EMB", False))
EXPORT_EMB_ARCH = str(_data_cfg.get("EXPORT_EMB_ARCH", "ResNet18_GAP"))


def run_cv_cnn(arch_name, model_factory, dataset_clean, dataset_noisy,
               dev_indices, dev_labels, folds, device,
               test_indices=None, test_labels=None,
               batch_size=BATCH_SIZE, accumulation=ACCUMULATION):
    log.info("\n%s\n  CNN: %s\n%s", "-" * 60, arch_name, "-" * 60)
    results = []
    fold_test_probs, oof_vp, oof_vl = [], [], []
    test_loader = None
    if test_indices is not None:
        tkw = {**_loader_kwargs(batch_size), "persistent_workers": False}
        test_loader = DataLoader(Subset(dataset_clean, test_indices), shuffle=False, **tkw)
    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        set_seed(SEED + fi)
        tr_abs = [dev_indices[i] for i in tr_rel]
        val_abs = [dev_indices[i] for i in val_rel]
        tr_lbl = [dev_labels[i] for i in tr_rel]
        val_lbl = [dev_labels[i] for i in val_rel]

        train_loader, val_loader, tr_loader_o = _make_loaders(
            Subset(dataset_noisy, tr_abs), Subset(dataset_clean, val_abs),
            Subset(dataset_clean, tr_abs), batch_size=batch_size)

        model, best_thr, epoch_count = _train_cnn(
            model_factory, train_loader, val_loader, device, tr_lbl,
            seed_offset=fi, accumulation=accumulation)

        criterion = FocalLoss(alpha=_class_weights(tr_lbl, device), gamma=2)
        _, vp, _ = evaluate_cnn(model, val_loader, criterion, device)
        _, tp, _ = evaluate_cnn(model, tr_loader_o, criterion, device)

        val_m = metrics_at_threshold(vp, np.array(val_lbl), best_thr)
        train_m = metrics_at_threshold(tp, np.array(tr_lbl), best_thr)

        log.info("  Fold %d | Train AUC=%.3f F1=%.3f | "
                 "Val AUC=%.3f F1=%.3f Sens=%.3f Spec=%.3f | Ep=%d",
                 fi, train_m["auc"], train_m["f1"],
                 val_m["auc"], val_m["f1"],
                 val_m["sensitivity"], val_m["specificity"], epoch_count)
        results.append({"train": train_m, "val": val_m,
                        "epochs": epoch_count, "threshold": best_thr})

        if test_loader is not None:
            ab = fit_platt(vp, np.array(val_lbl))
            _, te_p, _ = evaluate_cnn(model, test_loader, criterion, device)

            te_m = metrics_at_threshold(te_p, np.array(test_labels), best_thr)
            results[-1]["test"] = te_m

            fold_test_probs.append(apply_platt(te_p, ab))
            oof_vp.extend(apply_platt(vp, ab).tolist()); oof_vl.extend(list(val_lbl))
        if EXPORT_EMB and arch_name == EXPORT_EMB_ARCH:
            from emb_export import export_fold_embeddings
            export_fold_embeddings(
                model, dataset_clean, dataset_clean.df, device,
                int(os.getenv("NESTED_OUTER_FOLD", "0")), fi - 1, RESULTS_DIR,
                folds, dev_indices, test_indices, batch_size=batch_size,
                loader_kwargs={k: v for k, v in _loader_kwargs(batch_size).items()
                               if k != "batch_size"})
        del model
        _shutdown_loaders(train_loader, val_loader, tr_loader_o)
    if test_loader is not None:
        _shutdown_loaders(test_loader)
    tp_ens = np.mean(np.stack(fold_test_probs, 0), 0) if fold_test_probs else None
    return results, tp_ens, np.asarray(oof_vp), np.asarray(oof_vl)


def _carve_dev_val(dev_indices, dev_labels, manifest_df):
    if "IsInnerVal" in manifest_df.columns:
        val_mask = manifest_df.loc[dev_indices, "IsInnerVal"].fillna(False).astype(bool).values
    else:
        from sklearn.model_selection import train_test_split as _tts
        val_frac = float(_full_cfg["SPLIT"].get("INNER_VAL_FRAC", 0.15))
        _, val_rel = _tts(range(len(dev_indices)), test_size=val_frac,
                          stratify=dev_labels, random_state=SEED)
        val_set = set(val_rel)
        val_mask = [i in val_set for i in range(len(dev_indices))]
    tr_idx = [dev_indices[i] for i, v in enumerate(val_mask) if not v]
    val_idx = [dev_indices[i] for i, v in enumerate(val_mask) if v]
    tr_lab = [dev_labels[i] for i, v in enumerate(val_mask) if not v]
    val_lab = [dev_labels[i] for i, v in enumerate(val_mask) if v]
    return tr_idx, val_idx, tr_lab, val_lab


def _variant_checkpoint_path(arch_name: str) -> str:
    return os.path.join(RESULTS_DIR, f".{arch_name}_meta.json")


def _save_variant_meta(arch_name: str, fold_results: list) -> None:
    import json
    path = _variant_checkpoint_path(arch_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"arch": arch_name,
               "n_folds": len(fold_results),
               "mean_val_auc": float(np.mean([r["val"]["auc"] for r in fold_results]))}
    with open(path, "w") as f:
        json.dump(payload, f)


def save_patient_xlsx(arch_name: str, uuids: list, true_labels: np.ndarray,
                      probs: np.ndarray, threshold: float,
                      threshold_y: float | None = None) -> None:
    os.makedirs(PRED_DIR, exist_ok=True)
    preds = (probs >= threshold).astype(int)
    cols = {
        "UUID": uuids,
        "TrueLabel": true_labels,
        "PredLabel": preds,
        "RP_Probability": probs,
        "Threshold": threshold,
        "Correct": (preds == true_labels).astype(int),
    }
    if threshold_y is not None:
        cols["PredLabel_Youden"] = (probs >= threshold_y).astype(int)
        cols["Threshold_Youden"] = threshold_y
    df = pd.DataFrame(cols)
    path = os.path.join(PRED_DIR, f"{arch_name}_test_predictions.xlsx")
    df.to_excel(path, index=False)
    log.info("Saved patient predictions -> %s", path)


def print_fold_table(arch_name: str, results: list) -> None:
    log.info("\n  %-14s | %-26s | %-26s | %-6s", "Fold", "Train (AUC/F1/Sen/Spc)", "Val (AUC/F1/Sen/Spc)", "Ep")
    for i, r in enumerate(results, 1):
        tr_str = f"{r['train']['auc']:.3f}/{r['train']['f1']:.3f}/{r['train']['sensitivity']:.3f}/{r['train']['specificity']:.3f}"
        vl_str = f"{r['val']['auc']:.3f}/{r['val']['f1']:.3f}/{r['val']['sensitivity']:.3f}/{r['val']['specificity']:.3f}"
        log.info("  %-14d | %-26s | %-26s | %-6s", i, tr_str, vl_str, str(r["epochs"]))
    mean_val = np.mean([r["val"]["auc"] for r in results])
    log.info("  %-14s  %-6s  %-6s  %-6.3f", "Mean Val AUC", "", "", mean_val)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    sub_name = CURRENT_SUBMODALITY
    gen_key = CURRENT_GEN_KEY
    log.info("\n%s\n  Dose submodality: %s  (key=%s, modality_subdir=%s)\n%s",
             "=" * 60, sub_name, gen_key, DOSE_MODALITY_SUBDIR, "=" * 60)

    manifest_df_raw = read_manifest()
    log.info("Manifest: %d rows", len(manifest_df_raw))

    aug_cfg = _full_cfg["TRAIN"]["AUGMENTATION"]
    gen_cfg_base = _full_cfg["PREPROCESS"]["GENERATED_FILES"]
    gen_cfg = _gen_cfg_for_submodality(gen_cfg_base, gen_key)
    if gen_cfg is None:
        log.warning("Dose submodality %s: key %s missing - skipping.", sub_name, gen_key)
        return
    if not _submodality_available(sub_name, gen_cfg, _proc_dir, gen_key):
        log.warning("Dose submodality %s: not enough patients on disk - skipping.", sub_name)
        return

    transforms_clean = MedicalTransforms3D(
        mode="standard",
        p_flip=aug_cfg["P_FLIP"], p_rot=aug_cfg["P_ROT"],
        rot_range=aug_cfg["ROT_RANGE"], img_size=TARGET_SHAPE[0])
    transforms_noisy = MedicalTransforms3D(
        mode="augmented",
        p_flip=aug_cfg["P_FLIP"], p_rot=aug_cfg["P_ROT"],
        rot_range=aug_cfg["ROT_RANGE"], img_size=TARGET_SHAPE[0])

    dataset_clean = MedicalDataset(
        manifest_df_raw, _proc_dir, gen_cfg,
        transform=transforms_clean, target_shape=TARGET_SHAPE,
        window_width=100000, window_center=50000)
    dataset_noisy = MedicalDataset(
        manifest_df_raw, _proc_dir, gen_cfg,
        transform=transforms_noisy, target_shape=TARGET_SHAPE,
        window_width=100000, window_center=50000)

    dev_indices, test_indices, folds = resolve_dose_split(manifest_df_raw)
    dev_labels = [int(manifest_df_raw.iloc[i]["Group"] == "RP") for i in dev_indices]
    test_labels = [int(manifest_df_raw.iloc[i]["Group"] == "RP") for i in test_indices]
    test_uuids = [str(manifest_df_raw.iloc[i]["UUID"]) for i in test_indices]

    log.info("Dev=%d (RP+=%.1f%%)  Test=%d (RP+=%.1f%%)",
             len(dev_indices), 100 * np.mean(dev_labels),
             len(test_indices), 100 * np.mean(test_labels))

    all_results: dict[str, list[FoldResult]] = {}
    test_results: dict[str, dict] = {}

    for arch_name, factory in CNN_ARCHITECTURES.items():
        arch_overrides = ARCH_BATCH_OVERRIDES.get(arch_name, {})
        arch_bs = arch_overrides.get("batch_size", BATCH_SIZE)
        arch_acc = arch_overrides.get("accumulation", ACCUMULATION)

        fold_results, tp_ens, oof_vp, oof_vl = run_cv_cnn(
            arch_name, factory, dataset_clean, dataset_noisy,
            dev_indices, dev_labels, folds, device,
            test_indices, test_labels,
            batch_size=arch_bs, accumulation=arch_acc)
        all_results[arch_name] = fold_results
        print_fold_table(arch_name, fold_results)

        thr = optimal_f1_threshold(oof_vp, oof_vl) if len(oof_vp) else 0.5
        thr_y = youden_threshold(oof_vp, oof_vl) if len(oof_vp) else 0.5
        test_results[arch_name] = metrics_at_threshold(tp_ens, np.array(test_labels), thr)
        test_m_y = metrics_at_threshold(tp_ens, np.array(test_labels), thr_y)
        save_patient_xlsx(arch_name, test_uuids, np.array(test_labels), tp_ens, thr, thr_y)
        _save_variant_meta(arch_name, all_results[arch_name])
        log.info("  -> Test (5-model ens+Platt) AUC=%.3f | F1@%.2f: F1=%.3f Sens=%.3f Spec=%.3f"
                 " | Youden@%.2f: Sens=%.3f Spec=%.3f",
                 test_results[arch_name]["auc"], thr, test_results[arch_name]["f1"],
                 test_results[arch_name]["sensitivity"], test_results[arch_name]["specificity"],
                 thr_y, test_m_y["sensitivity"], test_m_y["specificity"])

    log.info("\n%s\n  FINAL SUMMARY - Dose  submodality=%s\n%s",
             "=" * 70, sub_name, "=" * 70)
    log.info("  %-24s | %-45s | %-45s | %-45s", "Model", "Train (AUC/F1/Sn/Sp/PR/Prc/Br)", "Val (AUC/F1/Sn/Sp/PR/Prc/Br)", "Test (AUC/F1/Sn/Sp/PR/Prc/Br)")
    for name, res in test_results.items():
        all_r = all_results[name]
        tr_auc = sum(r["train"]["auc"] for r in all_r)/len(all_r)
        tr_f1 = sum(r["train"]["f1"] for r in all_r)/len(all_r)
        tr_sen = sum(r["train"]["sensitivity"] for r in all_r)/len(all_r)
        tr_spe = sum(r["train"]["specificity"] for r in all_r)/len(all_r)
        tr_pra = sum(r["train"]["pr_auc"] for r in all_r)/len(all_r)
        tr_prc = sum(r["train"]["precision"] for r in all_r)/len(all_r)
        tr_br = sum(r["train"]["brier"] for r in all_r)/len(all_r)
        vl_auc = sum(r["val"]["auc"] for r in all_r)/len(all_r)
        vl_f1 = sum(r["val"]["f1"] for r in all_r)/len(all_r)
        vl_sen = sum(r["val"]["sensitivity"] for r in all_r)/len(all_r)
        vl_spe = sum(r["val"]["specificity"] for r in all_r)/len(all_r)
        vl_pra = sum(r["val"]["pr_auc"] for r in all_r)/len(all_r)
        vl_prc = sum(r["val"]["precision"] for r in all_r)/len(all_r)
        vl_br = sum(r["val"]["brier"] for r in all_r)/len(all_r)
        tr_str = f"{tr_auc:.2f}/{tr_f1:.2f}/{tr_sen:.2f}/{tr_spe:.2f}/{tr_pra:.2f}/{tr_prc:.2f}/{tr_br:.2f}"
        vl_str = f"{vl_auc:.2f}/{vl_f1:.2f}/{vl_sen:.2f}/{vl_spe:.2f}/{vl_pra:.2f}/{vl_prc:.2f}/{vl_br:.2f}"
        te_str = f"{res['auc']:.2f}/{res['f1']:.2f}/{res['sensitivity']:.2f}/{res['specificity']:.2f}/{res['pr_auc']:.2f}/{res['precision']:.2f}/{res['brier']:.2f}"
        log.info("  %-24s | %-45s | %-45s | %-45s", name, tr_str, vl_str, te_str)
    import util
    util.print_array_metrics(all_results, test_results, log)
    log.info("\nDone. Results -> %s", RESULTS_DIR)


def _run_all_submodalities(only: list[str] | None = None, quick: bool = False) -> None:
    g = globals()
    base_results = g["RESULTS_DIR"]
    base_pred = g["PRED_DIR"]
    use_nested = bool(_full_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False))
    n_outer = int(_full_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5))
    n_repeats = int(_full_cfg.get("SPLIT", {}).get("N_CV_REPEATS", 1))
    gen_cfg_base = _full_cfg["PREPROCESS"]["GENERATED_FILES"]

    if use_nested:
        from nested_cv import run_nested_cv

    for sub_name, gen_key in DOSE_SUBMODALITIES:
        if only is not None and sub_name not in only:
            continue
        sub_cfg = _gen_cfg_for_submodality(gen_cfg_base, gen_key)
        if sub_cfg is None:
            log.info("Skipping Dose submodality %s - %s not in GENERATED_FILES.",
                     sub_name, gen_key)
            continue
        if not _submodality_available(sub_name, sub_cfg, _proc_dir, gen_key):
            log.info("Skipping Dose submodality %s - not enough patients on disk.", sub_name)
            continue

        log.info("\n%s\n  Running Dose submodality: %s\n%s", "#" * 70, sub_name, "#" * 70)

        sub_results = os.path.join(base_results, sub_name)
        sub_pred = os.path.join(sub_results, "patient_predictions")
        os.makedirs(sub_results, exist_ok=True)
        os.makedirs(sub_pred, exist_ok=True)
        g["RESULTS_DIR"] = sub_results
        g["PRED_DIR"] = sub_pred
        g["CURRENT_SUBMODALITY"] = sub_name
        g["CURRENT_GEN_KEY"] = gen_key

        try:
            if use_nested:
                run_nested_cv(
                    main_fn=main, n_outer=n_outer, module_globals=g,
                    results_dir_key="RESULTS_DIR", pred_dir_key="PRED_DIR",
                    log=log, n_repeats=n_repeats, quick=quick)
            else:
                main()
        except Exception as exc:
            log.error("Dose submodality %s failed: %s", sub_name, exc, exc_info=True)
        finally:
            g["RESULTS_DIR"] = base_results
            g["PRED_DIR"] = base_pred


if __name__ == "__main__":
    import torch.multiprocessing as _torch_mp
    _torch_mp.set_sharing_strategy("file_system")

    import argparse
    parser = argparse.ArgumentParser(description="Train Dose volume models.")
    _valid = [s[0] for s in DOSE_SUBMODALITIES]
    parser.add_argument("--submodality", action="append", choices=_valid,
                        help=f"Run only this submodality. Default: all. Choices: {', '.join(_valid)}")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 1 outer fold for fast methodology checks.")
    args = parser.parse_args()
    _run_all_submodalities(only=args.submodality, quick=args.quick)
