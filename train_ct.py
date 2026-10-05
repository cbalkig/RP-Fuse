import gc
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
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
import shap
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
_cfg = _full_cfg["TRAIN_CT"]
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
TARGET_COL = _data_cfg.get("TARGET_COL", "RadyolojikRP")

DOSE_WEIGHTED_CROP = bool(_data_cfg.get("DOSE_WEIGHTED_CROP", False))
CROP_KEEP_FRACTION = float(_data_cfg.get("CROP_KEEP_FRACTION", 0.65))
CT_EXCEL_PATH = os.path.join(_input_dir,
                             _data_cfg.get("EXCEL_PATH", "tabular_data.xlsx"))


USE_PYRADIOMICS = bool(_data_cfg.get("USE_PYRADIOMICS", False))
PYRADIOMICS_TOP_K = int(_data_cfg.get("PYRADIOMICS_TOP_K", 30))
PYRADIOMICS_CACHE_DIR = os.path.join(_input_dir,
                                     _data_cfg.get("PYRADIOMICS_CACHE_DIR",
                                                   "pyradiomics_cache"))


_aug_cfg = _cfg.get("AUGMENTATION", {})
MIXUP_ALPHA = float(_aug_cfg.get("MIXUP_ALPHA", 0.0))
P_MIXUP = float(_aug_cfg.get("P_MIXUP", 0.0))

CT_VESSEL_COMPARISON = bool(_data_cfg.get("CT_VESSEL_COMPARISON", False))
CT_LUNG_NO_VESSELS = _data_cfg.get("CT_LUNG_NO_VESSELS",
                                   "lung_only_no_vessels.npy")


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
    "ResNet50_Attn": {"batch_size": 8, "accumulation": 4},
    "MedicalNet3D_R18": {"batch_size": 4, "accumulation": 8},
    "MedicalNet3D_R50": {"batch_size": 2, "accumulation": 16},
}
NUM_WORKERS = _data_cfg.get("NUM_WORKERS", 2)
TARGET_SHAPE = tuple(_data_cfg.get("TARGET_SHAPE", [128, 128, 128]))
WINDOW_WIDTH = _data_cfg.get("WINDOW_WIDTH", 1500)
WINDOW_CENTER = _data_cfg.get("WINDOW_CENTER", -600)
BACKBONE_SMALL = _model_cfg.get("BACKBONE_SMALL", "resnet18")
BACKBONE_LARGE = _model_cfg.get("BACKBONE_LARGE", "resnet50")

_MEDICALNET_R50_PATH: str = _model_cfg.get("MEDICALNET_PRETRAINED_R50", "") or ""
_MEDICALNET_R18_PATH: str = _model_cfg.get("MEDICALNET_PRETRAINED_R18", "") or ""

_log_dir = os.path.join(_output_dir, _full_cfg["OUTPUTS"]["LOGS"])
log = setup_logging(_log_dir, "train_ct_data.log")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


RADIOMICS_LABELS: list[str] = []

CT_SUBMODALITIES: list[tuple[str, str]] = [
    ("whole_body", "Has_Whole_Body"),
]

CURRENT_SUBMODALITY: str = "whole_body"
CURRENT_GEN_KEY: str = "Has_Whole_Body"

CT_MODALITY_KEY: str = "ct_plan"
CT_MODALITY_SUBDIR: str = (
    _full_cfg.get("PREPROCESS", {})
    .get("MODALITIES", {})
    .get(CT_MODALITY_KEY, {})
    .get("dest_subdir", "CT-Plan")
)

_STUDY_LAYOUT: bool = bool(_full_cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))
CT_STUDY_SUBDIR: str = os.path.join("studies", "*_PRIMARY", "CT")


def _subdir_for() -> str:
    return CT_STUDY_SUBDIR if _STUDY_LAYOUT else CT_MODALITY_SUBDIR


_DOSE_CROP_INCOMPATIBLE = {"whole_body", "tumor_only"}


def _gen_cfg_for_submodality(base_gen_cfg: dict, gen_key: str) -> dict | None:
    fname = base_gen_cfg.get(gen_key)
    if not fname:
        return None
    cfg = dict(base_gen_cfg)
    cfg["Active_Before_File"] = os.path.join(_subdir_for(), fname)
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
    rel = os.path.join(_subdir_for(), fname)
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


def resolve_ct_split(manifest_df: pd.DataFrame) -> tuple[list[int], list[int], list]:
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
    ct_split = df_split.copy()

    uuid_to_mrow = {}
    for mrow, uuid in enumerate(manifest_df["UUID"].values):
        uuid_to_mrow[str(uuid).strip()] = mrow

    test_mrows, dev_mrows = [], []
    dev_folds = []

    for _, row in ct_split.iterrows():
        uid = str(row["uuid"]).strip()
        if uid not in uuid_to_mrow:
            continue
        mrow = uuid_to_mrow[uid]
        if row["Split"] == "Test":
            test_mrows.append(mrow)
        else:
            dev_mrows.append(mrow)
            dev_folds.append(int(row["Fold"]))

    dev_mrows = sorted(dev_mrows)
    test_mrows = sorted(test_mrows)

    dev_fold_map = {uuid_to_mrow[str(r["uuid"]).strip()]: int(r["Fold"])
                    for _, r in ct_split[ct_split["Split"] == "Train_CV"].iterrows()
                    if str(r["uuid"]).strip() in uuid_to_mrow}
    dev_folds_sorted = [dev_fold_map.get(m, 0) for m in dev_mrows]

    folds = []
    unique_folds = sorted(set(dev_folds_sorted))
    for fi in unique_folds:
        val_rel = np.array([i for i, f in enumerate(dev_folds_sorted) if f == fi])
        train_rel = np.array([i for i, f in enumerate(dev_folds_sorted) if f != fi])
        folds.append((train_rel, val_rel))

    log.info("CT split (unified): dev=%d  test=%d  folds=%d",
             len(dev_mrows), len(test_mrows), len(folds))
    return dev_mrows, test_mrows, folds


RADIOMICS_FEATURE_NAMES = [
    "Mean_HU", "Std_HU", "Min_HU", "Max_HU",
    "P5", "P10", "P25", "Median", "P75", "P90", "P95",
    "Skewness", "Kurtosis",
    "Energy", "Entropy",
    "LungVoxelCount", "HighDensityFrac", "LowDensityFrac",
    "DynamicRange",
    "GradZ_Mean", "GradY_Mean", "GradX_Mean",
]


def extract_radiomics(volume: np.ndarray) -> np.ndarray:
    v = volume.ravel().astype(np.float64)
    percs = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    n = len(v)
    safe_v = v + 1e-12
    entropy = float(-np.sum(safe_v * np.log(safe_v)) / n)

    feat = [
        float(v.mean()),
        float(v.std()),
        float(v.min()),
        float(v.max()),
        float(percs[0]), float(percs[1]), float(percs[2]),
        float(percs[3]),
        float(percs[4]), float(percs[5]), float(percs[6]),
        float(scipy.stats.skew(v)),
        float(scipy.stats.kurtosis(v)),
        float(np.sum(v ** 2) / n),
        entropy,
        float((v > 0.1).sum()),
        float((v > 0.5).sum()) / (n + 1e-9),
        float((v < 0.05).sum()) / (n + 1e-9),
        float(v.max() - v.min()),
        float(np.mean(np.abs(np.diff(volume, axis=0)))),
        float(np.mean(np.abs(np.diff(volume, axis=1)))),
        float(np.mean(np.abs(np.diff(volume, axis=2)))),
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
            feats.append(extract_radiomics(vol))
            labels.append(int(lbl))
        except Exception as e:
            log.warning("Radiomics failed for index %d: %s", i, e)
            feats.append(np.zeros(len(RADIOMICS_FEATURE_NAMES), dtype=np.float32))
            labels.append(0)

    dataset.transform = orig_transform
    return np.vstack(feats), np.array(labels, dtype=np.int64)


def build_tumor_side_map(input_dir: str, excel_path: str) -> dict[str, int]:
    import unicodedata as _ud
    if not os.path.exists(excel_path):
        log.warning("[tumor_side] excel not found: %s - dose crop disabled.",
                    excel_path)
        return {}
    excel_df = pd.read_excel(excel_path)
    split = load_unified_split(input_dir)

    candidates = [
        "tumor_side",
        "tümörlokalizasyonu1sağsolakciğer",
        "tumorlokalizasyonu1sagsolakciger",
        "tümörlokalizasyonu1sagsolakciger",
    ]

    def _norm(s: str) -> str:
        return _ud.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode().lower()

    norm_to_real = {_norm(c): c for c in excel_df.columns}
    side_col = None
    for cand in candidates:
        key = _norm(cand)
        if key in norm_to_real:
            side_col = norm_to_real[key]
            break
    if side_col is None:
        log.warning("[tumor_side] laterality column not found in Excel - "
                    "dose crop disabled.")
        return {}

    out: dict[str, int] = {}
    for _, row in split.iterrows():
        try:
            eidx = int(row.get("excel_row_idx", -1))
        except (TypeError, ValueError):
            continue
        if eidx < 0 or eidx >= len(excel_df):
            continue
        v = excel_df.iloc[eidx][side_col]
        if pd.isna(v):
            continue
        try:
            out[str(row["uuid"]).strip()] = int(v)
        except (TypeError, ValueError):
            continue
    log.info("[tumor_side] mapped %d/%d UUIDs (left=%d right=%d bilateral=%d)",
             len(out), len(split),
             sum(1 for v in out.values() if v == 2),
             sum(1 for v in out.values() if v == 1),
             sum(1 for v in out.values() if v == 3))
    return out


_PYRADIOMICS_EXTRACTOR = None
_PYRADIOMICS_FEATURE_NAMES: list[str] = []


def _get_pyradiomics_extractor():
    global _PYRADIOMICS_EXTRACTOR
    if _PYRADIOMICS_EXTRACTOR is not None:
        return _PYRADIOMICS_EXTRACTOR
    from radiomics import featureextractor
    settings = {
        "binWidth": 50,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": [2.0, 2.0, 2.0],
        "label": 1,
        "force2D": False,
        "voxelArrayShift": 1000,
        "normalize": False,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("firstorder")
    for cls in ("glcm", "glrlm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    _PYRADIOMICS_EXTRACTOR = ext
    return ext


def _extract_pyradiomics_for_uuid(uuid: str, patient_dir: str,
                                  cache_dir: str,
                                  fname_volume: str) -> np.ndarray:
    global _PYRADIOMICS_FEATURE_NAMES
    submod = os.path.splitext(os.path.basename(fname_volume))[0]
    cache_path = os.path.join(cache_dir, f"{uuid}__{submod}.npz")
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            feats = data["feats"].astype(np.float32)
            names = data["names"].tolist()
            if feats.size == 0 or len(names) == 0:
                log.warning("[PyRadiomics] empty cache for %s - recomputing", uuid)
                os.remove(cache_path)
            else:
                if not _PYRADIOMICS_FEATURE_NAMES:
                    _PYRADIOMICS_FEATURE_NAMES = names
                return feats
        except Exception as e:
            log.warning("[PyRadiomics] cache read failed for %s (%s) - recomputing",
                        uuid, e)

    import SimpleITK as sitk
    vol_path = os.path.join(patient_dir, fname_volume)
    vol = np.load(vol_path).astype(np.float32)
    img = sitk.GetImageFromArray(vol)

    mask_arr = np.ones_like(vol, dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(img)

    ext = _get_pyradiomics_extractor()
    result = ext.execute(img, mask)

    pairs = []
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        pairs.append((k, fv))
    pairs.sort(key=lambda kv: kv[0])
    names = [k for k, _ in pairs]
    feats = np.array([v for _, v in pairs], dtype=np.float32)

    if feats.size == 0:
        raise RuntimeError(
            f"PyRadiomics returned 0 features for {uuid}. "
            f"result_keys[:5]={list(result.keys())[:5]}")

    if not _PYRADIOMICS_FEATURE_NAMES:
        _PYRADIOMICS_FEATURE_NAMES = names

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, feats=feats, names=np.array(names, dtype=object))
    return feats


def build_pyradiomics_matrix(dataset: MedicalDataset, indices: list[int],
                             fname_volume: str) -> tuple[np.ndarray, np.ndarray]:
    import time
    logging.getLogger("radiomics").setLevel(logging.WARNING)

    n_total = len(indices)
    n_cached = 0
    n_computed = 0
    t_start = time.time()

    feats_list, labels = [], []
    for ci, i in enumerate(indices, 1):
        row = dataset.df.iloc[i]
        uuid = str(row["UUID"]).strip()
        pdir = dataset.patient_path_map.get(uuid, "")
        cache_path = os.path.join(PYRADIOMICS_CACHE_DIR, f"{uuid}.npz")
        was_cached = os.path.exists(cache_path)

        t0 = time.time()
        try:
            f = _extract_pyradiomics_for_uuid(uuid, pdir,
                                              PYRADIOMICS_CACHE_DIR, fname_volume)
            feats_list.append(f)
            label = 1 if row.get("Group") == "RP" else 0
            labels.append(int(label))
        except Exception as e:
            log.warning("PyRadiomics failed for %s: %s - falling back to zeros", uuid, e)
            n = len(_PYRADIOMICS_FEATURE_NAMES) or 100
            feats_list.append(np.zeros(n, dtype=np.float32))
            labels.append(0)
        dt = time.time() - t0

        if was_cached:
            n_cached += 1
        else:
            n_computed += 1
            elapsed = time.time() - t_start
            est_remaining = elapsed / ci * (n_total - ci) if ci > 0 else 0
            log.info("  [PyRadiomics %d/%d] %s  (%.1fs)  "
                     "elapsed=%.0fs  ETA=%.0fs",
                     ci, n_total, uuid, dt, elapsed, est_remaining)

    if n_cached > 0:
        log.info("  [PyRadiomics] %d/%d patients loaded from cache, %d computed.",
                 n_cached, n_total, n_computed)

    max_len = max((len(f) for f in feats_list), default=0)
    if max_len == 0:
        raise RuntimeError(
            f"PyRadiomics produced no features for any of {n_total} patients. "
            f"Possible causes: stale cache (delete {PYRADIOMICS_CACHE_DIR}), "
            f"pyradiomics version mismatch, or volume/mask incompatibility.")
    feats_list = [np.pad(f, (0, max_len - len(f))) if len(f) < max_len else f
                  for f in feats_list]
    return np.vstack(feats_list), np.array(labels, dtype=np.int64)


def select_top_features(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.feature_selection import SelectKBest, f_classif
    k = min(k, X_train.shape[1])
    sel = SelectKBest(score_func=f_classif, k=k)
    var = X_train.var(axis=0)
    keep = var > 1e-10
    X_train_v = X_train[:, keep]
    X_test_v = X_test[:, keep]
    if X_train_v.shape[1] == 0:
        return X_train, X_test, np.arange(X_train.shape[1])
    k = min(k, X_train_v.shape[1])
    sel.set_params(k=k)
    Xtr_sel = sel.fit_transform(X_train_v, y_train)
    Xte_sel = sel.transform(X_test_v)
    keep_idx = np.where(keep)[0]
    selected = keep_idx[sel.get_support(indices=True)]
    return Xtr_sel, Xte_sel, selected


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def scale_rad(X_tr: np.ndarray,
              X_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_ev), sc


class ResNet18_GAP(nn.Module):
    def __init__(self, dropout_rate: float = DROPOUT, in_chans: int = 1) -> None:
        super().__init__()
        import timm
        self.in_chans = in_chans
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0,
            in_chans=in_chans, features_only=True)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, in_chans, 128, 128))[-1].shape[1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, c, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
            feat = self.pool(self.backbone(x)[-1]).flatten(1)
            feat = feat.view(b, d, -1).mean(dim=1)
        else:
            feat = self.pool(self.backbone(x)[-1]).flatten(1)
        return self.fc(feat)


def _make_resnet18_attn() -> PredictiveResNet:
    return PredictiveResNet(BACKBONE_SMALL, num_classes=2, dropout_rate=DROPOUT,
                            in_chans=1)


def _make_resnet50_attn() -> PredictiveResNet:
    return PredictiveResNet(BACKBONE_LARGE, num_classes=2, dropout_rate=DROPOUT,
                            in_chans=1)


def _make_resnet18_gap() -> ResNet18_GAP:
    return ResNet18_GAP(dropout_rate=DROPOUT, in_chans=1)


class MedicalNet3D_R50(nn.Module):
    def __init__(self, pretrained_path: str = "",
                 dropout_rate: float = DROPOUT, in_chans: int = 1) -> None:
        super().__init__()
        self.in_chans = in_chans
        try:
            from monai.networks.nets.resnet import resnet50 as monai_resnet50
        except ImportError as exc:
            raise ImportError(
                "monai is required for MedicalNet3D_R50. "
                "Install with: pip install monai"
            ) from exc

        self.backbone = monai_resnet50(
            spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="B",
            bias_downsample=False, num_classes=2,
        )

        if in_chans != 1:
            self.input_adapter = nn.Conv3d(in_chans, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0 / in_chans)
        else:
            self.input_adapter = nn.Identity()

        with torch.no_grad():
            _probe = self.backbone(torch.zeros(1, 1, 32, 32, 32))
            if isinstance(_probe, (list, tuple)):
                _probe = _probe[-1]
            if _probe.dim() > 2:
                _probe = _probe.mean(dim=tuple(range(2, _probe.dim())))
            feat_dim = _probe.shape[1]

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 2),
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_medicalnet(pretrained_path)
        elif pretrained_path:
            log.warning("MedicalNet3D_R50: pretrained path not found (%s) - "
                        "using random init.", pretrained_path)
        else:
            log.warning("MedicalNet3D_R50: no pretrained path set - "
                        "using random init (set CT_3D_PRETRAINED_PATH in config).")

    def _load_medicalnet(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        missing, unexpected = self.backbone.load_state_dict(clean, strict=False)
        log.info("MedicalNet3D_R50: loaded weights from %s "
                 "(missing=%d, unexpected=%d)",
                 os.path.basename(path), len(missing), len(unexpected))

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() > 2:
            feat = feat.mean(dim=tuple(range(2, feat.dim())))
        return self.head(feat)


def _make_medicalnet3d_r50() -> MedicalNet3D_R50:
    return MedicalNet3D_R50(pretrained_path=_MEDICALNET_R50_PATH,
                            dropout_rate=DROPOUT, in_chans=1)


class MedicalNet3D_R18(nn.Module):
    def __init__(self, pretrained_path: str = "",
                 dropout_rate: float = DROPOUT, in_chans: int = 1) -> None:
        super().__init__()
        self.in_chans = in_chans
        try:
            from monai.networks.nets.resnet import resnet18 as monai_resnet18
        except ImportError as exc:
            raise ImportError(
                "monai is required for MedicalNet3D_R18. "
                "Install with: pip install monai"
            ) from exc

        self.backbone = monai_resnet18(
            spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="A",
            bias_downsample=False, num_classes=2,
        )

        if in_chans != 1:
            self.input_adapter = nn.Conv3d(in_chans, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0 / in_chans)
        else:
            self.input_adapter = nn.Identity()

        with torch.no_grad():
            _probe = self.backbone(torch.zeros(1, 1, 32, 32, 32))
            if isinstance(_probe, (list, tuple)):
                _probe = _probe[-1]
            if _probe.dim() > 2:
                _probe = _probe.mean(dim=tuple(range(2, _probe.dim())))
            feat_dim = _probe.shape[1]

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 2),
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_medicalnet(pretrained_path)
        else:
            log.warning("MedicalNet3D_R18: no R18 pretrained weights at %r - "
                        "training from random init.", pretrained_path)

    def _load_medicalnet(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        missing, unexpected = self.backbone.load_state_dict(clean, strict=False)
        log.info("MedicalNet3D_R18: loaded weights from %s "
                 "(missing=%d, unexpected=%d)",
                 os.path.basename(path), len(missing), len(unexpected))

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() > 2:
            feat = feat.mean(dim=tuple(range(2, feat.dim())))
        return self.head(feat)


def _make_medicalnet3d_r18() -> MedicalNet3D_R18:
    return MedicalNet3D_R18(pretrained_path=_MEDICALNET_R18_PATH,
                            dropout_rate=DROPOUT, in_chans=1)


class Tiny3DCNN(nn.Module):
    def __init__(self, in_chans: int = 1, dropout_rate: float = DROPOUT) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.features = nn.Sequential(
            nn.Conv3d(in_chans, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

    def pooled_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled_feature(x))


def _make_tiny3dcnn() -> Tiny3DCNN:
    return Tiny3DCNN(in_chans=1, dropout_rate=DROPOUT)


CNN_ARCHITECTURES: dict[str, callable] = {
    "Tiny3DCNN":        _make_tiny3dcnn,
    "ResNet18_GAP":     _make_resnet18_gap,
    "ResNet18_Attn":    _make_resnet18_attn,
    "MedicalNet3D_R18": _make_medicalnet3d_r18,
    "MedicalNet3D_R50": _make_medicalnet3d_r50,
}

TREE_MODELS: dict[str, tuple] = {
}


def optimal_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
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


def _class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=device)
        return w / w.sum() * 2.0
    return torch.ones(2, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_cnn(model: nn.Module, loader: DataLoader,
                 criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, all_probs, all_labels = 0.0, [], []

    for vol_b, lbl in loader:
        vol_b, lbl = vol_b.to(device), lbl.to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
            out0 = model(vol_b)
            loss = criterion(out0, lbl)
            p_sum = torch.softmax(out0, dim=1)[:, 1]
            p_sum = p_sum + torch.softmax(model(torch.flip(vol_b, [3])), dim=1)[:, 1]
            p_sum = p_sum + torch.softmax(model(torch.flip(vol_b, [4])), dim=1)[:, 1]
        probs = (p_sum / 3.0).cpu().numpy()
        total_loss += loss.item() * vol_b.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(lbl.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.array(all_probs), np.array(all_labels, dtype=int)


def evaluate_cnn_mc(model: nn.Module, loader: DataLoader,
                    device: torch.device,
                    n_mc: int = N_MC_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for vol_b, lbl in loader:
            vol_b = vol_b.to(device)
            sample_probs = torch.stack([
                torch.softmax(model(vol_b).cpu(), dim=1)[:, 1]
                for _ in range(n_mc)
            ])
            all_probs.append(sample_probs.mean(0).numpy())
            all_labels.extend(lbl.numpy().tolist())
    return np.concatenate(all_probs), np.array(all_labels, dtype=int)


def _train_cnn(model_factory: callable,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               train_labels: list[int],
               seed_offset: int = 0,
               accumulation: int = ACCUMULATION) -> tuple[nn.Module, float, int]:
    set_seed(SEED + seed_offset)
    model = model_factory().to(device)

    for n, p in model.named_parameters():
        if "backbone" in n:
            p.requires_grad = False

    head_params = [p for n, p in model.named_parameters()
                   if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

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
            for p in backbone_params:
                p.requires_grad = True
            optimizer.add_param_group(
                {"params": backbone_params, "lr": LR * 0.1, "base_lr": LR * 0.1})
            backbone_unfrozen = True
            log.debug("  Epoch %d: backbone unfrozen.", epoch)

        model.train()
        optimizer.zero_grad()
        for step, (vol_b, lbl) in enumerate(train_loader, 1):
            vol_b, lbl = vol_b.to(device), lbl.to(device)

            use_mixup = (P_MIXUP > 0 and MIXUP_ALPHA > 0
                         and vol_b.size(0) > 1
                         and np.random.rand() < P_MIXUP)
            if use_mixup:
                vol_b, y_a, y_b, lam = mixup_batch(vol_b, lbl, MIXUP_ALPHA)
            else:
                y_a, y_b, lam = lbl, lbl, 1.0

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                out = model(vol_b)
                loss = (lam * criterion(out, y_a) +
                        (1 - lam) * criterion(out, y_b)) / accumulation
            if scaler:
                scaler.scale(loss).backward()
                if step % accumulation == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if step % accumulation == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        if len(train_loader) % accumulation != 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
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
            log.debug("  Early stop at epoch %d (no val-AUC gain for %d epochs).", epoch, no_imp)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_thr, epoch_count


FoldResult = dict


def _loader_kwargs(batch_size: int) -> dict:
    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def _shutdown_loaders(*loaders) -> None:
    for ld in loaders:
        if ld is None:
            continue
        it = getattr(ld, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            try:
                it._shutdown_workers()
            except Exception:
                pass
            ld._iterator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_loaders(ds_train, ds_val, ds_train_orig, batch_size=BATCH_SIZE):
    lkw = _loader_kwargs(batch_size)
    return (DataLoader(ds_train, shuffle=True, **lkw),
            DataLoader(ds_val, shuffle=False, **lkw),
            DataLoader(ds_train_orig, shuffle=False, **lkw))


EXPORT_EMB = bool(_data_cfg.get("EXPORT_EMB", False))
EXPORT_EMB_ARCH = str(_data_cfg.get("EXPORT_EMB_ARCH", "Tiny3DCNN"))


def run_cv_cnn(arch_name: str, model_factory: callable,
               dataset_clean: MedicalDataset, dataset_noisy: MedicalDataset,
               dev_indices: list[int], dev_labels: list[int],
               folds: list, device: torch.device,
               test_indices: list[int] | None = None,
               test_labels: list[int] | None = None,
               batch_size: int = BATCH_SIZE,
               accumulation: int = ACCUMULATION):
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
            Subset(dataset_noisy, tr_abs),
            Subset(dataset_clean, val_abs),
            Subset(dataset_clean, tr_abs),
            batch_size=batch_size,
        )

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
    if test_indices is None:
        return results
    tp_ens = np.mean(np.stack(fold_test_probs, 0), 0) if fold_test_probs else None
    return results, tp_ens, np.asarray(oof_vp), np.asarray(oof_vl)


def run_cv_mc_cnn(dataset_clean: MedicalDataset, dataset_noisy: MedicalDataset,
                  dev_indices: list[int], dev_labels: list[int],
                  folds: list, device: torch.device) -> list[FoldResult]:
    log.info("\n%s\n  MC-Dropout CNN (ResNet18_Attn)\n%s", "-" * 60, "-" * 60)
    results = []

    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        set_seed(SEED + fi)
        tr_abs = [dev_indices[i] for i in tr_rel]
        val_abs = [dev_indices[i] for i in val_rel]
        tr_lbl = [dev_labels[i] for i in tr_rel]
        val_lbl = [dev_labels[i] for i in val_rel]

        train_loader, val_loader, tr_loader_o = _make_loaders(
            Subset(dataset_noisy, tr_abs),
            Subset(dataset_clean, val_abs),
            Subset(dataset_clean, tr_abs),
        )

        model, best_thr, epoch_count = _train_cnn(
            _make_resnet18_attn, train_loader, val_loader, device, tr_lbl, seed_offset=fi)

        vp, _ = evaluate_cnn_mc(model, val_loader, device)
        tp, _ = evaluate_cnn_mc(model, tr_loader_o, device)

        val_m = metrics_at_threshold(vp, np.array(val_lbl), best_thr)
        train_m = metrics_at_threshold(tp, np.array(tr_lbl), best_thr)

        log.info("  Fold %d | Train AUC=%.3f F1=%.3f | "
                 "Val AUC=%.3f F1=%.3f Sens=%.3f Spec=%.3f | Ep=%d",
                 fi, train_m["auc"], train_m["f1"],
                 val_m["auc"], val_m["f1"],
                 val_m["sensitivity"], val_m["specificity"], epoch_count)
        results.append({"train": train_m, "val": val_m,
                        "epochs": epoch_count, "threshold": best_thr})
        del model
        _shutdown_loaders(train_loader, val_loader, tr_loader_o)
    return results


def run_cv_ensemble_cnn(dataset_clean: MedicalDataset, dataset_noisy: MedicalDataset,
                        dev_indices: list[int], dev_labels: list[int],
                        folds: list, device: torch.device,
                        n_members: int = N_ENSEMBLE) -> list[FoldResult]:
    name = f"Ensemble_R18x{n_members}"
    log.info("\n%s\n  CNN Ensemble: %s\n%s", "-" * 60, name, "-" * 60)
    results = []

    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        tr_abs = [dev_indices[i] for i in tr_rel]
        val_abs = [dev_indices[i] for i in val_rel]
        tr_lbl = [dev_labels[i] for i in tr_rel]
        val_lbl = [dev_labels[i] for i in val_rel]

        train_loader, val_loader, tr_loader_o = _make_loaders(
            Subset(dataset_noisy, tr_abs),
            Subset(dataset_clean, val_abs),
            Subset(dataset_clean, tr_abs),
        )
        criterion = FocalLoss(alpha=_class_weights(tr_lbl, device), gamma=2)

        member_vp, member_tp = [], []
        for m_idx in range(n_members):
            log.info("    Fold %d  member %d/%d ...", fi, m_idx + 1, n_members)
            model, _, _ = _train_cnn(
                _make_resnet18_attn, train_loader, val_loader,
                device, tr_lbl, seed_offset=fi * 100 + m_idx)
            _, vp, _ = evaluate_cnn(model, val_loader, criterion, device)
            _, tp, _ = evaluate_cnn(model, tr_loader_o, criterion, device)
            member_vp.append(vp)
            member_tp.append(tp)

        avg_vp = np.mean(member_vp, axis=0)
        avg_tp = np.mean(member_tp, axis=0)

        best_thr = optimal_f1_threshold(avg_vp, np.array(val_lbl))
        val_m = metrics_at_threshold(avg_vp, np.array(val_lbl), best_thr)
        train_m = metrics_at_threshold(avg_tp, np.array(tr_lbl), best_thr)

        log.info("  Fold %d | Train AUC=%.3f F1=%.3f | "
                 "Val AUC=%.3f F1=%.3f Sens=%.3f Spec=%.3f",
                 fi, train_m["auc"], train_m["f1"],
                 val_m["auc"], val_m["f1"],
                 val_m["sensitivity"], val_m["specificity"])
        results.append({"train": train_m, "val": val_m,
                        "epochs": "N/A", "threshold": best_thr})
        del model
        _shutdown_loaders(train_loader, val_loader, tr_loader_o)
    return results


def run_cv_tree(name: str, model_cls, model_kwargs: dict,
                X_dev: np.ndarray, y_dev: np.ndarray,
                folds: list) -> list[FoldResult]:
    log.info("\n%s\n  Radiomics Tree: %s\n%s", "-" * 60, name, "-" * 60)
    results = []
    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        set_seed(SEED + fi)
        Xtr, Xval = X_dev[tr_rel], X_dev[val_rel]
        ytr, yval = y_dev[tr_rel], y_dev[val_rel]

        Xtr_sc, Xval_sc, _ = scale_rad(Xtr, Xval)

        model = model_cls(**model_kwargs)
        model.fit(Xtr_sc, ytr)

        vp = model.predict_proba(Xval_sc)[:, 1]
        tp = model.predict_proba(Xtr_sc)[:, 1]

        thr = optimal_f1_threshold(vp, yval)
        val_m = metrics_at_threshold(vp, yval, thr)
        train_m = metrics_at_threshold(tp, ytr, thr)

        log.info("  Fold %d | Train AUC=%.3f F1=%.3f | "
                 "Val AUC=%.3f F1=%.3f Sens=%.3f Spec=%.3f",
                 fi, train_m["auc"], train_m["f1"],
                 val_m["auc"], val_m["f1"],
                 val_m["sensitivity"], val_m["specificity"])
        results.append({"train": train_m, "val": val_m,
                        "epochs": "N/A", "threshold": thr})
    return results


def _carve_dev_val_ct(dev_indices: list[int], dev_labels: list[int],
                      manifest_df: "pd.DataFrame") -> tuple[list[int], list[int],
list[int], list[int]]:
    if "IsInnerVal" in manifest_df.columns:
        val_mask = manifest_df.loc[dev_indices, "IsInnerVal"].fillna(False).astype(bool).values
    else:
        from sklearn.model_selection import train_test_split as _tts
        val_frac = float(_full_cfg["SPLIT"].get("INNER_VAL_FRAC", 0.15))
        n = len(dev_indices)
        _, val_rel = _tts(range(n), test_size=val_frac,
                          stratify=dev_labels, random_state=SEED)
        val_set = set(val_rel)
        val_mask = [i in val_set for i in range(n)]
    tr_idx = [dev_indices[i] for i, v in enumerate(val_mask) if not v]
    val_idx = [dev_indices[i] for i, v in enumerate(val_mask) if v]
    tr_lab = [dev_labels[i] for i, v in enumerate(val_mask) if not v]
    val_lab = [dev_labels[i] for i, v in enumerate(val_mask) if v]
    return tr_idx, val_idx, tr_lab, val_lab


def eval_on_test_mc_cnn(dataset_clean: MedicalDataset, dataset_noisy: MedicalDataset,
                        dev_indices: list[int], dev_labels: list[int],
                        test_indices: list[int], test_labels: list[int],
                        mean_thr: float, device: torch.device,
                        manifest_df: "pd.DataFrame") -> tuple[dict, np.ndarray]:
    set_seed(SEED)
    tr_idx, val_idx, tr_lab, _ = _carve_dev_val_ct(dev_indices, dev_labels, manifest_df)
    lkw = _loader_kwargs(BATCH_SIZE)
    test_lkw = {**lkw, "persistent_workers": False}
    train_loader = DataLoader(Subset(dataset_noisy, tr_idx), shuffle=True, **lkw)
    val_loader = DataLoader(Subset(dataset_clean, val_idx), shuffle=False, **lkw)
    test_loader = DataLoader(Subset(dataset_clean, test_indices), shuffle=False, **test_lkw)

    model, _, _ = _train_cnn(_make_resnet18_attn, train_loader, val_loader,
                             device, tr_lab, seed_offset=0)
    tp, _ = evaluate_cnn_mc(model, test_loader, device)
    metrics = metrics_at_threshold(tp, np.array(test_labels), mean_thr)
    del model
    _shutdown_loaders(train_loader, val_loader, test_loader)
    return metrics, tp


def eval_on_test_ensemble_cnn(dataset_clean: MedicalDataset, dataset_noisy: MedicalDataset,
                              dev_indices: list[int], dev_labels: list[int],
                              test_indices: list[int], test_labels: list[int],
                              mean_thr: float, device: torch.device,
                              manifest_df: "pd.DataFrame",
                              n_members: int = N_ENSEMBLE) -> tuple[dict, np.ndarray]:
    tr_idx, val_idx, tr_lab, _ = _carve_dev_val_ct(dev_indices, dev_labels, manifest_df)
    lkw = _loader_kwargs(BATCH_SIZE)
    test_lkw = {**lkw, "persistent_workers": False}
    train_loader = DataLoader(Subset(dataset_noisy, tr_idx), shuffle=True, **lkw)
    val_loader = DataLoader(Subset(dataset_clean, val_idx), shuffle=False, **lkw)
    test_loader = DataLoader(Subset(dataset_clean, test_indices), shuffle=False, **test_lkw)
    criterion = FocalLoss(alpha=_class_weights(dev_labels, device), gamma=2)

    member_probs = []
    for m_idx in range(n_members):
        model, _, _ = _train_cnn(_make_resnet18_attn, train_loader, val_loader,
                                 device, tr_lab, seed_offset=m_idx)
        _, tp, _ = evaluate_cnn(model, test_loader, criterion, device)
        member_probs.append(tp)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_tp = np.mean(member_probs, axis=0)
    metrics = metrics_at_threshold(avg_tp, np.array(test_labels), mean_thr)
    _shutdown_loaders(train_loader, val_loader, test_loader)
    return metrics, avg_tp


def eval_on_test_tree(model_cls, model_kwargs: dict,
                      X_dev: np.ndarray, y_dev: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      mean_thr: float) -> tuple[dict, np.ndarray]:
    set_seed(SEED)
    Xd_sc, Xt_sc, _ = scale_rad(X_dev, X_test)
    model = model_cls(**model_kwargs)
    model.fit(Xd_sc, y_dev)
    tp = model.predict_proba(Xt_sc)[:, 1]
    return metrics_at_threshold(tp, y_test, mean_thr), tp


def shap_for_tree(model_cls, model_kwargs: dict,
                  X_sc: np.ndarray, y: np.ndarray,
                  labels: list[str], save_dir: str) -> None:
    log.info("SHAP (TreeExplainer) on radiomics features ...")
    set_seed(SEED)
    model = model_cls(**model_kwargs)
    model.fit(X_sc, y)
    sv = shap.TreeExplainer(model).shap_values(X_sc)
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[..., 1]
    _save_shap_plots(sv, X_sc, labels, save_dir)


def _save_shap_plots(shap_vals: np.ndarray, X: np.ndarray,
                     labels: list[str], save_dir: str) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh([labels[i] for i in order[::-1]], mean_abs[order[::-1]],
                   color="#2980b9")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (impact on RP+ probability)")
    ax.set_title("Feature Importance (SHAP) - CT Radiomics Classifier")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=150)
    plt.close(fig)

    try:
        exp = shap.Explanation(values=shap_vals, data=X, feature_names=labels)
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(exp, max_display=len(labels), show=False, plot_size=None)
        plt.title("SHAP Beeswarm - CT Radiomics Classifier")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "shap_beeswarm.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning("Beeswarm skipped: %s", e)

    log.info("\n  SHAP Radiomics Ranking:")
    for rank, i in enumerate(order, 1):
        log.info("  %2d. %-22s  %.5f", rank, labels[i], mean_abs[i])


METRICS_KEYS = ["auc", "f1", "sensitivity", "specificity", "acc"]
METRICS_DISP = ["AUC", "F1", "Sens", "Spec", "Acc"]


def _mean_std(results: list[FoldResult], split: str) -> dict:
    return {m: (np.mean([r[split][m] for r in results]),
                np.std([r[split][m] for r in results]))
            for m in METRICS_KEYS}


def print_fold_table(name: str, results: list[FoldResult]) -> None:
    header = (f"  {'Fold':>4}  {'Split':>5}  " +
              "  ".join(f"{d:>6}" for d in METRICS_DISP) + "  Epochs")
    sep = "  " + "-" * (len(header) - 2)
    log.info("\n  -- %s %s", name, "-" * max(0, 55 - len(name)))
    log.info(header)
    log.info(sep)
    for fi, r in enumerate(results, 1):
        splits = ("train", "val", "test") if "test" in r else ("train", "val")
        for split in splits:
            m = r[split]
            ep = str(r["epochs"]) if split == "val" else ""
            row = (f"  {fi:>4}  {split:>5}  " +
                   "  ".join(f"{m[k]:.3f}" for k in METRICS_KEYS) +
                   f"  {ep}")
            log.info(row)
        log.info(sep)
    splits_avg = ("train", "val", "test") if results and "test" in results[0] else ("train", "val")
    for split in splits_avg:
        ms = _mean_std(results, split)
        row = (f"  {'Avg':>4}  {split:>5}  " +
               "  ".join(f"{ms[k][0]:.3f}" for k in METRICS_KEYS))
        if split == "val":
            ep_vals = [r["epochs"] for r in results if isinstance(r["epochs"], int)]
            if ep_vals:
                row += f"  Avg Ep={np.mean(ep_vals):.0f}"
        log.info(row)


def print_summary_table(all_results: dict, test_results: dict) -> None:
    W = 20
    header = (f"{'Model':<{W}}" +
              "".join(f"  {'Val ' + m:>12}" for m in METRICS_DISP) +
              "".join(f"  {'Test ' + m:>12}" for m in METRICS_DISP[:2]) +
              f"  {'Avg Ep':>7}")
    sep = "-" * len(header)
    log.info("\n\n%s", "=" * len(header))
    log.info("  5-Fold CV Summary  +  Test Set Metrics  (sorted by Test AUC)")
    log.info("=" * len(header))
    log.info(header)
    log.info(sep)

    sorted_names = sorted(all_results,
                          key=lambda n: test_results.get(n, {}).get("auc", -1.0),
                          reverse=True)
    best_name = max(all_results, key=lambda n: np.mean([r["val"]["auc"] for r in all_results[n]]))
    best_val_auc = np.mean([r["val"]["auc"] for r in all_results[best_name]])
    best_test_auc = -1.0
    for name in sorted_names:
        rs = all_results[name]
        ms = _mean_std(rs, "val")
        val_str = "".join(f"  {ms[k][0]:.3f}+/-{ms[k][1]:.3f}" for k in METRICS_KEYS)
        tm = test_results.get(name, {})
        test_auc = tm.get("auc", float("nan"))
        test_str = "".join(f"  {tm.get(m, float('nan')):.3f}      " for m in METRICS_KEYS[:2])
        ep_vals = [r["epochs"] for r in rs if isinstance(r["epochs"], int)]
        ep_str = f"  {np.mean(ep_vals):.0f}" if ep_vals else "  N/A"
        tag = "  <- BEST" if (test_auc == test_auc and test_auc > best_test_auc) else ""
        if test_auc == test_auc and test_auc > best_test_auc:
            best_test_auc = test_auc
        log.info(f"{name:<{W}}{val_str}{test_str}{ep_str}{tag}")

    log.info(sep)
    log.info("  Best model (by val AUC): %s  (%.3f)", best_name, best_val_auc)
    log.info("=" * len(header))


def save_patient_xlsx(model_name: str,
                      patient_uuids: list[str],
                      y_test: np.ndarray,
                      probs: np.ndarray,
                      threshold: float,
                      threshold_y: float | None = None) -> None:
    preds = (probs >= threshold).astype(int)
    cols = {
        "UUID": patient_uuids,
        "TrueLabel": y_test,
        "TrueLabel_text": ["RP" if v else "No RP" for v in y_test],
        "RP_Probability": np.round(probs, 4),
        "PredictedLabel": preds,
        "Predicted_text": ["RP" if v else "No RP" for v in preds],
        "Correct": (preds == y_test).astype(int),
    }
    if threshold_y is not None:
        cols["PredictedLabel_Youden"] = (probs >= threshold_y).astype(int)
        cols["Threshold_Youden"] = threshold_y
    df_out = pd.DataFrame(cols)
    safe = model_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(PRED_DIR, f"{safe}_test_predictions.xlsx")
    df_out.to_excel(path, index=False)


def generate_report_md(results_dir: str,
                       all_results: dict,
                       test_results: dict,
                       n_total: int, n_dev: int, n_test: int,
                       rp_rate: float, rp_rate_dev: float, rp_rate_test: float,
                       best_name: str,
                       submodality: str = "lung_only_no_vessels") -> str:
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")

    def fmt(v):
        if isinstance(v, float) and v != v:
            return "N/A"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    def ms(rs, split, key):
        vals = [r[split][key] for r in rs]
        return np.mean(vals), np.std(vals)

    sorted_names = sorted(all_results,
                          key=lambda n: np.mean([r["val"]["auc"] for r in all_results[n]]),
                          reverse=True)
    bv_auc, bv_std = ms(all_results[best_name], "val", "auc")
    bv_f1, _ = ms(all_results[best_name], "val", "f1")
    bv_sen, _ = ms(all_results[best_name], "val", "sensitivity")
    bv_spe, _ = ms(all_results[best_name], "val", "specificity")
    bv_acc, _ = ms(all_results[best_name], "val", "acc")
    bt = test_results.get(best_name, {})

    model_desc = {
        "Tiny3DCNN": "Tiny 3D CNN baseline (~500K params, 3 conv blocks, trained from scratch)",
        "ResNet18_Attn": "ResNet18 (ImageNet) + SliceAttention aggregator, 2D-slice pipeline",
        "ResNet50_Attn": "ResNet50 (ImageNet) + SliceAttention aggregator, 2D-slice pipeline",
        "ResNet18_GAP": "ResNet18 + GlobalAvgPool slice aggregation (ablation - no attention)",
        "MedicalNet3D_R18": "3D ResNet-18 (smaller capacity than R50; pretrained MedicalNet weights when available)",
        "MedicalNet3D_R50": "3D ResNet-50 pretrained on 23 medical datasets (MedicalNet/Tencent)",
        "MCDropout_R18": "ResNet18_Attn with MC-Dropout Bayesian inference (30 forward passes)",
        f"Ensemble_R18x{N_ENSEMBLE}": f"Ensemble of {N_ENSEMBLE}x ResNet18_Attn (seed-averaged)",
        "RF": "Random Forest (500 trees, balanced, radiomics features)",
        "ExtraTrees": "Extra-Trees (500 trees, balanced, radiomics features)",
        "GradBoost": "Gradient Boosting (300 estimators, lr=0.05, radiomics features)",
        "XGBoost": "XGBoost (300 rounds, radiomics features)",
        "LightGBM": "LightGBM (300 iterations, radiomics features)",
        "SVM_RBF": "SVM with RBF kernel (balanced, radiomics features)",
    }

    lines = []
    a = lines.append
    a("---")
    a(f'title: "Radiation Pneumonitis Prediction - Pre-RT CT Imaging (submodality: {submodality})"')
    a('subtitle: "5-Fold Cross-Validation Benchmark Report"')
    a(f'date: "{today}"')
    a('fontsize: 11pt')
    a('geometry: "margin=2.5cm"')
    a('colorlinks: true')
    a('linkcolor: blue')
    a("---")
    a("")
    a("# Executive Summary")
    a("")
    a("This report benchmarks machine learning models for binary prediction of "
      f"**Radiation Pneumonitis (RP)** from pre-RT CT imaging - submodality `{submodality}`. "
      f"A total of {n_total} patients were evaluated with the same 80/20 stratified "
      f"held-out split as experiments 6.1 and 6.2 ({n_dev} development, {n_test} held-out test). "
      f"All CNN models process 128x128x128 volumes via a 2D-slice pipeline "
      "(ResNet backbone + 3D aggregator). Radiomics baselines use 22 intensity/histogram "
      "statistics extracted from each volume.")
    a("")
    a(f"**Best model:** `{best_name}`")
    a("")
    a("| Metric | Validation (mean ± std) | Test |")
    a("|--------|------------------------|------|")
    a(f"| AUC    | {bv_auc:.3f} ± {bv_std:.3f} | {fmt(bt.get('auc', float('nan')))} |")
    a(f"| F1     | {fmt(bv_f1)} | {fmt(bt.get('f1', float('nan')))} |")
    a(f"| Sensitivity | {fmt(bv_sen)} | {fmt(bt.get('sensitivity', float('nan')))} |")
    a(f"| Specificity | {fmt(bv_spe)} | {fmt(bt.get('specificity', float('nan')))} |")
    a(f"| Accuracy    | {fmt(bv_acc)} | {fmt(bt.get('acc', float('nan')))} |")
    a("")
    a("# 1. Dataset")
    a("")
    a(f"Pre-RT CT volumes for submodality `{submodality}` were used for all patients. "
      "Volumes were normalised using a lung window "
      f"(centre={WINDOW_CENTER} HU, width={WINDOW_WIDTH} HU) and resized to "
      f"{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]} voxels.")
    a("")
    a("| Split | N | RP+ (%) |")
    a("|-------|---|---------|")
    a(f"| Total | {n_total} | {100 * rp_rate:.1f}% |")
    a(f"| Development (80%) | {n_dev} | {100 * rp_rate_dev:.1f}% |")
    a(f"| Held-out Test (20%) | {n_test} | {100 * rp_rate_test:.1f}% |")
    a("")
    a("The held-out test set was constructed by aligning patient identifiers "
      "with the Excel-based split used in experiments 6.1 and 6.2, ensuring "
      "direct comparability across modalities.")
    a("")
    a("# 2. Features")
    a("")
    a("## 2.1 CNN Pipeline Features")
    a("")
    a("CNN models learn features end-to-end from raw (normalised, resized) CT volumes. "
      "No hand-crafted features are required. Augmentation during training: "
      "random flips, rotations, elastic deformations, gamma correction.")
    a("")
    a("## 2.2 Radiomics Features (classical ML baselines)")
    a("")
    a(f"A total of **{len(RADIOMICS_FEATURE_NAMES)} intensity / histogram features** "
      "were extracted from each volume:")
    a("")
    for lbl in RADIOMICS_FEATURE_NAMES:
        a(f"- {lbl}")
    a("")
    a("# 3. Methods")
    a("")
    a("## 3.1 Experimental Setup")
    a("")
    a(f"- **Split:** Same held-out 20% test as 6.1 / 6.2 (patient-level alignment via UUID)")
    a(f"- **CV:** {N_FOLDS}-fold stratified CV on development set")
    a(f"- **Class imbalance:** Inverse-frequency class weights in FocalLoss (CNN); "
      "balanced class weights (tree models)")
    a(f"- **Threshold:** Youden-J optimal F1 threshold per fold; mean used at test time")
    a(f"- **Seed:** {SEED}")
    a("")
    a("## 3.2 CNN Training")
    a("")
    a(f"- **Architecture:** 2D-slice ResNet (timm, ImageNet pretrained, 1-channel) "
      "+ aggregator (SliceAttention or GlobalAvgPool)")
    a(f"- **Optimiser:** AdamW, lr={LR}, weight\\_decay={WEIGHT_DECAY}")
    a(f"- **Warmup:** {WARMUP_EPOCHS} epochs (backbone frozen, head only), "
      "then full unfreeze at 0.1x LR")
    a(f"- **Loss:** FocalLoss (gamma=2) with inverse-frequency class weights")
    a(f"- **Max epochs:** {MAX_EPOCHS}, gradient accumulation steps={ACCUMULATION}")
    a(f"- **Scheduler:** ReduceLROnPlateau (patience=7, factor=0.5) + NSS early stopping")
    a(f"- **TTA:** 3-view (original + H-flip + W-flip) at evaluation")
    a(f"- **MC Dropout:** {N_MC_SAMPLES} stochastic forward passes (BN in eval, Dropout active)")
    a(f"- **Ensemble:** {N_ENSEMBLE}x ResNet18_Attn with different random seeds, probability averaged")
    a("")
    a("## 3.3 Radiomics + Classical ML")
    a("")
    a("Radiomics features were StandardScaler-normalised per fold. "
      "Tree models trained on raw (unaugmented) volume statistics.")
    a("")
    a("## 3.4 Models Evaluated")
    a("")
    a("| Model | Description |")
    a("|-------|-------------|")
    for name in sorted_names:
        desc = model_desc.get(name, name)
        a(f"| `{name}` | {desc} |")
    a("")
    a("# 4. Cross-Validation Results")
    a("")
    a("## 4.1 Summary Table (sorted by Validation AUC)")
    a("")
    a("| Model | Val AUC | Val F1 | Val Sens | Val Spec | Test AUC | Test F1 |")
    a("|-------|---------|--------|----------|----------|----------|---------|")
    for name in sorted_names:
        rs = all_results[name]
        v_auc, v_auc_s = ms(rs, "val", "auc")
        v_f1, _ = ms(rs, "val", "f1")
        v_sen, _ = ms(rs, "val", "sensitivity")
        v_spe, _ = ms(rs, "val", "specificity")
        tm = test_results.get(name, {})
        star = " **\\*\\***" if name == best_name else ""
        a(f"| `{name}`{star} | {v_auc:.3f}±{v_auc_s:.3f} | {fmt(v_f1)} | "
          f"{fmt(v_sen)} | {fmt(v_spe)} | "
          f"{fmt(tm.get('auc', float('nan')))} | "
          f"{fmt(tm.get('f1', float('nan')))} |")
    a("")
    a("\\*\\* Best model by validation AUC.")
    a("")
    a(f"## 4.2 Best Model Fold Detail: `{best_name}`")
    a("")
    a("| Fold | Split | AUC | F1 | Sens | Spec | Acc |")
    a("|------|-------|-----|----|------|------|-----|")
    for fi, r in enumerate(all_results[best_name], 1):
        for split in ("train", "val"):
            m = r[split]
            ep_str = (f" (ep {r['epochs']})" if split == "val"
                                                and isinstance(r["epochs"], int) else "")
            a(f"| {fi} | {split}{ep_str} | {fmt(m['auc'])} | {fmt(m['f1'])} | "
              f"{fmt(m['sensitivity'])} | {fmt(m['specificity'])} | {fmt(m['acc'])} |")
    a("")
    a("## 4.3 Held-out Test Set Results (all models)")
    a("")
    a("| Model | AUC | F1 | Sensitivity | Specificity | Accuracy |")
    a("|-------|-----|----|-------------|-------------|----------|")
    for name in sorted_names:
        tm = test_results.get(name, {})
        a(f"| `{name}` | {fmt(tm.get('auc', float('nan')))} | "
          f"{fmt(tm.get('f1', float('nan')))} | "
          f"{fmt(tm.get('sensitivity', float('nan')))} | "
          f"{fmt(tm.get('specificity', float('nan')))} | "
          f"{fmt(tm.get('acc', float('nan')))} |")
    a("")
    a("# 5. Feature Importance (SHAP - Radiomics Best Model)")
    a("")
    a("SHAP (SHapley Additive exPlanations) values were computed for the best-performing "
      "radiomics / tree model using TreeExplainer on the development set.")
    a("")
    shap_bar = os.path.join(results_dir, "shap_bar.png")
    shap_bee = os.path.join(results_dir, "shap_beeswarm.png")
    if os.path.exists(shap_bar):
        a("![SHAP Feature Importance Bar Chart](shap_bar.png)")
        a("")
    if os.path.exists(shap_bee):
        a("![SHAP Beeswarm Plot](shap_beeswarm.png)")
        a("")
    a("# 6. Conclusion")
    a("")
    a(f"Across {len(sorted_names)} model configurations applied to pre-RT CT volumes, "
      f"**`{best_name}`** achieved the highest validation AUC of "
      f"**{bv_auc:.3f} ± {bv_std:.3f}** "
      f"and a test AUC of **{fmt(bt.get('auc', float('nan')))}** on the held-out set. "
      "Comparing with RT (6.1) and clinical (6.2) experiments allows assessment of the "
      "incremental predictive value of CT imaging versus dosimetric and demographic data. "
      "Results should be interpreted cautiously given the limited sample size.")
    a("")
    a("---")
    a(f"*Report generated automatically on {today}.*")

    md_path = os.path.join(results_dir, "CT_RP_Prediction_Report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("Markdown report written -> %s", md_path)
    return md_path


_META_SCHEMA_VERSION = 3
_FOLD_SPLIT_KEYS = ("auc", "f1", "sensitivity", "specificity", "acc")


def _save_variant_meta_ct(variant: str, cv_results: list,
                          op_threshold: float | None = None,
                          op_threshold_y: float | None = None) -> None:
    meta = {
        "schema_version": _META_SCHEMA_VERSION,
        "fold_thresholds": [r["threshold"] for r in cv_results],
        "fold_epochs": [r["epochs"] for r in cv_results],
        "fold_val": [{k: r["val"].get(k, 0.0) for k in _FOLD_SPLIT_KEYS}
                     for r in cv_results],
        "fold_train": [{k: r["train"].get(k, 0.0) for k in _FOLD_SPLIT_KEYS}
                       for r in cv_results],
        "op_threshold": op_threshold,
        "op_threshold_youden": op_threshold_y,
    }
    try:
        with open(os.path.join(PRED_DIR, f"{variant}_cv_meta.json"), "w") as fp:
            json.dump(meta, fp)
    except Exception as exc:
        log.warning("[RESUME] Could not save meta for %s: %s", variant, exc)


def _meta_is_complete(meta: dict) -> bool:
    if meta.get("schema_version") != _META_SCHEMA_VERSION:
        return False
    fold_val = meta.get("fold_val") or []
    if not fold_val:
        return False
    for split in fold_val:
        vals = [split.get(k, 0.0) for k in _FOLD_SPLIT_KEYS]
        if all(v == 0.0 for v in vals):
            return False
    return True


def _try_load_variant_ct(variant: str,
                         test_labels: list[int]) -> tuple[dict, np.ndarray, list] | None:
    xlsx_path = os.path.join(PRED_DIR, f"{variant}_test_predictions.xlsx")
    meta_path = os.path.join(PRED_DIR, f"{variant}_cv_meta.json")

    if not os.path.exists(xlsx_path) or not os.path.exists(meta_path):
        return None

    try:
        meta = json.load(open(meta_path))
    except Exception as exc:
        log.warning("  [RESUME] Failed to read meta for %s: %s", variant, exc)
        return None

    if not _meta_is_complete(meta):
        log.info("  [RESUME] %s: cache is incomplete (zero/missing val metrics) - "
                 "retraining to refresh cache.", variant)
        return None

    try:
        df_pred = pd.read_excel(xlsx_path)
        probs = df_pred["RP_Probability"].values.astype(float)
        y = np.array(test_labels, dtype=int)
        thr = meta.get("op_threshold")
        if thr is None:
            thr = optimal_f1_threshold(probs, y)
        metrics = metrics_at_threshold(probs, y, float(thr))

        fold_val = meta["fold_val"]
        fold_train = meta["fold_train"]
        fold_thr = meta["fold_thresholds"]
        fold_epochs = meta["fold_epochs"]

        cv_records = [
            {"val": {k: float(fv[k]) for k in _FOLD_SPLIT_KEYS},
             "train": {k: float(ft[k]) for k in _FOLD_SPLIT_KEYS},
             "epochs": fe, "threshold": float(fth), "state_dict": {}}
            for fv, ft, fth, fe in zip(fold_val, fold_train, fold_thr, fold_epochs)
        ]
        log.info("  [RESUME] %s: loaded from cache (Test AUC=%.3f, %d folds)",
                 variant, metrics["auc"], len(cv_records))
        return metrics, probs, cv_records
    except Exception as exc:
        log.warning("  [RESUME] Failed to load cache for %s: %s", variant, exc)
        return None


def main() -> None:
    global RADIOMICS_LABELS
    RADIOMICS_LABELS = RADIOMICS_FEATURE_NAMES[:]

    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    sub_name = CURRENT_SUBMODALITY
    gen_key = CURRENT_GEN_KEY
    log.info("\n%s\n  CT-Plan submodality: %s  (key=%s, modality_subdir=%s)\n%s",
             "=" * 60, sub_name, gen_key, CT_MODALITY_SUBDIR, "=" * 60)

    manifest_df_raw = read_manifest()
    log.info("Manifest: %d rows", len(manifest_df_raw))

    aug_cfg = _full_cfg["TRAIN"]["AUGMENTATION"]
    gen_cfg_base = _full_cfg["PREPROCESS"]["GENERATED_FILES"]
    gen_cfg = _gen_cfg_for_submodality(gen_cfg_base, gen_key)
    if gen_cfg is None:
        log.warning("Submodality %s: required key %s missing from "
                    "PREPROCESS.GENERATED_FILES - aborting this submodality.",
                    sub_name, gen_key)
        return
    if not _submodality_available(sub_name, gen_cfg, _proc_dir, gen_key):
        log.warning("Submodality %s: not enough patients with %s on disk "
                    "- aborting this submodality.",
                    sub_name, gen_cfg.get(gen_key))
        return

    sub_dose_crop = DOSE_WEIGHTED_CROP and (sub_name not in _DOSE_CROP_INCOMPATIBLE)
    if DOSE_WEIGHTED_CROP and not sub_dose_crop:
        log.info("Submodality %s: DOSE_WEIGHTED_CROP forced off "
                 "(only meaningful for lung-bbox-cropped volumes).", sub_name)

    transforms_clean = MedicalTransforms3D(
        mode="standard",
        p_flip=aug_cfg["P_FLIP"],
        p_rot=aug_cfg["P_ROT"],
        rot_range=aug_cfg["ROT_RANGE"],
        img_size=TARGET_SHAPE[0],
    )
    transforms_noisy = MedicalTransforms3D(
        mode="heavy",
        p_flip=aug_cfg["P_FLIP"],
        p_rot=aug_cfg["P_ROT"],
        rot_range=aug_cfg["ROT_RANGE"],
        p_elastic=aug_cfg["P_ELASTIC"],
        elastic_alpha=aug_cfg["ELASTIC_ALPHA"],
        elastic_sigma=aug_cfg["ELASTIC_SIGMA"],
        img_size=TARGET_SHAPE[0],
    )

    tumor_side_map: dict[str, int] = {}
    if sub_dose_crop:
        tumor_side_map = build_tumor_side_map(_input_dir, CT_EXCEL_PATH)
    log.info("CT input: before-RT only  dose_crop=%s (%d UUIDs)  in_chans=1",
             sub_dose_crop, len(tumor_side_map))

    ds_kwargs = dict(
        df=manifest_df_raw,
        root_dir=_proc_dir,
        generated_files_config=gen_cfg,
        target_shape=TARGET_SHAPE,
        window_width=WINDOW_WIDTH,
        window_center=WINDOW_CENTER,
        cache_data=True,
        tumor_side_map=tumor_side_map,
        dose_weighted_crop=sub_dose_crop,
        crop_keep_fraction=CROP_KEEP_FRACTION,
    )
    dataset_noisy = MedicalDataset(**ds_kwargs, transform=transforms_noisy)
    dataset_clean = MedicalDataset(**ds_kwargs, transform=transforms_clean)

    manifest_df = dataset_clean.df
    log.info("Patients on disk: %d / %d", len(manifest_df), len(manifest_df_raw))

    dev_indices, test_indices, folds = resolve_ct_split(manifest_df)

    all_labels = [1 if g == "RP" else 0 for g in manifest_df["Group"].values]
    dev_labels = [all_labels[i] for i in dev_indices]
    test_labels = [all_labels[i] for i in test_indices]

    test_uuids = [manifest_df.iloc[i]["UUID"] for i in test_indices]

    all_labels = list(dev_labels) + list(test_labels)
    rp_rate = np.mean(all_labels) if all_labels else 0
    rp_rate_dev = np.mean(dev_labels) if dev_labels else 0
    rp_rate_test = np.mean(test_labels) if test_labels else 0

    log.info("Dev=%d (RP+=%.1f%%)  Test=%d (RP+=%.1f%%)",
             len(dev_indices), 100 * rp_rate_dev,
             len(test_indices), 100 * rp_rate_test)

    all_results: dict[str, list[FoldResult]] = {}
    test_results: dict[str, dict] = {}

    for arch_name, factory in CNN_ARCHITECTURES.items():
        _cached = _try_load_variant_ct(arch_name, test_labels)
        if _cached is not None:
            test_results[arch_name], tp, all_results[arch_name] = _cached
            log.info("  -> Test  AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f  [cached]",
                     *[test_results[arch_name][k]
                       for k in ["auc", "f1", "sensitivity", "specificity"]])
            continue

        arch_overrides = ARCH_BATCH_OVERRIDES.get(arch_name, {})
        arch_bs = arch_overrides.get("batch_size", BATCH_SIZE)
        arch_acc = arch_overrides.get("accumulation", ACCUMULATION)
        fold_results, tp_ens, oof_vp, oof_vl = run_cv_cnn(
            arch_name, factory,
            dataset_clean, dataset_noisy,
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
        _save_variant_meta_ct(arch_name, fold_results, op_threshold=thr, op_threshold_y=thr_y)
        log.info("  -> Test (5-model ens+Platt) AUC=%.3f | F1@%.2f: F1=%.3f Sens=%.3f Spec=%.3f"
                 " | Youden@%.2f: Sens=%.3f Spec=%.3f",
                 test_results[arch_name]["auc"], thr, test_results[arch_name]["f1"],
                 test_results[arch_name]["sensitivity"], test_results[arch_name]["specificity"],
                 thr_y, test_m_y["sensitivity"], test_m_y["specificity"])

    if USE_PYRADIOMICS and sub_name == "whole_body":
        log.info("\nSkipping PyRadiomics for submodality 'whole_body' "
                 "(dilute signal, expensive). Lung-cropped submodalities still "
                 "run radiomics + tree models.")
    if USE_PYRADIOMICS and sub_name != "whole_body":
        log.info("\nExtracting PyRadiomics features (dev=%d, test=%d, cache=%s) ...",
                 len(dev_indices), len(test_indices), PYRADIOMICS_CACHE_DIR)
        rad_fname = gen_cfg["Active_Before_File"]
        try:
            X_dev_full, y_dev = build_pyradiomics_matrix(dataset_clean, dev_indices, rad_fname)
            X_test_full, y_test = build_pyradiomics_matrix(dataset_clean, test_indices, rad_fname)
            log.info("PyRadiomics matrix: dev=%s test=%s",
                     X_dev_full.shape, X_test_full.shape)
            X_dev, X_test, sel_idx = select_top_features(
                X_dev_full, y_dev, X_test_full, k=PYRADIOMICS_TOP_K)
            if _PYRADIOMICS_FEATURE_NAMES:
                RADIOMICS_LABELS = [_PYRADIOMICS_FEATURE_NAMES[i] for i in sel_idx]
            log.info("PyRadiomics: selected top-%d / %d features by ANOVA F-score",
                     X_dev.shape[1], X_dev_full.shape[1])
        except Exception as e:
            log.warning("PyRadiomics failed (%s) - falling back to hand-rolled.", e)
            X_dev, y_dev = build_radiomics_matrix(dataset_clean, dev_indices)
            X_test, y_test = build_radiomics_matrix(dataset_clean, test_indices)
    else:
        log.info("\nExtracting radiomics features (dev=%d, test=%d) ...",
                 len(dev_indices), len(test_indices))
        X_dev, y_dev = build_radiomics_matrix(dataset_clean, dev_indices)
        X_test, y_test = build_radiomics_matrix(dataset_clean, test_indices)
    log.info("Radiomics matrix: dev=%s  test=%s", X_dev.shape, X_test.shape)

    for name, (cls, kwargs) in TREE_MODELS.items():
        _cached_tree = _try_load_variant_ct(name, test_labels)
        if _cached_tree is not None:
            test_results[name], tp_t, all_results[name] = _cached_tree
            log.info("  -> Test  AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f  [cached]",
                     *[test_results[name][k]
                       for k in ["auc", "f1", "sensitivity", "specificity"]])
            continue

        tree_folds = [(tr_rel, val_rel) for tr_rel, val_rel in folds]
        all_results[name] = run_cv_tree(name, cls, kwargs, X_dev, y_dev, tree_folds)
        print_fold_table(name, all_results[name])
        mean_thr = float(np.mean([r["threshold"] for r in all_results[name]]))
        test_results[name], tp_t = eval_on_test_tree(
            cls, kwargs, X_dev, y_dev, X_test, y_test, mean_thr)
        save_patient_xlsx(name, test_uuids, np.array(test_labels), tp_t, mean_thr)
        _save_variant_meta_ct(name, all_results[name], op_threshold=mean_thr)
        log.info("  -> Test  AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f",
                 *[test_results[name][k]
                   for k in ["auc", "f1", "sensitivity", "specificity"]])

    print_summary_table(all_results, test_results)
    import util
    util.print_array_metrics(all_results, test_results, log)

    rows = []
    for name, rs in all_results.items():
        ms_tr = _mean_std(rs, "train")
        ms_val = _mean_std(rs, "val")
        ep_vals = [r["epochs"] for r in rs if isinstance(r["epochs"], int)]
        tm = test_results.get(name, {})
        row = {"model": name, "submodality": sub_name}
        for k in METRICS_KEYS:
            row[f"train_{k}_mean"] = ms_tr[k][0]
            row[f"val_{k}_mean"] = ms_val[k][0]
            row[f"val_{k}_std"] = ms_val[k][1]
        for k in METRICS_KEYS:
            row[f"test_{k}"] = tm.get(k, np.nan)
        row["avg_epochs"] = np.mean(ep_vals) if ep_vals else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "cv_results.csv"),
        index=False, float_format="%.4f")
    log.info("Results saved -> %s/cv_results.csv", RESULTS_DIR)

    best_name = max(all_results,
                    key=lambda n: np.mean([r["val"]["auc"] for r in all_results[n]]))
    log.info("\nBest model: %s", best_name)

    SHAP_COMPATIBLE = {"RF", "ExtraTrees", "GradBoost", "XGBoost", "LightGBM"}
    best_tree = max(
        (n for n in TREE_MODELS if n in all_results and n in SHAP_COMPATIBLE),
        key=lambda n: np.mean([r["val"]["auc"] for r in all_results[n]]),
        default=None,
    )
    if best_tree:
        log.info("Running SHAP for best radiomics model: %s", best_tree)
        cls, kwargs = TREE_MODELS[best_tree]
        sc = StandardScaler()
        Xsc = sc.fit_transform(X_dev.astype(np.float32))
        shap_for_tree(cls, kwargs, Xsc, y_dev, RADIOMICS_LABELS, RESULTS_DIR)

    import subprocess, shutil, tempfile, datetime

    report_md = generate_report_md(
        results_dir=RESULTS_DIR,
        all_results=all_results,
        test_results=test_results,
        n_total=len(manifest_df),
        n_dev=len(dev_indices),
        n_test=len(test_indices),
        rp_rate=rp_rate,
        rp_rate_dev=rp_rate_dev,
        rp_rate_test=rp_rate_test,
        best_name=best_name,
        submodality=sub_name,
    )

    report_pdf = os.path.join(RESULTS_DIR, "CT_RP_Prediction_Report.pdf")
    lua_filter = os.path.join(RESULTS_DIR, "squeeze_tables.lua")

    vars_yaml = os.path.join(tempfile.gettempdir(), "ct_report_vars.yaml")
    with open(vars_yaml, "w", encoding="utf-8") as fh:
        fh.write("---\n")
        fh.write(f'report_date: "{datetime.date.today().strftime("%Y-%m-%d")}"\n')
        fh.write(f'best_model: "{best_name}"\n')
        fh.write("---\n")

    if shutil.which("pandoc"):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                         delete=False, encoding="utf-8") as tmp:
            clean = open(report_md, encoding="utf-8").read()
            for ch, rep in [("★", "*"), ("◄", "<"), ("←", "<-"), ("→", "->"),
                            ("↔", "<->"), ("↕", "<->"), ("✓", "OK"), ("✗", "X"),
                            ("≥", ">="), ("×", "x")]:
                clean = clean.replace(ch, rep)
            tmp.write(clean)
            tmp_path = tmp.name

        base_cmd = ["pandoc", vars_yaml, tmp_path, "--pdf-engine=xelatex", "--toc",
                    "-o", report_pdf]
        if os.path.exists(lua_filter):
            base_cmd = ["pandoc", vars_yaml, tmp_path,
                        f"--lua-filter={lua_filter}",
                        "--pdf-engine=xelatex", "--toc", "-o", report_pdf]

        log.info("Generating PDF report...")
        result = subprocess.run(base_cmd, capture_output=True, text=True,
                                cwd=RESULTS_DIR)
        os.unlink(tmp_path)
        if result.returncode == 0:
            log.info("PDF saved -> %s", report_pdf)
        else:
            log.warning("PDF generation failed:\n%s", result.stderr[-800:])
    else:
        log.info("pandoc not found - markdown report saved, skipping PDF conversion")

    log.info("\nDone. Results -> %s", RESULTS_DIR)


def _run_all_submodalities(only: list[str] | None = None,
                           quick: bool = False) -> None:
    g = globals()
    base_results = g["RESULTS_DIR"]
    base_pred = g["PRED_DIR"]
    use_nested = bool(_full_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False))
    n_outer    = int(_full_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5))
    n_repeats  = int(_full_cfg.get("SPLIT", {}).get("N_CV_REPEATS", 1))
    gen_cfg_base = _full_cfg["PREPROCESS"]["GENERATED_FILES"]

    if use_nested:
        from nested_cv import run_nested_cv

    for sub_name, gen_key in CT_SUBMODALITIES:
        if only is not None and sub_name not in only:
            continue
        sub_cfg = _gen_cfg_for_submodality(gen_cfg_base, gen_key)
        if sub_cfg is None:
            log.info("Skipping submodality %s - files not configured "
                     "(missing %s in PREPROCESS.GENERATED_FILES).",
                     sub_name, gen_key)
            continue
        if not _submodality_available(sub_name, sub_cfg, _proc_dir, gen_key):
            log.info("Skipping submodality %s - not enough patients on disk "
                     "(looking for %s).", sub_name, sub_cfg.get(gen_key))
            continue

        log.info("\n%s\n  Running CT submodality: %s\n%s",
                 "#" * 70, sub_name, "#" * 70)

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
                    main_fn=main,
                    n_outer=n_outer,
                    module_globals=g,
                    results_dir_key="RESULTS_DIR",
                    pred_dir_key="PRED_DIR",
                    log=log,
                    n_repeats=n_repeats,
                    quick=quick,
                )
            else:
                main()
        except Exception as exc:
            log.error("Submodality %s failed: %s", sub_name, exc, exc_info=True)
        finally:
            g["RESULTS_DIR"] = base_results
            g["PRED_DIR"] = base_pred


if __name__ == "__main__":
    import torch.multiprocessing as _torch_mp

    _torch_mp.set_sharing_strategy("file_system")

    import argparse

    _valid = [s[0] for s in CT_SUBMODALITIES]
    parser = argparse.ArgumentParser(description="Train CT models per submodality.")
    parser.add_argument(
        "--submodality", action="append", choices=_valid,
        help=("Run only this submodality (repeatable). "
              f"Choices: {', '.join(_valid)}. Default: run all."),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run only 1 outer fold for fast methodology checks.")
    args = parser.parse_args()
    _run_all_submodalities(only=args.submodality, quick=args.quick)
