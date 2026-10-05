import gc
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.uname().sysname == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import glob
import json
import logging
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import wandb

import gate as _gate

from focal_loss import FocalLoss
from util import load_config, load_unified_split, read_table, setup_logging

_full_cfg = load_config()
_cfg = _full_cfg["TRAIN_RPFUSE"]
_data_cfg = _cfg["DATA"]
_model_cfg = _cfg["MODEL"]
_train_cfg = _cfg["TRAINING"]
_aug_cfg = _cfg.get("AUGMENTATION", {})

_input_dir = _full_cfg["INPUTS_DIR"]
_output_dir = _full_cfg["OUTPUTS_DIR"]
_cohort_dir = os.path.join(_input_dir, _full_cfg["INPUTS"]["PROCESSED"])

MANIFEST_FILE = os.path.join(_input_dir, _data_cfg["MANIFEST_FILE"])


def cohort_split():
    return load_unified_split(_input_dir), {"single"}


import endpoint as _endpoint
_EP = _endpoint.EndpointConfig(_full_cfg)


def read_manifest():
    return _endpoint.read_manifest(MANIFEST_FILE, _input_dir, _EP, log=log)


EXCEL_PATH = os.path.join(_input_dir, _data_cfg["EXCEL_PATH"])
_RESULTS_SUFFIX = str(_data_cfg.get("RESULTS_SUFFIX", ""))
TRAIN_FRACTION = float(os.environ.get("RPFUSE_TRAIN_FRACTION", _data_cfg.get("TRAIN_FRACTION", 1.0)))
TRAIN_FRACTION_SEED = int(os.environ.get("RPFUSE_TRAIN_FRACTION_SEED", _data_cfg.get("TRAIN_FRACTION_SEED", 0)))
if TRAIN_FRACTION < 1.0:
    _RESULTS_SUFFIX += f"_lc{int(round(TRAIN_FRACTION * 100))}_s{TRAIN_FRACTION_SEED}"
RESULTS_DIR = os.path.join(_output_dir, _data_cfg["RESULTS_DIR"] + _RESULTS_SUFFIX)
PRED_DIR = os.path.join(RESULTS_DIR,
                        os.path.basename(_data_cfg["PRED_DIR"].rstrip("/")) or
                        "patient_predictions")


TARGET_SHAPE = tuple(_data_cfg.get("TARGET_SHAPE", [128, 128, 128]))
WINDOW_WIDTH = _data_cfg.get("WINDOW_WIDTH", 1500)
WINDOW_CENTER = _data_cfg.get("WINDOW_CENTER", -600)
SUV_MAX_CLIP = float(_data_cfg.get("SUV_MAX_CLIP", 20.0))
DOSE_NORM_CGY = float(_data_cfg.get("DOSE_NORM_CGY", 6000.0))


def _cfg_alias(d: dict, new: str, old: str, default):
    if new in d:
        return d[new]
    return d.get(old, default)


def _flag(name: str, default: bool) -> bool:
    return bool(_data_cfg.get(name, default))


USE_IMAGING = _flag("USE_IMAGING", True)
USE_TABULAR = _flag("USE_TABULAR", True)
USE_PET = _flag("USE_PET", True)
USE_DVH = _flag("USE_DVH", True)
LUNG_CROP = bool(_data_cfg.get("LUNG_CROP", True))
LUNG_CROP_MARGIN = int(_data_cfg.get("LUNG_CROP_MARGIN", 8))
SINGLE_STUDY_ONLY = _flag("SINGLE_STUDY_ONLY", True)
EXCLUDE_MAJOR_REPLAN = True

USE_ANATOMY_GATE = _flag("USE_ANATOMY_GATE", True)
USE_LOBE_GATE = _flag("USE_LOBE_GATE", False)
USE_VULN_GATE = _flag("USE_VULN_GATE", False)
USE_VULN_DOSE_RESPONSE = bool(_cfg_alias(_data_cfg, "USE_VULN_DOSE_RESPONSE", "USE_VULN_NTCP", False))
VULN_BIDIR = _flag("VULN_BIDIR", False)
if USE_VULN_DOSE_RESPONSE:
    USE_VULN_GATE = True
USE_REGION_GATE = _flag("USE_REGION_GATE", False)


def _gate_config(K: int | None = None) -> "_gate.GateConfig":
    _, ic, iv = _anat_layout()
    return _gate.GateConfig(
        use_anatomy=USE_ANATOMY_GATE, use_lobe=USE_LOBE_GATE,
        use_region=USE_REGION_GATE, use_vuln_ntcp=USE_VULN_DOSE_RESPONSE,
        use_dose_gradient=USE_DOSE_GRADIENT, use_conv=USE_CONV_GATE,
        use_cross=USE_CROSS_GATE,
        n_points=DOSE_RESPONSE_K if K is None else K,
        init_d50_gy=DOSE_RESPONSE_INIT_GY, init_transition_gy=DOSE_RESPONSE_TRANSITION_GY,
        dose_norm_gy=DOSE_NORM_CGY / 100.0,
        off_init=GATE_OFF_INIT, idx_cc=ic, idx_vuln=iv)


def _anat_layout() -> tuple[int, int, int]:
    n = 2
    idx_cc = idx_vuln = -1
    if USE_LOBE_GATE or USE_REGION_GATE:
        idx_cc = n;
        n += 1
    if USE_VULN_GATE:
        idx_vuln = n;
        n += 1
    return n, idx_cc, idx_vuln


USE_CONV_GATE = _flag("USE_CONV_GATE", False)

GATE_EMB_DIR = str(_data_cfg.get("GATE_EMB_DIR", "") or "")
if GATE_EMB_DIR and not os.path.isabs(GATE_EMB_DIR):
    GATE_EMB_DIR = os.path.join(_output_dir, GATE_EMB_DIR)
_emb_dirs_cfg = _data_cfg.get("GATE_EMB_DIRS") or {}
GATE_EMB_DIRS: "dict[str, str]" = {}
if isinstance(_emb_dirs_cfg, dict):
    for _tok, _d in _emb_dirs_cfg.items():
        _d = str(_d or "")
        GATE_EMB_DIRS[str(_tok)] = _d if os.path.isabs(_d) else os.path.join(_output_dir, _d)
if not GATE_EMB_DIRS and GATE_EMB_DIR:
    GATE_EMB_DIRS = {"ct": GATE_EMB_DIR}
elif GATE_EMB_DIRS and not GATE_EMB_DIR:
    GATE_EMB_DIR = GATE_EMB_DIRS.get("ct") or next(iter(GATE_EMB_DIRS.values()))
CACHE_IMAGING = bool(GATE_EMB_DIRS)
TEST_MASK_PETE = bool(_data_cfg.get("TEST_MASK_PETE", False))
USE_IMAGING_SKLEARN_EXPERT = _flag("USE_IMAGING_SKLEARN_EXPERT", False)
IMAGING_EXPERT_RESIDUAL = _flag("IMAGING_EXPERT_RESIDUAL", False)
IMAGING_TOKEN_BOTTLENECK = int(_model_cfg.get("IMAGING_TOKEN_BOTTLENECK", 0))
_IMG_FEAT_DIM: int | None = None
_IMG_FEAT_DIMS: "dict[str, int]" = {}
GATE_LIVE_ENCODER = bool(_data_cfg.get("GATE_LIVE_ENCODER", False)) and CACHE_IMAGING
_LIVE_ENCODER = None
USE_DEEP_GATE = _flag("USE_DEEP_GATE", False)
USE_CROSS_GATE = _flag("USE_CROSS_GATE", False)
USE_DOSE_GRADIENT = _flag("USE_DOSE_GRADIENT", True)
PERI_TAU = float(_data_cfg.get("PERI_TAU", 20.0))
PERI_TAU_MM = float(_data_cfg.get("PERI_TAU_MM", 20.0))
PET_ROBUST = _flag("PET_ROBUST", True)
USE_PET_PLUS = _flag("USE_PET_PLUS", True)
USE_CT_RADIOMICS = _flag("USE_CT_RADIOMICS", False)
CT_RADIOMICS_DOSE_GY = float(_data_cfg.get("CT_RADIOMICS_DOSE_GY", 16.0))
CT_RADIOMICS_PER_FOLD_D50 = _flag("CT_RADIOMICS_PER_FOLD_D50", False)
CT_RADIOMICS_IMAGE = str(_data_cfg.get("CT_RADIOMICS_IMAGE", "raw")).lower()
CT_RADIOMICS_MASK = str(_data_cfg.get("CT_RADIOMICS_MASK", "dose_d50")).lower()
CT_RADIOMICS_TOP_K = int(_data_cfg.get("CT_RADIOMICS_TOP_K", 30))
CT_RADIOMICS_GATE_THRS = [float(t) for t in _data_cfg.get("CT_RADIOMICS_GATE_THRS", [])] \
    or [float(_data_cfg.get("CT_RADIOMICS_GATE_THR", 0.5))]
CT_RADIOMICS_CACHE_DIR = str(_data_cfg.get("CT_RADIOMICS_CACHE_DIR", "gated_radiomics_cache"))
CT_RADIOMICS_SPOTLIGHT_F = float(_data_cfg.get("CT_RADIOMICS_SPOTLIGHT_F", 1.0))
CT_RADIOMICS_EXCLUDE_GTV = _flag("CT_RADIOMICS_EXCLUDE_GTV", False)
CT_RADIOMICS_GTV_MARGIN_VOX = int(_data_cfg.get("CT_RADIOMICS_GTV_MARGIN_VOX", 0))
USE_CT_VOL = _flag("USE_CT_VOL", True)
USE_EARLY = _flag("USE_EARLY", True)
USE_EARLY_RAW = _flag("USE_EARLY_RAW", False)
USE_PET_VOL = _flag("USE_PET_VOL", False)
USE_DOSE_VOL = _flag("USE_DOSE_VOL", False)
GENERATE_GRADCAM = _flag("GRADCAM", True)
N_GRADCAM = int(_data_cfg.get("N_GRADCAM", 4))
_IMAGING_TOKENS = [t for t, on in [("ct", USE_CT_VOL), ("pet", USE_PET_VOL),
                                   ("dose", USE_DOSE_VOL), ("early", USE_EARLY),
                                   ("early_raw", USE_EARLY_RAW)] if on]
NUM_WORKERS = int(_data_cfg.get("NUM_WORKERS", 4))
CACHE_VOLUMES = bool(_data_cfg.get("CACHE_VOLUMES", True))

EMBED_DIM = int(_model_cfg.get("EMBED_DIM", 128))
FUSION_HIDDEN = int(_model_cfg.get("FUSION_HIDDEN", 128))
FUSION_ALPHA_INIT_RAW = float(_model_cfg.get("FUSION_ALPHA_INIT_RAW", -2.0))
FUSION_SUPERVISE_IMG = bool(_model_cfg.get("FUSION_SUPERVISE_IMG", False))
ALPHA_LR = float(_model_cfg.get("FUSION_ALPHA_LR", 0.0))
FUSION_LATE = bool(_model_cfg.get("FUSION_LATE", False))
MODALITY_DROP = float(_model_cfg.get("FUSION_MODALITY_DROP", 0.0))
FUSION_TEAM = bool(_model_cfg.get("FUSION_TEAM", False))
FUSION_TEAM_HEADS = int(_model_cfg.get("FUSION_TEAM_HEADS", 4))
FUSION_TEAM_LAYERS = int(_model_cfg.get("FUSION_TEAM_LAYERS", 1))
FUSION_MEAN_PROB = bool(_model_cfg.get("FUSION_MEAN_PROB", False))
FUSION_MEAN_LOGIT = bool(_model_cfg.get("FUSION_MEAN_LOGIT", False))
FUSION_MEDIAN_PROB = bool(_model_cfg.get("FUSION_MEDIAN_PROB", False))
FUSION_TRIM_PROB = bool(_model_cfg.get("FUSION_TRIM_PROB", False))
FUSION_MIN_PROB = bool(_model_cfg.get("FUSION_MIN_PROB", False))
FUSION_RESIDUAL_MEAN = bool(_model_cfg.get("FUSION_RESIDUAL_MEAN", False))
assert sum([FUSION_MEAN_PROB, FUSION_MEAN_LOGIT, FUSION_MEDIAN_PROB,
            FUSION_TRIM_PROB, FUSION_MIN_PROB, FUSION_RESIDUAL_MEAN]) <= 1, \
    "the parameter-free combiner flags are mutually exclusive"
FUSION_TEAM_LOGIT = bool(_model_cfg.get("FUSION_TEAM_LOGIT", False))
FUSION_IMAGING_RESIDUAL = bool(_model_cfg.get("FUSION_IMAGING_RESIDUAL", False))
EXPERT_PRETRAIN_EPOCHS = int(_model_cfg.get("EXPERT_PRETRAIN_EPOCHS", 0))
EXPERT_FREEZE = bool(_model_cfg.get("EXPERT_FREEZE", False))
SKLEARN_EXPERTS = bool(_model_cfg.get("SKLEARN_EXPERTS", False))
EXPERT_CROSSFIT = bool(_model_cfg.get("EXPERT_CROSSFIT", False))
EXPERT_CROSSFIT_FOLDS = int(_model_cfg.get("EXPERT_CROSSFIT_FOLDS", 5))
SPLIT_TAB = bool(_model_cfg.get("SPLIT_TAB", False))
TAB_KEEP = str(_model_cfg.get("TAB_KEEP", "") or "").strip().lower()
if TAB_KEEP not in ("", "clinical", "rt"):
    raise ValueError(f"TAB_KEEP must be '', 'clinical' or 'rt', got {TAB_KEEP!r}")
_CLIN_DIM = None
_DEFAULT_EXPERT_MODELS = {"dvh": "mlp_128_64_32", "dosim": "mlp_128_64_32",
                          "rt": "mlp_64_32_16",
                          "petplus": "dtree", "clinical": "lgbm", "tab": "mlp_128_64_32",
                          "ctrad": "rad_select_mlp"}
SKLEARN_EXPERT_MODELS = {**_DEFAULT_EXPERT_MODELS,
                         **(_model_cfg.get("SKLEARN_EXPERT_MODELS", {}) or {})}


class _CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.80, max_keep: int | None = None):
        self.threshold = threshold
        self.max_keep = max_keep

    def fit(self, X, y=None):
        from sklearn.feature_selection import f_classif
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]
        if y is not None:
            with np.errstate(all="ignore"):
                F, _ = f_classif(X, y)
            F = np.nan_to_num(F, nan=0.0)
            order = list(np.argsort(-F))
        else:
            order = list(range(n_feat))
        with np.errstate(all="ignore"):
            C = np.abs(np.corrcoef(X, rowvar=False))
        C = np.nan_to_num(C, nan=0.0)
        kept: list[int] = []
        for j in order:
            if any(C[j, k] > self.threshold for k in kept):
                continue
            kept.append(j)
            if self.max_keep is not None and len(kept) >= self.max_keep:
                break
        self.keep_idx_ = np.array(sorted(kept), dtype=int) if kept else np.arange(n_feat)
        self.n_features_in_ = n_feat
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_idx_]


def _make_sklearn(name: str):
    name = str(name).lower()
    if name == "mlp_128_64_32":
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, max_iter=3000, random_state=SEED)
    if name == "mlp_64_32_16":
        return MLPClassifier(hidden_layer_sizes=(64, 32, 16), alpha=1e-3, max_iter=3000, random_state=SEED)
    if name in ("extra_trees", "extratrees"):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=400, max_depth=4,
                                    class_weight="balanced", random_state=SEED)
    if name == "dtree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=4, random_state=SEED)
    if name == "qda":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return QuadraticDiscriminantAnalysis()
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              verbose=-1, n_jobs=1)
    if name in ("logreg", "logreg_l2"):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name in ("rf", "random_forest", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=-1)
    if name in ("nn_tiny", "nn_simple", "nn_deep", "nn_ft", "nn_attn",
                "nn_res", "nn_gated"):
        from tab_models import TorchMLPClassifier
        _arch = {"nn_tiny": "TinyMLP", "nn_simple": "SimpleMLP", "nn_deep": "DeepMLP",
                 "nn_ft": "FTTransformer", "nn_attn": "AttentionMLP",
                 "nn_res": "ResMLP", "nn_gated": "GatedMLP"}[name]
        return TorchMLPClassifier(arch=_arch, seed=SEED)
    if name in ("svm_rbf", "svc_rbf"):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=1.0, probability=True,
                                     class_weight="balanced", random_state=SEED))])
    if name == "rad_select_mlp":
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest, f_classif
        return Pipeline([
            ("sel", SelectKBest(f_classif, k=min(CT_RADIOMICS_TOP_K, 77))),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                  max_iter=3000, random_state=SEED)),
        ])
    if name == "rad_lasso":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                      solver="saga", max_iter=5000, random_state=SEED)),
        ])
    if name == "rad_lasso_corr":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ("scale", StandardScaler()),
            ("decorr", _CorrelationFilter(threshold=0.80)),
            ("lr", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                      solver="saga", max_iter=5000, random_state=SEED)),
        ])
    return MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, max_iter=3000, random_state=SEED)


SCALAR_ENCODER = str(_model_cfg.get("SCALAR_ENCODER", "linear")).lower()
DG_INIT_A = float(_model_cfg.get("DOSEGATE_INIT_A", 10.0))
DG_INIT_B = float(_model_cfg.get("DOSEGATE_INIT_B", 0.33))
DG_INIT_LAMBDA = float(_model_cfg.get("DOSEGATE_INIT_LAMBDA", 1.0))
USE_DOSE_RESPONSE_GATE = bool(_cfg_alias(_data_cfg, "USE_DOSE_RESPONSE_GATE", "USE_NTCP_GATE", False))
DOSE_RESPONSE_K = int(_cfg_alias(_model_cfg, "DOSE_RESPONSE_K", "NTCP_K", 4))
DOSE_RESPONSE_INIT_GY = float(_cfg_alias(_model_cfg, "DOSE_RESPONSE_INIT_GY", "NTCP_INIT_GY", 20.0))
DOSE_RESPONSE_TRANSITION_GY = float(_cfg_alias(_model_cfg, "DOSE_RESPONSE_TRANSITION_GY", "NTCP_TRANSITION_GY", 20.0))
SPOTLIGHT_INIT = float(_model_cfg.get("SPOTLIGHT_INIT", 0.4))
GATE_LR = float(_model_cfg.get("DOSEGATE_LR", 1e-2))
GATE_OFF_INIT = float(_model_cfg.get("GATE_OFF_INIT", -1.5))
GATE_ASSERT_LEARNABLE = _flag("GATE_ASSERT_LEARNABLE", True)
GATE_ASSERT_CURVE = _flag("GATE_ASSERT_CURVE", True)
GATE_MIN_CURVE_SPAN = float(_model_cfg.get("GATE_MIN_CURVE_SPAN", 0.5))
DEVICE_PREF = str(_model_cfg.get("DEVICE", "auto")).lower()
SHOW_PROGRESS = _flag("SHOW_PROGRESS", __import__("sys").stderr.isatty())
AUX_LOSS_WEIGHT = float(_model_cfg.get("AUX_LOSS_WEIGHT", 0.3))
GATE_PRETRAIN_EPOCHS = int(_model_cfg.get("GATE_PRETRAIN_EPOCHS", 0))
LOG_TRAIN_HISTORY = bool(_model_cfg.get("LOG_TRAIN_HISTORY", True))
_MEDICALNET_R18_PATH = _model_cfg.get("MEDICALNET_PRETRAINED_R18", "") or ""
_PROJECT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "medicalnet_weights")


BACKBONE_DEPTH = int(_model_cfg.get("BACKBONE_DEPTH", 18))
BACKBONE_WIDEN = int(_model_cfg.get("BACKBONE_WIDEN", 1))
BACKBONE_KIND = str(_model_cfg.get("BACKBONE_KIND", "resnet")).lower()
BACKBONE_SHORTCUT = str(_model_cfg.get("BACKBONE_SHORTCUT", "A"))
BACKBONE_BIAS_DOWNSAMPLE = bool(_model_cfg.get("BACKBONE_BIAS_DOWNSAMPLE", False))


def resolve_device() -> "torch.device":
    if DEVICE_PREF in ("cpu", "mps", "cuda"):
        return torch.device(DEVICE_PREF)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            return torch.device("mps")
        log.info("MPS is available but several 3D ops it lacks are used here; "
                 "falling back to CPU. Set PYTORCH_ENABLE_MPS_FALLBACK=1 (or "
                 "DEVICE: mps in the config) to use MPS with per-op CPU fallback.")
    return torch.device("cpu")


def _resolve_backbone_weights(depth: int) -> str:
    fname = f"resnet_{depth}_23dataset.pth"
    cands = []
    explicit = _model_cfg.get(f"MEDICALNET_PRETRAINED_R{depth}", "")
    if explicit:
        cands.append(explicit)
    if depth == 18:
        cands.append(_MEDICALNET_R18_PATH)
    elif _MEDICALNET_R18_PATH:
        cands.append(os.path.join(os.path.dirname(_MEDICALNET_R18_PATH), fname))
    cands.append(os.path.join(_PROJECT_WEIGHTS_DIR, fname))
    cands = [c for c in cands if c]
    for c in cands:
        if os.path.exists(c):
            if c != cands[0]:
                log.info("MedicalNet R%d: configured path missing, using repo-local "
                         "weights %s", depth, c)
            return c
    return cands[0] if cands else ""


SEED = int(_train_cfg["SEED"])
BATCH_SIZE = int(_train_cfg["BATCH_SIZE"])
MAX_EPOCHS = int(_train_cfg["MAX_EPOCHS"])
LR = float(_train_cfg["LR"])
WEIGHT_DECAY = float(_train_cfg["WEIGHT_DECAY"])
DROPOUT = float(_train_cfg["DROPOUT"])
N_MC_SAMPLES = int(_train_cfg["N_MC_SAMPLES"])
ACCUMULATION = int(_train_cfg["ACCUMULATION_STEPS"])
WARMUP_EPOCHS = int(_train_cfg["WARMUP_EPOCHS"])
FREEZE_BACKBONE = bool(_train_cfg.get("FREEZE_BACKBONE", False))
EARLY_STOP_PATIENCE = int(_train_cfg.get("EARLY_STOP_PATIENCE", 15))
EARLY_STOP_MIN_EPOCHS = int(_train_cfg.get("EARLY_STOP_MIN_EPOCHS", WARMUP_EPOCHS + 20))
CHECKPOINT_METRIC = str(_train_cfg.get("CHECKPOINT_METRIC", "loss")).lower()

FOCAL_GAMMA = float(_train_cfg.get("FOCAL_GAMMA", 2.0))
FOCAL_GAMMA_FINAL = float(_train_cfg.get("FOCAL_GAMMA_FINAL", FOCAL_GAMMA))
FOCAL_GAMMA_ANNEAL_FRAC = float(_train_cfg.get("FOCAL_GAMMA_ANNEAL_FRAC", 1.0))


def _focal_gamma_at(epoch: int, max_epochs: int) -> float:
    if FOCAL_GAMMA_FINAL == FOCAL_GAMMA or max_epochs <= 1:
        return FOCAL_GAMMA
    span = max(1, int(round(FOCAL_GAMMA_ANNEAL_FRAC * max_epochs)))
    frac = min(1.0, max(0.0, (epoch - 1) / span))
    return FOCAL_GAMMA + (FOCAL_GAMMA_FINAL - FOCAL_GAMMA) * frac


_AMP_STR = str(_train_cfg.get("AMP_DTYPE", "bf16")).lower()
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "f16": torch.float16, "float16": torch.float16,
             "bfloat16": torch.bfloat16}.get(_AMP_STR, torch.bfloat16)


P_FLIP = float(_aug_cfg.get("P_FLIP", 0.5))
P_NOISE = float(_aug_cfg.get("P_NOISE", 0.3))
NOISE_STD = float(_aug_cfg.get("NOISE_STD", 0.05))
MIXUP_ALPHA = float(_aug_cfg.get("MIXUP_ALPHA", 0.4))
P_MIXUP = float(_aug_cfg.get("P_MIXUP", 0.5))
P_ROT = float(_aug_cfg.get("P_ROT", 0.0))
ROT_RANGE = float(_aug_cfg.get("ROT_RANGE", 10.0))
P_SCALE = float(_aug_cfg.get("P_SCALE", 0.0))
SCALE_RANGE = float(_aug_cfg.get("SCALE_RANGE", 0.10))
P_ELASTIC = float(_aug_cfg.get("P_ELASTIC", 0.0))
ELASTIC_ALPHA = float(_aug_cfg.get("ELASTIC_ALPHA", 12.0))
ELASTIC_SIGMA = float(_aug_cfg.get("ELASTIC_SIGMA", 4.0))
P_GAMMA = float(_aug_cfg.get("P_GAMMA", 0.0))
GAMMA_RANGE = tuple(_aug_cfg.get("GAMMA_RANGE", [0.7, 1.5]))
P_BLUR = float(_aug_cfg.get("P_BLUR", 0.0))
BLUR_SIGMA = float(_aug_cfg.get("BLUR_SIGMA", 0.7))
AUG_BACKEND = str(_aug_cfg.get("BACKEND", "sync")).lower()
AUG_TRANSLATE = float(_aug_cfg.get("TRANSLATE", 0.0))
P_CUTOUT = float(_aug_cfg.get("P_CUTOUT", 0.0))
CUTOUT_HOLES = int(_aug_cfg.get("CUTOUT_HOLES", 4))
CUTOUT_FRAC = float(_aug_cfg.get("CUTOUT_FRAC", 0.12))

os.makedirs(RESULTS_DIR, exist_ok=True)
log = setup_logging(RESULTS_DIR, "train_rpfuse.log")

try:
    import shutil as _shutil
    _cfg_name = os.environ.get("RPFUSE_CONFIG", "config.yaml")
    _cfg_src = _cfg_name if os.path.exists(_cfg_name) else os.path.join("configs", _cfg_name)
    if os.path.exists(_cfg_src):
        _shutil.copy2(_cfg_src, os.path.join(RESULTS_DIR, os.path.basename(_cfg_src)))
        log.info("Config copied to results dir: %s", os.path.basename(_cfg_src))
except Exception as _e:
    log.warning("Could not copy config to results dir: %s", _e)


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _first(paths: list[str]) -> str | None:
    return sorted(paths)[0] if paths else None


def resolve_primary_study(uuid_dir: str) -> str | None:
    cands: list[tuple[str, dict]] = []
    for mp in sorted(glob.glob(os.path.join(uuid_dir, "studies", "*",
                                            "study_meta.json"))):
        sdir = os.path.dirname(mp)
        if not os.path.exists(os.path.join(sdir, "CT", "whole_body.npy")):
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            m = {}
        cands.append((sdir, m))
    if not cands:
        return None
    totals = [c for c in cands if c[1].get("is_total")]
    pool = totals or cands
    sdir, _ = max(pool, key=lambda c: (c[1].get("dose_max_cGy") or 0,
                                       c[1].get("lung_MLD_cGy") or 0))
    return sdir


def n_ct_studies(uuid_dir: str) -> int:
    n = 0
    for mp in glob.glob(os.path.join(uuid_dir, "studies", "*", "study_meta.json")):
        if os.path.exists(os.path.join(os.path.dirname(mp), "CT", "whole_body.npy")):
            n += 1
    return n


_REPLAN_MLD_TOL_GY = 3.0
_REPLAN_V20_TOL_PP = 8.0
_REPLAN_BOOST_V20_ABS = 5.0
_REPLAN_BOOST_V20_FRAC = 0.35


def _study_lung_dvh(study_dir: str):
    mld = v20 = None
    is_total = False
    try:
        d = json.load(open(os.path.join(study_dir, "DVH", "dvh.json")))
        s = d.get("structures", {}) or {}
        lung = next((s[k] for k in ("lung_minus_gtv", "lung") if s.get(k)), {}) or {}
        m = lung.get("Dmean_cGy")
        if m is not None and np.isfinite(m):
            mld = float(m) / 100.0
        v = lung.get("V20")
        if v is not None and np.isfinite(v):
            v20 = float(v)
    except Exception:
        pass
    try:
        meta = json.load(open(os.path.join(study_dir, "study_meta.json")))
        is_total = bool(meta.get("is_total", False))
    except Exception:
        pass
    return mld, v20, is_total


def is_major_replan(uuid_dir: str) -> bool:
    studies = sorted(os.path.dirname(os.path.dirname(p)) for p in
                     glob.glob(os.path.join(uuid_dir, "studies", "*", "DVH", "dvh.json")))
    if len(studies) < 2:
        return False
    met = [_study_lung_dvh(sd) for sd in studies]
    mlds = [m[0] for m in met]
    v20s = [m[1] for m in met]
    if any(m[2] for m in met):
        return False
    v20_vals = [v for v in v20s if v is not None]
    if not v20_vals:
        return False
    prim_v20 = max(v20_vals)
    boost_cut = max(_REPLAN_BOOST_V20_ABS, _REPLAN_BOOST_V20_FRAC * prim_v20)
    full = [i for i in range(len(studies)) if v20s[i] is not None and v20s[i] >= boost_cut]
    if len(full) < 2:
        return False

    def _spread(seq):
        vals = [seq[i] for i in full if seq[i] is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    return _spread(mlds) > _REPLAN_MLD_TOL_GY or _spread(v20s) > _REPLAN_V20_TOL_PP


def study_class(uuid_dir: str) -> str:
    if n_ct_studies(uuid_dir) <= 1:
        return "single"
    if is_major_replan(uuid_dir):
        return "major"
    sdir = resolve_primary_study(uuid_dir)
    try:
        if bool(json.load(open(os.path.join(sdir, "study_meta.json"))).get("is_total")):
            return "is_total"
    except Exception:
        pass
    return "boost"


def resolve_modality_paths(uuid_dir: str) -> dict[str, str | None]:
    sdir = resolve_primary_study(uuid_dir)

    def _p(*parts: str) -> str | None:
        if not sdir:
            return None
        p = os.path.join(sdir, *parts)
        return p if os.path.exists(p) else None

    pet = _first(glob.glob(os.path.join(uuid_dir, "studies", "*_PET",
                                        "PET", "whole_body_regCT.npy")))
    return {"ct": _p("CT", "whole_body.npy"),
            "dose": _p("Dose", "dose_regCT.npy"),
            "pet": pet,
            "lung": _p("Contours", "lung_mask.npy"),
            "gtv": _p("Contours", "gtv_mask.npy"),
            "ptv": _p("Contours", "ptv_mask.npy"),
            "liver": _p("Contours", "liver_mask.npy")}


def patient_has_required(uuid_dir: str) -> bool:
    p = resolve_modality_paths(uuid_dir)
    return p["ct"] is not None and p["dose"] is not None


def cohort_uuids(cohort_dir: str | None = None) -> list[str]:
    cohort_dir = cohort_dir or _cohort_dir
    split = load_unified_split(_input_dir)
    allowed = {"single"}
    out = []
    for _, r in split.iterrows():
        uid = str(r["uuid"]).strip()
        ud = os.path.join(cohort_dir, uid)
        if not (os.path.isdir(ud) and study_class(ud) in allowed):
            continue
        ofld = r.get("OuterFold", -1)
        if pd.notna(ofld) and int(ofld) >= 0:
            out.append(uid)
    return out


def ct_voxel_spacing(ct_path: str | None) -> tuple[float, float, float]:
    try:
        gj = os.path.join(os.path.dirname(ct_path), "grid.json")
        sp = json.load(open(gj))["spacing"]
        return float(sp[0]), float(sp[1]), float(sp[2])
    except Exception:
        return 3.0, 1.0, 1.0


def build_tabular_map() -> tuple[dict[str, np.ndarray], int]:
    if not USE_TABULAR:
        return {}, 0
    try:
        from fusion_utils import build_clinical_features, build_rt_features
        master = read_table(EXCEL_PATH)
        split = load_unified_split(_input_dir)
        X_clin, _ = build_clinical_features(master)
        X_rt, _ = build_rt_features(master)
        global _CLIN_DIM
        _CLIN_DIM = int(X_clin.shape[1])
        X_all = np.hstack([X_clin, X_rt]).astype(np.float32)
        tab_map: dict[str, np.ndarray] = {}
        for _, row in split.iterrows():
            eid = row.get("excel_row_idx")
            if pd.isna(eid):
                continue
            eid = int(eid)
            if 0 <= eid < len(X_all):
                tab_map[str(row["uuid"]).strip()] = X_all[eid]
        dim = X_all.shape[1]
        log.info("Tabular stream: %d feats/patient, %d uuids mapped.",
                 dim, len(tab_map))
        return tab_map, dim
    except Exception as exc:
        log.warning("Tabular stream disabled - build failed: %s", exc)
        return {}, 0


DVH_SOURCE = str(_data_cfg.get("DVH_SOURCE", "dvh_curve")).lower()
DVH_SOURCE2 = str(_data_cfg.get("DVH_SOURCE2", "") or "").lower()
_DERIVED_DIR = os.path.join(_input_dir, "Derivated_Modals")


def _cache_read_path(subdir: str, filename: str) -> str:
    sub = os.path.join(_DERIVED_DIR, subdir, filename)
    return sub if os.path.exists(sub) else os.path.join(_DERIVED_DIR, filename)


def _cache_write_path(subdir: str, filename: str) -> str:
    d = os.path.join(_DERIVED_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def build_dvh_map(source: str | None = None) -> tuple[dict[str, np.ndarray], int]:
    if not USE_DVH:
        return {}, 0
    src = (source or DVH_SOURCE)
    if src in ("dosimetrics", "dosimetrics_all"):
        path = _cache_read_path(src, f"{src}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"dose table not found: {path}. Run generate_dosimetrics.py.")
        z = np.load(path, allow_pickle=True)
        names, X, uu = list(z["names"]), z["X"].astype(np.float32), z["uuids"]
        dvh_map = {str(u).strip(): X[i] for i, u in enumerate(uu)}
        dim = len(names)
    else:
        path = _cache_read_path(src, f"{src}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"dose table not found: {path}. Run generate_{src}.py.")
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != "uuid"]
        dvh_map = {str(r["uuid"]).strip(): r[cols].to_numpy(np.float32) for _, r in df.iterrows()}
        dim = len(cols)
    log.info("Dose stream [%s] (LOADED): %d feats/patient, %d uuids mapped.",
             src, dim, len(dvh_map))
    return dvh_map, dim


_PET_PLUS_NAMES = ["SUVmax_tumour", "SUVmean_tumour", "MTV", "TLG",
                  "SUVmean_lung", "SUVmax_lung", "SUVp90_lung", "SUVp95_lung",
                  "SUVstd_lung", "SUV_CV_lung", "lung_frac_hi",
                  "SUVmean_peri", "lung_to_tumour_ratio"]
_PET_PLUS_DIM = len(_PET_PLUS_NAMES)
PETE_SCALE_INVARIANT = bool(_cfg_alias(_data_cfg, "PETE_SCALE_INVARIANT", "PET_SCALE_INVARIANT", True))
PETE_DOSE_GATED = _flag("PETE_DOSE_GATED", False)
PET_STREAM_RADIOMICS = _flag("PET_STREAM_RADIOMICS", False)
PET_RADIOMICS_TABLE = str(_data_cfg.get("PET_RADIOMICS_TABLE",
                          os.environ.get("PET_RADIOMICS_TABLE", "petomics_lung_si")))
_PET_RAD_CACHE: dict | None = None


def _pet_radiomics_table() -> tuple[dict, list[str]]:
    global _PET_RAD_CACHE
    if _PET_RAD_CACHE is None:
        f = os.path.join(_input_dir, "Derivated_Modals", f"{PET_RADIOMICS_TABLE}.npz")
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        X = z["X"].astype(np.float32)
        _PET_RAD_CACHE = ({str(u).strip(): X[i] for i, u in enumerate(z["uuids"])}, names)
    return _PET_RAD_CACHE


def _pet_plus_vector(pet_raw: np.ndarray, tumour: np.ndarray, lung: np.ndarray,
                    vox_cc: float = 1.0, liver: np.ndarray | None = None) -> np.ndarray:
    g = tumour > 0
    if not g.any() or not np.isfinite(pet_raw).any():
        return np.zeros(_PET_PLUS_DIM, np.float32)
    ref = 0.0
    if liver is not None and liver.shape == pet_raw.shape:
        lv = pet_raw[liver > 0]; lv = lv[np.isfinite(lv) & (lv > 0)]
        if lv.size >= 50:
            ref = float(np.median(lv))
    if ref <= 1e-6:
        allv = pet_raw[np.isfinite(pet_raw) & (pet_raw > 0)]
        ref = float(np.median(allv)) if allv.size else 1.0
    pet = pet_raw / max(ref, 1e-6)
    suv_t = pet[g];
    suv_t = suv_t[np.isfinite(suv_t)]
    suvmax = float(np.max(suv_t)) if suv_t.size else 0.0
    suvmean = float(np.mean(suv_t)) if suv_t.size else 0.0
    mtv = float((suv_t >= 0.4 * suvmax).sum()) * vox_cc
    tlg = mtv * suvmean
    normal = (lung > 0) & ~g
    suv_l = pet[normal];
    suv_l = suv_l[np.isfinite(suv_l)]
    if suv_l.size:
        lmean = float(np.mean(suv_l));
        lmax = float(np.max(suv_l))
        lp90 = float(np.percentile(suv_l, 90));
        lp95 = float(np.percentile(suv_l, 95))
        lstd = float(np.std(suv_l))
        lcv = lstd / lmean if lmean > 1e-6 else 0.0
        lfrac = float(np.mean(suv_l >= 1.5 * lmean)) if lmean > 1e-6 else 0.0
    else:
        lmean = lmax = lp90 = lp95 = lstd = lcv = lfrac = 0.0
    try:
        dist = scipy.ndimage.distance_transform_edt(~g)
        peri = normal & (dist <= PERI_TAU)
        pmean = float(np.nanmean(pet[peri])) if peri.any() else lmean
    except Exception:
        pmean = lmean
    ratio = lmean / suvmean if suvmean > 1e-6 else 0.0
    return np.array([suvmax, suvmean, mtv, tlg,
                     lmean, lmax, lp90, lp95, lstd, lcv, lfrac, pmean, ratio],
                    np.float32)


_PET_E_NAMES = ["MTV_frac_lung", "lung_CV", "lung_frac_hi",
                "lung_p90_over_mean", "lung_p95_over_mean", "lung_max_over_mean",
                "peri_over_lung", "lung_over_tumour", "tumour_max_over_lung_mean",
                "tumour_peak_ratio"]
_PET_E_DIM = len(_PET_E_NAMES)

_PET_E_GATED_NAMES = ["hd_over_lung", "ld_over_lung", "hd_over_ld", "hd_CV",
                      "hd_frac_hi", "hd_p90_over_mean", "hd_p95_over_mean",
                      "hd_max_over_mean"]


def pet_feature_names() -> list[str]:
    if PET_STREAM_RADIOMICS:
        return list(_pet_radiomics_table()[1])
    base = list(_PET_E_NAMES if PETE_SCALE_INVARIANT else _PET_PLUS_NAMES)
    if PETE_SCALE_INVARIANT and PETE_DOSE_GATED:
        base += _PET_E_GATED_NAMES
    return base


def pet_feature_dim() -> int:
    return len(pet_feature_names())


def _pet_e_vector(pet_raw, tumour, lung, vox_cc: float = 1.0):
    g = tumour > 0
    if not g.any() or not np.isfinite(pet_raw).any():
        return np.zeros(_PET_E_DIM, np.float32)
    suv_t = pet_raw[g]; suv_t = suv_t[np.isfinite(suv_t)]
    normal = (lung > 0) & ~g
    suv_l = pet_raw[normal]; suv_l = suv_l[np.isfinite(suv_l)]
    if not suv_t.size or not suv_l.size:
        return np.zeros(_PET_E_DIM, np.float32)
    tmax, tmean = float(np.max(suv_t)), float(np.mean(suv_t))
    lmean = float(np.mean(suv_l))
    if lmean <= 1e-9 or tmean <= 1e-9:
        return np.zeros(_PET_E_DIM, np.float32)
    mtv = float((suv_t >= 0.4 * tmax).sum())
    lung_vox = float(max(normal.sum(), 1))
    lp90, lp95 = float(np.percentile(suv_l, 90)), float(np.percentile(suv_l, 95))
    lmax, lstd = float(np.max(suv_l)), float(np.std(suv_l))
    try:
        dist = scipy.ndimage.distance_transform_edt(~g)
        peri = normal & (dist <= PERI_TAU)
        pmean = float(np.nanmean(pet_raw[peri])) if peri.any() else lmean
    except Exception:
        pmean = lmean
    return np.array([
        mtv / lung_vox,
        lstd / lmean,
        float(np.mean(suv_l >= 1.5 * lmean)),
        lp90 / lmean, lp95 / lmean, lmax / lmean,
        pmean / lmean,
        lmean / tmean,
        tmax / lmean,
        tmax / tmean,
    ], np.float32)


def _pet_e_gated_vector(pet_raw, tumour, lung, dose_gy):
    HD_GY, LD_GY, MIN_VOX = 20.0, 5.0, 30
    g = tumour > 0
    normal = (lung > 0) & ~g
    suv_l = pet_raw[normal]
    suv_l = suv_l[np.isfinite(suv_l)]
    if not normal.any() or suv_l.size < MIN_VOX:
        return np.zeros(len(_PET_E_GATED_NAMES), np.float32)
    lmean = float(np.mean(suv_l))
    if lmean <= 1e-9:
        return np.zeros(len(_PET_E_GATED_NAMES), np.float32)
    hd = normal & (dose_gy >= HD_GY)
    ld = normal & (dose_gy < LD_GY)
    suv_hd = pet_raw[hd]; suv_hd = suv_hd[np.isfinite(suv_hd)]
    suv_ld = pet_raw[ld]; suv_ld = suv_ld[np.isfinite(suv_ld)]
    if suv_hd.size < MIN_VOX:
        return np.zeros(len(_PET_E_GATED_NAMES), np.float32)
    hdmean = float(np.mean(suv_hd))
    ldmean = float(np.mean(suv_ld)) if suv_ld.size >= MIN_VOX else lmean
    return np.array([
        hdmean / lmean,
        ldmean / lmean,
        hdmean / max(ldmean, 1e-9),
        float(np.std(suv_hd)) / max(hdmean, 1e-9),
        float(np.mean(suv_hd >= 1.5 * lmean)),
        float(np.percentile(suv_hd, 90)) / lmean,
        float(np.percentile(suv_hd, 95)) / lmean,
        float(np.max(suv_hd)) / lmean,
    ], np.float32)


def build_pet_plus_map() -> tuple[dict[str, np.ndarray], int]:
    if not USE_PET_PLUS:
        return {}, 0
    if PET_STREAM_RADIOMICS:
        fmap, names = _pet_radiomics_table()
        log.info("PET stream = radiomics (%s): %d feats/patient, %d uuids mapped.",
                 PET_RADIOMICS_TABLE, len(names), len(fmap))
        return dict(fmap), len(names)

    _fn = ("pete_gated_cache.json" if (PETE_SCALE_INVARIANT and PETE_DOSE_GATED)
           else "pete_cache.json" if PETE_SCALE_INVARIANT else "petplus_cache_sur.json")
    cache_path = _cache_read_path("petplus", _fn)
    cached: dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
        except Exception as e:
            log.warning("Failed to load PET+ cache: %s", e)

    if not os.path.isdir(_cohort_dir):
        return cached, pet_feature_dim()

    want = cohort_uuids()
    todo = [u for u in want if u.strip() not in cached]
    if not todo:
        log.info("PET+ stream (LOADED FROM CACHE): %d feats/patient, %d uuids mapped.",
                 pet_feature_dim(), len(cached))
        return cached, pet_feature_dim()
    log.info("PET+ cache covers %d/%d of the cohort; generating the %d missing "
             "patient(s) from 3D distances...", len(want) - len(todo), len(want), len(todo))

    out: dict[str, np.ndarray] = dict(cached)
    n_misaligned = 0
    from tqdm import tqdm
    for uuid in tqdm(todo, desc="PET+ Generation", unit="patient"):
        ud = os.path.join(_cohort_dir, uuid)
        p = resolve_modality_paths(ud)
        tumour_path = p["gtv"] or p["ptv"]
        if not (p["pet"] and tumour_path):
            continue
        try:
            pet = np.load(p["pet"]).astype(np.float32)
            gtv = np.load(tumour_path)
            if pet.shape != gtv.shape:
                n_misaligned += 1
                continue
            lung = np.load(p["lung"]) if p["lung"] else np.ones_like(gtv)
            liver = None
            if p.get("liver"):
                lv = np.load(p["liver"])
                liver = lv if lv.shape == pet.shape else None
            vec = (_pet_e_vector(pet, gtv, lung)
                   if PETE_SCALE_INVARIANT else
                   _pet_plus_vector(pet, gtv, lung, liver=liver))
            if PETE_SCALE_INVARIANT and PETE_DOSE_GATED:
                gated = np.zeros(len(_PET_E_GATED_NAMES), np.float32)
                if p.get("dose"):
                    dose = np.load(p["dose"]).astype(np.float32) / 100.0
                    if dose.shape == pet.shape:
                        gated = _pet_e_gated_vector(pet, gtv, lung, dose)
                vec = np.concatenate([vec, gated])
            out[uuid.strip()] = vec
        except Exception:
            continue
    log.info("PET+ stream: %d feats/patient, %d uuids mapped (%d skipped: "
             "PET grid != primary-study grid).", pet_feature_dim(), len(out), n_misaligned)

    try:
        cache_out = _cache_write_path("petplus", _fn)
        with open(cache_out, "w") as f:
            json.dump({k: v.tolist() for k, v in out.items()}, f)
        log.info("PET+ cache saved -> %s (%d patients; reused on later runs).", cache_out, len(out))
    except Exception as e:
        log.warning("Failed to save PET+ cache: %s", e)

    return out, pet_feature_dim()


def build_ct_radiomics_map() -> tuple[dict[str, np.ndarray], int]:
    if not USE_CT_RADIOMICS:
        return {}, 0
    if not os.path.isdir(_cohort_dir):
        log.warning("CT-radiomics stream disabled - cohort dir missing: %s", _cohort_dir)
        return {}, 0
    try:
        import ct_radiomics_gated as ctr
    except Exception as e:
        log.warning("CT-radiomics stream disabled - import failed: %s", e)
        return {}, 0
    k = int(os.getenv("NESTED_OUTER_FOLD", "0"))
    dose_gy = CT_RADIOMICS_DOSE_GY
    per_fold = ""
    if CT_RADIOMICS_PER_FOLD_D50:
        if GATE_EMB_DIR:
            dose_gy = ctr.gate_d50_for_fold(GATE_EMB_DIR, k, default_gy=CT_RADIOMICS_DOSE_GY,
                                            dose_norm_cgy=DOSE_NORM_CGY, log=log.info)
            per_fold = f" [gate d50, outer fold {k}]"
        else:
            log.warning("CT_RADIOMICS_PER_FOLD_D50 set but GATE_EMB_DIR is empty; "
                        "falling back to fixed %.1f Gy.", CT_RADIOMICS_DOSE_GY)
    curve = None
    if (CT_RADIOMICS_MASK == "gate" or CT_RADIOMICS_IMAGE == "gated"):
        if GATE_EMB_DIR:
            curve = ctr.gate_curve_for_fold(GATE_EMB_DIR, k, log=log.info)
        if curve is None:
            log.warning("CT_RADIOMICS_IMAGE/MASK request the gate but its curve is "
                        "unavailable; falling back to raw CT + dose threshold.")
    if curve is not None and CT_RADIOMICS_IMAGE == "gated" and CT_RADIOMICS_SPOTLIGHT_F < 1.0:
        curve = {**curve, "f": CT_RADIOMICS_SPOTLIGHT_F}
        log.info("Spotlight radiomics: CT'=CT*(f+(1-f)*g) with fixed f=%.2f (fold %d).",
                 CT_RADIOMICS_SPOTLIGHT_F, k)
    cache = os.path.join(_input_dir, CT_RADIOMICS_CACHE_DIR)
    _gate_based = (CT_RADIOMICS_MASK == "gate" or CT_RADIOMICS_IMAGE == "gated"
                   or CT_RADIOMICS_PER_FOLD_D50)
    fold_tag = f"_k{k}" if (_gate_based and GATE_EMB_DIR) else ""
    if fold_tag and CT_RADIOMICS_IMAGE == "gated" and CT_RADIOMICS_SPOTLIGHT_F < 1.0:
        fold_tag += f"_f{int(round(CT_RADIOMICS_SPOTLIGHT_F * 100)):03d}"
    thrs = CT_RADIOMICS_GATE_THRS if CT_RADIOMICS_MASK == "gate" else [0.5]

    def _one(thr):
        fm, nm = ctr.build_map(_cohort_dir, dose_gy, cache, log=log.info,
                               uuids=cohort_uuids(),
                               image=CT_RADIOMICS_IMAGE, mask=CT_RADIOMICS_MASK,
                               curve=curve, dose_norm_cgy=DOSE_NORM_CGY,
                               exclude_gtv=CT_RADIOMICS_EXCLUDE_GTV,
                               gtv_margin_vox=CT_RADIOMICS_GTV_MARGIN_VOX,
                               fold_tag=fold_tag, gate_thr=thr)
        return fm, nm

    if len(thrs) == 1:
        feat_map, names = _one(thrs[0])
    else:
        maps, name_parts = [], []
        for thr in thrs:
            fm, nm = _one(thr)
            maps.append(fm)
            name_parts.append([f"g{int(round(thr * 100)):03d}__{n}" for n in nm])
        shared = set.intersection(*[set(m) for m in maps]) if maps else set()
        names = [n for part in name_parts for n in part]
        feat_map = {u: np.concatenate([maps[i][u] for i in range(len(thrs))]).astype(np.float32)
                    for u in shared}
        log.info("Multi-threshold gate radiomics: thresholds=%s -> %d bands x per-band feats "
                 "concatenated to %d feats/patient (%d uuids shared).",
                 thrs, len(thrs), len(names), len(feat_map))
    dim = len(names)
    log.info("CT-radiomics stream: %d feats/patient, %d uuids mapped "
             "(dose>=%.1f Gy%s | image=%s | mask=%s%s).",
             dim, len(feat_map), dose_gy, per_fold, CT_RADIOMICS_IMAGE, CT_RADIOMICS_MASK,
             (f" | GTV-excl(+{CT_RADIOMICS_GTV_MARGIN_VOX}vox)" if CT_RADIOMICS_EXCLUDE_GTV else ""))
    return feat_map, dim


def build_side_features() -> tuple[dict[str, np.ndarray], int, int, int, int, int]:
    tab_map, clinrt_dim = build_tabular_map()
    dvh_map, dvh_dim = build_dvh_map()
    dosim_map, dosim_dim = (build_dvh_map(DVH_SOURCE2) if DVH_SOURCE2 else ({}, 0))
    suv_map, petplus_dim = build_pet_plus_map()
    rad_map, ctrad_dim = build_ct_radiomics_map()
    global _PETE_UUIDS
    _PETE_UUIDS = {str(u).strip() for u in suv_map}
    side_map: dict[str, np.ndarray] = {}
    for u in set(tab_map) | set(dvh_map) | set(dosim_map) | set(suv_map) | set(rad_map):
        tv = tab_map.get(u, np.zeros(clinrt_dim, np.float32))
        dv = dvh_map.get(u, np.zeros(dvh_dim, np.float32))
        d2 = dosim_map.get(u, np.zeros(dosim_dim, np.float32))
        sv = suv_map.get(u, np.zeros(petplus_dim, np.float32))
        rv = rad_map.get(u, np.zeros(ctrad_dim, np.float32))
        side_map[u] = np.concatenate([tv, dv, d2, sv, rv]).astype(np.float32)
    return side_map, clinrt_dim, dvh_dim, dosim_dim, petplus_dim, ctrad_dim


_PETE_UUIDS: set[str] = set()


def _inplane_affine(arr, ip_axes, angle_deg, scale, order):
    nd = arr.ndim
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    inv = (1.0 / scale) * np.array([[c, s], [-s, c]], dtype=np.float64)
    M = np.eye(nd)
    a0, a1 = ip_axes
    M[a0, a0], M[a0, a1] = inv[0, 0], inv[0, 1]
    M[a1, a0], M[a1, a1] = inv[1, 0], inv[1, 1]
    center = (np.asarray(arr.shape) - 1) / 2.0
    offset = center - M @ center
    return scipy.ndimage.affine_transform(
        arr, M, offset=offset, order=order, mode="nearest").astype(np.float32)


def _elastic_coords(shape_dhw):
    d, h, w = shape_dhw
    dy = scipy.ndimage.gaussian_filter(
        np.random.rand(d, h, w).astype(np.float32) * 2 - 1, ELASTIC_SIGMA,
        mode="constant") * ELASTIC_ALPHA
    dx = scipy.ndimage.gaussian_filter(
        np.random.rand(d, h, w).astype(np.float32) * 2 - 1, ELASTIC_SIGMA,
        mode="constant") * ELASTIC_ALPHA
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    return np.stack([zz, yy + dy, xx + dx]).astype(np.float32)


def _elastic_apply(vol, coords, order):
    return scipy.ndimage.map_coordinates(
        vol, coords, order=order, mode="nearest").astype(np.float32)


class SyncAug:
    def __init__(self, mode: str = "clean"):
        self.heavy = mode == "heavy"

    def __call__(self, ct, pet, dose, anat):
        if not self.heavy:
            return (np.ascontiguousarray(ct), np.ascontiguousarray(pet),
                    np.ascontiguousarray(dose), np.ascontiguousarray(anat))

        for ax in (1, 2):
            if np.random.rand() < P_FLIP:
                ct, pet, dose = np.flip(ct, ax), np.flip(pet, ax), np.flip(dose, ax)
                anat = np.flip(anat, axis=ax + 1)
        ct, pet, dose, anat = ct.copy(), pet.copy(), dose.copy(), anat.copy()

        do_rot = P_ROT > 0 and np.random.rand() < P_ROT
        do_scale = P_SCALE > 0 and np.random.rand() < P_SCALE
        if do_rot or do_scale:
            ang = np.random.uniform(-ROT_RANGE, ROT_RANGE) if do_rot else 0.0
            scl = np.random.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE) if do_scale else 1.0
            ct = _inplane_affine(ct, (1, 2), ang, scl, 1)
            pet = _inplane_affine(pet, (1, 2), ang, scl, 1)
            dose = _inplane_affine(dose, (1, 2), ang, scl, 1)
            anat = _inplane_affine(anat, (2, 3), ang, scl, 1)
            anat[0] = (anat[0] > 0.5).astype(np.float32)

        if P_ELASTIC > 0 and np.random.rand() < P_ELASTIC:
            coords = _elastic_coords(ct.shape)
            ct = _elastic_apply(ct, coords, 1)
            pet = _elastic_apply(pet, coords, 1)
            dose = _elastic_apply(dose, coords, 1)
            anat = np.stack([_elastic_apply(anat[c], coords, 1)
                             for c in range(anat.shape[0])])
            anat[0] = (anat[0] > 0.5).astype(np.float32)

        if P_GAMMA > 0 and np.random.rand() < P_GAMMA:
            g = np.random.uniform(GAMMA_RANGE[0], GAMMA_RANGE[1])
            ct = np.power(np.clip(ct, 0, 1), g).astype(np.float32)
        if P_BLUR > 0 and np.random.rand() < P_BLUR:
            ct = scipy.ndimage.gaussian_filter(ct, BLUR_SIGMA).astype(np.float32)
        if np.random.rand() < P_NOISE:
            ct = np.clip(ct + np.random.normal(0, NOISE_STD, ct.shape).astype(np.float32), 0, 1)
        if np.random.rand() < P_NOISE:
            pet = np.clip(pet + np.random.normal(0, NOISE_STD, pet.shape).astype(np.float32), 0, 1)

        return (np.ascontiguousarray(ct), np.ascontiguousarray(pet),
                np.ascontiguousarray(dose), np.ascontiguousarray(anat))


class MonaiSyncAug:
    def __init__(self, mode: str = "clean"):
        self.heavy = (mode == "heavy")
        self.tf = None
        if not self.heavy:
            return
        from monai.transforms import (Compose, RandFlipd, RandAffined,
                                       RandGaussianNoised, RandCoarseDropoutd)
        keys = ["ct", "pet", "dose", "anat"]
        gmode = ["bilinear", "bilinear", "bilinear", "nearest"]
        D, H, W = TARGET_SHAPE
        tfs = [RandFlipd(keys=keys, prob=P_FLIP, spatial_axis=1),
               RandFlipd(keys=keys, prob=P_FLIP, spatial_axis=2)]
        _aff_p = max(P_ROT, P_SCALE, 0.5 if AUG_TRANSLATE > 0 else 0.0)
        if _aff_p > 0:
            tfs.append(RandAffined(
                keys=keys, prob=_aff_p,
                rotate_range=(float(np.radians(ROT_RANGE)) if P_ROT > 0 else 0.0, 0.0, 0.0),
                scale_range=(0.0, SCALE_RANGE if P_SCALE > 0 else 0.0,
                             SCALE_RANGE if P_SCALE > 0 else 0.0),
                translate_range=(0.0, AUG_TRANSLATE * H, AUG_TRANSLATE * W),
                mode=gmode, padding_mode="border"))
        if P_NOISE > 0:
            tfs.append(RandGaussianNoised(keys=["ct"], prob=P_NOISE, mean=0.0, std=NOISE_STD))
        if P_CUTOUT > 0:
            hole = (max(1, int(CUTOUT_FRAC * D)), max(1, int(CUTOUT_FRAC * H)), max(1, int(CUTOUT_FRAC * W)))
            tfs.append(RandCoarseDropoutd(keys=["ct"], prob=P_CUTOUT, holes=CUTOUT_HOLES,
                                          spatial_size=hole, fill_value=0.0))
        self.tf = Compose(tfs)

    def __call__(self, ct, pet, dose, anat):
        if not self.heavy:
            return (np.ascontiguousarray(ct), np.ascontiguousarray(pet),
                    np.ascontiguousarray(dose), np.ascontiguousarray(anat))
        d = {"ct": ct[None].astype(np.float32), "pet": pet[None].astype(np.float32),
             "dose": dose[None].astype(np.float32), "anat": anat.astype(np.float32)}
        out = self.tf(d)
        _np = lambda x: np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        ct2, pet2 = _np(out["ct"])[0], _np(out["pet"])[0]
        dose2, anat2 = _np(out["dose"])[0], _np(out["anat"])
        anat2[0] = (anat2[0] > 0.5).astype(np.float32)
        ct2 = np.clip(ct2, 0.0, 1.0)
        return ct2, pet2, dose2, anat2


class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cohort_dir: str,
                 tab_map: dict[str, np.ndarray], tab_dim: int,
                 mode: str = "clean", cache: bool = True):
        self.cohort_dir = cohort_dir
        self.tab_map = tab_map
        self.tab_dim = tab_dim
        self.aug = MonaiSyncAug(mode) if AUG_BACKEND == "monai" else SyncAug(mode)
        self.tab_scaler: StandardScaler | None = None
        self._cache: dict | None = {} if cache else None
        self._emb_map: dict | None = None

        df = df.copy()
        df["UUID"] = df["UUID"].astype(str).str.strip()
        keep = []
        for _, r in df.iterrows():
            ud = os.path.join(cohort_dir, r["UUID"])
            ok = os.path.isdir(ud) and patient_has_required(ud)
            if ok and SINGLE_STUDY_ONLY and n_ct_studies(ud) > 1:
                ok = False
            if ok and EXCLUDE_MAJOR_REPLAN and is_major_replan(ud):
                ok = False
            keep.append(ok)
        self.df = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
        if SINGLE_STUDY_ONLY:
            log.info("SINGLE_STUDY_ONLY: kept %d single-course patients (n=96 default cohort).",
                     len(self.df))
        elif EXCLUDE_MAJOR_REPLAN:
            log.info("EXCLUDE_MAJOR_REPLAN: kept %d patients (single-course + boosts + minor "
                     "re-plans; MAJOR adaptive re-plans dropped - dose-clean expanded cohort).",
                     len(self.df))
        else:
            log.info("Full cohort: kept %d patients (all single- and multi-course).", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def set_emb(self, emb_map: dict | None):
        if emb_map is not None and emb_map and all(
                isinstance(v, dict) for v in emb_map.values()):
            self._emb_map = dict(emb_map)
        else:
            self._emb_map = None if emb_map is None else {"ct": emb_map}

    def _side_vector(self, uuid: str) -> torch.Tensor:
        if self.tab_dim > 0:
            raw = self.tab_map.get(uuid)
            raw = np.zeros(self.tab_dim, np.float32) if raw is None else raw.astype(np.float32)
            if self.tab_scaler is not None:
                raw = self.tab_scaler.transform(raw[None]).astype(np.float32)[0]
            return torch.from_numpy(np.ascontiguousarray(raw))
        return torch.zeros(1)

    def _pete_mask(self, uuid: str) -> torch.Tensor:
        return torch.tensor(1.0 if str(uuid).strip() in _PETE_UUIDS else 0.0)

    def _norm_ct(self, v: np.ndarray) -> np.ndarray:
        lo = WINDOW_CENTER - WINDOW_WIDTH / 2
        hi = WINDOW_CENTER + WINDOW_WIDTH / 2
        return np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32)

    def _norm_pet(self, v: np.ndarray) -> np.ndarray:
        return np.clip(v / SUV_MAX_CLIP, 0, 1).astype(np.float32)

    def _norm_dose(self, v: np.ndarray) -> np.ndarray:
        return np.clip(v / DOSE_NORM_CGY, 0, 1).astype(np.float32)

    def _resize(self, v: np.ndarray) -> np.ndarray:
        if v.shape == TARGET_SHAPE:
            return v
        zoom = [t / c for t, c in zip(TARGET_SHAPE, v.shape)]
        return scipy.ndimage.zoom(v, zoom, order=1, prefilter=False).astype(np.float32)

    @staticmethod
    def _lung_bbox(lung_path: str | None, shape: tuple) -> tuple[slice, slice, slice] | None:
        if not LUNG_CROP or lung_path is None:
            return None
        try:
            m = np.load(lung_path) > 0
        except Exception:
            return None
        if not m.any():
            return None
        sl = []
        for ax, n in enumerate(shape):
            idx = np.any(m, axis=tuple(i for i in range(3) if i != ax)).nonzero()[0]
            lo = max(int(idx[0]) - LUNG_CROP_MARGIN, 0)
            hi = min(int(idx[-1]) + 1 + LUNG_CROP_MARGIN, n)
            sl.append(slice(lo, hi))
        return tuple(sl)

    def _lobe_map(self, ln: np.ndarray, lung_path: str | None) -> np.ndarray:
        cc = np.zeros_like(ln, dtype=np.float32)
        if ln is None or not ln.any():
            return cc
        slice_area = ln.reshape(ln.shape[0], -1).sum(1).astype(np.float32)
        zc = np.where(slice_area > 0)[0]
        z_lo, z_hi = int(zc[0]), int(zc[-1])
        zaxis = np.arange(ln.shape[0], dtype=np.float32)
        lung_zc = float((slice_area @ zaxis) / slice_area.sum())
        inferior_high = False
        heart_path = (os.path.join(os.path.dirname(lung_path), "heart_mask.npy")
                      if lung_path else None)
        if heart_path and os.path.exists(heart_path):
            h = np.load(heart_path) > 0
            if h.shape == ln.shape and h.any():
                ha = h.reshape(h.shape[0], -1).sum(1).astype(np.float32)
                heart_zc = float((ha @ zaxis) / ha.sum())
                inferior_high = heart_zc > lung_zc
        t = np.clip((zaxis - z_lo) / max(z_hi - z_lo, 1), 0.0, 1.0)
        prof = t if inferior_high else (1.0 - t)
        return (prof[:, None, None] * ln).astype(np.float32)

    def _vuln_map(self, ct_hu: np.ndarray, ln: np.ndarray) -> np.ndarray:
        if ln is None or not ln.any():
            return np.zeros_like(ct_hu, dtype=np.float32)
        hu = ct_hu.astype(np.float32)
        fib = np.clip((hu + 750.0) / 500.0, 0.0, 1.0)
        if VULN_BIDIR:
            emph = np.clip((-850.0 - hu) / 150.0, 0.0, 1.0)
            v = np.maximum(fib, emph)
        else:
            v = fib
        return (v * ln).astype(np.float32)

    def _load_volumes(self, uuid: str):
        ud = os.path.join(self.cohort_dir, uuid)
        n_anat = _anat_layout()[0]
        if not USE_IMAGING:
            z = np.zeros(TARGET_SHAPE, dtype=np.float32)
            anat = np.zeros((n_anat, *TARGET_SHAPE), dtype=np.float32)
            return z, z.copy(), z.copy(), 0.0, anat, 1.0

        paths = resolve_modality_paths(ud)
        ct_raw = np.load(paths["ct"]).astype(np.float32)
        bbox = self._lung_bbox(paths["lung"], ct_raw.shape)

        def _crop(v):
            return v[bbox] if bbox is not None else v

        def _prep(vol, norm):
            return self._resize(norm(_crop(vol)))

        ct = _prep(ct_raw, self._norm_ct)
        dose_raw = np.load(paths["dose"]).astype(np.float32)
        dose = _prep(dose_raw, self._norm_dose)

        gtv_raw = np.load(paths["gtv"]) > 0 if paths["gtv"] else None
        lung_raw = np.load(paths["lung"]) > 0 if paths["lung"] else None
        if lung_raw is not None:
            ln = lung_raw & (~gtv_raw if gtv_raw is not None else True)
        else:
            ln = np.ones_like(ct_raw, dtype=bool)
        lung_normal = self._resize(_crop(ln.astype(np.float32)))
        if gtv_raw is not None and gtv_raw.any():
            spacing = ct_voxel_spacing(paths["ct"])
            dist = scipy.ndimage.distance_transform_edt(
                ~gtv_raw, sampling=spacing).astype(np.float32)
            peri = np.exp(-dist / max(PERI_TAU_MM, 1e-3)).astype(np.float32)
        else:
            peri = np.zeros_like(ct_raw, dtype=np.float32)
        peri = self._resize(_crop(peri))
        anat_maps = [lung_normal, peri]
        if USE_LOBE_GATE or USE_REGION_GATE:
            anat_maps.append(self._resize(_crop(
                self._lobe_map(ln, paths.get("lung")))))
        if USE_VULN_GATE:
            anat_maps.append(self._resize(_crop(
                self._vuln_map(ct_raw, ln))))
        anat = np.stack(anat_maps, axis=0).astype(np.float32)

        pet_conf = 1.0
        if USE_PET and USE_PET_VOL and paths["pet"] is not None:
            pet_raw = np.load(paths["pet"]).astype(np.float32)
            if gtv_raw is not None and pet_raw.shape != gtv_raw.shape:
                pet = np.zeros_like(ct)
                pet_mask = 0.0
                pet_conf = 0.0
            else:
                pet = _prep(pet_raw, self._norm_pet)
                pet_mask = 1.0
                if PET_ROBUST and gtv_raw is not None and gtv_raw.any():
                    bgm = (lung_raw & ~gtv_raw) if lung_raw is not None else ~gtv_raw
                    s_g = float(pet_raw[gtv_raw].mean())
                    s_b = float(pet_raw[bgm].mean()) if bgm.any() else 0.0
                    ratio = s_g / s_b if s_b > 1e-6 else 0.0
                    r0 = float(_full_cfg.get("PREPROCESS", {}).get("PET_REG_QA_MIN_RATIO", 1.3))
                    pet_conf = float(np.clip((ratio - 1.0) / max(r0 - 1.0, 1e-3), 0.0, 1.0))
        else:
            pet = np.zeros_like(ct)
            pet_mask = 0.0
            pet_conf = 0.0
        return ct, pet, dose, pet_mask, anat, pet_conf

    def _empty_sample(self):
        z = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
        return {
            "ct": torch.from_numpy(z), "pet": torch.from_numpy(z.copy()),
            "dose": torch.from_numpy(z.copy()),
            "anat": torch.zeros((_anat_layout()[0], *TARGET_SHAPE)),
            "side": torch.zeros(max(self.tab_dim, 1)),
            "pet_mask": torch.tensor(0.0), "pet_conf": torch.tensor(0.0),
            "pete_mask": torch.tensor(0.0),
            "label": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uuid = row["UUID"]
        if self._emb_map is not None:
            _feats = {}
            for _tok, _m in self._emb_map.items():
                feat = _m.get(uuid)
                if feat is None:
                    raise KeyError(f"cached {_tok} embedding missing for uuid {uuid} "
                                   "(fold cache incomplete - export was not run for this patient)")
                _feats[f"{_tok}_feat"] = torch.from_numpy(
                    np.ascontiguousarray(feat.astype(np.float32)))
            return {
                **_feats,
                "side": self._side_vector(uuid),
                "pet_mask": torch.tensor(0.0), "pet_conf": torch.tensor(0.0),
                "pete_mask": self._pete_mask(uuid),
                "label": torch.tensor(
                    1 if str(row.get("Group", "")).strip() == "RP" else 0, dtype=torch.long),
            }
        if not _IMAGING_TOKENS:
            s = self._empty_sample()
            s["side"] = self._side_vector(uuid)
            s["pete_mask"] = self._pete_mask(uuid)
            s["label"] = torch.tensor(
                1 if str(row.get("Group", "")).strip() == "RP" else 0, dtype=torch.long)
            return s
        try:
            if self._cache is not None and uuid in self._cache:
                ct, pet, dose, pet_mask, anat, pet_conf = (
                    a.copy() if isinstance(a, np.ndarray) else a for a in self._cache[uuid])
            else:
                ct, pet, dose, pet_mask, anat, pet_conf = self._load_volumes(uuid)
                if self._cache is not None:
                    self._cache[uuid] = (ct.copy(), pet.copy(), dose.copy(),
                                         pet_mask, anat.copy(), pet_conf)
            ct, pet, dose, anat = self.aug(ct, pet, dose, anat)
        except Exception as exc:
            log.error("Load failed for %s: %s", uuid, exc)
            return self._empty_sample()

        side_t = self._side_vector(uuid)

        label = 1 if str(row.get("Group", "")).strip() == "RP" else 0
        return {
            "ct": torch.from_numpy(ct[None]), "pet": torch.from_numpy(pet[None]),
            "dose": torch.from_numpy(dose[None]), "anat": torch.from_numpy(anat),
            "side": side_t,
            "pet_mask": torch.tensor(float(pet_mask)),
            "pet_conf": torch.tensor(float(pet_conf)),
            "pete_mask": self._pete_mask(uuid),
            "label": torch.tensor(label, dtype=torch.long),
        }


class HandcraftedBank3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_dim = 8
        kx = torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 1, 3)
        self.register_buffer("_kx", kx)
        self.register_buffer("_ky", kx.transpose(3, 4).contiguous())
        self.register_buffer("_kz", kx.transpose(2, 4).contiguous())

    def _loc(self, x, k):
        pad = k // 2
        m = nn.functional.avg_pool3d(x, k, stride=1, padding=pad)
        m2 = nn.functional.avg_pool3d(x * x, k, stride=1, padding=pad)
        sd = (m2 - m * m).clamp_min(1e-6).sqrt()
        return m, sd

    def forward_map(self, x):
        x4 = nn.functional.avg_pool3d(x, 4)
        gx = nn.functional.conv3d(x4, self._kx, padding=(0, 0, 1))
        gy = nn.functional.conv3d(x4, self._ky, padding=(0, 1, 0))
        gz = nn.functional.conv3d(x4, self._kz, padding=(1, 0, 0))
        grad = (gx * gx + gy * gy + gz * gz).clamp_min(1e-6).sqrt()
        m3, sd3 = self._loc(x4, 3)
        m7, sd7 = self._loc(x4, 7)
        mx = nn.functional.max_pool3d(x4, 5, stride=1, padding=2)
        mn = -nn.functional.max_pool3d(-x4, 5, stride=1, padding=2)
        return torch.cat([x4, m3, sd3, m7, sd7, grad, mx, mn], dim=1)

    def forward(self, x):
        f = self.forward_map(x)
        return f.mean(dim=(2, 3, 4))


class Backbone3D(nn.Module):
    def __init__(self, pretrained_path: str = "", depth: int = 18):
        super().__init__()
        self.use_deep_gate = USE_DEEP_GATE
        from monai.networks.nets import resnet as _mn
        _ctor = {10: _mn.resnet10, 18: _mn.resnet18, 34: _mn.resnet34,
                 50: _mn.resnet50, 101: _mn.resnet101, 152: _mn.resnet152}
        if depth not in _ctor:
            raise ValueError(f"Unsupported BACKBONE_DEPTH={depth}; pick {sorted(_ctor)}")
        self.depth = depth
        self.net = _ctor[depth](
            spatial_dims=3, n_input_channels=1, feed_forward=False,
            shortcut_type=BACKBONE_SHORTCUT,
            bias_downsample=BACKBONE_BIAS_DOWNSAMPLE, num_classes=2,
            widen_factor=BACKBONE_WIDEN)
        with torch.no_grad():
            probe = self.net(torch.zeros(1, 1, 32, 32, 32))
            if isinstance(probe, (list, tuple)):
                probe = probe[-1]
            if probe.dim() > 2:
                probe = probe.mean(dim=tuple(range(2, probe.dim())))
            self.feat_dim = probe.shape[1]
        if pretrained_path and os.path.exists(pretrained_path):
            self._load(pretrained_path)
        else:
            log.warning("Backbone3D(R%d): no MedicalNet weights at %r - random init.",
                        depth, pretrained_path)

    def _load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("trunk_state_dict") or ckpt.get("state_dict", ckpt)
        clean = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        missing, unexpected = self.net.load_state_dict(clean, strict=False)
        log.info("Backbone3D: loaded %s (missing=%d, unexpected=%d)",
                 os.path.basename(path), len(missing), len(unexpected))

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        return self

    def forward(self, x: torch.Tensor, gate_map: torch.Tensor = None) -> torch.Tensor:
        if not getattr(self, "use_deep_gate", False) or gate_map is None:
            feat = self.net(x)
        else:
            n = self.net
            x = n.conv1(x)
            x = n.bn1(x)
            x = n.relu(x)
            x = n.maxpool(x)

            def apply_gate(feat, g):
                if g.shape[2:] != feat.shape[2:]:
                    import torch.nn.functional as F
                    g_down = F.interpolate(g, size=feat.shape[2:], mode='trilinear', align_corners=False)
                else:
                    g_down = g
                return feat * g_down

            x = n.layer1(x);
            x = apply_gate(x, gate_map)
            x = n.layer2(x);
            x = apply_gate(x, gate_map)
            x = n.layer3(x);
            x = apply_gate(x, gate_map)
            x = n.layer4(x);
            x = apply_gate(x, gate_map)

            if hasattr(n, 'avgpool'):
                x = n.avgpool(x)
            feat = x

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() > 2:
            feat = feat.mean(dim=tuple(range(2, feat.dim())))
        return feat

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        cap = {}
        h = self.net.layer4.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o))
        try:
            self.net(x)
        finally:
            h.remove()
        f = cap["f"]
        return f[-1] if isinstance(f, (list, tuple)) else f


class DoseGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.steepness = nn.Parameter(torch.tensor(float(DG_INIT_A)))
        self.dose_thresh = nn.Parameter(torch.tensor(float(DG_INIT_B)))
        self.amplify = nn.Parameter(torch.tensor(float(DG_INIT_LAMBDA)))
        _, _ic, _iv = _anat_layout()
        if USE_ANATOMY_GATE:
            self.prox_weight = nn.Parameter(torch.tensor(0.0))
        if USE_LOBE_GATE and _ic >= 0:
            self.lobe_raw = nn.Parameter(torch.tensor(GATE_OFF_INIT))
        if _iv >= 0:
            self.vuln_raw = nn.Parameter(torch.tensor(GATE_OFF_INIT))
        self.baseline = nn.Parameter(torch.tensor(1.0))

        if USE_CONV_GATE:
            self.spatial_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
            nn.init.dirac_(self.spatial_conv.weight)

        self.use_cross_gate = USE_CROSS_GATE
        if self.use_cross_gate:
            self.cross_conv = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(8, 1, kernel_size=1)
            )
            nn.init.normal_(self.cross_conv[-1].weight, std=1e-3)
            nn.init.zeros_(self.cross_conv[-1].bias)

    def gate(self, dose: torch.Tensor, anat: torch.Tensor | None, ct: torch.Tensor | None = None) -> torch.Tensor:
        if USE_CONV_GATE:
            dose = self.spatial_conv(dose)
        if getattr(self, "use_cross_gate", False) and ct is not None:
            offset = self.cross_conv(ct)
            dose = dose - offset
        logit = self.steepness * (dose - self.dose_thresh)
        if USE_ANATOMY_GATE and anat is not None:
            peri = anat[:, 1:2]
            extra = self.prox_weight * peri
            _, ic, iv = _anat_layout()
            if USE_LOBE_GATE and ic >= 0:
                extra = extra + nn.functional.softplus(self.lobe_raw) * anat[:, ic:ic + 1]
            if iv >= 0:
                extra = extra + nn.functional.softplus(self.vuln_raw) * anat[:, iv:iv + 1]
            g = torch.sigmoid(logit + extra)
            g = g * anat[:, 0:1]
            return g
        return torch.sigmoid(logit)

    def forward(self, vol, dose, anat=None):
        return vol * (self.baseline + self.amplify * self.gate(dose, anat, ct=vol))


class DoseResponseGate(nn.Module):
    def __init__(self, K: int = DOSE_RESPONSE_K, f_init: float = SPOTLIGHT_INIT):
        super().__init__()
        self.K = K

        if USE_CONV_GATE:
            self.spatial_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
            nn.init.dirac_(self.spatial_conv.weight)

        self.use_cross_gate = USE_CROSS_GATE
        if self.use_cross_gate:
            self.cross_conv = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(8, 1, kernel_size=1)
            )
            nn.init.normal_(self.cross_conv[-1].weight, std=1e-3)
            nn.init.zeros_(self.cross_conv[-1].bias)

        self.cfg = _gate_config(K=K)
        self.b_raw = nn.Parameter(_gate.init_control_points(self.cfg))
        self.a_raw = nn.Parameter(_gate.init_slope(self.cfg))
        self.w_raw = nn.Parameter(torch.zeros(K))
        self._optional = _gate.build_terms(self, self.cfg)
        self.f_raw = nn.Parameter(torch.logit(torch.tensor(float(f_init))))
        with torch.no_grad():
            self.init_steepness = float(self.steepness)
            self.init_dose_thresh = float(self.dose_thresh)

    def _phi(self, d, shift=None):
        return _gate.phi(d, self.a_raw, self.b_raw, self.w_raw, shift)

    def gate(self, dose, anat=None, ct=None):
        dose, vshift = _gate.apply_dose_terms(self, self.cfg, dose, anat, ct)
        g = self._phi(dose, shift=vshift)
        return _gate.apply_anatomy_terms(self, self.cfg, g, anat,
                                         mask_to_lung=USE_ANATOMY_GATE)

    def forward(self, vol, dose, anat=None):
        f = torch.sigmoid(self.f_raw)
        return vol * (f + (1.0 - f) * self.gate(dose, anat, ct=vol))

    def reg(self):
        d = torch.linspace(0, 1, 21, device=self.b_raw.device)
        phi = self._phi(d.unsqueeze(-1))
        return ((phi[2:] - 2 * phi[1:-1] + phi[:-2]) ** 2).mean()

    def _d50(self):
        d = torch.linspace(0, 1, 201, device=self.b_raw.device)
        phi = self._phi(d.unsqueeze(-1))
        return d, phi, int(torch.argmin((phi - 0.5).abs()))

    @property
    def dose_thresh(self):
        with torch.no_grad():
            d, _, i = self._d50();
            return float(d[i])

    @property
    def steepness(self):
        with torch.no_grad():
            d, phi, i = self._d50();
            i0, i1 = max(i - 1, 0), min(i + 1, 200)
            return float((phi[i1] - phi[i0]) / ((d[i1] - d[i0]) + 1e-6))

    @property
    def amplify(self):
        return 1.0

    @property
    def prox_weight(self):
        with torch.no_grad():
            return float(nn.functional.softplus(self.prox_raw))

    @property
    def lobe_weight(self):
        if not hasattr(self, "lobe_raw"):
            return 0.0
        with torch.no_grad():
            return float(nn.functional.softplus(self.lobe_raw))

    @property
    def vuln_weight(self):
        if not hasattr(self, "vuln_raw"):
            return 0.0
        with torch.no_grad():
            return float(nn.functional.softplus(self.vuln_raw))

    @property
    def region_weight(self):
        if not hasattr(self, "region_delta"):
            return 0.0
        with torch.no_grad():
            return float(self.region_delta)

    @property
    def vuln_shift_gy(self):
        if not hasattr(self, "vuln_shift_raw"):
            return 0.0
        with torch.no_grad():
            return float(nn.functional.softplus(self.vuln_shift_raw)) * 60.0

    @property
    def baseline(self):
        with torch.no_grad():
            return float(torch.sigmoid(self.f_raw))

    def curve_dict(self):
        with torch.no_grad():
            a = nn.functional.softplus(self.a_raw);
            w = torch.softmax(self.w_raw, 0)
            b = torch.sigmoid(self.b_raw)
            d = torch.linspace(0, 1, 101, device=self.b_raw.device)
            phi = self._phi(d.unsqueeze(-1))
        return {"K": self.K, "b_norm": [float(x) for x in b],
                "b_Gy": [float(x) * DOSE_NORM_CGY / 100.0 for x in b],
                "a": [float(x) for x in a], "w": [float(x) for x in w],
                "f": float(torch.sigmoid(self.f_raw)),
                "prox_weight": float(nn.functional.softplus(self.prox_raw)),
                "d_grid": [float(x) for x in d], "phi": [float(x) for x in phi]}


def _mlp_head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 2))


_TEAM_VOCAB = {"ct": 0, "pet": 1, "dose": 2, "early": 3, "early_raw": 4,
               "tab": 5, "dvh": 6, "petplus": 7, "clinical": 8, "rt": 9, "ctrad": 10,
               "dosim": 11}
_TEAM_CLS_ID = len(_TEAM_VOCAB)

_resolve_gate = _gate.resolve_gate
gate_param_snapshot = _gate.param_snapshot
gate_grad_snapshot = _gate.grad_snapshot


def assert_gate_params_learnable(model, tag: str = "") -> list:
    return _gate.assert_learnable(model, strict=GATE_ASSERT_LEARNABLE, log=log, tag=tag)


def assert_gate_curve_active(model, tag: str = "") -> float:
    return _gate.assert_curve_active(model, min_span=GATE_MIN_CURVE_SPAN,
                                     strict=GATE_ASSERT_CURVE, log=log, tag=tag)


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int, heads: int, layers: int, ffn: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_emb = nn.Embedding(_TEAM_CLS_ID + 1, embed_dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, ffn), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(ffn, embed_dim), nn.Dropout(dropout))
            for _ in range(layers)])

    def forward(self, tokens, type_ids, key_padding_mask=None):
        B = tokens.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        tids = torch.cat([torch.full((1,), _TEAM_CLS_ID, device=tokens.device,
                                     dtype=torch.long), type_ids])
        x = x + self.type_emb(tids).unsqueeze(0)
        if key_padding_mask is not None:
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
                             key_padding_mask], dim=1)
        else:
            kpm = None
        last_attn = None
        for mha, ln1, ln2, ffn in zip(self.attn, self.ln1, self.ln2, self.ffn):
            h = ln1(x)
            a, w = mha(h, h, h, key_padding_mask=kpm, need_weights=True,
                       average_attn_weights=True)
            x = x + a
            x = x + ffn(ln2(x))
            last_attn = w
        cls_out = x[:, 0]
        tok_w = last_attn[:, 0, 1:]
        tok_w = tok_w / (tok_w.sum(dim=1, keepdim=True) + 1e-8)
        return cls_out, tok_w


class BBFusionNet(nn.Module):
    def __init__(self, clinrt_dim: int = 0, dvh_dim: int = 0, dosim_dim: int = 0,
                 petplus_dim: int = 0, ctrad_dim: int = 0, dropout: float = DROPOUT,
                 pretrained_path: str = _MEDICALNET_R18_PATH):
        super().__init__()
        self.clinrt_dim, self.dvh_dim, self.petplus_dim = clinrt_dim, dvh_dim, petplus_dim
        self.dosim_dim = dosim_dim
        self.ctrad_dim = ctrad_dim
        self.img_tokens = list(_IMAGING_TOKENS) if USE_IMAGING else []
        self.use_imaging = len(self.img_tokens) > 0
        self.has_side = (clinrt_dim + dvh_dim + dosim_dim + petplus_dim + ctrad_dim) > 0
        if not (self.use_imaging or self.has_side):
            raise ValueError("BBFusionNet needs at least one stream - all disabled.")

        if self.use_imaging and CACHE_IMAGING:
            _missing = [t for t in self.img_tokens if t not in GATE_EMB_DIRS]
            if _missing:
                raise ValueError(
                    f"CACHE_IMAGING has no export for imaging token(s) {_missing}; got "
                    f"tokens {self.img_tokens} and exports {sorted(GATE_EMB_DIRS)}. Give "
                    "each token its own directory under GATE_EMB_DIRS, or switch the "
                    "token off (USE_CT_VOL/USE_DOSE_VOL/USE_PET_VOL/USE_EARLY).")

            def _img_proj(fdim: int):
                if fdim is None:
                    raise RuntimeError("CACHE_IMAGING is on but the cached feature dim is "
                                       "unknown - load a fold cache before building the model.")
                if IMAGING_TOKEN_BOTTLENECK > 0:
                    _b = IMAGING_TOKEN_BOTTLENECK
                    return nn.Sequential(nn.Linear(fdim, _b), nn.GELU(),
                                         nn.Linear(_b, EMBED_DIM), nn.GELU())
                return nn.Sequential(nn.Linear(fdim, EMBED_DIM), nn.GELU())

            if self.img_tokens == ["ct"]:
                self.proj = _img_proj(_IMG_FEAT_DIM)
                self.projs = None
            else:
                self.projs = nn.ModuleDict(
                    {t: _img_proj(_IMG_FEAT_DIMS.get(t)) for t in self.img_tokens})
                self.proj = None
        elif self.use_imaging:
            _bb_w = _resolve_backbone_weights(BACKBONE_DEPTH) or pretrained_path
            self.backbone = Backbone3D(_bb_w, BACKBONE_DEPTH)
            fdim = self.backbone.feat_dim
            self.dose_gate = DoseResponseGate() if USE_DOSE_RESPONSE_GATE else DoseGate()
            if ({"early", "early_raw"} & set(self.img_tokens)):
                self.early_in = 2 if PET_ROBUST else 3
                self.early_adapter = nn.Conv3d(self.early_in, 1, 1, bias=False)
                with torch.no_grad():
                    self.early_adapter.weight.fill_(1.0 / self.early_in)
            self.proj = nn.Sequential(
                nn.Linear(fdim, FUSION_HIDDEN), nn.LayerNorm(FUSION_HIDDEN), nn.GELU(),
                nn.Linear(FUSION_HIDDEN, EMBED_DIM), nn.GELU()
            )

        def _tab_mlp(d_in):
            if SCALAR_ENCODER == "linear":
                return nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout),
                                     nn.Linear(d_in, EMBED_DIM))
            return nn.Sequential(
                nn.Linear(d_in, FUSION_HIDDEN), nn.LayerNorm(FUSION_HIDDEN),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(FUSION_HIDDEN, EMBED_DIM), nn.GELU())

        self._split_tab = SPLIT_TAB and clinrt_dim > 0
        self._clin_dim = (_CLIN_DIM if _CLIN_DIM else 30) if self._split_tab else 0
        self._rt_dim = (clinrt_dim - self._clin_dim) if self._split_tab else 0
        self._keep_clin = self._split_tab and TAB_KEEP != "rt"
        self._keep_rt = self._split_tab and TAB_KEEP != "clinical"
        if self._split_tab:
            self.clinical_enc = _tab_mlp(self._clin_dim) if self._keep_clin else None
            self.rt_enc = _tab_mlp(self._rt_dim) if self._keep_rt else None
            self.tab_enc = None
        else:
            self.tab_enc = _tab_mlp(clinrt_dim) if clinrt_dim > 0 else None
            self.clinical_enc = self.rt_enc = None
        self.dvh_enc = _tab_mlp(dvh_dim) if dvh_dim > 0 else None
        self.dosim_enc = _tab_mlp(dosim_dim) if dosim_dim > 0 else None
        self.petplus_enc = _tab_mlp(petplus_dim) if petplus_dim > 0 else None
        self.ctrad_enc = _tab_mlp(ctrad_dim) if ctrad_dim > 0 else None

        self._scalar_slices, _o = {}, 0
        if self._split_tab:
            if self._keep_clin:
                self._scalar_slices["clinical"] = (0, self._clin_dim)
            if self._keep_rt:
                self._scalar_slices["rt"] = (self._clin_dim, clinrt_dim)
            _o = clinrt_dim
        elif clinrt_dim > 0:
            self._scalar_slices["tab"] = (0, clinrt_dim);
            _o = clinrt_dim
        for _nm, _d in (("dvh", dvh_dim), ("dosim", dosim_dim),
                        ("petplus", petplus_dim), ("ctrad", ctrad_dim)):
            if _d > 0:
                self._scalar_slices[_nm] = (_o, _o + _d);
                _o += _d
        self.sk_experts: dict = {}
        self._cf_logits: dict = {}

        self.token_names = list(self.img_tokens)
        if clinrt_dim > 0:
            if self._split_tab:
                self.token_names += ([n for n, keep in (("clinical", self._keep_clin),
                                                        ("rt", self._keep_rt)) if keep])
            else:
                self.token_names.append("tab")
        if dvh_dim > 0:     self.token_names.append("dvh")
        if dosim_dim > 0:    self.token_names.append("dosim")
        if petplus_dim > 0:  self.token_names.append("petplus")
        if ctrad_dim > 0:   self.token_names.append("ctrad")

        self.scorer = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM // 2), nn.Tanh(),
                                    nn.Linear(EMBED_DIM // 2, 1))
        self.fuse_in = nn.Linear(EMBED_DIM, FUSION_HIDDEN)
        self.fuse_res = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN))
        self.fuse_out = _mlp_head(FUSION_HIDDEN, dropout)
        self.aux_heads = nn.ModuleDict({n: _mlp_head(EMBED_DIM, dropout)
                                        for n in self.token_names})

        self._scalar_set = {"tab", "clinical", "rt", "dvh", "dosim", "petplus", "ctrad"}
        self.has_scalar_token = any(n in self._scalar_set for n in self.token_names)
        self.alpha_raw = nn.Parameter(torch.tensor(float(FUSION_ALPHA_INIT_RAW)))
        self.img_head = _mlp_head(EMBED_DIM, dropout) if self.use_imaging else None

        self.use_team = FUSION_TEAM
        if self.use_team:
            self.team = CrossModalFusion(EMBED_DIM, FUSION_TEAM_HEADS,
                                         FUSION_TEAM_LAYERS, FUSION_HIDDEN, dropout)
            self.team_head = _mlp_head(EMBED_DIM, dropout)
            self.team_beta = nn.Parameter(torch.tensor(-2.0))

    def _encode_imaging(self, ct, pet, dose, anat, pet_mask, pet_conf):
        bb, proj = self.backbone, self.proj
        out: dict = {}
        if ({"ct", "early"} & set(self.img_tokens)):
            ct_g = self.dose_gate(ct, dose, anat)
            g_map = self.dose_gate.gate(dose, anat, ct=ct) if bb.use_deep_gate else None
        else:
            ct_g, g_map = None, None

        if "ct" in self.img_tokens:
            out["ct"] = proj(bb(ct_g, gate_map=g_map))
        if "dose" in self.img_tokens:
            out["dose"] = proj(bb(dose))
        if "pet" in self.img_tokens:
            e_pet = proj(bb(pet))
            if pet_mask is not None:
                e_pet = e_pet * pet_mask.view(-1, 1)
            if PET_ROBUST and pet_conf is not None:
                e_pet = e_pet * pet_conf.view(-1, 1)
            out["pet"] = e_pet
        if "early" in self.img_tokens:
            stack = (torch.cat([ct_g, dose], dim=1) if PET_ROBUST
                     else torch.cat([ct_g, pet, dose], dim=1))
            out["early"] = proj(bb(self.early_adapter(stack), gate_map=g_map))
        if "early_raw" in self.img_tokens:
            stack_r = (torch.cat([ct, dose], dim=1) if PET_ROBUST
                       else torch.cat([ct, pet, dose], dim=1))
            out["early_raw"] = proj(bb(self.early_adapter(stack_r)))
        return out

    def gradcam_early(self, batch: dict, device) -> "torch.Tensor | None":
        tok = ("early" if "early" in self.img_tokens
               else "ct" if "ct" in self.img_tokens else None)
        if not (self.use_imaging and tok is not None and hasattr(self, "backbone")):
            return None
        self.eval()
        ct = batch["ct"].to(device)
        dose = batch["dose"].to(device)
        anat = batch.get("anat")
        anat = anat.to(device) if anat is not None else None
        acts: dict = {}

        def _fhook(_m, _i, o):
            o.retain_grad()
            acts["f"] = o

        handle = self.backbone.net.layer4.register_forward_hook(_fhook)
        try:
            with torch.enable_grad():
                ctg = self.dose_gate(ct, dose, anat)
                if tok == "early":
                    stack = (torch.cat([ctg, dose], dim=1) if PET_ROBUST
                             else torch.cat([ctg, batch["pet"].to(device), dose], dim=1))
                    e = self.proj(self.backbone(self.early_adapter(stack)))
                else:
                    e = self.proj(self.backbone(ctg))
                logit = self.aux_heads[tok](e)
                self.zero_grad(set_to_none=True)
                logit[:, 1].sum().backward()
                fmap = acts.get("f")
                if fmap is None or fmap.grad is None:
                    return None
                w = fmap.grad.mean(dim=(2, 3, 4), keepdim=True)
                cam = torch.relu((w * fmap).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=ct.shape[2:], mode="trilinear",
                                    align_corners=False)
                return (cam / (cam.amax() + 1e-6)).detach()
        except Exception:
            return None
        finally:
            handle.remove()

    def forward(self, b: dict) -> dict:
        embeds: dict = {}
        if self.use_imaging:
            if CACHE_IMAGING:
                for _tok in self.img_tokens:
                    feat = b.get(f"{_tok}_feat")
                    if feat is None:
                        if _tok != "ct" or _LIVE_ENCODER is None:
                            raise RuntimeError(
                                f"no cached '{_tok}' feature in the batch and no LIVE "
                                "gate encoder is loaded")
                        with torch.no_grad():
                            feat = _LIVE_ENCODER.pooled_feature(
                                {"ct": b["ct"], "dose": b["dose"],
                                 "pet": b.get("pet"), "anat": b.get("anat")})
                    embeds[_tok] = (self.proj(feat) if self.projs is None
                                    else self.projs[_tok](feat))
            else:
                embeds.update(self._encode_imaging(
                    b["ct"], b["pet"], b["dose"], b.get("anat"),
                    b.get("pet_mask"), b.get("pet_conf")))
        side = b.get("side")
        if side is not None:
            o = 0
            if self._split_tab:
                if self.clinical_enc is not None:
                    embeds["clinical"] = self.clinical_enc(side[:, 0:self._clin_dim])
                if self.rt_enc is not None:
                    embeds["rt"] = self.rt_enc(side[:, self._clin_dim:self.clinrt_dim])
                o = self.clinrt_dim
            elif self.tab_enc is not None:
                embeds["tab"] = self.tab_enc(side[:, o:o + self.clinrt_dim]);
                o += self.clinrt_dim
            if self.dvh_enc is not None:
                embeds["dvh"] = self.dvh_enc(side[:, o:o + self.dvh_dim]);
                o += self.dvh_dim
            if self.dosim_enc is not None:
                embeds["dosim"] = self.dosim_enc(side[:, o:o + self.dosim_dim]);
                o += self.dosim_dim
            if self.petplus_enc is not None:
                embeds["petplus"] = self.petplus_enc(side[:, o:o + self.petplus_dim]);
                o += self.petplus_dim
            if self.ctrad_enc is not None:
                embeds["ctrad"] = self.ctrad_enc(side[:, o:o + self.ctrad_dim])

        names = [n for n in self.token_names if n in embeds]

        if self.use_team:
            toks = torch.stack([embeds[n] for n in names], dim=1)
            type_ids = torch.tensor([_TEAM_VOCAB[n] for n in names],
                                    device=toks.device, dtype=torch.long)
            B, T = toks.shape[0], len(names)
            kpm = torch.zeros(B, T, dtype=torch.bool, device=toks.device)
            if "pet" in names and b.get("pet_mask") is not None:
                kpm[:, names.index("pet")] = b["pet_mask"] < 0.5
            if "petplus" in names and b.get("pete_mask") is not None:
                kpm[:, names.index("petplus")] = b["pete_mask"] < 0.5
            if self.training and MODALITY_DROP > 0 and T > 1:
                drop = torch.rand(B, T, device=toks.device) < MODALITY_DROP
                keep_one = drop.all(dim=1)
                if keep_one.any():
                    drop[keep_one, 0] = False
                kpm = kpm | drop
            cls_out, tok_w = self.team(toks, type_ids, key_padding_mask=kpm)
            use_sk = SKLEARN_EXPERTS and bool(self.sk_experts)
            aux_logits, trainable_aux = {}, []
            _img_experts = getattr(self, "img_experts", None) or {}
            for n in names:
                _nf = b.get(f"{n}_feat")
                if n in _img_experts and _nf is not None:
                    aux_logits[n] = self._frozen_logit(
                        _img_experts[n], _nf, self._img_cfs.get(n, {}))
                elif use_sk and n in self.sk_experts and side is not None:
                    aux_logits[n] = self._sklearn_aux(n, side)
                else:
                    aux_logits[n] = self.aux_heads[n](embeds[n])
                    trainable_aux.append(n)
            _team_dbg = {}
            if (FUSION_MEAN_PROB or FUSION_MEAN_LOGIT or FUSION_MEDIAN_PROB
                    or FUSION_TRIM_PROB or FUSION_MIN_PROB or FUSION_RESIDUAL_MEAN):
                stacked = torch.stack([aux_logits[n] for n in names], dim=1)
                probs = torch.softmax(stacked, dim=-1)[..., 1]
                probs = probs.clamp(1e-6, 1.0 - 1e-6)
                keep = (~kpm).float()
                if FUSION_MEAN_LOGIT:
                    zs = torch.log(probs / (1.0 - probs))
                    z = (zs * keep).sum(dim=1) / (keep.sum(dim=1) + 1e-8)
                elif FUSION_MEDIAN_PROB:
                    pm = probs.masked_fill(kpm, float("nan"))
                    p = pm.nanmedian(dim=1).values.clamp(1e-6, 1.0 - 1e-6)
                    z = torch.log(p / (1.0 - p))
                elif FUSION_RESIDUAL_MEAN:
                    scal_keep = keep.clone()
                    for _ti, _n in enumerate(names):
                        if _n == "ct":
                            scal_keep[:, _ti] = 0.0
                    pbar = (probs * scal_keep).sum(dim=1) / (scal_keep.sum(dim=1) + 1e-8)
                    r = torch.zeros_like(pbar)
                    if getattr(self, "img_residual", None) is not None and _img_feat is not None:
                        _sci, _ridge = self.img_residual
                        _rn = _ridge.predict(_sci.transform(
                            _img_feat.detach().float().cpu().numpy()))
                        r = torch.from_numpy(_rn.astype(np.float32)).to(
                            device=pbar.device, dtype=pbar.dtype)
                    p = (pbar + r).clamp(1e-6, 1.0 - 1e-6)
                    z = torch.log(p / (1.0 - p))
                elif FUSION_MIN_PROB:
                    pm = probs.masked_fill(kpm, float("inf"))
                    p = pm.min(dim=1).values.clamp(1e-6, 1.0 - 1e-6)
                    z = torch.log(p / (1.0 - p))
                elif FUSION_TRIM_PROB:
                    pm = probs.masked_fill(kpm, float("-inf"))
                    top = pm.argmax(dim=1)
                    keep_t = keep.clone()
                    rows = torch.arange(keep.size(0), device=keep.device)
                    multi = keep.sum(dim=1) > 1
                    keep_t[rows[multi], top[multi]] = 0.0
                    p = (probs * keep_t).sum(dim=1) / (keep_t.sum(dim=1) + 1e-8)
                    p = p.clamp(1e-6, 1.0 - 1e-6)
                    z = torch.log(p / (1.0 - p))
                else:
                    p = (probs * keep).sum(dim=1) / (keep.sum(dim=1) + 1e-8)
                    p = p.clamp(1e-6, 1.0 - 1e-6)
                    z = torch.log(p / (1.0 - p))
                logit_final = torch.stack([torch.zeros_like(z), z], dim=1)
            elif FUSION_TEAM_LOGIT:
                _mix = names
                _img_extra = None
                if FUSION_IMAGING_RESIDUAL:
                    _mix = [n for n in names if n in self._scalar_set]
                    _img = [n for n in names if n not in self._scalar_set]
                    if _mix and _img:
                        _idx = [names.index(n) for n in _mix]
                        stacked = torch.stack([aux_logits[n] for n in _mix], dim=1)
                        w = tok_w[:, _idx].masked_fill(kpm[:, _idx], 0.0)
                        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
                        experts = (stacked * w.unsqueeze(-1)).sum(dim=1)
                        _img_extra = sum(aux_logits[n] for n in _img) / len(_img)
                if _img_extra is None:
                    stacked = torch.stack([aux_logits[n] for n in names], dim=1)
                    w = tok_w.masked_fill(kpm, 0.0)
                    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
                    experts = (stacked * w.unsqueeze(-1)).sum(dim=1)
                beta = nn.functional.softplus(self.team_beta)
                _inter = self.team_head(cls_out)
                logit_final = experts + beta * _inter
                _team_dbg = {"team_experts": experts, "team_interaction": beta * _inter,
                             "team_beta": beta.detach(), "team_mix_weights": w,
                             "team_mix_names": list(_mix)}
                if _img_extra is not None:
                    _alpha = nn.functional.softplus(self.alpha_raw)
                    logit_final = logit_final + _alpha * _img_extra
                    _team_dbg["team_img_term"] = _alpha * _img_extra
                    _team_dbg["team_alpha"] = _alpha.detach()
            else:
                logit_final = self.team_head(cls_out)
            if (FUSION_MEAN_PROB or FUSION_MEAN_LOGIT or FUSION_MEDIAN_PROB
                    or FUSION_TRIM_PROB or FUSION_MIN_PROB or FUSION_RESIDUAL_MEAN):
                keep = (~kpm).float()
                wlog = keep / (keep.sum(dim=1, keepdim=True) + 1e-8)
            else:
                wlog = tok_w * (~kpm).float()
            out = {"logit_final": logit_final, "fusion_weights": wlog,
                   "token_names": names, "alpha": torch.tensor(1.0, device=toks.device),
                   "trainable_aux": trainable_aux}
            for n in names:
                out[f"logit_{n}"] = aux_logits[n]
            out.update(_team_dbg)
            return out

        scalar_names = [n for n in names if n in self._scalar_set]
        img_names = [n for n in names if n not in self._scalar_set]

        def _attn_pool(nms, drop=False):
            if not nms:
                return None, None
            toks = torch.stack([embeds[n] for n in nms], dim=1)
            sc = self.scorer(toks).squeeze(-1)
            if ("pet" in nms and b.get("pet_mask") is not None) or \
                    ("petplus" in nms and b.get("pete_mask") is not None):
                mask = torch.zeros_like(sc, dtype=torch.bool)
                if "pet" in nms and b.get("pet_mask") is not None:
                    mask[:, nms.index("pet")] = b["pet_mask"] < 0.5
                if "petplus" in nms and b.get("pete_mask") is not None:
                    mask[:, nms.index("petplus")] = b["pete_mask"] < 0.5
                sc = sc.masked_fill(mask, float("-inf"))
            if drop and self.training and MODALITY_DROP > 0 and len(nms) > 1:
                keep = torch.rand(sc.shape[0], len(nms), device=sc.device) >= MODALITY_DROP
                sc = sc.masked_fill(~keep, float("-inf"))
                allinf = torch.isinf(sc).all(dim=1)
                if allinf.any():
                    sc = sc.masked_fill(allinf.unsqueeze(1), 0.0)
            ww = torch.softmax(sc, dim=1)
            return (toks * ww.unsqueeze(-1)).sum(dim=1), ww

        scal_fused, scal_w = _attn_pool(scalar_names, drop=True)
        scalar_logit = None
        if scal_fused is not None:
            if FUSION_LATE:
                slog = torch.stack([self.aux_heads[n](embeds[n]) for n in scalar_names],
                                   dim=1)
                scalar_logit = (slog * scal_w.unsqueeze(-1)).sum(dim=1)
            else:
                h = self.fuse_in(scal_fused)
                h = h + self.fuse_res(h)
                scalar_logit = self.fuse_out(h)

        img_fused, img_w = _attn_pool(img_names)
        alpha = nn.functional.softplus(self.alpha_raw)
        img_logit = self.img_head(img_fused) if img_fused is not None else None

        if scalar_logit is not None and img_logit is not None:
            logit_final = scalar_logit + alpha * img_logit
        elif scalar_logit is not None:
            logit_final = scalar_logit
        else:
            logit_final = img_logit

        wlog = torch.zeros(logit_final.shape[0], len(names), device=logit_final.device)
        for j, n in enumerate(names):
            if n in scalar_names and scal_w is not None:
                wlog[:, j] = scal_w[:, scalar_names.index(n)]
            elif n in img_names and img_w is not None:
                wlog[:, j] = float(alpha.detach()) * img_w[:, img_names.index(n)]

        out = {"logit_final": logit_final, "fusion_weights": wlog,
               "token_names": names, "alpha": alpha}
        if img_logit is not None and FUSION_SUPERVISE_IMG:
            out["logit_img"] = img_logit
        for n in names:
            out[f"logit_{n}"] = self.aux_heads[n](embeds[n])
        return out

    def fit_sklearn_experts(self, side_np: "np.ndarray", y_np: "np.ndarray",
                            side_val: "np.ndarray | None" = None,
                            y_val: "np.ndarray | None" = None,
                            pete_np: "np.ndarray | None" = None,
                            pete_val: "np.ndarray | None" = None,
                            img_np: "np.ndarray | None" = None,
                            img_val: "np.ndarray | None" = None) -> None:
        import inspect
        self.sk_experts = {}
        self._cf_logits = {}
        self.img_expert = None
        self.img_experts: dict = {}
        self._img_cfs: dict = {}
        self.img_residual = None
        self._img_cf = {}
        if len(set(y_np.tolist())) < 2:
            return
        has_val = (side_val is not None and y_val is not None
                   and len(set(np.asarray(y_val).tolist())) > 1)
        for nm, (s, e) in self._scalar_slices.items():
            Xa, ya = side_np, y_np
            Xv, yv = side_val, y_val
            if nm == "petplus" and pete_np is not None:
                keep = np.asarray(pete_np) > 0.5
                if keep.sum() < 4 or len(set(np.asarray(ya)[keep].tolist())) < 2:
                    continue
                Xa, ya = side_np[keep], np.asarray(y_np)[keep]
                if Xv is not None and pete_val is not None:
                    kv = np.asarray(pete_val) > 0.5
                    Xv, yv = (Xv[kv], np.asarray(yv)[kv]) if kv.any() else (None, None)
            X = Xa[:, s:e]
            sc = StandardScaler().fit(X)
            clf = _make_sklearn(SKLEARN_EXPERT_MODELS.get(nm, "mlp_128_64_32"))
            use_val = (has_val and Xv is not None and yv is not None
                       and len(set(np.asarray(yv).tolist())) > 1)
            accepts_val = use_val and "X_val" in inspect.signature(clf.fit).parameters
            if accepts_val:
                clf.fit(sc.transform(X), ya,
                        X_val=sc.transform(Xv[:, s:e]), y_val=np.asarray(yv))
            else:
                clf.fit(sc.transform(X), ya)
            self.sk_experts[nm] = (sc, clf)

            if EXPERT_CROSSFIT:
                self._cf_logits[nm] = self._crossfit_stream(
                    nm, X, ya, Xv[:, s:e] if (accepts_val and Xv is not None) else None,
                    np.asarray(yv) if accepts_val else None,
                    sc.transform(X), clf)

        _img_map = ({} if not USE_IMAGING_SKLEARN_EXPERT or img_np is None
                    else (dict(img_np) if isinstance(img_np, dict) else {"ct": img_np}))
        _ival_map = (img_val if isinstance(img_val, dict)
                     else ({} if img_val is None else {"ct": img_val}))
        for _tok, _xi in _img_map.items():
            if _xi is None or len(_xi) != len(y_np):
                continue
            Xi = np.asarray(_xi, np.float32)
            img_val = _ival_map.get(_tok)
            if IMAGING_EXPERT_RESIDUAL:
                assert EXPERT_CROSSFIT, "IMAGING_EXPERT_RESIDUAL needs EXPERT_CROSSFIT"
                pbar = np.zeros(len(y_np), np.float64); cnt = np.zeros(len(y_np))
                for _nm, (_s, _e) in self._scalar_slices.items():
                    _cf = self._cf_logits.get(_nm) or {}
                    for _j in range(len(y_np)):
                        _v = _cf.get(side_np[_j, _s:_e].tobytes())
                        if _v is not None:
                            pbar[_j] += 1.0 / (1.0 + np.exp(-_v)); cnt[_j] += 1
                ok = cnt > 0
                pbar[ok] /= cnt[ok]
                from sklearn.linear_model import Ridge
                sci_r = StandardScaler().fit(Xi[ok])
                ridge = Ridge(alpha=1.0).fit(sci_r.transform(Xi[ok]),
                                             (np.asarray(y_np, float) - pbar)[ok])
                self.img_residual = (sci_r, ridge)
                log.info("    residual-expert | target on %d/%d train patients, "
                         "|r| mean %.3f", int(ok.sum()), len(y_np),
                         float(np.abs((np.asarray(y_np, float) - pbar)[ok]).mean()))
                return
            sci = StandardScaler().fit(Xi)
            clfi = _make_sklearn(SKLEARN_EXPERT_MODELS.get(_tok, "rad_select_mlp"))
            accepts_val_i = (img_val is not None and y_val is not None
                             and len(set(np.asarray(y_val).tolist())) > 1
                             and "X_val" in inspect.signature(clfi.fit).parameters)
            if accepts_val_i:
                clfi.fit(sci.transform(Xi), y_np,
                         X_val=sci.transform(np.asarray(img_val, np.float32)),
                         y_val=np.asarray(y_val))
            else:
                clfi.fit(sci.transform(Xi), y_np)
            self.img_experts[_tok] = (sci, clfi)
            if _tok == "ct":
                self.img_expert = (sci, clfi)
            if EXPERT_CROSSFIT:
                _cf_img = self._crossfit_stream(
                    _tok, Xi, y_np,
                    np.asarray(img_val, np.float32) if accepts_val_i else None,
                    np.asarray(y_val) if accepts_val_i else None,
                    sci.transform(Xi), clfi)
                self._img_cfs[_tok] = _cf_img
                if _tok == "ct":
                    self._img_cf = _cf_img

    def _crossfit_stream(self, name, X, y, X_val, y_val, X_full_scaled, full_clf) -> dict:
        import inspect
        from sklearn.model_selection import StratifiedKFold
        n_fold = max(2, min(EXPERT_CROSSFIT_FOLDS, int(np.bincount(y).min())))
        if n_fold < 2 or len(y) < 8:
            return {}
        oof = np.full(len(y), np.nan, dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
        for tr, te in skf.split(X, y):
            if len(set(y[tr].tolist())) < 2:
                continue
            sc_i = StandardScaler().fit(X[tr])
            clf_i = _make_sklearn(SKLEARN_EXPERT_MODELS.get(name, "mlp_128_64_32"))
            if (X_val is not None and y_val is not None
                    and "X_val" in inspect.signature(clf_i.fit).parameters):
                clf_i.fit(sc_i.transform(X[tr]), y[tr],
                          X_val=sc_i.transform(X_val), y_val=y_val)
            else:
                clf_i.fit(sc_i.transform(X[tr]), y[tr])
            p = np.clip(clf_i.predict_proba(sc_i.transform(X[te]))[:, 1], 1e-6, 1 - 1e-6)
            oof[te] = np.log(p / (1 - p))

        ok = ~np.isnan(oof)
        if not ok.any():
            return {}
        try:
            p_in = full_clf.predict_proba(X_full_scaled)[:, 1]
            if len(set(y.tolist())) > 1:
                log.info("    crossfit[%s] | expert AUC on its own train set %.3f -> "
                         "out-of-fold %.3f (%d/%d patients)", name,
                         roc_auc_score(y, p_in), roc_auc_score(y[ok], oof[ok]),
                         int(ok.sum()), len(y))
        except Exception:
            pass
        return {X[i].tobytes(): float(oof[i]) for i in np.flatnonzero(ok)}

    def _frozen_logit(self, expert, feats: "torch.Tensor", cf: dict) -> "torch.Tensor":
        sc, clf = expert
        raw = feats.detach().float().cpu().numpy()
        p = np.clip(clf.predict_proba(sc.transform(raw))[:, 1], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).astype(np.float32)
        if cf:
            for i in range(len(z)):
                v = cf.get(raw[i].tobytes())
                if v is not None:
                    z[i] = v
        out2 = np.stack([np.zeros_like(z), z], axis=1)
        return torch.from_numpy(out2).to(device=feats.device, dtype=feats.dtype)

    def _sklearn_aux(self, name: str, side: "torch.Tensor") -> "torch.Tensor":
        s, e = self._scalar_slices[name]
        sc, clf = self.sk_experts[name]
        raw = side[:, s:e].detach().float().cpu().numpy()
        p = np.clip(clf.predict_proba(sc.transform(raw))[:, 1], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).astype(np.float32)
        cf = self._cf_logits.get(name)
        if cf:
            for i in range(len(z)):
                v = cf.get(raw[i].tobytes())
                if v is not None:
                    z[i] = v
        out2 = np.stack([np.zeros_like(z), z], axis=1)
        return torch.from_numpy(out2).to(device=side.device, dtype=side.dtype)

    @property
    def alpha_value(self) -> float:
        with torch.no_grad():
            return float(nn.functional.softplus(self.alpha_raw))


def make_model(clinrt_dim: int, dvh_dim: int, dosim_dim: int = 0, petplus_dim: int = 0,
               ctrad_dim: int = 0) -> BBFusionNet:
    return BBFusionNet(clinrt_dim=clinrt_dim, dvh_dim=dvh_dim, dosim_dim=dosim_dim,
                       petplus_dim=petplus_dim, ctrad_dim=ctrad_dim, dropout=DROPOUT,
                       pretrained_path=_MEDICALNET_R18_PATH)


def optimal_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(labels, probs)
    f1 = np.where((p + r) == 0, 0, 2 * p * r / (p + r + 1e-8))
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    return t if np.isfinite(t) else 0.5


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


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 1.0
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / torch.exp(logT), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(torch.exp(logT).item())
        return T if np.isfinite(T) and 0.05 < T < 20 else 1.0
    except Exception:
        return 1.0


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / max(T, 1e-6)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 1.0, 0.0
    n_pos = int(labels.sum());
    n_neg = int(len(labels) - n_pos)
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    t = np.where(labels == 1, t_pos, t_neg).astype(np.float32)
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    y = torch.tensor(t, dtype=torch.float32)
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(a * z + b, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        av, bv = float(a.detach().item()), float(b.detach().item())
        if not (np.isfinite(av) and np.isfinite(bv)) or not (0.05 < av < 20.0):
            return 1.0, 0.0
        return av, bv
    except Exception:
        return 1.0, 0.0


def apply_platt(probs: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    a, b = ab
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.clip(a * np.log(p / (1 - p)) + b, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=device)
        return w / w.sum() * 2.0
    return torch.ones(2, dtype=torch.float32, device=device)


_VOL_KEYS = ("ct", "pet", "dose", "anat")


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _flip_batch(b, ax):
    nb = dict(b)
    for k in _VOL_KEYS:
        nb[k] = torch.flip(b[k], [ax])
    return nb


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, all_p, all_y = 0.0, [], []
    for batch in loader:
        b = _to_device(batch, device)
        lbl = b["label"]
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
            out = model(b)
            loss = criterion(out["logit_final"], lbl)
            p = torch.softmax(out["logit_final"], 1)[:, 1]

            if not CACHE_IMAGING:
                for ax in (3, 4):
                    p = p + torch.softmax(model(_flip_batch(b, ax))["logit_final"], 1)[:, 1]
                p = p / 3.0

        all_p.extend(p.float().cpu().numpy().tolist())
        all_y.extend(lbl.cpu().numpy().tolist())
        total_loss += loss.item() * lbl.size(0)
    return total_loss / max(len(loader.dataset), 1), np.array(all_p), np.array(all_y, dtype=int)


@torch.no_grad()
def evaluate_mc(model, loader, device, n_mc: int = N_MC_SAMPLES):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    all_mean, all_std, all_y = [], [], []
    for batch in tqdm(loader, desc=f"MC-dropout x{n_mc}", unit="batch", leave=False,
                      dynamic_ncols=True, disable=not SHOW_PROGRESS):
        b = _to_device(batch, device)
        samples = torch.stack([torch.softmax(model(b)["logit_final"], 1)[:, 1].cpu()
                               for _ in range(n_mc)])
        all_mean.append(samples.mean(0).numpy())
        all_std.append(samples.std(0).numpy())
        all_y.extend(b["label"].cpu().numpy().tolist())
    return (np.concatenate(all_mean), np.concatenate(all_std),
            np.array(all_y, dtype=int))


def _mixup(b: dict, alpha: float):
    lbl = b["label"]
    if alpha <= 0 or lbl.size(0) < 2:
        return b, lbl, lbl, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(lbl.size(0), device=lbl.device)
    nb = dict(b)
    for k in ("ct", "pet", "dose", "anat", "side", "pet_mask", "pet_conf", "pete_mask"):
        nb[k] = lam * b[k] + (1 - lam) * b[k][perm]
    return nb, lbl, lbl[perm], lam


def _aux_loss(out, criterion, y_a, y_b, lam, pmask, pemask=None) -> torch.Tensor:
    losses = []
    _gate = {"pet": pmask, "petplus": pemask}
    for n in out.get("trainable_aux", out["token_names"]):
        k = f"logit_{n}"
        if n in _gate:
            msk = _gate[n]
            if msk is None or msk.sum() == 0:
                continue
            m = msk > 0.5
            losses.append(lam * criterion(out[k][m], y_a[m]) +
                          (1 - lam) * criterion(out[k][m], y_b[m]))
        else:
            losses.append(lam * criterion(out[k], y_a) + (1 - lam) * criterion(out[k], y_b))
    if "logit_img" in out:
        losses.append(lam * criterion(out["logit_img"], y_a) +
                      (1 - lam) * criterion(out["logit_img"], y_b))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=y_a.device)


def train_one(train_loader, val_loader, device, train_labels,
              clinrt_dim, dvh_dim, dosim_dim=0, petplus_dim=0, ctrad_dim=0, seed_offset=0,
              accumulation=ACCUMULATION, max_epochs=MAX_EPOCHS,
              train_eval_loader=None, ext_outer=0, ext_inner=0):
    set_seed(SEED + seed_offset)
    model = make_model(clinrt_dim, dvh_dim, dosim_dim, petplus_dim, ctrad_dim).to(device)

    wandb.init(
        project="rp-fuse",
        group="rpfuse",
        name=f"fold_O{ext_outer}_I{ext_inner}",
        reinit=True
    )

    for n, p in model.named_parameters():
        if "backbone" in n:
            p.requires_grad = False
    gate_params = [p for n, p in model.named_parameters()
                   if "dose_gate" in n and p.requires_grad]
    use_alpha_grp = ALPHA_LR > 0
    alpha_params = [p for n, p in model.named_parameters()
                    if n.endswith("alpha_raw") and p.requires_grad] if use_alpha_grp else []
    head_params = [p for n, p in model.named_parameters()
                   if "backbone" not in n and "dose_gate" not in n and p.requires_grad
                   and not (use_alpha_grp and n.endswith("alpha_raw"))]
    groups = [{"params": head_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
    if gate_params:
        groups.append({"params": gate_params, "lr": GATE_LR, "weight_decay": 0.0})
    if alpha_params:
        groups.append({"params": alpha_params, "lr": ALPHA_LR, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups, lr=LR, weight_decay=WEIGHT_DECAY)

    if SKLEARN_EXPERTS and getattr(model, "_scalar_slices", None):
        _sides, _ys, _pes, _imgs = [], [], [], {}
        for _b in train_loader:
            _s = _b.get("side")
            if _s is not None:
                _sides.append(_s.cpu().numpy());
                _ys.append(_b["label"].cpu().numpy())
                _m = _b.get("pete_mask")
                _pes.append(_m.cpu().numpy() if _m is not None
                            else np.ones(len(_s), np.float32))
                for _tok in (_IMAGING_TOKENS if CACHE_IMAGING else ()):
                    _f = _b.get(f"{_tok}_feat")
                    if _f is not None:
                        _imgs.setdefault(_tok, []).append(_f.cpu().numpy())
        _vsides, _vys, _vpes, _vimgs = [], [], [], {}
        for _b in val_loader:
            _s = _b.get("side")
            if _s is not None:
                _vsides.append(_s.cpu().numpy());
                _vys.append(_b["label"].cpu().numpy())
                _m = _b.get("pete_mask")
                _vpes.append(_m.cpu().numpy() if _m is not None
                             else np.ones(len(_s), np.float32))
                for _tok in (_IMAGING_TOKENS if CACHE_IMAGING else ()):
                    _f = _b.get(f"{_tok}_feat")
                    if _f is not None:
                        _vimgs.setdefault(_tok, []).append(_f.cpu().numpy())
        if _sides:
            _val_side = np.concatenate(_vsides) if _vsides else None
            _val_y = np.concatenate(_vys) if _vys else None
            _pe = np.concatenate(_pes) if _pes else None
            _val_pe = np.concatenate(_vpes) if _vpes else None
            if (_pe is not None and _pe.min() < 0.5
                    and "petplus" in getattr(model, "_scalar_slices", {})):
                log.info("  PET-E present in %d/%d train patients - the PET-E expert is "
                         "fitted on those only and the token is masked for the rest.",
                         int((_pe > 0.5).sum()), len(_pe))
            model.fit_sklearn_experts(np.concatenate(_sides), np.concatenate(_ys),
                                      side_val=_val_side, y_val=_val_y,
                                      pete_np=_pe, pete_val=_val_pe,
                                      img_np={t: np.concatenate(v) for t, v in _imgs.items()}
                                      if _imgs else None,
                                      img_val={t: np.concatenate(v) for t, v in _vimgs.items()}
                                      if _vimgs else None)
            for _tok in (getattr(model, "img_experts", None) or {}):
                log.info("  Imaging expert fit [%s=%s] on the same %d train patients - the "
                         "%s stream now enters frozen, like the scalar streams.", _tok,
                         SKLEARN_EXPERT_MODELS.get(_tok, "rad_select_mlp"),
                         sum(len(y) for y in _ys), _tok)
            _expert_str = ", ".join(f"{nm}={SKLEARN_EXPERT_MODELS.get(nm, 'mlp_128_64_32')}"
                                     for nm in model.sk_experts.keys())
            log.info("  Scalar-stream experts fit [%s] on %d train patients%s.",
                     _expert_str, sum(len(y) for y in _ys),
                     f" (+{len(_val_y)} inner-val for NN early-stop)" if _val_y is not None else "")

    criterion = FocalLoss(alpha=_class_weights(train_labels, device), gamma=FOCAL_GAMMA)
    if FOCAL_GAMMA_FINAL != FOCAL_GAMMA:
        log.info("  Focal gamma annealing ON: %.2f -> %.2f over %.0f%% of %d epochs",
                 FOCAL_GAMMA, FOCAL_GAMMA_FINAL, 100 * FOCAL_GAMMA_ANNEAL_FRAC, max_epochs)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and AMP_DTYPE == torch.float16
    scaler = GradScaler("cuda") if use_scaler else None

    sched_mode = "min" if CHECKPOINT_METRIC == "loss" else "max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, factor=0.5, patience=7, min_lr=1e-7)

    best_auc, best_loss, best_state, best_thr, epoch_count = -1.0, float("inf"), None, 0.5, 0
    no_improve = 0
    auc_hist: list[float] = []
    loss_hist: list[float] = []
    gate_pretrain = 0 if max_epochs <= 1 else GATE_PRETRAIN_EPOCHS
    expert_pretrain = 0 if max_epochs <= 1 else EXPERT_PRETRAIN_EPOCHS
    min_epochs = max(EARLY_STOP_MIN_EPOCHS, gate_pretrain + 10, expert_pretrain + 10)
    if gate_pretrain:
        log.info("  Gate pre-training: %d epochs on the imaging aux head(s) before fusion",
                 gate_pretrain)
    if expert_pretrain:
        log.info("  Expert pre-training: %d epochs training each stream's encoder+aux "
                 "head standalone%s before the handshake trains on top.",
                 expert_pretrain, " (experts then FROZEN)" if EXPERT_FREEZE else "")
    experts_frozen = False
    unfrozen = False
    gate_hist: list[dict] = []
    _gate_checked = False
    _gate_init = (gate_param_snapshot(model.dose_gate)
                  if hasattr(model, "dose_gate") else {})
    if _gate_init:
        log.info("[gate] %s init: %s", type(model.dose_gate).__name__,
                 {k: round(v, 4) for k, v in _gate_init.items()})

    if FREEZE_BACKBONE:
        log.info("  Backbone FROZEN for the whole run (FREEZE_BACKBONE=true) - "
                 "only the gate/fusion/heads learn.")
    for epoch in range(1, max_epochs + 1):
        criterion.gamma = _focal_gamma_at(epoch, max_epochs)
        if (EXPERT_FREEZE and expert_pretrain and epoch == expert_pretrain + 1
                and not experts_frozen):
            frozen_n = 0
            for mod in (model.tab_enc, model.clinical_enc, model.rt_enc,
                        model.dosim_enc, model.ctrad_enc, model.dvh_enc, model.petplus_enc,
                        getattr(model, "proj", None), model.aux_heads):
                if mod is not None:
                    for p in mod.parameters():
                        if p.requires_grad:
                            p.requires_grad = False;
                            frozen_n += p.numel()
            experts_frozen = True
            log.info("  Experts FROZEN after pre-training (%d params) - only the "
                     "handshake (attention + CLS + beta) trains now.", frozen_n)
        if not FREEZE_BACKBONE and epoch == WARMUP_EPOCHS + 1 and not unfrozen:
            bb = [p for n, p in model.named_parameters() if "backbone" in n]
            if bb:
                for p in bb:
                    p.requires_grad = True
                optimizer.add_param_group({"params": bb, "lr": LR * 0.1})
            unfrozen = True

        model.train()
        optimizer.zero_grad()
        n_steps = len(train_loader)
        epoch_loss = 0.0
        _bar = tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs}", unit="batch",
                    leave=False, dynamic_ncols=True, disable=not SHOW_PROGRESS)
        for step, batch in enumerate(_bar, 1):
            b = _to_device(batch, device)
            lbl = b["label"]
            use_mix = (P_MIXUP > 0 and MIXUP_ALPHA > 0 and lbl.size(0) > 1
                       and np.random.rand() < P_MIXUP)
            if use_mix:
                b, y_a, y_b, lam = _mixup(b, MIXUP_ALPHA)
            else:
                y_a, y_b, lam = lbl, lbl, 1.0

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                out = model(b)
                if epoch <= expert_pretrain:
                    loss = _aux_loss(out, criterion, y_a, y_b, lam, b.get("pet_mask"),
                                     b.get("pete_mask"))
                    loss = loss / accumulation
                elif epoch <= gate_pretrain:
                    imk = [k for k in ("logit_ct", "logit_early") if k in out]
                    loss = sum(lam * criterion(out[k], y_a)
                               + (1 - lam) * criterion(out[k], y_b) for k in imk)
                    loss = loss / max(len(imk), 1) / accumulation
                else:
                    main = lam * criterion(out["logit_final"], y_a) + \
                           (1 - lam) * criterion(out["logit_final"], y_b)
                    aux = _aux_loss(out, criterion, y_a, y_b, lam, b.get("pet_mask"),
                                    b.get("pete_mask"))
                    loss = (main + AUX_LOSS_WEIGHT * aux) / accumulation

            if loss.requires_grad:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if not _gate_checked:
                    _gate_checked = True
                    assert_gate_params_learnable(model)
                    if hasattr(model, "dose_gate"):
                        log.info("[gate] dose-response span over 0-%.0f Gy: %.3f",
                                 DOSE_NORM_CGY / 100.0,
                                 assert_gate_curve_active(model.dose_gate))
                        log.info("[gate] first-step |grad|: %s",
                                 {k: f"{v:.2e}" for k, v in
                                  gate_grad_snapshot(model.dose_gate).items()})
            epoch_loss += loss.item() * accumulation
            _bar.set_postfix(loss=f"{loss.item() * accumulation:.4f}", refresh=False)
            if loss.requires_grad and (step % accumulation == 0 or step == n_steps):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

        val_loss, vp, vl = evaluate(model, val_loader, criterion, device)
        val_auc = (roc_auc_score(vl, vp)
                   if len(set(vl.tolist())) > 1 else 0.5)
        epoch_count = epoch

        thr_e = optimal_f1_threshold(vp, vl)
        vm = metrics_at_threshold(vp, np.asarray(vl), thr_e)
        rec = {"epoch": epoch, "train_auc": float("nan"), "train_f1": float("nan"),
               "val_auc": float(val_auc), "val_f1": float(vm["f1"])}
        train_eval_loss = None
        if LOG_TRAIN_HISTORY and train_eval_loader is not None:
            train_eval_loss, tpe, tle = evaluate(model, train_eval_loader, criterion, device)
            tm = metrics_at_threshold(tpe, np.asarray(tle), thr_e)
            rec["train_auc"], rec["train_f1"] = float(tm["auc"]), float(tm["f1"])
        if hasattr(model, "dose_gate"):
            dg = model.dose_gate
            rec.update({"steepness": float(dg.steepness), "dose_thresh": float(dg.dose_thresh),
                        "amplify": float(dg.amplify),
                        "prox_weight": float(getattr(dg, "prox_weight", 0.0)),
                        "baseline": float(dg.baseline),
                        "threshold_Gy": float(dg.dose_thresh) * DOSE_NORM_CGY / 100.0})
            _snap = gate_param_snapshot(dg)
            for k, v in _snap.items():
                rec[f"gate.{k}"] = v
                rec[f"gate.d_{k}"] = v - _gate_init.get(k, v)
            _moved = {k: round(rec[f"gate.d_{k}"], 5) for k in _snap}
            log.info("[gate] epoch %d drift-from-init: %s", epoch, _moved)
        gate_hist.append(rec)

        avg_train_loss = epoch_loss / max(n_steps, 1)
        log_dict = {
            "Loss/train_batch_avg": avg_train_loss,
            "Loss/val": val_loss,
            "AUC/val": val_auc,
            "F1/val": rec["val_f1"],
        }
        if train_eval_loss is not None:
            log_dict["Loss/train_eval"] = train_eval_loss
        if not np.isnan(rec["train_auc"]):
            log_dict["AUC/train"] = rec["train_auc"]
            log_dict["F1/train"] = rec["train_f1"]
        if hasattr(model, "dose_gate"):
            for k in ("steepness", "dose_thresh", "amplify", "prox_weight", "baseline", "threshold_Gy"):
                if k in rec:
                    log_dict[f"Gate/{k}"] = rec[k]
            for k, v in rec.items():
                if k.startswith("gate."):
                    log_dict[f"GateRaw/{k[5:]}"] = v
        wandb.log(log_dict, step=epoch)

        if epoch <= gate_pretrain:
            continue
        auc_hist.append(val_auc)
        loss_hist.append(val_loss)
        sm_auc = float(np.mean(auc_hist[-3:]))
        sm_loss = float(np.mean(loss_hist[-3:]))

        if CHECKPOINT_METRIC == "loss":
            scheduler.step(sm_loss)
            is_better = sm_loss < best_loss - 1e-4
        else:
            scheduler.step(sm_auc)
            is_better = sm_auc > best_auc + 1e-4

        if epoch > WARMUP_EPOCHS and is_better:
            best_loss = sm_loss
            best_auc, best_thr = sm_auc, optimal_f1_threshold(vp, vl)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif epoch > WARMUP_EPOCHS:
            no_improve += 1
        if epoch >= min_epochs and no_improve >= EARLY_STOP_PATIENCE:
            log.debug("  Early stop @ epoch %d (best val metric [%s]: loss=%.4f, AUC=%.3f)",
                      epoch, CHECKPOINT_METRIC, best_loss, best_auc)
            break

    if best_state:
        model.load_state_dict(best_state)

    if getattr(model, "use_team", False) and hasattr(model, "team_beta"):
        with torch.no_grad():
            _b = float(nn.functional.softplus(model.team_beta))
        log.info("  Fusion readout | beta=%.4f (init 0.1269, %+.1f%% from init)",
                 _b, 100.0 * (_b / 0.12692801654338837 - 1.0))
    model._gate_history = gate_hist
    wandb.finish()
    return model, best_thr, epoch_count


def _generate_gradcam(model, dataset, test_idx, test_uuids, test_labels,
                      test_probs, thr, device, out_dir):
    pos = [j for j, (lb, pr) in enumerate(zip(test_labels, test_probs))
           if int(lb) == 1 and pr >= thr]
    pos.sort(key=lambda j: test_probs[j], reverse=True)
    pos = pos[:N_GRADCAM]
    if not pos:
        log.info("Grad-CAM: no true-positive test cases this fold.")
        return
    os.makedirs(out_dir, exist_ok=True)
    dose20 = 2000.0 / DOSE_NORM_CGY
    cases = []
    for j in pos:
        sample = dataset[test_idx[j]]
        batch = {k: v.unsqueeze(0) for k, v in sample.items()
                 if isinstance(v, torch.Tensor)}
        cam = model.gradcam_early(batch, device)
        if cam is None:
            continue
        cam_np = cam.squeeze().cpu().numpy()
        dose_np = batch["dose"].squeeze().cpu().numpy()
        anat_np = batch["anat"].squeeze().cpu().numpy()
        normal = anat_np[0] > 0.5
        cam_hot = (cam_np >= 0.5) & normal
        dose_hot = (dose_np >= dose20) & normal
        inter = float(np.logical_and(cam_hot, dose_hot).sum())
        union = float(np.logical_or(cam_hot, dose_hot).sum())
        iou = inter / union if union > 0 else 0.0
        uuid = test_uuids[j]
        np.savez_compressed(os.path.join(out_dir, f"gradcam_{uuid}.npz"),
                            ct=batch["ct"].squeeze().cpu().numpy(), dose=dose_np,
                            normal_lung=anat_np[0], peri=anat_np[1], cam=cam_np)
        cases.append({"uuid": uuid, "prob": float(test_probs[j]),
                      "label": int(test_labels[j]), "iou_cam_vs_20Gy": round(iou, 3)})
        log.info("Grad-CAM %s: prob=%.3f  IoU(cam,>=20Gy normal lung)=%.3f",
                 uuid, float(test_probs[j]), iou)
    if cases:
        mean_iou = round(float(np.mean([c["iou_cam_vs_20Gy"] for c in cases])), 3)
        with open(os.path.join(out_dir, "summary.json"), "w") as fp:
            json.dump({"n_cases": len(cases), "mean_iou_cam_vs_20Gy": mean_iou,
                       "cases": cases}, fp, indent=2)
        log.info("Grad-CAM: %d cases, mean IoU vs >=20 Gy normal lung = %.3f",
                 len(cases), mean_iou)


def resolve_split(manifest_df: pd.DataFrame):
    import pandas as pd
    import os
    from util import load_config

    _cfg_for_split = load_config()
    cohort_dir = os.path.join(_cfg_for_split["INPUTS_DIR"], _cfg_for_split["INPUTS"]["PROCESSED"])
    raw_split, allowed = cohort_split()

    keep = []
    for _, r in raw_split.iterrows():
        ud = os.path.join(cohort_dir, str(r["uuid"]).strip())
        ok_cls = os.path.isdir(ud) and study_class(ud) in allowed
        ofld = r.get("OuterFold", -1)
        keep.append(bool(ok_cls) and pd.notna(ofld) and int(ofld) >= 0)

    split = raw_split[pd.Series(keep, index=raw_split.index)].reset_index(drop=True)
    uuid_to_row = {str(u).strip(): i for i, u in enumerate(manifest_df["UUID"].values)}

    dev_rows, test_rows, dev_fold_of_row = [], [], {}
    for _, r in split.iterrows():
        uid = str(r["uuid"]).strip()
        if uid not in uuid_to_row:
            continue
        mrow = uuid_to_row[uid]
        if r["Split"] == "Test":
            test_rows.append(mrow)
        else:
            dev_rows.append(mrow)
            dev_fold_of_row[mrow] = int(r["Fold"])

    dev_rows = sorted(dev_rows)
    test_rows = sorted(test_rows)
    dev_folds = [dev_fold_of_row.get(m, 0) for m in dev_rows]
    if TRAIN_FRACTION < 1.0:
        lab_of = {m: (1 if str(manifest_df.iloc[m]["Group"]).strip() == "RP" else 0)
                  for m in dev_rows}
        rng = np.random.default_rng(1000 * TRAIN_FRACTION_SEED
                                    + int(os.getenv("NESTED_OUTER_FOLD", "0")))
        keep = set()
        for fi in sorted(set(dev_folds)):
            for lab in (0, 1):
                members = [m for m, f in zip(dev_rows, dev_folds) if f == fi and lab_of[m] == lab]
                n_keep = max(1, int(round(TRAIN_FRACTION * len(members)))) if members else 0
                if n_keep:
                    keep.update(int(x) for x in rng.choice(members, n_keep, replace=False))
        n_full = len(dev_rows)
        dev_rows = [m for m in dev_rows if m in keep]
        dev_folds = [dev_fold_of_row.get(m, 0) for m in dev_rows]
        log.info("TRAIN_FRACTION=%.2f (seed %d): development set subsampled %d -> %d patients "
                 "(RP+=%d); outer test fold untouched.", TRAIN_FRACTION, TRAIN_FRACTION_SEED,
                 n_full, len(dev_rows), sum(lab_of[m] for m in dev_rows))

    folds = []
    for fi in sorted(set(dev_folds)):
        val_rel = np.array([i for i, f in enumerate(dev_folds) if f == fi])
        tr_rel = np.array([i for i, f in enumerate(dev_folds) if f != fi])
        folds.append((tr_rel, val_rel))
    log.info("Split sizes | cohort=%d  dev=%d  test(outer held-out)=%d  inner_folds=%d",
             len(dev_rows) + len(test_rows), len(dev_rows), len(test_rows), len(folds))
    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        log.info("    inner fold %d | train=%d  val=%d", fi, len(tr_rel), len(val_rel))
    return dev_rows, test_rows, folds


def _load_fold_cache(outer_k: int, inner_i: int, tok: str = "ct"):
    global _IMG_FEAT_DIM
    tag = f"fold{outer_k}_fold{inner_i}"
    d = os.path.join(GATE_EMB_DIRS.get(tok, GATE_EMB_DIR), f"outer_{outer_k}")
    npz_path = os.path.join(d, f"imaging_emb_{tag}.npz")
    part_path = os.path.join(d, f"fold_partition_{tag}.json")
    if not (os.path.exists(npz_path) and os.path.exists(part_path)):
        raise FileNotFoundError(
            f"gate_ct embedding cache missing for {tag}: expected {npz_path} and "
            f"{part_path}. Re-run the conv gate_ct with TRAIN_GATE_CT.EXPORT_EMB=true.")
    z = np.load(npz_path, allow_pickle=True)
    uuids = [str(u).strip() for u in z["uuids"].tolist()]
    feats = z["feats"]
    emb_map = {u: feats[j] for j, u in enumerate(uuids)}
    with open(part_path) as fp:
        part = json.load(fp)
    fdim = int(part["feat_dim"])
    _prev = _IMG_FEAT_DIMS.get(tok)
    if _prev is None:
        _IMG_FEAT_DIMS[tok] = fdim
    elif _prev != fdim:
        raise RuntimeError(f"[cache] feature-dim drift at {tag} ({tok}): {_prev} vs {fdim}")
    if tok == "ct" or _IMG_FEAT_DIM is None:
        _IMG_FEAT_DIM = fdim
    return emb_map, part


def _load_gate_encoder(outer_k: int, inner_i: int, device):
    global _IMG_FEAT_DIM
    import train_gate_ct as tg
    tag = f"fold{outer_k}_fold{inner_i}"
    d = os.path.join(GATE_EMB_DIR, f"outer_{outer_k}")
    ckpt_path = os.path.join(d, f"gatenet_{tag}.pt")
    part_path = os.path.join(d, f"fold_partition_{tag}.json")
    if not (os.path.exists(ckpt_path) and os.path.exists(part_path)):
        raise FileNotFoundError(
            f"LIVE gate encoder missing for {tag}: expected {ckpt_path} and {part_path}. "
            "Re-run train_gate_ct with EXPORT_EMB=true (writes gatenet_*.pt with __arch__).")
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.pop("__arch__", None)
    if arch is None:
        raise RuntimeError(f"{ckpt_path} has no __arch__ block - re-export with the current "
                           "train_gate_ct (older gatenet_*.pt cannot be rebuilt).")
    if not arch.get("backbone_frozen", True):
        raise RuntimeError(
            "LIVE mode needs a FROZEN-backbone gate (TRAIN_ONLY_GATE=true); this gate "
            "fine-tuned its backbone, whose weights were not exported. Re-run the gate frozen.")
    _sv = {k: globals()[k] for k in ("PET_ROBUST", "BACKBONE_DEPTH",
                                     "USE_ANATOMY_GATE", "USE_LOBE_GATE", "USE_REGION_GATE")}
    _tg_keys = ("GATE_NTCP_K", "GATE_AS_CHANNEL", "GATE_EARLY_CT_ONLY", "GATE_NTCP",
                "GATE_SPOTLIGHT", "GATE_SPOTLIGHT_INIT", "GATE_DISPERSION", "GATE_EMBED",
                "TOKEN", "HARD_MASK", "NO_GATE", "_HARD_MASK_THR", "TRAIN_ONLY_GATE",
                "GATE_WARMUP_UNFREEZE", "GATE_INPUT_VOL")
    _tgsv = {k: getattr(tg, k) for k in _tg_keys}
    try:
        globals()["PET_ROBUST"] = arch["pet_robust"]
        globals()["BACKBONE_DEPTH"] = arch["backbone_depth"]
        globals()["USE_ANATOMY_GATE"] = arch.get("use_anatomy_gate", False)
        globals()["USE_LOBE_GATE"] = arch.get("use_lobe_gate", False)
        globals()["USE_REGION_GATE"] = arch.get("use_region_gate", False)
        tg.GATE_DOSE_RESPONSE_K = arch["ntcp_k"];          tg.GATE_AS_CHANNEL = arch["gate_as_channel"]
        tg.GATE_EARLY_CT_ONLY = arch["early_ct_only"]; tg.GATE_DOSE_RESPONSE = arch["ntcp"]
        tg.GATE_SPOTLIGHT = arch["spotlight"];    tg.GATE_SPOTLIGHT_INIT = arch["spotlight_init"]
        tg.GATE_DISPERSION = arch["dispersion"];  tg.GATE_EMBED = arch["embed"]
        tg.TOKEN = arch["token"];                 tg.HARD_MASK = arch["hard_mask"]
        tg.NO_GATE = arch["no_gate"];             tg._HARD_MASK_THR = arch["hard_mask_thr"]
        tg.GATE_INPUT_VOL = arch.get("input_vol", "ct")
        tg.TRAIN_ONLY_GATE = True;                tg.GATE_WARMUP_UNFREEZE = False
        enc = tg.GateNet(dropout=arch["dropout"])
    finally:
        for k, v in _sv.items():   globals()[k] = v
        for k, v in _tgsv.items(): setattr(tg, k, v)
    missing, unexpected = enc.load_state_dict(ckpt, strict=False)
    missing = [m for m in missing if not m.startswith("backbone.")]
    _opt = ("gate.prox_raw", "gate.lobe_raw", "gate.vuln_raw", "gate.region_delta",
            "gate.vuln_shift_raw", "gate.grad_gamma_raw")
    unexpected = [u for u in unexpected if u not in _opt]
    if missing or unexpected:
        raise RuntimeError(f"gate encoder weight mismatch at {tag}: missing={missing} "
                           f"unexpected={unexpected}")
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    _IMG_FEAT_DIM = enc.proj[0].in_features
    with open(part_path) as fp:
        part = json.load(fp)
    return enc, part


def _set_live_encoder(enc):
    global _LIVE_ENCODER
    _LIVE_ENCODER = enc


def _assert_fold_alignment(part, manifest_df, tr_abs, val_abs, test_idx, outer_k, inner_i):
    def _us(idxs):
        return set(str(manifest_df.iloc[i]["UUID"]).strip() for i in idxs)

    exp = {"train": _us(tr_abs), "val": _us(val_abs), "test": _us(test_idx)}
    got = {"train": set(part["train"]), "val": set(part["val"]), "test": set(part["test"])}
    for name in ("train", "val", "test"):
        _ok = (exp[name] <= got[name]) if (TRAIN_FRACTION < 1.0 and name != "test") \
              else (exp[name] == got[name])
        if not _ok:
            raise RuntimeError(
                f"[leakage guard] fold(k={outer_k},i={inner_i}) {name} UUIDs differ between "
                f"fusion and gate_ct cache - inner folds NOT aligned. "
                f"missing_in_cache={sorted(exp[name] - got[name])[:5]}, "
                f"extra_in_cache={sorted(got[name] - exp[name])[:5]}. Abort.")
    leak = (got["train"] & got["val"]) | (got["train"] & got["test"])
    if leak:
        raise RuntimeError(f"[leakage guard] fold(k={outer_k},i={inner_i}) encoder train set "
                           f"overlaps val/test: {sorted(leak)[:5]}. Abort.")


def _fit_tab_scaler(datasets, manifest_df, dev_idx, tab_map, tab_dim):
    if tab_dim == 0:
        return
    uuids = [str(manifest_df.iloc[i]["UUID"]).strip() for i in dev_idx]
    X = np.array([tab_map.get(u, np.zeros(tab_dim, np.float32)) for u in uuids],
                 dtype=np.float32)
    sc = StandardScaler().fit(X)
    for ds in datasets:
        ds.tab_scaler = sc


def _loader_kwargs(bs):
    nw = 0 if CACHE_IMAGING else NUM_WORKERS
    kw = dict(batch_size=bs, num_workers=nw, pin_memory=True)
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def _shutdown(*loaders):
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


def save_patient_xlsx(uuids, y_true, probs, threshold, unc=None, threshold_y=None):
    os.makedirs(PRED_DIR, exist_ok=True)
    preds = (probs >= threshold).astype(int)
    cols = {
        "UUID": uuids,
        "TrueLabel": y_true,
        "TrueLabel_text": ["RP" if v else "No RP" for v in y_true],
        "RP_Probability": np.round(probs, 4),
        "PredictedLabel": preds,
        "Predicted_text": ["RP" if v else "No RP" for v in preds],
        "Correct": (preds == y_true).astype(int),
    }
    if threshold_y is not None:
        cols["PredictedLabel_Youden"] = (probs >= threshold_y).astype(int)
    if unc is not None:
        cols["Uncertainty_MC_std"] = np.round(unc, 4)
    pd.DataFrame(cols).to_excel(
        os.path.join(PRED_DIR, "RP-Fuse_test_predictions.xlsx"), index=False)


def _mean_std(results, key):
    out = {}
    for k in ["auc", "f1", "sensitivity", "specificity", "acc"]:
        vals = [r[key][k] for r in results]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


class _MaskPETEAtTest(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        s = dict(self.base[i])
        s["pete_mask"] = torch.tensor(0.0)
        return s


@torch.no_grad()
def log_patient_contributions(model, loader, device, uuids, y_true) -> None:
    model.eval()
    rows, i = [], 0
    for batch in loader:
        b = _to_device(batch, device)
        out = model(b)
        if "team_experts" not in out:
            log.info("patient_contributions: no team-logit terms in this model, skipped.")
            return
        names = list(out["token_names"])
        B = int(out["logit_final"].size(0))
        _d = lambda t: (t[:, 1] - t[:, 0]).detach().cpu().numpy()
        z_final, z_exp, z_int = _d(out["logit_final"]), _d(out["team_experts"]), _d(out["team_interaction"])
        w_all = out["fusion_weights"].detach().cpu().numpy()
        mix, w_mix = list(out["team_mix_names"]), out["team_mix_weights"].detach().cpu().numpy()
        z_img = _d(out["team_img_term"]) if "team_img_term" in out else None
        z_tok = {n: _d(out[f"logit_{n}"]) for n in names}
        for j in range(B):
            r = {"uuid": uuids[i + j], "label": int(y_true[i + j]),
                 "logit_final": float(z_final[j]),
                 "prob_final": float(1.0 / (1.0 + np.exp(-z_final[j]))),
                 "experts_term": float(z_exp[j]), "interaction_term": float(z_int[j]),
                 "beta": float(out["team_beta"])}
            if z_img is not None:
                r["imaging_term"] = float(z_img[j]); r["alpha_vol"] = float(out["team_alpha"])
            for t, n in enumerate(names):
                r[f"attn_w_{n}"] = float(w_all[j, t]); r[f"z_{n}"] = float(z_tok[n][j])
                if n in mix:
                    wm = float(w_mix[j, mix.index(n)])
                    r[f"w_{n}"] = wm; r[f"contrib_{n}"] = wm * float(z_tok[n][j])
            rows.append(r)
        i += B
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "patient_contributions.csv"), index=False)
    log.info("patient_contributions.csv: %d test patients, streams=%s", len(rows), names)


@torch.no_grad()
def log_component_effects(model, loader, device, y_true) -> None:
    model.eval()
    names = list(model.token_names)
    w_sum = {n: 0.0 for n in names}
    aux_p = {n: [] for n in names}
    pete_flag: list[float] = []
    n_obs = 0
    for batch in loader:
        b = _to_device(batch, device)
        out = model(b)
        w = out["fusion_weights"]
        for j, n in enumerate(out["token_names"]):
            w_sum[n] += float(w[:, j].sum())
            aux_p[n].extend(torch.softmax(out[f"logit_{n}"], 1)[:, 1].cpu().numpy().tolist())
        _pm = b.get("pete_mask")
        pete_flag.extend(_pm.cpu().numpy().tolist() if _pm is not None
                         else [1.0] * int(w.size(0)))
        n_obs += w.size(0)

    rows = []
    for n in names:
        probs = np.array(aux_p[n])
        valid = len(probs) == len(y_true) and len(set(y_true.tolist())) > 1
        if valid:
            thr = optimal_f1_threshold(probs, y_true)
            m = metrics_at_threshold(probs, y_true, thr)
        else:
            m = {k: np.nan for k in ("auc", "pr_auc", "f1", "f2", "sensitivity", "specificity", "brier")}

        def _r(x):
            return round(float(x), 4) if np.isfinite(x) else np.nan

        rows.append({"component": n,
                     "mean_fusion_weight": round(w_sum[n] / max(n_obs, 1), 4),
                     "standalone_auc": _r(m["auc"]), "pr_auc": _r(m["pr_auc"]),
                     "f1": _r(m["f1"]), "f2": _r(m["f2"]),
                     "sensitivity": _r(m["sensitivity"]), "specificity": _r(m["specificity"]),
                     "brier": _r(m["brier"])})
    df = pd.DataFrame(rows).sort_values("mean_fusion_weight", ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "component_effects.csv"), index=False)
    try:
        diag = {"label": list(np.asarray(y_true).tolist())}
        if len(pete_flag) == len(y_true):
            diag["has_pete"] = pete_flag
        for n in names:
            if len(aux_p[n]) == len(y_true):
                diag[f"aux_{n}"] = aux_p[n]
        pd.DataFrame(diag).to_csv(os.path.join(RESULTS_DIR, "aux_oof.csv"), index=False)
    except Exception as _e:
        log.info("    [diag] aux_oof dump skipped: %s", _e)
    log.info("Component effects - per-stream standalone success (fusion weight | metrics @ F1-opt threshold):")

    def _c(x):
        return f"{x:.3f}" if pd.notna(x) else " n/a "

    for _, r in df.iterrows():
        log.info("    %-7s w=%.3f | AUC=%s PR=%s F1=%s F2=%s Sn=%s Sp=%s BS=%s",
                 r["component"], r["mean_fusion_weight"], _c(r["standalone_auc"]),
                 _c(r["pr_auc"]), _c(r["f1"]), _c(r["f2"]), _c(r["sensitivity"]),
                 _c(r["specificity"]), _c(r["brier"]))


def main(smoke: bool = False) -> None:
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    device = resolve_device()
    log.info("Device: %s | smoke=%s | imaging_tokens=%s (lung_crop=%s) | "
             "Tabular=%s | DVH=%s | anatomy_gate=%s | pet_robust=%s | pet_plus=%s | "
             "scalar_encoder=%s | petplus_feats=%d",
             device, smoke, _IMAGING_TOKENS if USE_IMAGING else [], LUNG_CROP,
             USE_TABULAR, USE_DVH, USE_ANATOMY_GATE, PET_ROBUST, USE_PET_PLUS,
             SCALAR_ENCODER, pet_feature_dim())

    manifest_df_raw = read_manifest()
    side_map, clinrt_dim, dvh_dim, dosim_dim, petplus_dim, ctrad_dim = build_side_features()
    side_dim = clinrt_dim + dvh_dim + dosim_dim + petplus_dim + ctrad_dim

    use_cache = CACHE_VOLUMES and not smoke
    ds_clean = MultiModalDataset(manifest_df_raw, _cohort_dir, side_map, side_dim,
                                 mode="clean", cache=use_cache)
    ds_noisy = MultiModalDataset(manifest_df_raw, _cohort_dir, side_map, side_dim,
                                 mode="heavy", cache=use_cache)
    manifest_df = ds_clean.df
    log.info("Patients on disk (CT+Dose): %d / %d", len(manifest_df), len(manifest_df_raw))

    dev_idx, test_idx, folds = resolve_split(manifest_df)
    labels = [1 if str(g).strip() == "RP" else 0 for g in manifest_df["Group"].values]
    dev_labels = [labels[i] for i in dev_idx]
    test_labels = [labels[i] for i in test_idx]
    test_uuids = [str(manifest_df.iloc[i]["UUID"]) for i in test_idx]
    log.info("Dev=%d (RP+=%.0f%%)  Test=%d (RP+=%.0f%%)",
             len(dev_idx), 100 * np.mean(dev_labels) if dev_labels else 0,
             len(test_idx), 100 * np.mean(test_labels) if test_labels else 0)

    bs = max(BATCH_SIZE, 2) if not smoke else 2
    max_epochs = 1 if smoke else MAX_EPOCHS
    if smoke:
        folds = folds[:1]

    _ds_test = _MaskPETEAtTest(ds_clean) if TEST_MASK_PETE else ds_clean
    if TEST_MASK_PETE:
        log.info("TEST_MASK_PETE: outer-test patients are scored with the PET-E token masked.")
    te = (DataLoader(Subset(_ds_test, test_idx), shuffle=False, **_loader_kwargs(bs))
          if (test_idx and not smoke) else None)
    cv_results = []
    best = None
    fold_test_probs = []
    fold_test_aucs = []
    oof_val_probs, oof_val_lbls = [], []
    for fi, (tr_rel, val_rel) in enumerate(
            tqdm(folds, desc="CV folds", unit="fold", dynamic_ncols=True,
                 disable=not SHOW_PROGRESS), 1):
        set_seed(SEED + fi)
        tr_abs = [dev_idx[i] for i in tr_rel]
        val_abs = [dev_idx[i] for i in val_rel]
        tr_lbl = [labels[i] for i in tr_abs]
        val_lbl = [labels[i] for i in val_abs]
        if smoke:
            tr_abs, val_abs = tr_abs[:3], val_abs[:2]
            tr_lbl, val_lbl = [labels[i] for i in tr_abs], [labels[i] for i in val_abs]

        log.info("  Fold %d sizes | train=%d (RP+=%d/%.0f%%)  val=%d (RP+=%d/%.0f%%)  "
                 "test(outer)=%d", fi,
                 len(tr_abs), sum(tr_lbl), 100 * np.mean(tr_lbl) if tr_lbl else 0,
                 len(val_abs), sum(val_lbl), 100 * np.mean(val_lbl) if val_lbl else 0,
                 len(test_idx))
        _fit_tab_scaler([ds_clean, ds_noisy], manifest_df, tr_abs, side_map, side_dim)

        if CACHE_IMAGING and _IMAGING_TOKENS:
            _ok = int(os.getenv("NESTED_OUTER_FOLD", "0"))
            if GATE_LIVE_ENCODER:
                _enc, _part = _load_gate_encoder(_ok, fi - 1, device)
                _assert_fold_alignment(_part, manifest_df, tr_abs, val_abs, test_idx, _ok, fi - 1)
                _set_live_encoder(_enc)
                ds_clean.set_emb(None)
                ds_noisy.set_emb(None)
            else:
                _emb_maps = {}
                for _tok in _IMAGING_TOKENS:
                    _emb_maps[_tok], _part = _load_fold_cache(_ok, fi - 1, _tok)
                    _assert_fold_alignment(_part, manifest_df, tr_abs, val_abs, test_idx,
                                           _ok, fi - 1)
                ds_clean.set_emb(_emb_maps)
                ds_noisy.set_emb(_emb_maps)

        tl = DataLoader(Subset(ds_noisy, tr_abs), shuffle=True, **_loader_kwargs(bs))
        vl = DataLoader(Subset(ds_clean, val_abs), shuffle=False, **_loader_kwargs(bs))
        to = DataLoader(Subset(ds_clean, tr_abs), shuffle=False, **_loader_kwargs(bs))

        ext_outer = int(os.getenv("NESTED_OUTER_FOLD", "0"))
        model, thr, ep = train_one(tl, vl, device, tr_lbl, clinrt_dim, dvh_dim,
                                   dosim_dim, petplus_dim, ctrad_dim, seed_offset=fi, accumulation=ACCUMULATION,
                                   max_epochs=max_epochs, train_eval_loader=to,
                                   ext_outer=ext_outer, ext_inner=fi - 1)
        criterion = FocalLoss(alpha=_class_weights(tr_lbl, device), gamma=2)
        _, vp, _ = evaluate(model, vl, criterion, device)
        _, tp, _ = evaluate(model, to, criterion, device)
        val_m = metrics_at_threshold(vp, np.array(val_lbl), thr)
        train_m = metrics_at_threshold(tp, np.array(tr_lbl), thr)
        log.info(
            "  Fold %d | Train (AUC=%.3f F1=%.3f Sn=%.3f Sp=%.3f PR=%.3f Prc=%.3f Br=%.3f) | Val (AUC=%.3f F1=%.3f Sn=%.3f Sp=%.3f PR=%.3f Prc=%.3f Br=%.3f) | Ep=%d",
            fi, train_m["auc"], train_m["f1"], train_m["sensitivity"], train_m["specificity"], train_m["pr_auc"],
            train_m["precision"], train_m["brier"], val_m["auc"], val_m["f1"], val_m["sensitivity"],
            val_m["specificity"], val_m["pr_auc"], val_m["precision"], val_m["brier"], ep)
        cv_results.append({"train": train_m, "val": val_m, "epochs": ep, "threshold": thr})

        if te is not None:
            ab_k = fit_platt(np.asarray(vp), np.asarray(val_lbl))
            _, tp_det_k, _ = evaluate(model, te, criterion, device)
            tp_cal_k = apply_platt(tp_det_k, ab_k)
            fold_test_probs.append(tp_cal_k)
            oof_val_probs.extend(apply_platt(np.asarray(vp), ab_k).tolist())
            oof_val_lbls.extend(list(val_lbl))
            auc_k = (roc_auc_score(test_labels, tp_cal_k)
                     if len(set(test_labels)) > 1 else 0.5)
            fold_test_aucs.append(auc_k)
            tem = metrics_at_threshold(tp_cal_k, np.array(test_labels), thr)
            cv_results[-1]["test"] = tem
            log.info("    [diag] inner fold %d -> outer-test AUC=%.3f  (val AUC=%.3f)",
                     fi, auc_k, val_m["auc"])

        if not smoke and (best is None or val_m["auc"] > best["auc"]):
            if best is not None:
                del best["model"]
            model.to("cpu")
            best = {"auc": val_m["auc"], "model": model, "fold": fi,
                    "tr_abs": list(tr_abs), "tr_lbl": list(tr_lbl)}
        else:
            del model
        _shutdown(tl, vl, to)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ms_val = _mean_std(cv_results, "val")
    log.info("CV val: AUC=%.3f±%.3f  F1=%.3f±%.3f",
             ms_val["auc"][0], ms_val["auc"][1], ms_val["f1"][0], ms_val["f1"][1])

    test_metrics = {}
    if te is not None and best is not None and fold_test_probs:
        oof_v, oof_l = np.asarray(oof_val_probs), np.asarray(oof_val_lbls)
        cal_thr = optimal_f1_threshold(oof_v, oof_l)
        cal_thr_y = youden_threshold(oof_v, oof_l)
        tp_ens = np.mean(np.stack(fold_test_probs, axis=0), axis=0)
        test_metrics = metrics_at_threshold(tp_ens, np.array(test_labels), cal_thr)
        log.info("Inner-model ensemble | %d models  test AUC=%.3f  "
                 "(per-model test AUC: %s)  thr=%.3f (pooled inner-OOF, n=%d)",
                 len(fold_test_probs), test_metrics["auc"],
                 ", ".join(f"{a:.3f}" for a in fold_test_aucs), cal_thr, len(oof_l))

        _fit_tab_scaler([ds_clean, ds_noisy], manifest_df, best["tr_abs"], side_map, side_dim)
        if CACHE_IMAGING and _IMAGING_TOKENS:
            _okb = int(os.getenv("NESTED_OUTER_FOLD", "0"))
            if GATE_LIVE_ENCODER:
                _encb, _ = _load_gate_encoder(_okb, best["fold"] - 1, device)
                _set_live_encoder(_encb)
                ds_clean.set_emb(None)
            else:
                _emb_b = {t: _load_fold_cache(_okb, best["fold"] - 1, t)[0]
                          for t in _IMAGING_TOKENS}
                ds_clean.set_emb(_emb_b)
        tr_lbl = best["tr_lbl"]
        model = best["model"].to(device)
        log.info("Explainability model | best-val inner fold=%d (val AUC=%.3f)  "
                 "test=%d (RP+=%d)  [cohort=%d]", best["fold"], best["auc"],
                 len(test_idx), sum(labels[i] for i in test_idx), len(manifest_df))
        if hasattr(model, "dose_gate"):
            dg = model.dose_gate
            steep_, thr_, amp_, pw_, base_ = (float(dg.steepness), float(dg.dose_thresh),
                                              float(dg.amplify), float(getattr(dg, "prox_weight", 0.0)),
                                              float(dg.baseline))
            thr_gy = thr_ * DOSE_NORM_CGY / 100.0
            init_steep_ = float(getattr(dg, "init_steepness", DG_INIT_A))
            init_thr_ = float(getattr(dg, "init_dose_thresh", DG_INIT_B))
            log.info("Learned dose-gate: steepness=%.3f  dose_thresh=%.3f (~%.1f Gy)  "
                     "amplify=%.3f  prox_weight=%.3f  baseline=%.3f   "
                     "[init: steepness=%.3f dose_thresh=%.2f amplify=%.1f prox_weight=0.0 baseline=1.0]",
                     steep_, thr_, thr_gy, amp_, pw_, base_, init_steep_, init_thr_, DG_INIT_LAMBDA)
            try:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(os.path.join(RESULTS_DIR, "dose_gate_params.json"), "w") as fp:
                    json.dump({"steepness": steep_, "dose_thresh": thr_, "amplify": amp_,
                               "prox_weight": pw_, "baseline": base_,
                               "dose_norm_cGy": DOSE_NORM_CGY, "threshold_Gy": thr_gy,
                               "init": {"steepness": init_steep_, "dose_thresh": init_thr_,
                                        "amplify": DG_INIT_LAMBDA, "prox_weight": 0.0, "baseline": 1.0}}, fp, indent=2)
                if USE_DOSE_RESPONSE_GATE and hasattr(dg, "curve_dict"):
                    with open(os.path.join(RESULTS_DIR, "ntcp_gate.json"), "w") as fp:
                        json.dump(dg.curve_dict(), fp, indent=2)
                hist = getattr(model, "_gate_history", [])
                if hist:
                    import csv as _csv
                    with open(os.path.join(RESULTS_DIR, "dose_gate_history.csv"), "w",
                              newline="") as fp:
                        wr = _csv.DictWriter(fp, fieldnames=list(hist[0].keys()))
                        wr.writeheader();
                        wr.writerows(hist)
            except Exception as exc:
                log.warning("Could not write dose-gate params/history: %s", exc)
        if getattr(model, "use_team", False):
            log.info("Fusion: cooperative cross-modal attention (handshake) - %d layer(s), "
                     "%d head(s); no alpha (all streams are peers).",
                     FUSION_TEAM_LAYERS, FUSION_TEAM_HEADS)
        if SKLEARN_EXPERTS and getattr(model, "_scalar_slices", None):
            log.info("Scalar-stream expert networks: %s",
                     " | ".join(f"{nm}={SKLEARN_EXPERT_MODELS.get(nm, 'mlp_128_64_32')}"
                                for nm in model._scalar_slices.keys()))
        if hasattr(model, "alpha_value") and (FUSION_IMAGING_RESIDUAL
                                              or not getattr(model, "use_team", False)):
            try:
                a_ = float(model.alpha_value)
                log.info("Learned fusion gain: alpha=%.4f  (init~%.3f; logit=scalar_backbone + alpha*imaging)",
                         a_, float(F.softplus(torch.tensor(FUSION_ALPHA_INIT_RAW))))
                os.makedirs(RESULTS_DIR, exist_ok=True)
                _afile = os.path.join(RESULTS_DIR, "fusion_alpha.json")
                _hist = []
                if os.path.exists(_afile):
                    try:
                        _hist = json.load(open(_afile)).get("alpha_per_fold", [])
                    except Exception:
                        _hist = []
                _hist.append(a_)
                json.dump({"alpha_per_fold": _hist,
                           "alpha_mean": float(np.mean(_hist)),
                           "alpha_init": float(F.softplus(torch.tensor(FUSION_ALPHA_INIT_RAW)))},
                          open(_afile, "w"), indent=2)
            except Exception as exc:
                log.warning("Could not log/persist fusion alpha: %s", exc)
        if GENERATE_GRADCAM and not CACHE_IMAGING:
            try:
                _generate_gradcam(model, ds_clean, test_idx, test_uuids,
                                  test_labels, tp_ens, cal_thr, device,
                                  os.path.join(RESULTS_DIR, "gradcam"))
            except Exception as exc:
                log.warning("Grad-CAM generation skipped: %s", exc)
        elif CACHE_IMAGING:
            log.info("Grad-CAM skipped (frozen-encoder cache mode: no 3D backbone in the "
                     "fusion; spatial gate validation lives in the gate_ct run).")
        _, tp_std, _ = evaluate_mc(model, te, device)
        log.info("TEST | AUC=%.3f F1=%.3f Sens=%.3f Spec=%.3f "
                 "(ensemble of %d inner models; thr=%.3f from pooled inner-OOF)",
                 test_metrics["auc"], test_metrics["f1"], test_metrics["sensitivity"],
                 test_metrics["specificity"], len(fold_test_probs), cal_thr)
        save_patient_xlsx(test_uuids, np.array(test_labels), tp_ens, cal_thr,
                          unc=tp_std, threshold_y=cal_thr_y)
        log_component_effects(model, te, device, np.array(test_labels))
        try:
            log_patient_contributions(model, te, device, test_uuids, np.array(test_labels))
        except Exception as exc:
            log.warning("patient_contributions export skipped: %s", exc)
        del model, best
        _shutdown(te)

    ms_tr = _mean_std(cv_results, "train")
    row = {"model": "RP-Fuse",
           "imaging_tokens": "+".join(_IMAGING_TOKENS) if USE_IMAGING else "none",
           "lung_crop": LUNG_CROP, "use_tabular": USE_TABULAR, "use_dvh": USE_DVH,
           "anatomy_gate": USE_ANATOMY_GATE, "pet_robust": PET_ROBUST,
           "use_pet_plus": USE_PET_PLUS}
    for k in ["auc", "f1", "sensitivity", "specificity", "acc"]:
        row[f"val_{k}_mean"], row[f"val_{k}_std"] = ms_val[k]
        row[f"train_{k}_mean"] = ms_tr[k][0]
        row[f"test_{k}"] = test_metrics.get(k, np.nan)
    row["avg_epochs"] = float(np.mean([r["epochs"] for r in cv_results])) if cv_results else 0
    pd.DataFrame([row]).to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"),
                               index=False, float_format="%.4f")
    with open(os.path.join(PRED_DIR, "RP-Fuse_cv_meta.json"), "w") as fp:
        json.dump({"fold_thresholds": [r["threshold"] for r in cv_results],
                   "fold_epochs": [r["epochs"] for r in cv_results],
                   "fold_val": [r["val"] for r in cv_results]}, fp, indent=2)
    import util
    util.print_array_metrics({"RP-Fuse": cv_results}, {"RP-Fuse": test_metrics}, log)
    log.info("Done. Results -> %s", RESULTS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny CPU run: 1 fold, <=3 train patients, 1 epoch.")
    args = ap.parse_args()

    use_nested = bool(_full_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False))
    if use_nested and not args.smoke and os.environ.get("NESTED_OUTER_FOLD") is None:
        try:
            from nested_cv import run_nested_cv

            n_outer = int(_full_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5))
            run_nested_cv(main_fn=lambda: main(False), n_outer=n_outer,
                          module_globals=globals(), results_dir_key="RESULTS_DIR",
                          pred_dir_key="PRED_DIR", log=log,
                          n_repeats=int(os.environ.get("NESTED_CV_REPEATS", "1")))
        except Exception as exc:
            log.warning("Nested-CV runner unavailable (%s) - single split.", exc)
            main(args.smoke)
    else:
        main(args.smoke)
