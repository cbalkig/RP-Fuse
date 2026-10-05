#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

import train_tabular_compare as tc
import train_rpfuse as dg
from fusion_utils import build_clinical_features, build_rt_features
from util import load_config

log = logging.getLogger("train_sota")
logging.basicConfig(level=logging.INFO, format="%(message)s")

_cfg = load_config()
SEED = 42
RESULTS_DIR = os.path.join(_cfg["OUTPUTS_DIR"], "results_sota")
if os.environ.get("NESTED_OUTER_FOLD") is not None:
    RESULTS_DIR = os.path.join(RESULTS_DIR, f"outer_{os.environ['NESTED_OUTER_FOLD']}")
PRED_DIR = os.path.join(RESULTS_DIR, "patient_predictions")

_OMICS_CACHE = os.path.join(_cfg["INPUTS_DIR"],
                            _cfg.get("DATA", {}).get("SOTA_OMICS_CACHE",
                                                     "sota_omics_cache"))


def _roi_mask(paths: dict, kind: str) -> np.ndarray | None:
    def _load(k):
        p = paths.get(k)
        if not p or not os.path.exists(p):
            return None
        return (np.load(p) > 0)

    lung, gtv, ptv = _load("lung"), _load("gtv"), _load("ptv")
    if kind == "lung":
        m = lung
    elif kind == "gtv":
        m = gtv
    elif kind == "ptv":
        m = ptv if ptv is not None else gtv
    elif kind == "lung_minus_gtv":
        if lung is None:
            return None
        m = lung & ~gtv if gtv is not None else lung
    elif kind == "lung_minus_ptv":
        if lung is None:
            return None
        sub = ptv if ptv is not None else gtv
        m = lung & ~sub if sub is not None else lung
    elif kind == "ptv_minus_gtv":
        base = ptv if ptv is not None else gtv
        if base is None:
            return None
        m = base & ~gtv if gtv is not None else base
    elif kind == "peri10":
        if lung is None or gtv is None:
            return None
        import scipy.ndimage as ndi
        m = ndi.binary_dilation(gtv, iterations=10) & lung & ~gtv
    elif kind.startswith("lung_d"):
        if lung is None:
            return None
        dp = paths.get("dose")
        if not dp or not os.path.exists(dp):
            return None
        lo, hi = (float(x) for x in kind[len("lung_d"):].split("_"))
        dose_gy = np.load(dp).astype(np.float32) / 100.0
        healthy = lung & ~gtv if gtv is not None else lung
        m = healthy & (dose_gy >= lo) & (dose_gy < hi)
    else:
        raise ValueError(f"unknown ROI kind {kind!r}")
    if m is None or m.sum() < 8:
        return None
    return m.astype(np.uint8)


_OMICS_SETTINGS = {
    "ct":   {"binWidth": 25},
    "dose": {"binWidth": 100},
    "pet":  {"binCount": 32},
}
_OMICS_BASE = {
    "interpolator": "sitkBSpline",
    "resampledPixelSpacing": [2.0, 2.0, 2.0],
    "label": 1,
    "force2D": False,
    "normalize": False,
}
_EXTRACTORS: dict = {}
_OMICS_NAMES: dict = {}


def _omics_tag(image_type: str, roi: str) -> str:
    s = _OMICS_SETTINGS[image_type]
    disc = f"bw{s['binWidth']}" if "binWidth" in s else f"bc{s['binCount']}"
    return f"{image_type}_{roi}_{disc}"


def _get_extractor(image_type: str):
    if image_type in _EXTRACTORS:
        return _EXTRACTORS[image_type]
    from radiomics import featureextractor
    settings = dict(_OMICS_BASE)
    settings.update(_OMICS_SETTINGS[image_type])
    if image_type == "ct":
        settings["voxelArrayShift"] = 1000
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("firstorder")
    for cls in ("glcm", "glrlm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    _EXTRACTORS[image_type] = ext
    return ext


def _extract_omics_one(uuid: str, vol_path: str, mask_arr: np.ndarray,
                       image_type: str, roi: str) -> np.ndarray:
    import SimpleITK as sitk
    tag = _omics_tag(image_type, roi)
    cache_path = os.path.join(_OMICS_CACHE, f"{uuid}__{tag}.npz")
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=True)
            feats, names = d["feats"].astype(np.float32), d["names"].tolist()
            if feats.size and names:
                _OMICS_NAMES.setdefault(tag, names)
                return feats
            os.remove(cache_path)
        except Exception:
            pass

    vol = np.load(vol_path).astype(np.float32)
    if vol.shape != mask_arr.shape:
        raise ValueError(f"{uuid}: vol {vol.shape} != mask {mask_arr.shape}")
    img = sitk.GetImageFromArray(vol)
    msk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    msk.CopyInformation(img)
    result = _get_extractor(image_type).execute(img, msk)

    pairs = []
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            pairs.append((f"{image_type}.{roi}.{k}", fv))
    pairs.sort(key=lambda kv: kv[0])
    names = [k for k, _ in pairs]
    feats = np.array([v for _, v in pairs], dtype=np.float32)
    if feats.size == 0:
        raise RuntimeError(f"{uuid}: 0 omics features for {tag}")
    _OMICS_NAMES.setdefault(tag, names)
    os.makedirs(_OMICS_CACHE, exist_ok=True)
    np.savez(cache_path, feats=feats, names=np.array(names, dtype=object))
    return feats


def build_omics_map(split: pd.DataFrame, image_type: str, roi: str
                    ) -> tuple[dict[str, np.ndarray], list[str]]:
    tag = _omics_tag(image_type, roi)
    out: dict[str, np.ndarray] = {}
    for u in split["uuid"]:
        udir = os.path.join(dg._cohort_dir, str(u))
        paths = dg.resolve_modality_paths(udir)
        vol_path = paths.get(image_type)
        mask = _roi_mask(paths, roi)
        if not vol_path or mask is None:
            continue
        try:
            out[str(u)] = _extract_omics_one(str(u), vol_path, mask, image_type, roi)
        except Exception as exc:
            log.warning("  omics[%s/%s] failed for %s: %s: %s",
                        image_type, roi, u, type(exc).__name__, exc)
    names = _OMICS_NAMES.get(tag, [])
    log.info("  omics[%s/%s]: %d/%d patients, %d features",
             image_type, roi, len(out), len(split), len(names))
    return out, names


_DVH_METRIC_TO_STD = {"Dmean_cGy": "lung.MLD_cGy", "volume_cc": "lung.volume_cc",
                      "V5": "lung.V5", "V20": "lung.V20", "V30": "lung.V30"}


def _dvh_map(split):
    z = np.load(os.path.join(dg._input_dir, "Derivated_Modals", "dosimetrics.npz"),
                allow_pickle=True)
    names, X, uu = list(z["names"]), z["X"].astype(np.float32), z["uuids"]
    fmap = {str(u).strip(): X[i] for i, u in enumerate(uu)}
    return fmap, names


def _clinical_map(split):
    master = dg.read_table(dg.EXCEL_PATH)
    Xb, names = build_clinical_features(master)
    out = {}
    for _, r in split.iterrows():
        eid = int(r.get("excel_row_idx", -1))
        if 0 <= eid < len(Xb):
            out[str(r["uuid"])] = Xb[eid].astype(np.float32)
    return out, list(names)


def _rt_map(split):
    master = dg.read_table(dg.EXCEL_PATH)
    Xb, names = build_rt_features(master)
    out = {}
    for _, r in split.iterrows():
        eid = int(r.get("excel_row_idx", -1))
        if 0 <= eid < len(Xb):
            out[str(r["uuid"])] = Xb[eid].astype(np.float32)
    return out, list(names)


def _dvh_subset(split, fields: list[str]):
    fmap, names = _dvh_map(split)
    want: list[str] = []
    for f in fields:
        metric = f.split(".")[-1]
        col = _DVH_METRIC_TO_STD.get(metric)
        if col and col not in want:
            want.append(col)
    idx = [names.index(c) for c in want]
    out = {u: v[idx].astype(np.float32) for u, v in fmap.items()}
    return out, want


def _assemble(split: pd.DataFrame, blocks: list[tuple[dict, list[str]]]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    for _, nm in blocks:
        names += list(nm)
    rows, ys, fs, us = [], [], [], []
    y_all = split["rp_label"].astype(int).to_numpy()
    f_all = split["OuterFold"].astype(int).to_numpy()
    for i, u in enumerate(split["uuid"].astype(str)):
        vecs = [b.get(u) for b, _ in blocks]
        if any(v is None for v in vecs):
            continue
        rows.append(np.concatenate([np.asarray(v, np.float32) for v in vecs]))
        ys.append(y_all[i]); fs.append(f_all[i]); us.append(u)
    if not rows:
        raise RuntimeError("no complete-case patients for this method")
    X = np.vstack(rows).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(ys, int), np.array(fs, int), np.array(us, object), names


def feats_zhang2022(split):
    dvh_fields = [f"lung_minus_gtv.{m}" for m in ("Dmean_cGy", "V5", "V20", "V30")]
    dvh_fields += [f"lung.{m}" for m in ("Dmean_cGy", "V5", "V20")]
    dvh = _dvh_subset(split, dvh_fields)
    clin = _clinical_map(split)
    return _assemble(split, [clin, dvh])


def feats_zhang2023(split):
    ct = build_omics_map(split, "ct", "lung")
    dose = build_omics_map(split, "dose", "lung")
    dvh = _dvh_map(split)
    clin = _clinical_map(split)
    return _assemble(split, [ct, dose, dvh, clin])


def feats_wang2025(split):
    ct = build_omics_map(split, "ct", "lung_minus_ptv")
    dose = build_omics_map(split, "dose", "lung_minus_ptv")
    clin_fields = [f"lung_minus_gtv.{m}" for m in ("V20", "V30", "Dmean_cGy")]
    clin = _dvh_subset(split, clin_fields)
    return _assemble(split, [ct, dose, clin])


def feats_ai2025(split):
    ct = build_omics_map(split, "ct", "lung")
    pet = build_omics_map(split, "pet", "lung")
    dose = build_omics_map(split, "dose", "lung")
    return _assemble(split, [ct, pet, dose])


def feats_su2024(split):
    ct = build_omics_map(split, "ct", "lung")
    dose = build_omics_map(split, "dose", "lung")
    return _assemble(split, [ct, dose])


def _deep(split, modality, roi):
    import train_sota_cnn as cnn
    return cnn.build_deep_feature_map(split, modality, roi)


def feats_lee2024(split):
    ct = _deep(split, "ct", "lung")
    clin = _clinical_map(split)
    dvh = _dvh_map(split)
    return _assemble(split, [ct, clin, dvh])


def feats_kong2025(split):
    d1 = _deep(split, "ct", "lung_minus_ptv")
    d2 = _deep(split, "ct", "ptv_minus_gtv")
    v30 = _dvh_subset(split, ["lung_minus_gtv.V30"])
    return _assemble(split, [d1, d2, v30])


def feats_wang2025_dl(split):
    ct = build_omics_map(split, "ct", "lung_minus_ptv")
    dose = build_omics_map(split, "dose", "lung_minus_ptv")
    deep = _deep(split, "ct", "lung_minus_ptv")
    clin = _dvh_subset(split, [f"lung_minus_gtv.{m}" for m in ("V20", "V30", "Dmean_cGy")])
    return _assemble(split, [ct, dose, deep, clin])


def feats_ai2025_dl(split):
    ct = build_omics_map(split, "ct", "lung")
    pet = build_omics_map(split, "pet", "lung")
    dose = build_omics_map(split, "dose", "lung")
    deep = _deep(split, "ct", "lung")
    return _assemble(split, [ct, pet, dose, deep])


def feats_su2024_dl(split):
    ct = build_omics_map(split, "ct", "lung")
    dose = build_omics_map(split, "dose", "lung")
    deep = _deep(split, "ct", "lung")
    return _assemble(split, [ct, dose, deep])


_TANG_DEEP_DIM = 2048
_TANG_AE_DIM = 15


def feats_tang2025(split):
    from train_sota_cnn import build_deep_feature_map
    deep = build_deep_feature_map(split, "ct", "lung", arch="r50")
    dose = _dvh_subset(split, [f"lung.{m}" for m in ("V5", "V20", "V30", "Dmean_cGy")])
    return _assemble(split, [deep, dose])


class _BlockPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_deep=_TANG_DEEP_DIM, k=_TANG_AE_DIM):
        self.n_deep, self.k = n_deep, k
    def fit(self, X, y=None):
        from sklearn.decomposition import PCA
        n = min(self.n_deep, X.shape[1])
        self.n_ = n
        self.pca_ = PCA(n_components=min(self.k, n, X.shape[0] - 1), random_state=SEED)
        self.pca_.fit(X[:, :n])
        return self
    def transform(self, X):
        return np.hstack([self.pca_.transform(X[:, :self.n_]), X[:, self.n_:]])


def _tang_mlp():
    from sklearn.neural_network import MLPClassifier
    return Pipeline([
        ("ae", _BlockPCA()),
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000,
                              random_state=SEED)),
    ])


_KAWA_REGIONS = ["gtv", "peri10", "lung_minus_gtv",
                 "lung_d05_10", "lung_d10_20", "lung_d20_30", "lung_d30_40", "lung_d40_99"]
_KAWA_BLOCKS: list = []
_KAWA_CLIN: tuple = (0, 0)


def feats_kawahara2025(split):
    global _KAWA_BLOCKS, _KAWA_CLIN
    blocks = []
    for roi in _KAWA_REGIONS:
        for img in ("ct", "dose"):
            fmap, names = build_omics_map(split, img, roi)
            if fmap and names:
                blocks.append((fmap, names))
            else:
                log.warning("  kawahara2025: region %s/%s empty, skipped", img, roi)
    clin = _clinical_map(split)
    all_u = [str(u) for u in split["uuid"]]
    for fmap, names in blocks:
        z = np.zeros(len(names), np.float32)
        for u in all_u:
            fmap.setdefault(u, z)
    out = _assemble(split, blocks + [clin])
    _KAWA_BLOCKS, pos = [], 0
    for _, names in blocks:
        _KAWA_BLOCKS.append((pos, pos + len(names))); pos += len(names)
    _KAWA_CLIN = (pos, pos + len(clin[1]))
    return out


class _RegionAttentionEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks=None, clin=None, k=10, inner=5):
        self.blocks, self.clin, self.k, self.inner = blocks, clin, k, inner
    def _base(self):
        return Pipeline([("sc", StandardScaler()),
                         ("sel", SelectKBest(f_classif, k=self.k)),
                         ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                                    class_weight="balanced", max_iter=5000,
                                                    random_state=SEED))])
    def _region_logits(self, X, models):
        cols = []
        for (a, b), m in zip(self.blocks, models):
            p = np.clip(m.predict_proba(X[:, a:b])[:, 1], 1e-4, 1 - 1e-4)
            cols.append(np.log(p / (1 - p)))
        return np.column_stack(cols)
    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        self.classes_ = np.unique(y)
        blocks = self.blocks or _KAWA_BLOCKS
        clin = self.clin or _KAWA_CLIN
        self.blocks, self.clin = blocks, clin
        oof = np.zeros((len(y), len(blocks)), np.float32)
        skf = StratifiedKFold(n_splits=self.inner, shuffle=True, random_state=SEED)
        for tr, va in skf.split(X, y):
            ms = [clone(self._base()).fit(X[tr][:, a:b], y[tr]) for a, b in blocks]
            oof[va] = self._region_logits(X[va], ms)
        self.models_ = [self._base().fit(X[:, a:b], y) for a, b in blocks]
        self.clin_sc_ = StandardScaler().fit(X[:, clin[0]:clin[1]])
        Z = np.hstack([oof, self.clin_sc_.transform(X[:, clin[0]:clin[1]])])
        self.meta_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                        random_state=SEED).fit(Z, y)
        return self
    def predict_proba(self, X):
        Z = np.hstack([self._region_logits(X, self.models_),
                       self.clin_sc_.transform(X[:, self.clin[0]:self.clin[1]])])
        return self.meta_.predict_proba(Z)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _kawahara_ensemble():
    return _RegionAttentionEnsemble()


class _SkEst:
    def __init__(self, base):
        self.base = base
    def fit(self, X, y, **_ignored):
        self.base.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.base.predict_proba(X)
    def predict(self, X):
        return self.base.predict(X)


def _lasso_lr():
    return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                              class_weight="balanced", max_iter=5000,
                              random_state=SEED)


def _xgb_selected():
    from xgboost import XGBClassifier
    return Pipeline([
        ("sel", SelectKBest(f_classif, k=16)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=SEED,
                              verbosity=0)),
    ])


def _rf_zhang():
    return RandomForestClassifier(n_estimators=2000, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def _rd_selected():
    return Pipeline([
        ("sel", SelectKBest(f_classif, k=20)),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                   class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


METHODS = {
    "zhang2022":    (feats_zhang2022,     _rf_zhang),
    "zhang2023":    (feats_zhang2023,     _lasso_lr),
    "wang2025":     (feats_wang2025,      _rd_selected),
    "ai2025":       (feats_ai2025,        _xgb_selected),
    "su2024":       (feats_su2024,        _rd_selected),
    "lee2024":      (feats_lee2024,       _rd_selected),
    "kong2025":     (feats_kong2025,      _rd_selected),
    "wang2025_dl":  (feats_wang2025_dl,   _rd_selected),
    "ai2025_dl":    (feats_ai2025_dl,     _xgb_selected),
    "su2024_dl":    (feats_su2024_dl,     _rd_selected),
    "tang2025":     (feats_tang2025,     _tang_mlp),
    "kawahara2025": (feats_kawahara2025, _kawahara_ensemble),
}


def run_method(name: str, split: pd.DataFrame) -> dict | None:
    builder, make = METHODS[name]
    log.info("\n=== SoTA method: %s ===", name)
    try:
        X, y, folds, uuids, names = builder(split)
    except Exception as exc:
        log.error("  %s: feature build failed (%s) - skipping", name, exc)
        return None
    log.info("  %s: X=%s  RP+=%d/%d  (%d features)",
             name, X.shape, int(y.sum()), len(y), len(names))

    outer_k = int(os.environ.get("NESTED_OUTER_FOLD", -1))
    try:
        preds, te_mask, tr_m, val_m, te_m = tc.nested_oof_auc(
            lambda: _SkEst(make()), X, y, folds)
    except Exception as exc:
        log.error("  %s: nested CV failed (%s) - skipping", name, exc)
        return None

    if outer_k >= 0:
        os.makedirs(PRED_DIR, exist_ok=True)
        pd.DataFrame({
            "UUID":           uuids[te_mask],
            "TrueLabel":      y[te_mask],
            "RP_Probability": preds[te_mask],
        }).to_excel(os.path.join(PRED_DIR, f"{name}_test_predictions.xlsx"),
                    index=False)
    _prev = float(np.mean(y[te_mask])) if te_mask.any() else 0.345
    _bs_ref = _prev * (1.0 - _prev)
    _bss = 1.0 - float(te_m[6]) / _bs_ref if _bs_ref > 0 else float("nan")
    log.info("  %s: Test AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  BrierSkill=%+.3f  "
             "| Val AUC=%.3f  Train AUC=%.3f",
             name, te_m[0], te_m[4], te_m[6], _bss, val_m[0], tr_m[0])
    return {"method": name, "test_auc": round(float(te_m[0]), 4),
            "test_pr_auc": round(float(te_m[4]), 4),
            "test_brier": round(float(te_m[6]), 4),
            "test_brier_skill": round(_bss, 4),
            "val_auc": round(float(val_m[0]), 4),
            "train_auc": round(float(tr_m[0]), 4)}


CNN_METHOD_NAMES = ["liang2020", "kapoor2023", "midroni2026", "kapoor2023_gated", "liang2020_gated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",
                    choices=list(METHODS) + CNN_METHOD_NAMES
                            + ["all", "all_feat", "all_cnn"],
                    default="all")
    args = ap.parse_args()
    split = tc._cohort()
    if args.method == "all":
        names = list(METHODS) + CNN_METHOD_NAMES
    elif args.method == "all_feat":
        names = list(METHODS)
    elif args.method == "all_cnn":
        names = list(CNN_METHOD_NAMES)
    else:
        names = [args.method]
    rows = []
    for m in names:
        if m in METHODS:
            r = run_method(m, split)
        else:
            import train_sota_cnn as cnn
            r = cnn.cnn_run_method(m, split, PRED_DIR)
        if r:
            rows.append(r)
    out_dir = globals().get("RESULTS_DIR")
    os.makedirs(out_dir, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "sota_methods_summary.csv"), index=False)


if __name__ == "__main__":
    if (bool(_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False))
            and os.environ.get("NESTED_OUTER_FOLD") is None):
        from nested_cv import run_nested_cv
        run_nested_cv(main_fn=main,
                      n_outer=int(_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5)),
                      module_globals=globals(),
                      results_dir_key="RESULTS_DIR", pred_dir_key="PRED_DIR",
                      log=log)
    else:
        main()
