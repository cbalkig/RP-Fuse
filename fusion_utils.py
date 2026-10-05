from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from util import load_config

_full_cfg = load_config()
RT_BASE_COLS   = _full_cfg["TRAIN_RT"]["DATA"]["BASE_COLS"]
CLIN_BASE_COLS = _full_cfg["TRAIN_CLINICAL"]["DATA"]["BASE_COLS"]


def _normalize_col(s: str) -> str:
    s = s.replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str:
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        nc = _normalize_col(cand)
        if nc in norm_map:
            return norm_map[nc]
    raise KeyError(
        f"None of {list(candidates)!r} found. "
        f"Available: {list(df.columns[:10])}")


def _get(df: pd.DataFrame, *cands: str, fill: float = 0.0) -> np.ndarray:
    col = _resolve_col(df, *cands)
    return df[col].fillna(fill).values.astype(np.float32)


def build_rt_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    v5  = _get(df, "lung_V5",  "AkciğerV5", "AkcigerV5")
    v10 = _get(df, "lung_V10", "AkcV10")
    v20 = _get(df, "lung_V20", "Akc.V20")
    mld = _get(df, "MLD",      "MLDmeanlungdose")
    ptv = _get(df, "PTV_cc",   "PTVölümcc", "PTVvolumcc")
    gtv = _get(df, "GTV_cc",   "GTVölümcc", "GTVvolumcc")

    resolved_base = [_resolve_col(df, col) for col in RT_BASE_COLS]
    X_base = df[resolved_base].fillna(0).values.astype(np.float32)

    composites = np.column_stack([
        v5 + v10,
        v10 + v20,
        v5 + v10 + v20,
        mld * ptv,
        gtv / (ptv + 1e-8),
        v20 / (v5 + 1e-8),
    ]).astype(np.float32)

    X = np.hstack([X_base, composites])
    labels = [
        "RT Technique", "GTV Volume", "PTV Volume",
        "Lung V5", "Lung V10", "Lung V20", "Mean Lung Dose",
        "V5+V10", "V10+V20", "V5+V10+V20",
        "MLD x PTV", "GTV/PTV", "V20/V5",
    ]
    return X, labels


def build_clinical_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    age      = _get(df, "age", "yas")
    ecog     = _get(df, "ECOG_performance", "ECOGperformans")
    smk_pk   = _get(df, "smoking_packyear",
                        "tedaviöncesisigarahikayesipaketyil",
                        "tedavioncessigarahikayesipaketyil")
    smk_act  = _get(df, "active_smoking",
                        "Aktifsigaraiçimitedavisirasinda",
                        "Aktifsigaraicimitedavisirasinda")
    t_diam   = _get(df, "tumor_size_cm",
                        "tümörçapicm", "tumorcapicm")
    t_stage  = _get(df, "T_stage", "Tevresi")
    n_stage  = _get(df, "N_stage", "Nevresi")
    tnm      = _get(df, "TNM_stage", "TNMEvre")
    dm       = _get(df, "DM", "Diabetesmellitusvaryok")
    copd     = _get(df, "COPD", "KOAHvaryok")
    colvask  = _get(df, "collagen_vascular_disease",
                        "kollajenvask.Hast.varyok")
    emphysem = _get(df, "centrilobular_emphysema",
                        "Sentrilobüleramfizemvarligi",
                        "Sentrilobuleramfizemvarligi")
    bronsekt = _get(df, "bronchiectasis", "Bronsektazivarligi")
    reticul  = _get(df, "reticular_pattern",
                        "Retikülerpatern", "Retikulerpatern")
    cystic   = _get(df, "cystic_lung_disease", "KistikAChastaligi")
    asbestos = _get(df, "asbestos_findings", "Asbestozbulgulari")

    composites = np.column_stack([
        age * ecog,
        smk_pk * smk_act,
        smk_pk * copd,
        dm + copd + colvask,
        emphysem + bronsekt + reticul + cystic + asbestos,
        t_diam * t_stage,
        n_stage * tnm,
    ]).astype(np.float32)

    resolved_base = [_resolve_col(df, col) for col in CLIN_BASE_COLS]
    X_base = df[resolved_base].fillna(0).values.astype(np.float32)
    X = np.hstack([X_base, composites])

    labels = [
        "Age", "Gender", "ECOG", "Weight", "Smoking PY", "Active Smoking",
        "T Stage", "N Stage", "TNM Stage", "Tumour Diam", "Tumour Side",
        "Tumour Lobe", "Histology", "CT Cycles", "PET Fusion",
        "Diabetes", "COPD", "Collagen Vasc.", "Emphysema", "Bronchiectasis",
        "Reticular Pattern", "Cystic AC", "Asbestos",
        "Age x ECOG", "SmkPY x ActiveSmk", "SmkPY x COPD",
        "Comorbidity Idx", "Imaging Score", "Diam x T-Stage", "N x TNM",
    ]
    return X, labels
