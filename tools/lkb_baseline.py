from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import endpoint as _endpoint
import train_rpfuse as dg
from util import load_unified_split

N_LKB, M_LKB, TD50_LKB = 0.99, 0.37, 30.8
STRUCT = "lung"
METHOD = "lkb"
GRID_MAX_GY, GRID_STEP_GY = 80.0, 0.25


def _cohort() -> pd.DataFrame:
    split = load_unified_split(dg._input_dir).copy()
    split["uuid"] = split["uuid"].astype(str).str.strip()
    keep = []
    for _, r in split.iterrows():
        ud = os.path.join(dg._cohort_dir, r["uuid"])
        ofld = r.get("OuterFold", -1)
        keep.append(os.path.isdir(ud) and dg.study_class(ud) == "single"
                    and pd.notna(ofld) and int(ofld) >= 0)
    return split[pd.Series(keep, index=split.index)].reset_index(drop=True)


def _read_curve(uuid: str, grid_gy: np.ndarray) -> np.ndarray | None:
    sdir = dg.resolve_primary_study(os.path.join(dg._cohort_dir, uuid))
    if not sdir:
        return None
    p = os.path.join(sdir, "DVH", "dvh.json")
    if not os.path.exists(p):
        return None
    with open(p) as fp:
        dvh = json.load(fp)
    native = np.asarray(dvh.get("dose_grid_cGy", []), float) / 100.0
    cum = (dvh.get("structures", {}) or {}).get(STRUCT, {}).get("dvh_pct")
    if cum is None or len(native) < 2 or len(native) != len(cum):
        return None
    return np.interp(grid_gy, native, np.clip(np.asarray(cum, float), 0.0, 100.0))


def cumulative_to_differential(cum_pct: np.ndarray, grid_gy: np.ndarray):
    c = np.clip(np.asarray(cum_pct, float), 0.0, 100.0) / 100.0
    v = np.clip(-np.diff(c, axis=1), 0.0, None)
    d = 0.5 * (grid_gy[:-1] + grid_gy[1:])
    tot = v.sum(1, keepdims=True)
    v = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
    return v, d


def geud(v: np.ndarray, d: np.ndarray, n: float) -> np.ndarray:
    a = 1.0 / n
    return np.sum(v * np.power(np.clip(d, 0, None), a)[None, :], axis=1) ** (1.0 / a)


def lkb(g: np.ndarray, td50: float, m: float) -> np.ndarray:
    return 0.5 * (1.0 + erf((g - td50) / (m * td50 * np.sqrt(2.0))))


def build_inputs(split: pd.DataFrame):
    grid = np.arange(0.0, GRID_MAX_GY + GRID_STEP_GY, GRID_STEP_GY)
    curves, y, folds, uuids = [], [], [], []
    for _, r in split.iterrows():
        c = _read_curve(r["uuid"], grid)
        if c is None:
            continue
        curves.append(c)
        y.append(int(r["rp_label"]))
        folds.append(int(r["OuterFold"]))
        uuids.append(r["uuid"])
    if not curves:
        raise RuntimeError(f"no {STRUCT} DVH curves found")
    v, d = cumulative_to_differential(np.vstack(curves), grid)
    keep = v.sum(1) > 0
    return v[keep], d, np.asarray(y)[keep], np.asarray(folds)[keep], np.asarray(uuids)[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_sota_lkb")
    a = ap.parse_args()
    out = os.path.join(dg._output_dir, a.results)

    ep = _endpoint.EndpointConfig(dg._full_cfg)
    split = _endpoint.apply_to_split(_cohort(), dg.read_table(dg.EXCEL_PATH), ep)
    v, d, y, folds, uuids = build_inputs(split)
    p = lkb(geud(v, d, N_LKB), TD50_LKB, M_LKB)

    for k in np.unique(folds):
        te = folds == k
        pred_dir = os.path.join(out, f"outer_{int(k)}", "patient_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({"UUID": uuids[te], "TrueLabel": y[te],
                      "RP_Probability": p[te]}).to_excel(
            os.path.join(pred_dir, f"{METHOD}_test_predictions.xlsx"), index=False)

    fold_auc = [roc_auc_score(y[folds == k], p[folds == k]) for k in np.unique(folds)]
    prev = float(y.mean())
    print(f"LKB (n={N_LKB}, m={M_LKB}, TD50={TD50_LKB} Gy, {STRUCT} DVH): n={len(y)} events={int(y.sum())}  "
          f"pooled AUC={roc_auc_score(y, p):.3f}  fold-wise {np.mean(fold_auc):.3f}+/-{np.std(fold_auc, ddof=1):.2f}  "
          f"BSS={1.0 - float(np.mean((p - y) ** 2)) / (prev * (1.0 - prev)):+.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
