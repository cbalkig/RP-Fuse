from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_tabular_compare as tc

N_BOOT = 2000


def _ci(y, p, fn, seed=0, n=N_BOOT):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.choice(len(y), len(y), replace=True)
        if len(set(y[s].tolist())) < 2:
            continue
        vals.append(fn(y[s], p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(modality, split):
    X, y, folds, names, rows = tc.build_modality(modality, split)
    print(f"\n=== {modality}: X={X.shape}, RP+={int(y.sum())}/{len(y)} ===", flush=True)
    best = None
    table = []
    for name, make in tc._models().items():
        try:
            preds, te_mask, tr_m, val_m, te_m = tc.nested_oof_auc(make, X, y, folds)
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
            continue
        table.append({"model": name, "val_auc": val_m[0], "oof_auc": te_m[0]})
        print(f"  {name:18s} val {val_m[0]:.3f}  oof {te_m[0]:.3f}", flush=True)
        if best is None or val_m[0] > best["val_auc"]:
            best = {"model": name, "val_auc": val_m[0], "preds": preds.copy(),
                    "mask": te_mask.copy()}
    if best is None:
        print(f"{modality}: no model completed")
        return None

    m = best["mask"]
    yv, pv, fv = y[m], best["preds"][m], folds[m]
    prev = float(yv.mean())
    auc, pr = roc_auc_score(yv, pv), average_precision_score(yv, pv)
    lo, hi = _ci(yv, pv, roc_auc_score)
    plo, phi = _ci(yv, pv, average_precision_score)
    bss = 1.0 - float(np.mean((pv - yv) ** 2)) / (prev * (1.0 - prev))
    per_fold = [roc_auc_score(yv[fv == k], pv[fv == k]) for k in sorted(set(fv.tolist()))
                if len(set(yv[fv == k].tolist())) > 1]
    pd.DataFrame(table).sort_values("val_auc", ascending=False).to_csv(
        os.path.join(tc._OUT, f"{modality}_model_comparison.csv"), index=False)
    out = dict(modality=modality, n=int(m.sum()), features=X.shape[1],
               model=best["model"], auc=auc, lo=lo, hi=hi,
               fold_mean=float(np.mean(per_fold)), fold_sd=float(np.std(per_fold)),
               pr=pr, pr_lo=plo, pr_hi=phi, bss=bss)
    print(f"\n{modality}: {best['model']}  AUC {auc:.3f} [{lo:.2f}-{hi:.2f}]  "
          f"fold-wise {np.mean(per_fold):.3f}+/-{np.std(per_fold):.2f}  "
          f"PR {pr:.3f} [{plo:.2f}-{phi:.2f}]  BSS {bss:+.3f}")
    print(f"LATEX  & {best['model']} & {auc:.3f}~[{lo:.2f}--{hi:.2f}] & "
          f"{np.mean(per_fold):.3f}$\\pm${np.std(per_fold):.2f} & "
          f"{pr:.3f}~[{plo:.2f}--{phi:.2f}] & {bss:+.3f} \\\\")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", nargs="*", default=list(tc._PETOMICS_MODALITIES))
    a = ap.parse_args()
    os.makedirs(tc._OUT, exist_ok=True)
    split = tc._cohort()
    res = [r for r in (run(mod, split) for mod in a.modality) if r]
    if res:
        pd.DataFrame(res).to_csv(os.path.join(tc._OUT, "petomics_rows.csv"), index=False)
        print("\nwrote", os.path.join(tc._OUT, "petomics_rows.csv"))
