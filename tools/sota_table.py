from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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


def _pool(results_dir, method):
    frames = []
    for k in range(5):
        f = os.path.join(results_dir, f"outer_{k}", "patient_predictions",
                         f"{method}_test_predictions.xlsx")
        if os.path.exists(f):
            frames.append(pd.read_excel(f))
    if not frames:
        return None, 0
    return pd.concat(frames, ignore_index=True), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_sota")
    ap.add_argument("--root", default=os.environ.get(
        "RPFUSE_OUTPUTS", "data/Outputs"))
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    a = ap.parse_args()
    rd = a.results if os.path.isabs(a.results) else os.path.join(a.root, a.results)

    methods = sorted({re.sub(r"_test_predictions\.xlsx$", "", os.path.basename(f))
                      for f in glob.glob(os.path.join(rd, "outer_*", "patient_predictions",
                                                      "*_test_predictions.xlsx"))})
    if not methods:
        raise SystemExit(f"no predictions under {rd}")

    rows = []
    for m in methods:
        df, n_folds = _pool(rd, m)
        y = df["TrueLabel"].to_numpy(int)
        p = df["RP_Probability"].to_numpy(float)
        if len(set(y.tolist())) < 2:
            continue
        prev = float(y.mean())
        auc, pr = roc_auc_score(y, p), average_precision_score(y, p)
        a1, a2 = _ci(y, p, roc_auc_score)
        p1, p2 = _ci(y, p, average_precision_score)
        bss = 1.0 - float(np.mean((p - y) ** 2)) / (prev * (1.0 - prev))
        rows.append(dict(method=m, n=len(df), folds=n_folds, auc=auc, lo=a1, hi=a2,
                         pr=pr, pr_lo=p1, pr_hi=p2, bss=bss))

    rows.sort(key=lambda r: -r["auc"])
    if a.latex:
        for r in rows:
            flag = "" if r["folds"] == 5 else f"  % INCOMPLETE: {r['folds']}/5 folds"
            print(f"        {r['method']:<28} & {r['n']} & {r['auc']:.3f}~[{r['lo']:.2f}--{r['hi']:.2f}]"
                  f" & --- & {r['pr']:.3f}~[{r['pr_lo']:.2f}--{r['pr_hi']:.2f}]"
                  f" & {r['bss']:+.3f} \\\\{flag}")
    else:
        print(f"{'method':<28} {'n':>4} {'folds':>6}  {'AUC [95% CI]':<22} {'PR-AUC [95% CI]':<22} {'BSS':>7}")
        print("-" * 96)
        for r in rows:
            warn = "" if r["folds"] == 5 else "   <-- incomplete"
            print(f"{r['method']:<28} {r['n']:>4} {r['folds']:>4}/5  "
                  f"{r['auc']:.3f} [{r['lo']:.2f}-{r['hi']:.2f}]      "
                  f"{r['pr']:.3f} [{r['pr_lo']:.2f}-{r['pr_hi']:.2f}]      "
                  f"{r['bss']:+.3f}{warn}")


if __name__ == "__main__":
    main()
