from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posthoc_analysis import _pool_oof

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


def _row(name, y, p):
    prev = float(y.mean())
    a1, a2 = _ci(y, p, roc_auc_score)
    p1, p2 = _ci(y, p, average_precision_score)
    bss = 1.0 - float(np.mean((p - y) ** 2)) / (prev * (1.0 - prev))
    return (f"{name:<10} {roc_auc_score(y, p):.3f} [{a1:.2f}-{a2:.2f}]      "
            f"{average_precision_score(y, p):.3f} [{p1:.2f}-{p2:.2f}]      {bss:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar", default="results_rpfuse_scalar")
    ap.add_argument("--gated", default="results_rpfuse_gated")
    ap.add_argument("--root", default=os.environ.get("RPFUSE_OUTPUTS", "data/Outputs"))
    ap.add_argument("--variant", default="RP-Fuse")
    ap.add_argument("--n-outer", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional xlsx path for the ensemble predictions")
    a = ap.parse_args()

    dirs = [d if os.path.isabs(d) else os.path.join(a.root, d) for d in (a.scalar, a.gated)]
    s = _pool_oof(dirs[0], a.variant, a.n_outer, 1)
    g = _pool_oof(dirs[1], a.variant, a.n_outer, 1)

    m = s.merge(g, on="UUID", suffixes=("_s", "_g"))
    if len(m) != len(s) or len(m) != len(g):
        raise SystemExit(f"patients differ: scalar n={len(s)}, gated n={len(g)}, matched n={len(m)}")
    if not (m["TrueLabel_s"] == m["TrueLabel_g"]).all():
        raise SystemExit("labels disagree between the two runs")

    y = m["TrueLabel_s"].to_numpy(int)
    ps, pg = m["RP_Probability_s"].to_numpy(float), m["RP_Probability_g"].to_numpy(float)
    pe = 0.5 * (ps + pg)

    print(f"n={len(m)}  events={int(y.sum())}")
    print(f"{'model':<10} {'AUC [95% CI]':<22} {'PR-AUC [95% CI]':<22} {'BSS':>7}")
    print("-" * 66)
    for name, p in (("Scalar", ps), ("Gated", pg), ("Ensemble", pe)):
        print(_row(name, y, p))

    if a.out:
        pd.DataFrame({"UUID": m["UUID"], "TrueLabel": y, "RP_Probability": pe}).to_excel(a.out, index=False)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
