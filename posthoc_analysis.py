#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pool_oof(results_dir: str, variant: str, n_outer: int, n_repeats: int
              ) -> pd.DataFrame:
    def _repeat_frame(repeat_dir: str) -> pd.DataFrame | None:
        frames = []
        for k in range(n_outer):
            pat = os.path.join(repeat_dir, f"outer_{k}", "patient_predictions",
                               f"{variant}_test_predictions.xlsx")
            if os.path.exists(pat):
                frames.append(pd.read_excel(pat))
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True, sort=False)

    rep_frames = []
    if n_repeats > 1:
        for r in range(n_repeats):
            rf = _repeat_frame(os.path.join(results_dir, f"repeat_{r}"))
            if rf is not None:
                rep_frames.append(rf)
    else:
        rf = _repeat_frame(results_dir)
        if rf is not None:
            rep_frames.append(rf)
    if not rep_frames:
        raise FileNotFoundError(
            f"No {variant}_test_predictions.xlsx under {results_dir} "
            f"(n_outer={n_outer}, n_repeats={n_repeats}).")

    allrep = pd.concat(rep_frames, ignore_index=True, sort=False)
    agg = (allrep.groupby("UUID")
                 .agg(TrueLabel=("TrueLabel", "first"),
                      RP_Probability=("RP_Probability", "mean"))
                 .reset_index())
    return agg


def _ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, pts = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.mean()
        ece += w * abs(acc - conf)
        pts.append((conf, acc, int(m.sum())))
    return float(ece), pts


def calibration(y, p, out_dir, tag):
    brier = float(np.mean((p - y) ** 2))
    ece, pts = _ece(y, p, 10)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfectly calibrated")
    if pts:
        cx = [c for c, _, _ in pts]; cy = [a for _, a, _ in pts]
        ax.plot(cx, cy, "o-", color="#1f77b4", label=f"{tag}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed RP frequency")
    ax.set_title(f"Reliability  (Brier={brier:.3f}, ECE={ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"calibration_{tag}.png"), dpi=200)
    plt.close(fig)
    return {"brier": brier, "ece": ece}


def dca(y, p, out_dir, tag, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 60)
    n = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        odds = pt / (1 - pt)
        tp = np.sum((p >= pt) & (y == 1)) / n
        fp = np.sum((p >= pt) & (y == 0)) / n
        nb_model = tp - fp * odds
        nb_all = prev - (1 - prev) * odds
        rows.append({"threshold": pt, "net_benefit_model": nb_model,
                     "net_benefit_all": nb_all, "net_benefit_none": 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"dca_{tag}.csv"), index=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(df.threshold, df.net_benefit_model, "-", color="#1f77b4", label=f"{tag}")
    ax.plot(df.threshold, df.net_benefit_all, "--", color="gray", label="Treat all")
    ax.plot(df.threshold, df.net_benefit_none, ":", color="black", label="Treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_ylim(min(-0.02, df.net_benefit_model.min()), max(0.02, prev + 0.02))
    ax.set_title("Decision-curve analysis"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"dca_{tag}.png"), dpi=200)
    plt.close(fig)
    return df


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y, p1, p2):
    from scipy import stats
    order = np.argsort(-y)
    ys = y[order]; m = int(ys.sum())
    preds = np.vstack([p1[order], p2[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, pval = 0.0, 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_model": float(aucs[0]), "auc_baseline": float(aucs[1]),
            "auc_diff": float(aucs[0] - aucs[1]), "z": float(z), "p_value": float(pval)}


def alpha_readout(results_dir, n_outer, n_repeats):
    vals = []
    for f in glob.glob(os.path.join(results_dir, "**", "fusion_alpha.json"), recursive=True):
        try:
            vals.extend(json.load(open(f)).get("alpha_per_fold", []))
        except Exception:
            pass
    if not vals:
        return None
    return {"alpha_mean": float(np.mean(vals)), "alpha_std": float(np.std(vals)),
            "alpha_per_fold": [float(v) for v in vals], "n": len(vals)}


def attention_bars(results_dir, out_dir):
    frames = [pd.read_csv(f) for f in
              glob.glob(os.path.join(results_dir, "**", "component_effects.csv"), recursive=True)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby("component")
           .agg(w_mean=("mean_fusion_weight", "mean"),
                w_std=("mean_fusion_weight", "std"),
                auc_mean=("standalone_auc", "mean"))
           .reset_index().sort_values("w_mean", ascending=False))
    g["w_std"] = g["w_std"].fillna(0.0)
    g.to_csv(os.path.join(out_dir, "attention_weights.csv"), index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(g.component, g.w_mean, yerr=g.w_std, color="#1f77b4", capsize=3)
    ax.set_ylabel("Mean fusion attention weight"); ax.set_xlabel("Token")
    ax.set_title("Per-token attention (mean±SD over folds)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "attention_weights.png"), dpi=200)
    plt.close(fig)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="fusion run orig results dir")
    ap.add_argument("--baseline_dir", default=None, help="baseline run dir (for DeLong)")
    ap.add_argument("--variant", default="RP-Fuse")
    ap.add_argument("--baseline_variant", default=None,
                    help="prediction-file variant in baseline_dir (default: --variant)")
    ap.add_argument("--n_outer", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=1)
    a = ap.parse_args()

    out_dir = os.path.join(a.results_dir, "posthoc")
    os.makedirs(out_dir, exist_ok=True)

    oof = _pool_oof(a.results_dir, a.variant, a.n_outer, a.n_repeats)
    y = oof["TrueLabel"].values.astype(int)
    p = oof["RP_Probability"].values.astype(float)
    auc = roc_auc_score(y, p)
    print(f"[posthoc] pooled OOF: n={len(y)}  RP+={y.sum()}  AUC={auc:.3f}")

    summary = {"variant": a.variant, "n_patients": int(len(y)),
               "n_pos": int(y.sum()), "pooled_oof_auc": float(auc)}

    summary["calibration"] = calibration(y, p, out_dir, a.variant)
    print(f"[posthoc] calibration: Brier={summary['calibration']['brier']:.3f}  "
          f"ECE={summary['calibration']['ece']:.3f}")

    dca(y, p, out_dir, a.variant)
    print("[posthoc] DCA written.")

    al = alpha_readout(a.results_dir, a.n_outer, a.n_repeats)
    if al:
        summary["alpha"] = al
        print(f"[posthoc] fusion alpha = {al['alpha_mean']:.4f} ± {al['alpha_std']:.4f} "
              f"(n={al['n']} folds)")
    else:
        print("[posthoc] no fusion_alpha.json found (skipping alpha readout).")

    ab = attention_bars(a.results_dir, out_dir)
    if ab is not None:
        summary["attention"] = ab.to_dict(orient="records")
        print("[posthoc] attention weights:")
        for _, r in ab.iterrows():
            print(f"    {r['component']:<8} w={r['w_mean']:.3f}±{r['w_std']:.3f}  "
                  f"AUC={r['auc_mean']:.3f}")
    else:
        print("[posthoc] no component_effects.csv found (skipping attention bars).")

    if a.baseline_dir:
        bvar = a.baseline_variant or a.variant
        base = _pool_oof(a.baseline_dir, bvar, a.n_outer, a.n_repeats)
        merged = oof.merge(base[["UUID", "RP_Probability"]], on="UUID",
                           suffixes=("", "_base"))
        if len(merged) < len(oof):
            print(f"[posthoc] DeLong: aligned {len(merged)}/{len(oof)} patients by UUID.")
        d = delong_test(merged["TrueLabel"].values.astype(int),
                        merged["RP_Probability"].values.astype(float),
                        merged["RP_Probability_base"].values.astype(float))
        summary["delong"] = d
        print(f"[posthoc] DeLong  fusion AUC={d['auc_model']:.3f}  "
              f"baseline AUC={d['auc_baseline']:.3f}  "
              f"delta={d['auc_diff']:+.3f}  p={d['p_value']:.3f}")

    json.dump(summary, open(os.path.join(out_dir, "posthoc_summary.json"), "w"), indent=2)
    print(f"[posthoc] summary -> {os.path.join(out_dir, 'posthoc_summary.json')}")


if __name__ == "__main__":
    main()
