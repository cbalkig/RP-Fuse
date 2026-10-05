#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import load_config, read_table

MLD_TOL_GY = 3.0
V20_TOL_PP = 8.0


def _lung_struct(structs: dict) -> dict:
    for k in ("lung_minus_gtv", "lung"):
        if k in structs:
            return structs[k] or {}
    return {}


def _ct_studies(uuid_dir: str) -> list[str]:
    dvhs = glob.glob(os.path.join(uuid_dir, "studies", "*", "DVH", "dvh.json"))
    return sorted(os.path.dirname(os.path.dirname(p)) for p in dvhs)


def _study_metrics(study_dir: str) -> dict:
    out = {"mld": None, "v20": None, "gtv": None, "plan": "", "is_total": False}
    try:
        d = json.load(open(os.path.join(study_dir, "DVH", "dvh.json")))
        s = d.get("structures", {}) or {}
        lung = _lung_struct(s)
        mld = lung.get("Dmean_cGy")
        out["mld"] = round(float(mld) / 100.0, 2) if mld is not None and np.isfinite(mld) else None
        v20 = lung.get("V20")
        out["v20"] = round(float(v20), 2) if v20 is not None and np.isfinite(v20) else None
        gtv = (s.get("gtv", {}) or {}).get("volume_cc")
        out["gtv"] = round(float(gtv), 1) if gtv is not None and np.isfinite(gtv) else None
    except Exception:
        pass
    try:
        m = json.load(open(os.path.join(study_dir, "study_meta.json")))
        out["plan"] = str(m.get("plan", "") or "")
        out["is_total"] = bool(m.get("is_total", False))
    except Exception:
        pass
    return out


def _name_map(input_dir: str) -> dict:
    for fn in ("patient_manifest.xlsx", "patient_name_to_uuid.xlsx", "unified_split.xlsx"):
        p = os.path.join(input_dir, fn)
        try:
            df = read_table(p)
            cols = {c.lower(): c for c in df.columns}
            uc = cols.get("uuid"); nc = cols.get("patient_name")
            if uc and nc:
                return {str(u).strip(): str(n) for u, n in zip(df[uc], df[nc])}
        except Exception:
            continue
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="replan_drift.csv", help="output table path")
    ap.add_argument("--mld-tol", type=float, default=MLD_TOL_GY)
    ap.add_argument("--v20-tol", type=float, default=V20_TOL_PP)
    ap.add_argument("--boost-v20-abs", type=float, default=5.0,
                    help="lung V20 (pp) below this marks a study as a boost phase")
    ap.add_argument("--boost-v20-frac", type=float, default=0.35,
                    help="...or below this fraction of the primary plan's V20")
    args = ap.parse_args()

    cfg = load_config()
    input_dir = cfg["INPUTS_DIR"]
    cohort = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    if not os.path.isdir(cohort):
        sys.exit(f"Cohort dir not found (drive mounted?): {cohort}")
    names = _name_map(input_dir)

    rows, n_multi = [], 0
    for uuid in sorted(os.listdir(cohort)):
        ud = os.path.join(cohort, uuid)
        if not os.path.isdir(ud):
            continue
        studies = _ct_studies(ud)
        if len(studies) < 2:
            continue
        n_multi += 1
        met = [_study_metrics(sd) for sd in studies]
        mlds = [m["mld"] for m in met]
        v20s = [m["v20"] for m in met]
        gtvs = [m["gtv"] for m in met]
        is_total = [m["is_total"] for m in met]
        plans = [m["plan"] for m in met]

        v20_vals = [v for v in v20s if v is not None]
        prim_v20 = max(v20_vals) if v20_vals else None
        prim_idx = int(np.argmax([v if v is not None else -1 for v in v20s])) if v20_vals else 0
        boost_cut = max(args.boost_v20_abs, args.boost_v20_frac * prim_v20) if prim_v20 else args.boost_v20_abs

        def _is_boost(i):
            return v20s[i] is not None and prim_v20 is not None and v20s[i] < boost_cut

        full_idx = [i for i in range(len(studies)) if v20s[i] is not None and not _is_boost(i)]

        def _maxspread(seq, idxs):
            vals = [seq[i] for i in idxs if seq[i] is not None]
            return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan

        d_mld = _maxspread(mlds, full_idx)
        d_v20 = _maxspread(v20s, full_idx)
        base_gtv = gtvs[0]
        d_gtv = (max(abs(g - base_gtv) / base_gtv * 100 for g in gtvs[1:] if g and base_gtv)
                 if base_gtv and any(g for g in gtvs[1:]) else np.nan)

        has_total = any(is_total)
        n_full = len(full_idx)
        major_replan = (n_full >= 2 and
                        ((np.isfinite(d_mld) and d_mld > args.mld_tol) or
                         (np.isfinite(d_v20) and d_v20 > args.v20_tol)))

        if has_total:
            verdict = "use is_total plan"
        elif n_full <= 1:
            verdict = (f"primary+boost (keep, use CT{prim_idx + 1})"
                       if prim_idx != 0 else "primary+boost (keep, use CT1)")
        elif major_replan:
            verdict = "MAJOR re-plan (review)"
        else:
            verdict = "minor re-plan (keep, use CT1)"

        fmt = lambda seq, f: " / ".join(f.format(x) if x is not None else "-" for x in seq)
        rows.append({
            "uuid": uuid,
            "patient_name": names.get(uuid, ""),
            "n_ct_studies": len(studies),
            "n_full_plans": n_full,
            "primary_study": prim_idx + 1,
            "plans": " / ".join(p or "-" for p in plans),
            "MLD_Gy_per_study": fmt(mlds, "{:.1f}"),
            "V20_pct_per_study": fmt(v20s, "{:.1f}"),
            "dMLD_Gy_full": round(d_mld, 2) if np.isfinite(d_mld) else None,
            "dV20_pp_full": round(d_v20, 2) if np.isfinite(d_v20) else None,
            "dGTV_pct": round(d_gtv, 1) if np.isfinite(d_gtv) else None,
            "verdict": verdict,
        })

    if not rows:
        print(f"No multi-study patients found under {cohort}.")
        print("Have you rebuilt studies with PREPROCESS.MAX_MEANINGFUL_CT >= 3?")
        return

    df = pd.DataFrame(rows).sort_values(["verdict", "dMLD_Gy_full"], ascending=[True, False])
    df.to_csv(args.csv, index=False, float_format="%.2f")
    counts = df["verdict"].value_counts()
    n_keep = int((~df["verdict"].str.startswith("MAJOR")).sum())

    print(f"\nRe-plan drift over {n_multi} multi-study patient(s)  "
          f"(re-plan tol: dMLD <= {args.mld_tol} Gy, dV20 <= {args.v20_tol} pp; "
          f"boost if lung V20 < max({args.boost_v20_abs} pp, "
          f"{args.boost_v20_frac:g}x primary))")
    for v, n in counts.items():
        print(f"  {v:<34}: {n}")
    print(f"  {'-' * 34}  ----")
    print(f"  {'KEEP (not a genuine re-plan drift)':<34}: {n_keep} / {len(df)}")
    show = ["uuid", "patient_name", "n_ct_studies", "n_full_plans", "primary_study",
            "plans", "V20_pct_per_study", "dMLD_Gy_full", "dV20_pp_full", "verdict"]
    print("\n" + df[show].to_string(index=False))
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
