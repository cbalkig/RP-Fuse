#!/usr/bin/env python
import os
import sys

import glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import load_config, read_table, load_unified_split


def _n_ct_studies(uuid_dir):
    return sum(os.path.exists(os.path.join(os.path.dirname(m), "CT", "whole_body.npy"))
               for m in glob.glob(os.path.join(uuid_dir, "studies", "*", "study_meta.json")))


CODEBOOK = {
    "gender":         {0: "Male", 1: "Female"},
    "histopathology": {1: "Squamous cell carcinoma", 2: "Adenocarcinoma",
                       3: "Other", 4: "Other"},
    "active_smoking": {0: "No", 1: "Yes"},
    "COPD":           {0: "No", 1: "Yes"},
    "DM":             {0: "No", 1: "Yes"},
    "RT_technique":   {0: "3D-CRT", 1: "IMRT", 2: "VMAT", 3: "VMAT"},
    "TNM_stage":      {0: "Unknown", 1: "I", 2: "II", 3: "IIIA",
                       4: "IIIB", 5: "IIIC", 6: "IV"},
}
TNM_COARSE = {"I": "I", "II": "II", "IIIA": "III", "IIIB": "III",
              "IIIC": "III", "IV": "IV", "Unknown": "Unknown"}


RP_COL = "rp_label_grade2"

CONT = "cont"; CAT = "cat"; DERIVED = "derived"
SPEC = [
    ("Age (years)",                 "age",             CONT,  None),
    ("Sex",                         "gender",          CAT,   "gender"),
    ("ECOG performance",            "ECOG_performance",CAT,   None),
    ("Smoking (pack-years)",        "smoking_packyear",CONT,  None),
    ("Active smoking",              "active_smoking",  CAT,   "active_smoking"),
    ("COPD",                        "COPD",            CAT,   "COPD"),
    ("Diabetes mellitus",           "DM",              CAT,   "DM"),
    ("Histology",                   "histopathology",  CAT,   "histopathology"),
    ("Disease stage",               "TNM_stage",       CAT,   "TNM_coarse"),
    ("T stage",                     "T_stage",         CAT,   None),
    ("N stage",                     "N_stage",         CAT,   None),
    ("Tumour size (cm)",            "tumor_size_cm",   CONT,  None),
    ("GTV (cc)",                    "GTV_cc",          CONT,  None),
    ("PTV (cc)",                    "PTV_cc",          CONT,  None),
    ("RT technique",                "RT_technique",    CAT,   "RT_technique"),
    ("Total RT dose (Gy)",          "RT_dose_total",   CONT,  None),
    ("Lung V5 (%)",                 "lung_V5",         CONT,  None),
    ("Lung V20 (%)",                "lung_V20",        CONT,  None),
    ("Mean lung dose (Gy)",         "MLD",             CONT,  None),
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _fmt_cont(x):
    x = _num(x).dropna()
    if x.empty:
        return "--", x
    med, lo, hi = x.median(), x.quantile(.25), x.quantile(.75)
    return f"{med:.1f} [{lo:.1f}-{hi:.1f}]", x


def _label(col, code, opt):
    if opt == "TNM_coarse":
        return TNM_COARSE.get(CODEBOOK["TNM_stage"].get(code, str(code)), str(code))
    if opt and opt in CODEBOOK:
        return CODEBOOK[opt].get(code, f"code {code}")
    return str(code)


def _derived_series(df, opt):
    if opt == "chemo":
        cyc = _num(df["chemo_cycles"]).fillna(0)
        return (cyc > 0).map({True: "Yes", False: "No"})
    raise ValueError(opt)


def main():
    cfg = load_config()
    inp = cfg["INPUTS_DIR"]
    master = read_table(os.path.join(inp, cfg["INPUTS"]["TABULAR_DATA"]))
    os.environ.setdefault("RPFUSE_CONFIG", "config.yaml")
    import train_tabular_compare as tc
    split = tc._cohort()
    eid = next(c for c in split.columns if c.lower() == "excel_row_idx")
    idx = [int(i) for i in split[eid].dropna()]
    coh = master.iloc[idx].reset_index(drop=True)
    coh[RP_COL] = split["rp_label"].to_numpy(int)

    rp = _num(coh[RP_COL])
    pos, neg = rp == 1, rp == 0
    n, npos, nneg = len(coh), int(pos.sum()), int(neg.sum())
    print(f"Cohort n={n}  (RP+ {npos} / RP- {nneg})\n")

    rows = []
    raw_dump = []

    for label, col, kind, opt in SPEC:
        if col not in coh.columns:
            rows.append((label, "n/a (col missing)", "", "", ""))
            continue

        if kind == CONT:
            ov, _ = _fmt_cont(coh[col])
            pv, xp = _fmt_cont(coh.loc[pos, col])
            nv, xn = _fmt_cont(coh.loc[neg, col])
            try:
                p = stats.mannwhitneyu(_num(coh.loc[pos, col]).dropna(),
                                       _num(coh.loc[neg, col]).dropna(),
                                       alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            rows.append((label, ov, pv, nv, p))
            continue

        if kind == DERIVED:
            ser = _derived_series(coh, opt)
            levels = ["Yes", "No"]
        else:
            ser = coh[col]
            if opt and (opt in CODEBOOK or opt == "TNM_coarse"):
                ser = _num(ser).map(lambda c: _label(col, c, opt) if pd.notna(c) else np.nan)
            levels = [l for l in pd.unique(ser.dropna())]
            try:
                levels = sorted(levels)
            except Exception:
                pass

        ct = pd.crosstab(ser, np.where(pos, "pos", np.where(neg, "neg", "na")))
        ct = ct.drop(columns=[c for c in ct.columns if c == "na"], errors="ignore")
        try:
            if ct.shape == (2, 2) and (stats.chi2_contingency(ct)[3] < 5).any():
                p = stats.fisher_exact(ct.values)[1]
            else:
                p = stats.chi2_contingency(ct)[1]
        except Exception:
            p = np.nan

        rows.append((f"{label}, n (%)", "", "", "", p))
        for lv in levels:
            sub = ser == lv
            o = int((sub).sum())
            pp = int((sub & pos).sum())
            nn = int((sub & neg).sum())
            rows.append((f"  {lv}",
                         f"{o} ({100*o/n:.0f})",
                         f"{pp} ({100*pp/max(npos,1):.0f})",
                         f"{nn} ({100*nn/max(nneg,1):.0f})", ""))

        if opt and opt in CODEBOOK:
            raw = _num(coh[col]).value_counts(dropna=False).sort_index()
            raw_dump.append((label, col, raw.to_dict()))

    def pval(p):
        return "" if p == "" or (isinstance(p, float) and np.isnan(p)) \
            else ("<0.001" if p < 0.001 else f"{p:.3f}")
    w = max(len(r[0]) for r in rows)
    hdr = f"{'Variable':<{w}}  {'Overall':>14}  {'RP+':>12}  {'RP-':>12}  {'p':>7}"
    print(hdr); print("-" * len(hdr))
    for lab, ov, pv, nv, p in rows:
        print(f"{lab:<{w}}  {ov:>14}  {pv:>12}  {nv:>12}  {pval(p):>7}")

    print("\n=== RAW CODE COUNTS (verify CODEBOOK against these) ===")
    for label, col, d in raw_dump:
        print(f"  {label} [{col}]: {d}")

    out_csv = os.path.join(os.path.dirname(__file__), "table1.csv")
    pd.DataFrame(rows, columns=["variable", f"overall_n{n}", f"rp_pos_n{npos}",
                                f"rp_neg_n{nneg}", "p_value"]).to_csv(out_csv, index=False)

    out_tex = os.path.join(os.path.dirname(__file__), "table1_generated.tex")
    with open(out_tex, "w") as fp:
        fp.write("% Auto-generated by tools/make_table1.py - verify CODEBOOK labels.\n")
        fp.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        fp.write(f"\\textbf{{Variable}} & \\textbf{{Overall (n={n})}} & "
                 f"\\textbf{{RP$+$ (n={npos})}} & \\textbf{{RP$-$ (n={nneg})}} & "
                 f"\\textbf{{$p$}} \\\\\n\\midrule\n")
        for lab, ov, pv, nv, p in rows:
            esc = lab.replace("%", "\\%").replace("_", "\\_").lstrip()
            ind = "\\quad " if lab.startswith("  ") else ""
            pl = pval(p).replace("<", "$<$")
            fp.write(f"{ind}{esc} & {ov} & {pv} & {nv} & {pl} \\\\\n")
        fp.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_tex}")
    print("\nNOTE: FEV1 / baseline pulmonary function is NOT in the dataset; that "
          "template row cannot be populated from these data.")


if __name__ == "__main__":
    main()
