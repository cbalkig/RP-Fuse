#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import load_config


def _first(pats):
    m = sorted(glob.glob(pats))
    return m[0] if m else None


def _paths(uuid_dir):
    return {
        "pet":  _first(os.path.join(uuid_dir, "studies", "*_PET", "PET", "whole_body_regCT.npy")),
        "gtv":  _first(os.path.join(uuid_dir, "studies", "*_CT1", "Contours", "gtv_mask.npy")),
        "lung": _first(os.path.join(uuid_dir, "studies", "*_CT1", "Contours", "lung_mask.npy")),
        "meta": _first(os.path.join(uuid_dir, "studies", "*_PET", "PET", "pet_reg_meta.json")),
    }


def _audit_one(uuid_dir):
    p = _paths(uuid_dir)
    if not (p["pet"] and p["gtv"]):
        return None
    pet = np.load(p["pet"]).astype(np.float32)
    gtv = np.load(p["gtv"]) > 0
    if gtv.sum() < 10 or not np.isfinite(pet).any():
        return None

    lung = (np.load(p["lung"]) > 0) if p["lung"] else np.ones_like(gtv)
    bg = lung & ~gtv
    if bg.sum() < 10:
        bg = ~gtv

    suv_gtv = float(pet[gtv].mean())
    suv_bg = float(pet[bg].mean())
    avidity = suv_gtv / suv_bg if suv_bg > 1e-6 else float("nan")

    gtv_mirror = np.flip(gtv, axis=-1)
    suv_mirror = float(pet[gtv_mirror].mean())
    mirror = suv_gtv / suv_mirror if suv_mirror > 1e-6 else float("nan")

    qa_pass, qa_ratio = None, None
    if p["meta"]:
        try:
            md = json.load(open(p["meta"]))
            qa_pass = md.get("qa_pass")
            qa_ratio = md.get("ratio", md.get("qa_ratio", md.get("uptake_ratio")))
        except Exception:
            pass

    return dict(suv_gtv=suv_gtv, suv_bg=suv_bg, avidity=avidity,
                mirror=mirror, qa_pass=qa_pass, qa_ratio=qa_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="optional path to write the per-patient table")
    ap.add_argument("--min-ratio", type=float, default=None,
                    help="pass threshold (default: PREPROCESS.PET_REG_QA_MIN_RATIO)")
    args = ap.parse_args()

    cfg = load_config()
    cohort = os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"]["PROCESSED"])
    min_ratio = (args.min_ratio if args.min_ratio is not None
                 else float(cfg.get("PREPROCESS", {}).get("PET_REG_QA_MIN_RATIO", 1.3)))
    if not os.path.isdir(cohort):
        sys.exit(f"Cohort dir not found (drive mounted?): {cohort}")

    rows = []
    for uuid in sorted(os.listdir(cohort)):
        ud = os.path.join(cohort, uuid)
        if not os.path.isdir(ud):
            continue
        r = _audit_one(ud)
        if r is not None:
            r["uuid"] = uuid
            rows.append(r)

    if not rows:
        sys.exit("No patients with PET + GTV found.")

    rows.sort(key=lambda r: (r["mirror"] if np.isfinite(r["mirror"]) else 0))
    print(f"\nPET-CT functional alignment audit  ({len(rows)} patients with PET+GTV)")
    print(f"pass threshold (mirror ratio) = {min_ratio}\n")
    print(f"{'uuid':<14}{'SUV_gtv':>9}{'SUV_lung':>9}{'avidity':>9}{'mirror':>8}"
          f"{'qa_pass':>9}{'flag':>7}")
    print("-" * 65)
    n_fail = 0
    for r in rows:
        bad = not (np.isfinite(r["mirror"]) and r["mirror"] >= min_ratio)
        n_fail += bad
        print(f"{r['uuid']:<14}{r['suv_gtv']:>9.2f}{r['suv_bg']:>9.2f}"
              f"{r['avidity']:>9.2f}{r['mirror']:>8.2f}"
              f"{str(r['qa_pass']):>9}{'  <-- ' if bad else '':>7}")

    av = np.array([r["avidity"] for r in rows if np.isfinite(r["avidity"])])
    mr = np.array([r["mirror"] for r in rows if np.isfinite(r["mirror"])])
    print("-" * 65)
    print(f"\nSummary ({len(rows)} patients):")
    print(f"  avidity (GTV/lung SUV) : median {np.median(av):.2f}  "
          f"[{np.percentile(av,25):.2f}-{np.percentile(av,75):.2f}]")
    print(f"  mirror  (GTV/mirror)   : median {np.median(mr):.2f}  "
          f"[{np.percentile(mr,25):.2f}-{np.percentile(mr,75):.2f}]")
    print(f"  FLAGGED (mirror < {min_ratio}): {n_fail} / {len(rows)} "
          f"({100*n_fail/len(rows):.0f}%) - PET likely misaligned/non-avid")
    meta_n = sum(1 for r in rows if r["qa_pass"] is not None)
    if meta_n:
        meta_fail = sum(1 for r in rows if r["qa_pass"] is False)
        print(f"  pet_reg_meta qa_pass=false: {meta_fail} / {meta_n} with a meta file")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["uuid", "suv_gtv", "suv_bg",
                                               "avidity", "mirror", "qa_pass", "qa_ratio"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
