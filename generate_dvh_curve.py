import os
import json
import numpy as np
import pandas as pd

import train_rpfuse as dg

_CURVE_STRUCTS = ["lung", "heart", "esophagus", "spinal_cord", "liver", "ptv", "gtv"]
_CURVE_DOSES_GY = list(range(0, 71, 5))

_VOLUME_CAP_CC = {"lung": 9000.0, "lung_minus_gtv": 9000.0, "heart": 1500.0,
                  "esophagus": 200.0, "spinal_cord": 400.0, "liver": 4000.0,
                  "ptv": 5000.0, "gtv": 2000.0}


def cap_volume(struct: str, v: float) -> float:
    cap = _VOLUME_CAP_CC.get(struct)
    return min(float(v), cap) if (cap is not None and v is not None and np.isfinite(v)) else v


def _curve_columns() -> list[str]:
    cols = []
    for s in _CURVE_STRUCTS:
        cols += [f"{s}.V{d}Gy" for d in _CURVE_DOSES_GY]
        cols.append(f"{s}.volume_cc")
    return cols


def _patient_curve(dvh: dict) -> dict:
    grid = np.asarray(dvh.get("dose_grid_cGy", []), float)
    structs = dvh.get("structures", {}) or {}
    out = {}
    for s in _CURVE_STRUCTS:
        st = structs.get(s, {}) or {}
        cum = st.get("dvh_pct")
        if cum is not None and len(grid) == len(cum) and len(grid) > 1:
            c = np.clip(np.asarray(cum, float), 0.0, 100.0)
            for d in _CURVE_DOSES_GY:
                out[f"{s}.V{d}Gy"] = float(np.interp(d * 100.0, grid, c))
        else:
            for d in _CURVE_DOSES_GY:
                out[f"{s}.V{d}Gy"] = 0.0
        v = st.get("volume_cc")
        out[f"{s}.volume_cc"] = cap_volume(s, v) if v is not None and np.isfinite(v) else 0.0
    return out


def build_dvh_curve_table() -> pd.DataFrame:
    cols = _curve_columns()
    rows = []
    cohort = dg._cohort_dir
    for uuid in dg.cohort_uuids(cohort):
        udir = os.path.join(cohort, uuid)
        if not os.path.isdir(udir):
            continue
        sdir = dg.resolve_primary_study(udir)
        if not sdir:
            continue
        p = os.path.join(sdir, "DVH", "dvh.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fp:
                rows.append({"uuid": uuid.strip(), **_patient_curve(json.load(fp))})
        except Exception as e:
            print(f"  [skip] {uuid}: {e}")
    return pd.DataFrame(rows, columns=["uuid"] + cols)


if __name__ == "__main__":
    out_csv = dg._cache_write_path("dvh_curve", "dvh_curve.csv")
    df = build_dvh_curve_table()
    df.to_csv(out_csv, index=False)
    n_feat = len(_curve_columns())
    print(f"DVH curve : {len(df)} patients x {n_feat} features "
          f"({len(_CURVE_STRUCTS)} structures x {len(_CURVE_DOSES_GY)} dose steps + volumes) -> {out_csv}")
    print(f"structures: {_CURVE_STRUCTS}")
    print(f"dose steps (Gy): {_CURVE_DOSES_GY}")
