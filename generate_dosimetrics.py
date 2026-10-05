import os
import json
import math
import numpy as np

import train_rpfuse as dg

_GEUD_N = 0.87
_NTCP_TD50, _NTCP_M = 24.5, 0.18

_ORGANS = ["lung", "lung_minus_gtv", "heart", "esophagus", "spinal_cord", "liver", "ptv", "gtv"]
_ORGAN_INDEX_FIELDS = ["volume_cc", "MLD_cGy", "V5", "V10", "V20", "V30", "D2_cGy", "D50_cGy", "gEUD_Gy"]


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else 0.0


def _dose_at_volume(grid_cgy, cum_pct, x_pct):
    g = np.asarray(grid_cgy, float); c = np.asarray(cum_pct, float)
    if len(g) < 2 or len(g) != len(c):
        return 0.0
    return float(np.interp(x_pct, c[::-1], g[::-1]))


def _v_at_dose(grid_cgy, cum_pct, d_gy):
    g = np.asarray(grid_cgy, float); c = np.asarray(cum_pct, float)
    if len(g) < 2 or len(g) != len(c):
        return 0.0
    return float(np.interp(d_gy * 100.0, g, c))


def _geud(grid_cgy, cum_pct, n):
    g = np.asarray(grid_cgy, float); c = np.clip(np.asarray(cum_pct, float), 0, 100) / 100.0
    if len(g) < 2 or len(g) != len(c):
        return 0.0
    d_gy = g / 100.0
    dvol = np.clip(-np.diff(c), 0.0, None)
    dmid = 0.5 * (d_gy[:-1] + d_gy[1:])
    tot = dvol.sum()
    if tot <= 0:
        return 0.0
    dvol = dvol / tot
    a = 1.0 / n
    return float(np.sum(dvol * np.power(np.clip(dmid, 0, None), a)) ** (1.0 / a))


def _ntcp(geud_gy):
    if geud_gy <= 0:
        return 0.0
    t = (geud_gy - _NTCP_TD50) / (_NTCP_M * _NTCP_TD50)
    return float(0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))


def _organ_indices(organ, grid, st: dict) -> dict:
    import generate_dvh_curve as gc
    c = st.get("dvh_pct")
    geud = _geud(grid, c, _GEUD_N) if c else 0.0
    return {
        "volume_cc": gc.cap_volume(organ, _num(st.get("volume_cc"))),
        "MLD_cGy":   _num(st.get("Dmean_cGy")),
        "V5":        _num(st.get("V5")),
        "V10":       _v_at_dose(grid, c, 10) if c else 0.0,
        "V20":       _num(st.get("V20")),
        "V30":       _num(st.get("V30")),
        "D2_cGy":    _dose_at_volume(grid, c, 2) if c else 0.0,
        "D50_cGy":   _dose_at_volume(grid, c, 50) if c else 0.0,
        "gEUD_Gy":   geud,
    }


_LUNG_NAMES = [f"lung.{f}" for f in _ORGAN_INDEX_FIELDS] + ["lung.LKB_NTCP"]
_ALL_NAMES = ([f"{o}.{f}" for o in _ORGANS for f in _ORGAN_INDEX_FIELDS]
              + ["lung.LKB_NTCP", "lung_minus_gtv.LKB_NTCP"])


def _vectors(dvh: dict):
    grid = dvh.get("dose_grid_cGy", [])
    S = dvh.get("structures", {}) or {}
    idx = {o: _organ_indices(o, grid, S.get(o, {}) or {}) for o in _ORGANS}
    lung_vec = [idx["lung"][f] for f in _ORGAN_INDEX_FIELDS] + [_ntcp(idx["lung"]["gEUD_Gy"])]
    all_vec = ([idx[o][f] for o in _ORGANS for f in _ORGAN_INDEX_FIELDS]
               + [_ntcp(idx["lung"]["gEUD_Gy"]), _ntcp(idx["lung_minus_gtv"]["gEUD_Gy"])])
    return lung_vec, all_vec


if __name__ == "__main__":
    cohort = dg._cohort_dir
    uuids, X_lung, X_all = [], [], []
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
                lv, av = _vectors(json.load(fp))
            uuids.append(uuid.strip()); X_lung.append(lv); X_all.append(av)
        except Exception as e:
            print(f"  [skip] {uuid}: {e}")
    uu = np.array(uuids, dtype=object)
    np.savez_compressed(dg._cache_write_path("dosimetrics", "dosimetrics.npz"),
                        uuids=uu, X=np.asarray(X_lung, np.float32),
                        names=np.array(_LUNG_NAMES, dtype=object))
    np.savez_compressed(dg._cache_write_path("dosimetrics_all", "dosimetrics_all.npz"),
                        uuids=uu, X=np.asarray(X_all, np.float32),
                        names=np.array(_ALL_NAMES, dtype=object))
    print(f"dosimetrics      : {len(uuids)} x {len(_LUNG_NAMES)} (whole lung) -> dosimetrics.npz")
    print(f"dosimetrics_all  : {len(uuids)} x {len(_ALL_NAMES)} (8 organs)   -> dosimetrics_all.npz")
