import os
import glob
import json
import numpy as np

_EXTRACTOR = None


def _get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR
    from radiomics import featureextractor
    settings = {
        "binWidth": 25,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,
        "label": 1,
        "force2D": False,
        "normalize": False,
        "resegmentRange": [-1000, 400],
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("firstorder")
    for cls in ("glcm", "glrlm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    _EXTRACTOR = ext
    return ext


def _gate_g(dose_cgy, lung, curve, dose_norm_cgy: float = 6000.0):
    dn = np.clip(np.asarray(dose_cgy, np.float32) / float(dose_norm_cgy), 0.0, 1.0)
    g = np.interp(dn, np.asarray(curve["d_grid"], np.float64),
                  np.asarray(curve["phi"], np.float64)).astype(np.float32)
    return g * (np.asarray(lung) > 0)


def extract_one(ct, dose_cgy, lung, dose_gy: float, min_voxels: int = 200,
                image: str = "raw", mask: str = "dose_d50",
                curve=None, dose_norm_cgy: float = 6000.0,
                gtv=None, exclude_gtv: bool = False, gtv_margin_vox: int = 0,
                gate_thr: float = 0.5):
    import SimpleITK as sitk
    ct = np.asarray(ct, np.float32)
    dose_cgy = np.asarray(dose_cgy, np.float32)
    lung = (np.asarray(lung) > 0)
    if exclude_gtv and gtv is not None:
        gtv = (np.asarray(gtv) > 0)
        if gtv.shape == lung.shape and gtv.any():
            if gtv_margin_vox and gtv_margin_vox > 0:
                from scipy.ndimage import binary_dilation
                gtv = binary_dilation(gtv, iterations=int(gtv_margin_vox))
            lung = lung & (~gtv)
    g = None
    if (mask == "gate" or image == "gated") and curve is not None:
        g = _gate_g(dose_cgy, lung, curve, dose_norm_cgy)
    if mask == "lung":
        roi = lung.astype(np.uint8)
    elif mask == "gate" and g is not None:
        roi = (g >= gate_thr).astype(np.uint8)
    else:
        roi = (lung & (dose_cgy >= dose_gy * 100.0)).astype(np.uint8)
    if int(roi.sum()) < min_voxels:
        roi = lung.astype(np.uint8)
    if int(roi.sum()) < min_voxels:
        return None, None
    ct_used = ct
    if image == "gated" and g is not None:
        f = float(curve.get("f", 1.0))
        ct_used = (ct * (f + (1.0 - f) * g)).astype(np.float32)
    img = sitk.GetImageFromArray(ct_used)
    msk = sitk.GetImageFromArray(roi)
    msk.CopyInformation(img)
    result = _get_extractor().execute(img, msk)
    names, vals = [], []
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            names.append(k)
            vals.append(fv)
    if not names:
        return None, None
    return names, np.asarray(vals, np.float32)


def _resolve_paths(uuid_dir: str):
    pool = []
    for mp in glob.glob(os.path.join(uuid_dir, "studies", "*", "study_meta.json")):
        sdir = os.path.dirname(mp)
        if not os.path.exists(os.path.join(sdir, "CT", "whole_body.npy")):
            continue
        try:
            meta = json.load(open(mp))
        except Exception:
            meta = {}
        pool.append((sdir, meta))
    if not pool:
        return None

    def _key(c):
        return (c[1].get("dose_max_cGy") or 0, c[1].get("lung_MLD_cGy") or 0)
    sdir, _ = max(pool, key=_key)

    def _p(*a):
        f = os.path.join(sdir, *a)
        return f if os.path.exists(f) else None
    return {"ct": _p("CT", "whole_body.npy"),
            "dose": _p("Dose", "dose_regCT.npy"),
            "lung": _p("Contours", "lung_mask.npy"),
            "gtv": _p("Contours", "gtv_mask.npy")}


def gate_d50_for_fold(gate_dir: str, outer_fold: int, default_gy: float = 16.0,
                      dose_norm_cgy: float = 6000.0, log=print) -> float:
    _d = os.path.join(gate_dir, f"outer_{outer_fold}")
    path = os.path.join(_d, "dose_response_gate.json")
    if not os.path.exists(path):
        path = os.path.join(_d, "ntcp_gate.json")
    if not os.path.exists(path):
        log(f"[radiomics] gate curve not found ({path}); using fixed {default_gy:.1f} Gy")
        return float(default_gy)
    try:
        j = json.load(open(path))
        dg = np.asarray(j["d_grid"], np.float64)
        phi = np.asarray(j["phi"], np.float64)
        d50 = float(dg[int(np.argmin(np.abs(phi - 0.5)))]) * (dose_norm_cgy / 100.0)
        log(f"[radiomics] outer fold {outer_fold}: gate-learned d50 = {d50:.1f} Gy")
        return d50
    except Exception as e:
        log(f"[radiomics] failed to read gate d50 ({e}); using fixed {default_gy:.1f} Gy")
        return float(default_gy)


def gate_curve_for_fold(gate_dir: str, outer_fold: int, log=print):
    _d = os.path.join(gate_dir, f"outer_{outer_fold}")
    path = os.path.join(_d, "dose_response_gate.json")
    if not os.path.exists(path):
        path = os.path.join(_d, "ntcp_gate.json")
    if not os.path.exists(path):
        log(f"[radiomics] gate curve not found ({path}); gate mask/image unavailable")
        return None
    try:
        j = json.load(open(path))
        return {"d_grid": j["d_grid"], "phi": j["phi"], "f": float(j.get("f", 1.0))}
    except Exception as e:
        log(f"[radiomics] failed to read gate curve ({e})")
        return None


def build_map(cohort_dir: str, dose_gy: float, cache_dir: str,
              uuids=None, log=print, image: str = "raw", mask: str = "dose_d50",
              curve=None, dose_norm_cgy: float = 6000.0,
              exclude_gtv: bool = False, gtv_margin_vox: int = 0,
              fold_tag: str = "", gate_thr: float = 0.5):
    os.makedirs(cache_dir, exist_ok=True)
    if uuids is None:
        uuids = sorted(u for u in os.listdir(cohort_dir)
                       if os.path.isdir(os.path.join(cohort_dir, u)))
    tag = "" if (image == "raw" and mask == "dose_d50") else f"_{image}_{mask}"
    if mask == "gate" and abs(gate_thr - 0.5) > 1e-6:
        tag += f"_g{int(round(gate_thr * 100)):03d}"
    if exclude_gtv:
        tag += f"_nogtv{int(gtv_margin_vox)}"
    tag += fold_tag
    feat_map, names_ref, n_ok, n_fail = {}, None, 0, 0
    from tqdm import tqdm
    for u in tqdm(uuids, desc=f"radiomics ({image}/{mask})", unit="pt"):
        cache = os.path.join(cache_dir, f"{u}__gy{dose_gy:.1f}{tag}.npz")
        if os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=True)
                nm, fv = z["names"].tolist(), z["feats"].astype(np.float32)
                if len(nm) and fv.size:
                    feat_map[u] = fv
                    names_ref = names_ref or nm
                    n_ok += 1
                    continue
            except Exception:
                pass
        paths = _resolve_paths(os.path.join(cohort_dir, u))
        if not paths or not all(paths.get(k) for k in ("ct", "dose", "lung")):
            n_fail += 1
            continue
        gtv_arr = None
        if exclude_gtv and paths.get("gtv"):
            try:
                gtv_arr = np.load(paths["gtv"])
            except Exception:
                gtv_arr = None
        try:
            nm, fv = extract_one(np.load(paths["ct"]), np.load(paths["dose"]),
                                 np.load(paths["lung"]), dose_gy,
                                 image=image, mask=mask, curve=curve,
                                 dose_norm_cgy=dose_norm_cgy,
                                 gtv=gtv_arr, exclude_gtv=exclude_gtv,
                                 gtv_margin_vox=gtv_margin_vox, gate_thr=gate_thr)
        except Exception as e:
            log(f"[radiomics] {u} failed: {e}")
            nm, fv = None, None
        if nm is None:
            n_fail += 1
            continue
        np.savez_compressed(cache, names=np.array(nm), feats=fv)
        feat_map[u] = fv
        names_ref = names_ref or nm
        n_ok += 1
    log(f"[radiomics] extracted {n_ok} ok, {n_fail} failed; F={len(names_ref or [])} features")
    return feat_map, (names_ref or [])


if __name__ == "__main__":
    from util import load_config
    c = load_config()
    inp = c["INPUTS_DIR"]
    out_dir = c.get("OUTPUTS_DIR", inp)
    cohort = os.path.join(inp, c["INPUTS"]["PROCESSED"])
    dcfg = c.get("TRAIN_RPFUSE", {}).get("DATA", {})
    fixed_gy = float(dcfg.get("CT_RADIOMICS_DOSE_GY", 16.0))
    cache = os.path.join(inp, str(dcfg.get("CT_RADIOMICS_CACHE_DIR", "gated_radiomics_cache")))

    image = str(dcfg.get("CT_RADIOMICS_IMAGE", "raw")).lower()
    mask  = str(dcfg.get("CT_RADIOMICS_MASK", "dose_d50")).lower()
    thrs  = [float(t) for t in dcfg.get("CT_RADIOMICS_GATE_THRS", [])] \
        or [float(dcfg.get("CT_RADIOMICS_GATE_THR", 0.5))]
    if mask != "gate":
        thrs = [0.5]
    norm  = float(dcfg.get("DOSE_NORM_CGY", 6000.0))
    excl_gtv = bool(dcfg.get("CT_RADIOMICS_EXCLUDE_GTV", False))
    gtv_marg = int(dcfg.get("CT_RADIOMICS_GTV_MARGIN_VOX", 0))
    needs_curve = (image == "gated" or mask == "gate")
    spotlight_f = float(dcfg.get("CT_RADIOMICS_SPOTLIGHT_F", 1.0))
    gate_dir = str(dcfg.get("GATE_EMB_DIR", "") or "")
    if gate_dir and not os.path.isabs(gate_dir):
        gate_dir = os.path.join(out_dir, gate_dir)

    def _inject_f(crv):
        if crv is not None and image == "gated" and spotlight_f < 1.0:
            return {**crv, "f": spotlight_f}
        return crv

    per_fold = bool(dcfg.get("CT_RADIOMICS_PER_FOLD_D50", False))
    gate_based = needs_curve or per_fold
    jobs = []
    if per_fold and gate_dir:
        folds = sorted(int(os.path.basename(d).split("_")[1])
                       for d in glob.glob(os.path.join(gate_dir, "outer_*")))
        for k in folds:
            gy = round(gate_d50_for_fold(gate_dir, k, fixed_gy, norm), 1)
            crv = _inject_f(gate_curve_for_fold(gate_dir, k)) if needs_curve else None
            jobs.append((k, gy, crv))
    if not jobs:
        k0 = int(os.getenv("NESTED_OUTER_FOLD", "0"))
        crv = (_inject_f(gate_curve_for_fold(gate_dir, k0)) if (needs_curve and gate_dir) else None)
        jobs.append((k0 if (gate_based and gate_dir) else None, fixed_gy, crv))
    import train_rpfuse as dg
    cohort_ids = dg.cohort_uuids(cohort)
    print(f"cohort={cohort}\ncache={cache}\nimage={image} mask={mask}  "
          f"jobs(fold,dose_gy)={[(k, g) for k, g, _ in jobs]}"
          f"\nrestricted to n={len(cohort_ids)} cohort patients")

    names = []
    for k_job, gy, crv in jobs:
        ftag = f"_k{k_job}" if k_job is not None else ""
        if ftag and image == "gated" and spotlight_f < 1.0:
            ftag += f"_f{int(round(spotlight_f * 100)):03d}"
        for thr in thrs:
            print(f"\n=== precomputing radiomics: dose>={gy:.1f} Gy | image={image} | mask={mask}"
                  f"{(' | gate_thr ' + format(thr, '.2f')) if mask == 'gate' else ''}"
                  f"{(' | fold ' + str(k_job)) if k_job is not None else ''} ===")
            fmap, names = build_map(cohort, gy, cache, image=image, mask=mask, uuids=cohort_ids,
                                    curve=crv, dose_norm_cgy=norm,
                                    exclude_gtv=excl_gtv, gtv_margin_vox=gtv_marg,
                                    fold_tag=ftag, gate_thr=thr)
            print(f"  {len(fmap)} patients, {len(names)} features/patient")
    if names:
        print("first 8 feature names:", names[:8])
