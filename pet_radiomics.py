import os
import glob
import numpy as np

import train_rpfuse as dg

BIN_COUNT = 32
MIN_VOX = {"gtv": 64, "lung": 200}
CACHE_DIR = "pet_radiomics_cache"
_EXTRACTOR = None


def _get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR
    from radiomics import featureextractor
    ext = featureextractor.RadiomicsFeatureExtractor(**{
        "binCount": BIN_COUNT,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,
        "label": 1,
        "force2D": False,
        "normalize": False,
        "resegmentRange": None,
    })
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("firstorder")
    for cls in ("glcm", "glrlm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    _EXTRACTOR = ext
    return ext


def extract_one(pet, roi):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.asarray(pet, np.float32))
    msk = sitk.GetImageFromArray(np.asarray(roi, np.uint8))
    msk.CopyInformation(img)
    names, vals = [], []
    for k, v in _get_extractor().execute(img, msk).items():
        if k.startswith("diagnostics_"):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            names.append(k)
            vals.append(fv)
    return (names, np.asarray(vals, np.float32)) if names else (None, None)


def rois_of(uuid_dir):
    p = dg.resolve_modality_paths(uuid_dir)
    if not p["pet"] or not p["lung"]:
        return None, None
    pet = np.load(p["pet"]).astype(np.float32)
    lung = np.load(p["lung"]) > 0
    gtv = (np.load(p["gtv"]) > 0) if p["gtv"] else np.zeros_like(lung)
    if pet.shape != lung.shape:
        return None, None
    return pet, {"gtv": gtv, "lung": lung & ~gtv}


def build_cache(cohort_dir, cache_dir, force=False):
    os.makedirs(cache_dir, exist_ok=True)
    uuids = dg.cohort_uuids(cohort_dir)
    print(f"cohort: {len(uuids)} patients")
    done = skipped = 0
    for i, u in enumerate(uuids, 1):
        todo = [r for r in ("gtv", "lung")
                if force or not os.path.exists(os.path.join(cache_dir, f"{u}__{r}.npz"))]
        if not todo:
            done += 1
            continue
        pet, rois = rois_of(os.path.join(cohort_dir, u))
        if pet is None:
            print(f"  [{i:3d}/{len(uuids)}] {u}: no registered PET, skipped")
            skipped += 1
            continue
        for r in todo:
            m = rois[r]
            if int(m.sum()) < MIN_VOX[r]:
                print(f"  [{i:3d}/{len(uuids)}] {u}/{r}: ROI too small ({int(m.sum())})")
                continue
            names, feats = extract_one(pet, m)
            if names is None:
                print(f"  [{i:3d}/{len(uuids)}] {u}/{r}: extraction returned nothing")
                continue
            np.savez_compressed(os.path.join(cache_dir, f"{u}__{r}.npz"),
                                names=np.array(names, object), feats=feats)
        done += 1
        print(f"  [{i:3d}/{len(uuids)}] {u}: ok", flush=True)
    print(f"cached {done}, skipped {skipped}")


def assemble(cache_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for roi in ("gtv", "lung"):
        fs = sorted(glob.glob(os.path.join(cache_dir, f"*__{roi}.npz")))
        names_ref, uu, X = None, [], []
        for f in fs:
            u = os.path.basename(f).split("__")[0]
            z = np.load(f, allow_pickle=True)
            nm = [str(x) for x in z["names"].tolist()]
            fv = np.asarray(z["feats"], np.float32)
            if names_ref is None:
                names_ref = nm
            if nm != names_ref or len(fv) != len(names_ref):
                print(f"  [skip] {u}/{roi}: feature-name mismatch")
                continue
            uu.append(u)
            X.append(fv)
        if not uu:
            print(f"petomics_{roi}: nothing to assemble")
            continue
        out = os.path.join(out_dir, f"petomics_{roi}.npz")
        np.savez_compressed(out, uuids=np.array(uu, object),
                            X=np.asarray(X, np.float32), names=np.array(names_ref, object))
        print(f"petomics_{roi}: {len(uu)} patients x {len(names_ref)} features -> {out}")


if __name__ == "__main__":
    cache = os.path.join(dg._input_dir, CACHE_DIR)
    build_cache(dg._cohort_dir, cache)
    assemble(cache, os.path.join(dg._input_dir, "Derivated_Modals"))
