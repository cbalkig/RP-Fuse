import os
import glob
import numpy as np

import train_rpfuse as dg

_ROI_CACHE = {
    "radiomics_lung":      ("lung_radiomics_cache_norm", "*.npz"),
    "radiomics_v20":       ("v20_radiomics_cache_fixed", "*.npz"),
    "radiomics_v20_nogtv": ("v20_nogtv_radiomics_cache", "*.npz"),
    "radiomics_gate":      ("gated_radiomics_cache", "*__gy21.0_*.npz"),
}


def _assemble(cache_dir: str, pattern: str = "*.npz") -> tuple[list[str], np.ndarray, list[str]]:
    fs = sorted(glob.glob(os.path.join(cache_dir, pattern)))
    if not fs:
        raise FileNotFoundError(f"no radiomics cache files in {cache_dir}")
    names_ref, uuids, X = None, [], []
    for f in fs:
        uuid = os.path.basename(f).split("__")[0].strip()
        z = np.load(f, allow_pickle=True)
        nm = [str(x) for x in z["names"].tolist()]
        fv = np.asarray(z["feats"], np.float32)
        if names_ref is None:
            names_ref = nm
        if nm != names_ref or len(fv) != len(names_ref):
            print(f"  [skip] {uuid}: feature-name/len mismatch ({len(fv)} vs {len(names_ref)})")
            continue
        uuids.append(uuid)
        X.append(fv)
    return uuids, np.asarray(X, np.float32), names_ref


if __name__ == "__main__":
    out_dir = os.path.join(dg._input_dir, "Derivated_Modals")
    os.makedirs(out_dir, exist_ok=True)
    for modality, (cache, pat) in _ROI_CACHE.items():
        cache_dir = os.path.join(dg._input_dir, cache)
        if not os.path.isdir(cache_dir):
            print(f"{modality:20s}: cache dir MISSING ({cache}) -- skipped")
            continue
        uuids, X, names = _assemble(cache_dir, pat)
        out = os.path.join(out_dir, f"{modality}.npz")
        np.savez_compressed(out, uuids=np.array(uuids, dtype=object),
                            X=X, names=np.array(names, dtype=object))
        print(f"{modality:20s}: {len(uuids)} x {len(names)}  ({cache}) -> {os.path.basename(out)}")
