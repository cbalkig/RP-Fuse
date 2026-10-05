import os
import numpy as np

import train_rpfuse as dg
import ct_radiomics_gated as ctr

_dcfg = dg._data_cfg
GATED_CT_FOLD  = int(_dcfg.get("GATED_CT_FOLD", 0))
SPOTLIGHT_F    = float(_dcfg.get("GATED_CT_SPOTLIGHT_F", 0.2))
GATED_CT_OUT   = str(_dcfg.get("GATED_CT_OUT_DIR", "gated_ct_export"))


def main():
    if not dg.GATE_EMB_DIR:
        raise SystemExit("Set TRAIN_RPFUSE.DATA.GATE_EMB_DIR to the trained gate's results "
                         "dir (it must contain outer_<k>/ntcp_gate.json).")
    curve = ctr.gate_curve_for_fold(dg.GATE_EMB_DIR, GATED_CT_FOLD, log=print)
    if curve is None:
        raise SystemExit(f"No gate curve at {dg.GATE_EMB_DIR}/outer_{GATED_CT_FOLD}/ntcp_gate.json")

    out_root = os.path.join(dg._output_dir, GATED_CT_OUT)
    os.makedirs(out_root, exist_ok=True)
    f = SPOTLIGHT_F
    print(f"[gated-ct] gate fold {GATED_CT_FOLD} | spotlight f={f} | out -> {out_root}")

    n_ok = n_skip = 0
    from tqdm import tqdm
    for uuid in tqdm(dg.cohort_uuids(), desc="export gated CT", unit="pt"):
        paths = ctr._resolve_paths(os.path.join(dg._cohort_dir, uuid))
        if not paths or not all(paths.get(k) for k in ("ct", "dose", "lung")):
            n_skip += 1
            continue
        try:
            ct   = np.load(paths["ct"]).astype(np.float32)
            dose = np.load(paths["dose"]).astype(np.float32)
            lung = np.load(paths["lung"])
        except Exception as e:
            print(f"  [skip] {uuid}: load failed ({e})")
            n_skip += 1
            continue
        if not (ct.shape == dose.shape == lung.shape):
            print(f"  [skip] {uuid}: shape mismatch ct{ct.shape} dose{dose.shape} lung{lung.shape}")
            n_skip += 1
            continue

        g = ctr._gate_g(dose, lung, curve, dg.DOSE_NORM_CGY)
        ct_spot = (ct * (f + (1.0 - f) * g)).astype(np.float32)

        pdir = os.path.join(out_root, uuid)
        os.makedirs(pdir, exist_ok=True)
        np.save(os.path.join(pdir, "g.npy"), g.astype(np.float32))
        np.save(os.path.join(pdir, "ct_spotlight.npy"), ct_spot)
        np.save(os.path.join(pdir, "ct_raw.npy"), ct)

        gl = g[np.asarray(lung) > 0]
        print(f"  {uuid}: g mean={float(g.mean()):.3f} max={float(g.max()):.3f} "
              f"| in-lung mean={float(gl.mean()) if gl.size else 0:.3f} "
              f"gated-frac(g>0.5)={float((gl > 0.5).mean()) if gl.size else 0:.3f}")
        n_ok += 1

    print(f"[gated-ct] wrote {n_ok} patients ({n_skip} skipped) -> {out_root}")
    print("[gated-ct] g.npy = the model's gate map; ct_spotlight.npy = visible render "
          f"(f={f}); ct_raw.npy = reference.")


if __name__ == "__main__":
    main()
