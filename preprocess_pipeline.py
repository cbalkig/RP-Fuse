import argparse
import csv
import datetime
import gc
import glob
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import re
import shutil
import sys
import warnings
from concurrent.futures import (ProcessPoolExecutor, as_completed)

import unicodedata

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from util import load_config, read_table, table_exists

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s",
                    stream=sys.stdout)


def _name_to_uuid(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def _read_dicom(fp, **kwargs):
    ds = pydicom.dcmread(fp, force=True, **kwargs)
    fm = getattr(ds, "file_meta", None)
    if fm is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        fm = ds.file_meta
    if "TransferSyntaxUID" not in fm:
        fm.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    return ds


def _read_rt_demographic_name(patient_folder, hasta_no):
    demo_path = os.path.join(patient_folder, f"demographic.{hasta_no}")
    if not os.path.isfile(demo_path):
        return ""
    try:
        with open(demo_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "^" in line:
                    return line
    except Exception:
        pass
    return ""


def _collect_rt_patients(raw_rt_dir):
    if not os.path.isdir(raw_rt_dir):
        return {}
    patients = {}
    for pf in sorted(os.listdir(raw_rt_dir)):
        if not pf.startswith("1~"):
            continue
        hasta_no = pf[2:].strip()
        pp = os.path.join(raw_rt_dir, pf)
        if not os.path.isdir(pp):
            continue
        ct_dcm = os.path.join(pp, "1~CT1", "DCMData")
        pt_dcm = os.path.join(pp, "1~PT1", "DCMData")
        patients[hasta_no] = {
            "src": pp,
            "ct_plan_dir": ct_dcm if os.path.isdir(ct_dcm) else None,
            "pet_plan_dir": pt_dcm if os.path.isdir(pt_dcm) else None,
            "demographic_name": _read_rt_demographic_name(pp, hasta_no),
        }
    return patients


def _scan_patient_imaging(patient_dir: str) -> dict:
    def _series_meta(series_dir):
        dcm = os.path.join(series_dir, "DCMData")
        files = glob.glob(os.path.join(dcm, "*")) if os.path.isdir(dcm) else []
        n = sum(1 for f in files if f.lower().endswith(".dcm"))
        date, desc, modality = "", "", ""
        for f in files:
            try:
                x = _read_dicom(f, stop_before_pixels=True)
                if hasattr(x, "ImagePositionPatient"):
                    d = str(x.get("AcquisitionDate") or x.get("StudyDate") or "")
                    date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
                    desc = str(x.get("SeriesDescription", ""))[:24]
                    modality = str(x.get("Modality", ""))
                    break
            except Exception:
                continue
        has_contours = os.path.isfile(os.path.join(series_dir, "contournames"))
        return {"n": n, "date": date, "desc": desc, "modality": modality,
                "contours": has_contours}

    inv = {}
    plan_files = sorted(glob.glob(os.path.join(patient_dir, "plan", "*", "plan")))
    plan_descr = []
    planned_refs = set()
    combined_refs = set()
    for pf in plan_files:
        name = os.path.basename(os.path.dirname(pf))
        try:
            L = open(pf, encoding="ascii", errors="ignore").read().splitlines()
            ctref = (L[6].strip() if len(L) > 6 else "?")
        except Exception:
            ctref = "?"
        combined = "+" in name
        if ctref and ctref != "?":
            planned_refs.add(ctref.upper())
            if combined:
                combined_refs.add(ctref.upper())
        plan_descr.append(f"{name}->{ctref}" + ("*" if combined else ""))
    inv["n_plans"] = len(plan_files)

    metas = []
    for cd in glob.glob(os.path.join(patient_dir, "1~CT*")):
        m = _series_meta(cd)
        m["name"] = os.path.basename(cd)
        m["folder_num"] = int("".join(c for c in m["name"] if c.isdigit()) or 0)
        m["planned"] = m["name"].replace("1~", "").upper() in planned_refs
        m["combined"] = m["name"].replace("1~", "").upper() in combined_refs
        m["meaningful"] = m["contours"] and m["planned"]
        metas.append(m)
    metas.sort(key=lambda m: (m["date"] == "", m["date"], m["folder_num"]))
    inv["n_ct"] = len(metas)
    inv["n_contoured_ct"] = sum(1 for m in metas if m["contours"])
    inv["n_meaningful_ct"] = sum(1 for m in metas if m["meaningful"])
    for i, m in enumerate(metas, 1):
        inv[f"ct{i}_name"] = m["name"]
        inv[f"ct{i}_date"] = m["date"]
        inv[f"ct{i}_slices"] = m["n"]
        inv[f"ct{i}_contours"] = "Y" if m["contours"] else "N"
        inv[f"ct{i}_planned"] = "Y" if m["planned"] else "N"
        inv[f"ct{i}_meaningful"] = ("Y" if m["meaningful"] else "N") + (
            " (total)" if m["combined"] else "")

    pet_dir = os.path.join(patient_dir, "1~PT1")
    if os.path.isdir(pet_dir):
        m = _series_meta(pet_dir)
        inv["pet_date"] = m["date"]
        inv["pet_slices"] = m["n"]
    inv["plans"] = "; ".join(plan_descr)

    inv["has_phantom"] = "Y" if os.path.isdir(
        os.path.join(patient_dir, "1~MonacoPhantom")) else "N"
    return inv


def step_0_organize(cfg):
    print("\n" + "=" * 60)
    print("  STEP 0 - Build Patient Manifest")
    print("=" * 60)
    input_dir = cfg["INPUTS_DIR"]
    raw_dir = cfg["RAW_DIR"]
    raw_rt_dir = os.path.join(raw_dir, "RTs")

    print(f"Source root: {raw_dir}")

    rt_patients = _collect_rt_patients(raw_rt_dir)
    print(f"RTs/: found {len(rt_patients)} patient folder(s)  "
          f"(ct_plan: {sum(1 for v in rt_patients.values() if v['ct_plan_dir'])}, "
          f"pet_plan: {sum(1 for v in rt_patients.values() if v['pet_plan_dir'])})")

    modalities = _get_modalities(cfg)
    modality_keys = list(modalities.keys())

    def _modality_present(modality_cfg: dict, row: dict) -> bool:
        src = _resolve_source(modality_cfg["source"], row, raw_dir)
        if not src or not os.path.exists(src):
            return False
        if modality_cfg.get("type") in ("passthrough", "contours", "dose"):
            patterns = modality_cfg.get("patterns", [])
            if patterns:
                for pat in patterns:
                    for path in glob.glob(os.path.join(src, pat)):
                        if os.path.isfile(path):
                            return True
                return False
            return bool(os.listdir(src))
        if not os.path.isdir(src):
            return False
        return any(f.lower().endswith(".dcm") for f in os.listdir(src))

    manifest_rows = []
    inventory_rows = []
    for hasta_no in sorted(rt_patients.keys()):
        rt_info = rt_patients[hasta_no]
        demo_name = _norm_name(rt_info.get("demographic_name", "")) or hasta_no
        uuid = _name_to_uuid(hasta_no)
        row = {
            "uuid": uuid,
            "patient_name": demo_name,
            "dicom_patient_name": rt_info.get("demographic_name", ""),
            "hasta_no": hasta_no,
            "rp_status": "",
        }
        for mk, mcfg in modalities.items():
            row[f"has_{mk}"] = _modality_present(mcfg, row)
        manifest_rows.append(row)
        inv = {"uuid": uuid, "patient_name": demo_name, "hasta_no": hasta_no}
        inv.update(_scan_patient_imaging(rt_info["src"]))
        inventory_rows.append(inv)

    os.makedirs(input_dir, exist_ok=True)
    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    base_cols = ["uuid", "patient_name", "dicom_patient_name", "hasta_no",
                 "rp_status"]
    has_cols = [f"has_{mk}" for mk in modality_keys]
    cols = base_cols + has_cols
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in manifest_rows:
            w.writerow([r.get(c, "") for c in cols])
    _to_excel(pd.read_csv(manifest_path), manifest_path.replace(".csv", ".xlsx"))

    inv_df = pd.DataFrame(inventory_rows)
    _front = ["uuid", "patient_name", "hasta_no", "n_ct"]
    inv_cols = _front + [c for c in inv_df.columns if c not in _front]
    inv_df = inv_df.reindex(columns=inv_cols)
    inv_path = os.path.join(input_dir, "patient_imaging_inventory.csv")
    inv_df.to_csv(inv_path, index=False)
    _to_excel(inv_df, inv_path.replace(".csv", ".xlsx"))
    _max_ct = inv_df["n_ct"].max() if len(inv_df) else 0
    print(f"Imaging inventory: {inv_path}  "
          f"(max {_max_ct} CT series/patient; "
          f"{int((inv_df['n_ct'] > 1).sum())} patients with >1 CT, "
          f"{int((inv_df['n_plans'] > 1).sum())} with >1 plan)")

    legacy_path = os.path.join(input_dir, "patient_name_to_uuid.csv")
    with open(legacy_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_name", "uuid", "rp_status",
                    "has_before_rt", "has_after_rt",
                    "dicom_patient_name", "dicom_patient_id"])
        for r in manifest_rows:
            w.writerow([r["patient_name"], r["uuid"], r["rp_status"],
                        True, False,
                        r["dicom_patient_name"], r["hasta_no"]])
    _to_excel(pd.read_csv(legacy_path), legacy_path.replace(".csv", ".xlsx"))

    print(f"\nManifest: {manifest_path}")
    print(f"  {len(manifest_rows)} patients")
    for mk in modality_keys:
        n = sum(1 for r in manifest_rows if r[f"has_{mk}"])
        print(f"    has_{mk}: {n}")
    print(f"Legacy: {legacy_path}")


def _check_sequence(file_list):
    if not file_list:
        return False, "Empty Folder"
    numbers = []
    for f in file_list:
        m = re.search(r"_(\d+)\.dcm$", f.lower())
        if m:
            numbers.append(int(m.group(1)))
    if not numbers:
        return True, "OK (unnumbered)"
    numbers.sort()
    unique = set(numbers)
    if (numbers[-1] - numbers[0] + 1) == len(unique):
        return True, "OK"
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - unique)
    return False, (f"Missing: {missing[:5]}..." if len(missing) > 5
                   else f"Missing: {missing}")


def _validate_one(args):
    uuid, modality_key, mtype, expected_mod, src_dir = args
    record = {
        "UUID": uuid,
        "Modality": modality_key,
        "Type": mtype,
        "Source": src_dir,
        "Slice_Count": 0,
        "Sequence": "N/A",
        "Wrong_Modality_Count": 0,
        "Expected_Modality": expected_mod,
    }
    if not src_dir or not os.path.isdir(src_dir):
        record["Sequence"] = "missing source dir"
        return record

    if mtype in ("passthrough", "contours", "dose"):
        n_files = sum(len(fs) for _, _, fs in os.walk(src_dir))
        record["Slice_Count"] = n_files
        record["Sequence"] = "OK" if n_files > 0 else "Empty Folder"
        return record

    dcm = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".dcm"))
    record["Slice_Count"] = len(dcm)
    _, msg = _check_sequence(dcm)
    record["Sequence"] = msg
    if dcm:
        try:
            ds = _read_dicom(os.path.join(src_dir, dcm[0]),
                             stop_before_pixels=True)
            mod = getattr(ds, "Modality", "Unknown")
            if mod != expected_mod:
                record["Wrong_Modality_Count"] = len(dcm)
        except Exception:
            record["Wrong_Modality_Count"] = len(dcm)
    return record


def step_1_validate(cfg):
    print("\n" + "=" * 60)
    print("  STEP 1 - Validate DICOMs")
    print("=" * 60)
    raw_dir = cfg["RAW_DIR"]
    input_dir = cfg["INPUTS_DIR"]

    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    if not table_exists(manifest_path):
        print(f"Error: manifest not found ({manifest_path}). Run step 0 first.")
        return

    manifest = read_table(manifest_path).fillna("").to_dict("records")
    manifest = _filter_manifest_only(manifest, cfg)
    modalities = _get_modalities(cfg)
    enabled = {k: v for k, v in modalities.items() if v.get("enabled", True)}

    type_to_mod = {"ct": "CT", "pet": "PT"}

    work = []
    for row in manifest:
        for modality_key, modality_cfg in enabled.items():
            has_flag = f"has_{modality_key}"
            if not row.get(has_flag, False):
                continue
            src = _resolve_source(modality_cfg["source"], row, raw_dir)
            mtype = modality_cfg.get("type", "ct")
            expected_mod = type_to_mod.get(mtype, "CT")
            work.append((str(row["uuid"]), modality_key, mtype,
                         expected_mod, src))

    print(f"Validating {len(work)} (patient, modality) pairs...")
    if not work:
        print("Nothing to validate - manifest has no patients with enabled "
              "modalities. Did step 0 find any data?")
        return
    results = []
    with ProcessPoolExecutor() as pool:
        futs = {pool.submit(_validate_one, w): w for w in work}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Validating"):
            results.append(f.result())

    df = pd.DataFrame(results)
    out = os.path.join(input_dir, "patient_dcom_validation.csv")
    df.to_csv(out, index=False)
    _to_excel(df, out.replace(".csv", ".xlsx"))
    print(f"Saved: {out}")

    if df.empty:
        print("No validation rows produced.")
        return
    seq_ok = df["Sequence"].astype(str).str.startswith("OK")
    problems = (df["Wrong_Modality_Count"] > 0) | (~seq_ok)
    if problems.any():
        print(f"Issues found in {problems.sum()} (patient, modality) pair(s):")
        print(df[problems][["UUID", "Modality", "Slice_Count",
                            "Wrong_Modality_Count", "Sequence"]]
              .head(20).to_string(index=False))
    else:
        print("No issues found.")


def _norm_name(s: str) -> str:
    s = str(s).strip()
    if "^" in s:
        s = " ".join(reversed(s.split("^")))
    s = s.lower()
    s = (s.replace("ı", "i").replace("İ", "i")
         .replace("ğ", "g").replace("Ğ", "g")
         .replace("ş", "s").replace("Ş", "s")
         .replace("ç", "c").replace("Ç", "c")
         .replace("ö", "o").replace("Ö", "o")
         .replace("ü", "u").replace("Ü", "u"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.split())


TR_TO_EN_COLS = {
    "AdSoyad": "patient_name",
    "HastaNo": "patient_id",
    "yaş": "age",
    "cinsiyet": "gender",
    "ECOGperformans": "ECOG_performance",
    "kilo": "weight_kg",
    "tedaviöncesisigarahikayesipaketyıl": "smoking_packyear",
    "Aktifsigaraiçimitedavisırasında": "active_smoking",
    "Tevresi": "T_stage",
    "Nevresi": "N_stage",
    "TNMEvre": "TNM_stage",
    "tümörçapıcm": "tumor_size_cm",
    "tümörlokalizasyonu1sağsolakciğer": "tumor_side",
    "tümörlokalizasyonu2altortaüstlob": "tumor_lobe",
    "patolojihistopatolojiktanı": "histopathology",
    "KTajanıkullanılanajanınınadıvedozu": "chemo_agent_dose",
    "KaçkürKTalınankürsayısı": "chemo_cycles",
    "RTteknik3DCRTIMRT": "RT_technique",
    "RTdoztoplamdozfrx.dozu": "RT_dose_total_frac",
    "RTaldığızaman": "RT_date",
    "PETfüzyonvaryok": "PET_fusion",
    "Diabetesmellitusvaryok": "DM",
    "KOAHvaryok": "COPD",
    "kollajenvask.Hast.varyok": "collagen_vascular_disease",
    "Sentrilobüleramfizemvarlığı": "centrilobular_emphysema",
    "Bronşektazivarlığı": "bronchiectasis",
    "Retikülerpatern": "reticular_pattern",
    "KistikAChastalığı": "cystic_lung_disease",
    "Asbestozbulguları": "asbestos_findings",
    "GTVvolümcc": "GTV_cc",
    "PTVvolümcc": "PTV_cc",
    "AkciğerV5": "lung_V5",
    "AkcV10": "lung_V10",
    "Akc.V20": "lung_V20",
    "MLDmeanlungdose": "MLD",
    "Radyasyonpnömonisi": "RP_raw",
    "RadyolojikRP": "RP_radiologic",
    "KlinikRP": "RP_clinical",
    "Radyasyonpnömonisigrade234": "RP_grade_234",
    "RPtanısıradyolojikklinik": "RP_diagnosis_type",
    "RPgelişmezamanıtedavidenkaçaysonra": "RP_onset_months",
    "RPtedavi": "RP_treatment",
    "RPgelişenlerdeprogresyonsuzsağkalım": "PFS_RP_pos",
    "RPgelişenlerdegenelsağkalım": "OS_RP_pos",
    "RPgelişmeyenlerdeprogresyonsuzsağkalım": "PFS_RP_neg",
    "RPgelişmeyenlerdegenelsağkalım": "OS_RP_neg",
    "Sonkontroltarihi": "last_followup_date",
    "Exolanlarınölümtarihi": "death_date",
    "Sondurum": "final_status",
}


def step_build_master(cfg):
    import glob

    print("\n" + "=" * 60)
    print("  STEP 5 - Build Master Clinical Excel")
    print("=" * 60)

    src_cfg = cfg.get("PREPROCESS", {}).get("SOURCE_EXCELS", {})
    src_dir_rel = src_cfg.get("DIR", "Data")
    master_glob = src_cfg.get("MASTER_FILE", "rp 18*.xlsx")
    master_sheet = src_cfg.get("MASTER_SHEET", "Sayfa1")
    clean_glob = src_cfg.get("CLEAN_MASTER_FILE", "Veriler BALKI.xlsx")
    out_name = cfg.get("PREPROCESS", {}).get("MASTER_OUTPUT",
                                             "master_clinical_data.xlsx")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = cfg.get("RAW_DIR") or ""
    input_dir = cfg.get("INPUTS_DIR") or ""
    data_root = os.path.dirname(os.path.abspath(input_dir)) if input_dir else ""
    files_dir = os.path.join(data_root, "Files") if data_root else ""

    if os.path.isabs(src_dir_rel):
        src_dir = src_dir_rel
    else:
        candidates = [
            files_dir,
            os.path.join(files_dir, src_dir_rel) if files_dir else "",
            os.path.join(raw_dir, src_dir_rel) if raw_dir else "",
            os.path.join(repo_root, src_dir_rel),
        ]
        src_dir = next((c for c in candidates
                        if c and os.path.isdir(c)),
                       candidates[-1])

    def _resolve(pattern, role):
        if any(ch in pattern for ch in "*?["):
            cands = sorted(glob.glob(os.path.join(src_dir, pattern)))
        else:
            literal = os.path.join(src_dir, pattern)
            cands = [literal] if os.path.exists(literal) else []
        if not cands:
            tried_ext = (os.path.join(raw_dir, src_dir_rel)
                         if raw_dir and not os.path.isabs(src_dir_rel) else None)
            extra = (f"\n  RAW_DIR candidate: {tried_ext} "
                     f"(exists={os.path.isdir(tried_ext) if tried_ext else 'n/a'})"
                     if tried_ext else "")
            raise FileNotFoundError(
                f"No {role} file matching {pattern!r} found in {src_dir}"
                f"{extra}\n"
                f"Adjust PREPROCESS.SOURCE_EXCELS in config.yaml, or check "
                f"that the external drive holding Raw/ is mounted.")
        return cands[0]

    master_path = _resolve(master_glob, "MASTER_FILE")
    clean_path = _resolve(clean_glob, "CLEAN_MASTER_FILE")

    print(f"  Master source        : {os.path.basename(master_path)}")
    print(f"  Clean-master source  : {os.path.basename(clean_path)}")

    df_master = pd.read_excel(master_path, sheet_name=master_sheet)
    df_clean = pd.read_excel(clean_path)

    print(f"  Master rows:        {len(df_master)} x {len(df_master.columns)} cols")
    print(f"  Clean-master rows:  {len(df_clean)} x {len(df_clean.columns)} cols")

    df_master.columns = [str(c).strip() for c in df_master.columns]
    df_clean.columns = [str(c).strip() for c in df_clean.columns]

    fp_master_cols = ["GTV volüm(cc)", "PTV volüm(cc)", "Akciğer V5 (%)",
                      "Akc V10 (%)", "Akc. V20 (%)", "MLD (mean lung dose)"]
    fp_clean_cols = ["GTVvolümcc", "PTVvolümcc", "AkciğerV5",
                     "AkcV10", "Akc.V20", "MLDmeanlungdose"]

    for c in fp_master_cols:
        if c not in df_master.columns:
            raise KeyError(f"Master is missing fingerprint column: {c!r}")
    for c in fp_clean_cols:
        if c not in df_clean.columns:
            raise KeyError(f"Clean-master is missing fingerprint column: {c!r}")

    def _fp(row, cols):
        try:
            return tuple(round(float(row[c]), 2) for c in cols)
        except (TypeError, ValueError):
            return None

    df_master["__fp"] = df_master.apply(lambda r: _fp(r, fp_master_cols), axis=1)
    df_clean["__fp"] = df_clean.apply(lambda r: _fp(r, fp_clean_cols), axis=1)

    n_master_unique = df_master["__fp"].dropna().nunique()
    n_clean_unique = df_clean["__fp"].dropna().nunique()
    print(f"  Master fingerprints:        {n_master_unique} unique / {len(df_master)} rows")
    print(f"  Clean-master fingerprints:  {n_clean_unique} unique / {len(df_clean)} rows")
    if n_clean_unique < len(df_clean):
        print("  Warning: non-unique fingerprints in clean-master - "
              "ambiguous joins possible. Inspect manually.")

    derived_cols = ["KlinikRP", "RadyolojikRP", "TNMEvre"]
    available = [c for c in derived_cols if c in df_clean.columns]
    missing = [c for c in derived_cols if c not in df_clean.columns]
    if missing:
        print(f"  Warning: Clean-master missing expected derived columns: {missing}")

    df_clean_subset = df_clean[["__fp"] + available].drop_duplicates(subset="__fp")

    df_out = df_master.merge(df_clean_subset, on="__fp", how="left",
                             indicator="_clean_match")
    n_matched = (df_out["_clean_match"] == "both").sum()
    n_unmatched = (df_out["_clean_match"] == "left_only").sum()
    print(f"  Joined: {n_matched} master rows enriched from clean-master, "
          f"{n_unmatched} master rows unmatched")

    curated_glob = src_cfg.get("CURATED_FILE")

    def _norm_id(s):
        return (s.astype(str).str.strip().str.upper()
                .str.replace(r"\s+", "", regex=True).replace({"NAN": None}))

    if curated_glob:
        curated_path = _resolve(curated_glob, "CURATED_FILE")
        curated_sheet = src_cfg.get("CURATED_SHEET", "Sayfa1")
        df_cur = pd.read_excel(curated_path, sheet_name=curated_sheet)
        df_cur.columns = [str(c).strip() for c in df_cur.columns]
        print(f"  Curated source       : {os.path.basename(curated_path)} "
              f"({len(df_cur)} rows)")

        keep_ids = set(_norm_id(df_cur["Hasta No"]).dropna())
        keep_mask = _norm_id(df_out["Hasta No"]).isin(keep_ids)
        print(f"  Curated cohort filter: keeping {int(keep_mask.sum())} / "
              f"{len(df_out)} (dropping {int((~keep_mask).sum())} excluded patient(s))")

        if "Radyasyon pnömonisi (grade2-3-4)" in df_out.columns and \
                "RadyolojikRP" in df_out.columns:
            mp = df_out["Radyasyon pnömonisi (grade2-3-4)"].fillna(0) >= 1
            cp = df_out["RadyolojikRP"].fillna(-1) == 1
            dis = (mp != cp) & df_out["RadyolojikRP"].notna() & keep_mask
            if int(dis.sum()) > 0:
                print(f"  Warning: {int(dis.sum())} kept patient(s) still have a master/clean "
                      f"RP-label disagreement (kept per curated file - review):")
                cc = [c for c in ["Ad Soyad", "Hasta No",
                                  "Radyasyon pnömonisi (grade2-3-4)", "RadyolojikRP"]
                      if c in df_out.columns]
                print(df_out.loc[dis, cc].head(10).to_string(index=False))

        df_out = df_out[keep_mask].reset_index(drop=True)

        if "TNM Evre" in df_cur.columns:
            tnm_map = dict(zip(_norm_id(df_cur["Hasta No"]), df_cur["TNM Evre"]))
            new_tnm = _norm_id(df_out["Hasta No"]).map(tnm_map)
            df_out["TNMEvre"] = (new_tnm.fillna(df_out["TNMEvre"])
                                 if "TNMEvre" in df_out.columns else new_tnm)
            print(f"  TNM Evre: sourced from curated file for "
                  f"{int(new_tnm.notna().sum())} patient(s)")
    else:
        if "Radyasyon pnömonisi (grade2-3-4)" in df_out.columns and \
                "RadyolojikRP" in df_out.columns:
            master_pos = df_out["Radyasyon pnömonisi (grade2-3-4)"].fillna(0) >= 1
            clean_pos = df_out["RadyolojikRP"].fillna(-1) == 1
            both_known = df_out["RadyolojikRP"].notna()
            disagree = (master_pos != clean_pos) & both_known
            n_disagree = int(disagree.sum())
            if n_disagree > 0:
                print(f"  Dropping {n_disagree} ambiguous patient(s) "
                      f"(master grade>=2 vs clean-master RadyolojikRP disagree):")
                cols = ["Ad Soyad", "Hasta No",
                        "Radyasyon pnömonisi (grade2-3-4)", "RadyolojikRP"]
                cols = [c for c in cols if c in df_out.columns]
                print(df_out.loc[disagree, cols].head(10).to_string(index=False))
                df_out = df_out[~disagree].reset_index(drop=True)

    df_out = df_out.drop(columns=["__fp", "_clean_match"])

    def _norm_col(c):
        s = str(c).strip()
        return re.sub(r"[\s()%/\-]", "", s)

    df_out.columns = [_norm_col(c) for c in df_out.columns]

    seen, dup = {}, []
    for c in df_out.columns:
        if c in seen:
            dup.append(c)
        seen[c] = True
    if dup:
        print(f"  Warning: Normalization produced duplicate column names: {dup}")

    df_out = df_out.rename(columns=TR_TO_EN_COLS)
    untranslated = [c for c in df_out.columns
                    if not c.replace("_", "").isascii()]
    if untranslated:
        print(f"  Warning: {len(untranslated)} column(s) left in original Turkish "
              f"(not in TR_TO_EN_COLS): {untranslated}")

    def _clean_numeric_with_suffix(series, suffixes):
        s = series.astype(str).str.strip()
        for suf in suffixes:
            s = s.str.replace(suf, "", regex=False, case=False)
        s = s.str.strip().replace({"nan": None, "": None, "None": None})
        return pd.to_numeric(s, errors="coerce")

    if "smoking_packyear" in df_out.columns:
        df_out["smoking_packyear"] = _clean_numeric_with_suffix(
            df_out["smoking_packyear"], ["p.y", "py", "paket/yıl", "paket/yil"])
    for col in ("PFS_RP_pos", "OS_RP_pos", "PFS_RP_neg", "OS_RP_neg",
                "RP_onset_months"):
        if col in df_out.columns:
            df_out[col] = _clean_numeric_with_suffix(df_out[col], ["ay"])

    def _leading_digit(series):
        s = series.astype(str).str.strip()
        return pd.to_numeric(s.str.extract(r"^(\d+)", expand=False), errors="coerce")

    for col in ("T_stage", "N_stage"):
        if col in df_out.columns:
            df_out[col] = _leading_digit(df_out[col])

    if "chemo_cycles" in df_out.columns:
        df_out["chemo_cycles"] = _leading_digit(df_out["chemo_cycles"])

    if "RT_dose_total_frac" in df_out.columns:
        parts = df_out["RT_dose_total_frac"].astype(str).str.split("/", n=1, expand=True)
        df_out["RT_dose_total"] = pd.to_numeric(
            parts[0].str.strip().replace({"nan": None}), errors="coerce")
        df_out["RT_dose_frac"] = pd.to_numeric(
            parts[1].str.strip().replace({"nan": None}) if 1 in parts.columns else None,
            errors="coerce")

    if "RP_grade_234" in df_out.columns:
        df_out["rp_label_grade2"] = (df_out["RP_grade_234"].fillna(0) >= 1).astype(int)
        n_pos = int(df_out["rp_label_grade2"].sum())
        print(f"  Added 'rp_label_grade2' = (RP_grade_234 >= 1).astype(int)  "
              f"-> {n_pos}/{len(df_out)} positive")
    else:
        print("  Warning: Cannot add rp_label_grade2: 'RP_grade_234' not in columns")

    front = [c for c in ["patient_name", "patient_id", "rp_label_grade2"]
             if c in df_out.columns]
    rest = [c for c in df_out.columns if c not in front]
    df_out = df_out[front + rest]

    out_path = os.path.join(cfg["INPUTS_DIR"], out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_excel(out_path, index=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df_out.shape[0]} rows x {df_out.shape[1]} cols")

    if "rp_label_grade2" in df_out.columns:
        n = int(df_out["rp_label_grade2"].sum())
        print(f"  CTCAE grade >=2 (literature standard):  {n} / {len(df_out)}")
    if "RP_raw" in df_out.columns:
        n = int((df_out["RP_raw"].fillna(0) == 1).sum())
        print(f"  Raw RP flag (any RP):                  {n} / {len(df_out)}")
    if "RP_radiologic" in df_out.columns:
        n = int((df_out["RP_radiologic"].fillna(-1) == 1).sum())
        print(f"  RP_radiologic=1 (where known):         {n}")
    if "RP_clinical" in df_out.columns:
        n = int((df_out["RP_clinical"].fillna(-1) == 1).sum())
        print(f"  RP_clinical=1 (where known):           {n}")


def step_export_patient_tabular(cfg):
    import json as _json

    print("\n" + "=" * 60)
    print("  STEP 6 - Export Per-Patient Tabular JSON")
    print("=" * 60)

    input_dir = cfg["INPUTS_DIR"]
    processed_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])

    master_name = cfg.get("PREPROCESS", {}).get("MASTER_OUTPUT",
                                                "master_clinical_data.xlsx")
    master_path = os.path.join(input_dir, master_name)
    if not table_exists(master_path):
        print(f"Warning: {master_name} not found at {master_path} - run step 5 first.")
        return

    map_path = os.path.join(input_dir, "patient_name_to_uuid.csv")
    if not table_exists(map_path):
        print(f"Warning: patient_name_to_uuid.csv not found - run step 0 first.")
        return

    df_master = pd.read_excel(master_path)
    df_map = read_table(map_path, dtype=str).fillna("")

    clin_cols = cfg.get("TRAIN_CLINICAL", {}).get("DATA", {}).get("BASE_COLS", []) or []
    rt_cols = cfg.get("TRAIN_RT", {}).get("DATA", {}).get("BASE_COLS", []) or []
    target_col = cfg.get("TRAIN_CLINICAL", {}).get("DATA", {}).get(
        "TARGET_COL", "rp_label_grade2")

    clin_present = [c for c in clin_cols if c in df_master.columns]
    rt_present = [c for c in rt_cols if c in df_master.columns]
    print(f"Clinical features : {len(clin_present)}/{len(clin_cols)} cols in master")
    print(f"RT-Tabular features: {len(rt_present)}/{len(rt_cols)} cols in master")

    if "patient_name" not in df_master.columns:
        print("Warning: master_clinical_data.xlsx missing 'patient_name' column - aborting.")
        return
    name_to_row = {str(n).strip(): i for i, n in enumerate(df_master["patient_name"].values)}
    norm_name_to_row = {_norm_name(n): i
                        for i, n in enumerate(df_master["patient_name"].values)}

    def _hasta_no_stem(s):
        if pd.isna(s) or not str(s).strip():
            return None
        digits = "".join(c for c in str(s).strip() if c.isdigit())
        return int(digits) if digits else None

    pid_to_row: dict[int, int] = {}
    if "patient_id" in df_master.columns:
        _pid_int = pd.to_numeric(df_master["patient_id"], errors="coerce")
        for ridx, pid in enumerate(_pid_int.values):
            if pd.notna(pid):
                pid_to_row[int(pid)] = ridx

    def _to_jsonable(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    n_clin_written = 0
    n_rt_written = 0
    n_skipped_no_match = 0
    for _, mr in df_map.iterrows():
        uuid = str(mr["uuid"]).strip()
        name = str(mr["patient_name"]).strip()
        if not uuid or not name:
            continue
        hasta_stem = _hasta_no_stem(
            mr.get("dicom_patient_id", "") or mr.get("hasta_no", ""))
        row_idx = pid_to_row.get(hasta_stem) if hasta_stem is not None else None
        if row_idx is None:
            row_idx = name_to_row.get(name)
        if row_idx is None:
            row_idx = norm_name_to_row.get(_norm_name(name))
        if row_idx is None:
            n_skipped_no_match += 1
            continue
        row = df_master.iloc[row_idx]
        label = _to_jsonable(row[target_col]) if target_col in row.index else None

        clin_dir = os.path.join(processed_dir, uuid, "Clinical")
        os.makedirs(clin_dir, exist_ok=True)
        clin_data = {
            "uuid": uuid,
            "patient_name": name,
            "rp_label": label,
            "features": {c: _to_jsonable(row[c]) for c in clin_present},
        }
        with open(os.path.join(clin_dir, "clinical_features.json"), "w") as f:
            _json.dump(clin_data, f, indent=2, ensure_ascii=False)
        n_clin_written += 1

        rt_dir = os.path.join(processed_dir, uuid, "RT-Tabular")
        os.makedirs(rt_dir, exist_ok=True)
        rt_data = {
            "uuid": uuid,
            "patient_name": name,
            "rp_label": label,
            "features": {c: _to_jsonable(row[c]) for c in rt_present},
        }
        with open(os.path.join(rt_dir, "rt_tabular_features.json"), "w") as f:
            _json.dump(rt_data, f, indent=2, ensure_ascii=False)
        n_rt_written += 1

    print(f"\nWrote clinical_features.json     : {n_clin_written} patients")
    print(f"Wrote rt_tabular_features.json   : {n_rt_written} patients")
    if n_skipped_no_match:
        print(f"Skipped (no master Excel match): {n_skipped_no_match} patients")


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            dl = curr[j - 1] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dl, sub))
        prev = curr
    return prev[-1]


def step_generate_split(cfg):
    from sklearn.model_selection import StratifiedKFold

    print("\n" + "=" * 60)
    print("  STEP 7 - Generate Unified Split (Nested-CV)")
    print("=" * 60)
    input_dir = cfg["INPUTS_DIR"]
    split_cfg = cfg["SPLIT"]
    seed = split_cfg["SEED"]
    out_name = cfg.get("PREPROCESS", {}).get("MASTER_OUTPUT",
                                             "master_clinical_data.xlsx")

    df_map = read_table(os.path.join(input_dir, "patient_name_to_uuid.csv"))
    print(f"Folder mapping: {len(df_map)} patients")

    master_path = os.path.join(input_dir, out_name)
    if not table_exists(master_path):
        raise FileNotFoundError(
            f"Master Excel not found: {master_path}\n"
            "Run 'preprocess_pipeline.py --step 4' first to build it.")
    df_xls = pd.read_excel(master_path)
    print(f"Master Excel : {len(df_xls)} rows x {len(df_xls.columns)} cols")

    if "patient_id" not in df_xls.columns:
        raise KeyError("Master Excel missing 'patient_id' column - "
                       "rebuild via step 4.")
    if "rp_label_grade2" not in df_xls.columns:
        raise KeyError("Master Excel missing 'rp_label_grade2' column - "
                       "rebuild via step 4.")

    df_xls["_pid_int"] = pd.to_numeric(df_xls["patient_id"], errors="coerce")
    pid_to_row: dict[int, int] = {}
    for ridx, pid in enumerate(df_xls["_pid_int"].values):
        if pd.notna(pid):
            pid_to_row[int(pid)] = ridx

    _parent_dir = os.path.dirname(os.path.abspath(input_dir))
    _id_alias_path = os.path.join(_parent_dir, "Files", "patient_id_aliases.csv")
    _id_aliases: dict[str, int] = {}
    if os.path.isfile(_id_alias_path):
        try:
            with open(_id_alias_path, encoding="utf-8") as af:
                for ln in af:
                    s = ln.strip()
                    if not s or s.startswith("#") or s.startswith("manifest_hasta_no"):
                        continue
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) >= 2 and parts[0] and parts[1]:
                        try:
                            _id_aliases[parts[0]] = int(parts[1])
                        except ValueError:
                            print(f"  Warning: id_aliases: non-integer master_patient_id "
                                  f"{parts[1]!r} for {parts[0]!r}, skipping")
            print(f"ID alias map ({_id_alias_path}): {len(_id_aliases)} entries loaded.")
        except Exception as _e:
            print(f"[!] Failed to read patient_id_aliases.csv: {_e}")
    else:
        print(f"(No patient_id_aliases.csv at {_id_alias_path} - using "
              f"dicom_patient_id + hasta_no stem + name only.)")

    def _hasta_no_stem(s) -> int | None:
        if pd.isna(s) or not str(s).strip():
            return None
        digits = "".join(c for c in str(s).strip() if c.isdigit())
        try:
            return int(digits) if digits else None
        except ValueError:
            return None

    name_to_row: dict[str, int] = {}
    if "patient_name" in df_xls.columns:
        for ridx, pname in enumerate(df_xls["patient_name"].values):
            if pd.notna(pname):
                name_to_row[_norm_name(str(pname))] = ridx

    proc_dir = os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"]["PROCESSED"])
    _modalities = _get_modalities(cfg)
    ct_subdir = _modalities.get("ct_plan", {}).get("dest_subdir", "CT-Plan")
    req_file = "whole_body.npy"
    if os.path.isdir(proc_dir):
        n_uuid = sum(1 for d in os.listdir(proc_dir)
                     if os.path.isdir(os.path.join(proc_dir, d)))
        print(f"Processed dir : {n_uuid} UUID dirs on disk")
    else:
        print(f"Warning: processed dir not found ({proc_dir}) - "
              f"has_valid_ct=False for all")

    study_layout = bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))

    def _has_valid_ct(uuid: str) -> bool:
        root = os.path.join(proc_dir, str(uuid).strip())
        if study_layout:
            meta = os.path.join(root, "patient_meta.json")
            if os.path.exists(meta):
                try:
                    m = json.load(open(meta))
                    if not m.get("excluded") and int(m.get("n_studies", 0)) > 0:
                        return True
                except Exception:
                    pass
            return bool(glob.glob(os.path.join(root, "studies", "[0-9]*",
                                               "CT", "whole_body.npy")))
        return os.path.exists(os.path.join(root, ct_subdir, req_file))

    clin_cols = cfg.get("TRAIN_CLINICAL", {}).get("DATA", {}).get("BASE_COLS", [])
    rt_cols = cfg.get("TRAIN_RT", {}).get("DATA", {}).get("BASE_COLS", [])
    clin_present = [c for c in clin_cols if c in df_xls.columns]
    rt_present = [c for c in rt_cols if c in df_xls.columns]
    if len(clin_present) < len(clin_cols):
        missing = set(clin_cols) - set(clin_present)
        print(f"Warning: Master Excel missing clinical cols {sorted(missing)} - "
              f"has_clinical computed from {len(clin_present)} of {len(clin_cols)} cols.")
    if len(rt_present) < len(rt_cols):
        missing = set(rt_cols) - set(rt_present)
        print(f"Warning: Master Excel missing RT cols {sorted(missing)} - "
              f"has_rt_tabular computed from {len(rt_present)} of {len(rt_cols)} cols.")

    def _has_tabular(row_idx: int, cols: list[str]) -> bool:
        if row_idx < 0 or row_idx >= len(df_xls):
            return False
        if not cols:
            return False
        row = df_xls.iloc[row_idx]
        return bool(all(pd.notna(row.get(c)) for c in cols))

    records = []
    n_pid_match = 0
    n_hasta_match = 0
    n_alias_match = 0
    n_name_match = 0
    n_id_rejected = 0
    n_unmatched = 0

    for _, mr in df_map.iterrows():
        uuid = str(mr["uuid"]).strip()
        dicom_pid_raw = mr.get("dicom_patient_id")
        try:
            dicom_pid = int(float(dicom_pid_raw)) if pd.notna(dicom_pid_raw) else None
        except (TypeError, ValueError):
            dicom_pid = None

        folder_name_norm = _norm_name(str(mr.get("patient_name", "")))
        hasta_no_raw = mr.get("hasta_no", "") or mr.get("dicom_patient_id", "")
        hasta_stem = _hasta_no_stem(hasta_no_raw)

        master_row = -1

        _key_str = str(hasta_no_raw).strip() if hasta_no_raw else ""
        _key_stem = str(hasta_stem) if hasta_stem is not None else ""
        if _key_str and _key_str in _id_aliases:
            target_pid = _id_aliases[_key_str]
            if target_pid in pid_to_row:
                master_row = pid_to_row[target_pid]
                n_alias_match += 1
        elif _key_stem and _key_stem in _id_aliases:
            target_pid = _id_aliases[_key_stem]
            if target_pid in pid_to_row:
                master_row = pid_to_row[target_pid]
                n_alias_match += 1

        if master_row < 0 and dicom_pid is not None and dicom_pid in pid_to_row:
            cand_row = pid_to_row[dicom_pid]
            cand_name_norm = _norm_name(str(df_xls.iloc[cand_row].get("patient_name", "")))
            f_first = folder_name_norm.split()[0] if folder_name_norm else ""
            m_first = cand_name_norm.split()[0] if cand_name_norm else ""
            f_dist = _edit_distance(f_first, m_first) if f_first and m_first else 99
            if f_first and f_dist <= 2:
                master_row = cand_row
                n_pid_match += 1
            else:
                print(f"  Warning: DICOM id {dicom_pid} for folder "
                      f"{mr.get('patient_name', '')!r} -> master "
                      f"{df_xls.iloc[cand_row].get('patient_name', '')!r}: "
                      f"first-name distance={f_dist}, trying hasta_no/name fallback")
                n_id_rejected += 1

        if master_row < 0 and hasta_stem is not None and hasta_stem in pid_to_row:
            cand_row = pid_to_row[hasta_stem]
            cand_name_norm = _norm_name(str(df_xls.iloc[cand_row].get("patient_name", "")))
            f_first = folder_name_norm.split()[0] if folder_name_norm else ""
            m_first = cand_name_norm.split()[0] if cand_name_norm else ""
            f_dist = _edit_distance(f_first, m_first) if f_first and m_first else 99
            if not f_first or f_dist <= 3:
                master_row = cand_row
                n_hasta_match += 1

        if master_row < 0:
            if folder_name_norm and folder_name_norm in name_to_row:
                master_row = name_to_row[folder_name_norm]
                n_name_match += 1
            else:
                n_unmatched += 1

        if master_row >= 0:
            rp_label = int(df_xls.iloc[master_row]["rp_label_grade2"])
        else:
            rp_label = 0

        records.append({
            "patient_name": mr["patient_name"],
            "dicom_patient_name": mr.get("dicom_patient_name", ""),
            "dicom_patient_id": dicom_pid if dicom_pid is not None else "",
            "uuid": uuid,
            "rp_status": "RP" if rp_label == 1 else "RP Not Present",
            "rp_label": rp_label,
            "has_before_rt": mr["has_before_rt"],
            "has_after_rt": mr["has_after_rt"],
            "has_valid_ct": _has_valid_ct(uuid),
            "has_clinical": _has_tabular(master_row, clin_present),
            "has_rt_tabular": _has_tabular(master_row, rt_present),
            "excel_row_idx": master_row,
        })

    print(f"Linkage : {n_alias_match} by id-alias, "
          f"{n_pid_match} by dicom_pid+name, "
          f"{n_hasta_match} by hasta_no stem, "
          f"{n_name_match} by name only, "
          f"{n_id_rejected} id rejected (name mismatch), "
          f"{n_unmatched} unmatched")

    df_u = pd.DataFrame(records)

    n_before = len(df_u)
    df_u = df_u[df_u["excel_row_idx"] >= 0].reset_index(drop=True)
    if len(df_u) < n_before:
        print(f"Dropped {n_before - len(df_u)} folder patient(s) with no master "
              f"link (master excludes them - e.g. ambiguous-label rows from step 4).")

    dup_mask = df_u["excel_row_idx"].duplicated(keep=False)
    if dup_mask.any():
        dups = df_u[dup_mask].sort_values("excel_row_idx")
        print(f"Warning: {dup_mask.sum()} folder patients share an excel_row_idx - "
              f"dropping all conflicting rows:")
        print(dups[["patient_name", "dicom_patient_id", "excel_row_idx"]]
              .to_string(index=False))
        df_u = df_u[~dup_mask].reset_index(drop=True)

    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    if table_exists(manifest_path):
        df_man = read_table(manifest_path)
        df_man["uuid"] = df_man["uuid"].astype(str).str.strip()
        man_has_cols = [c for c in df_man.columns if c.startswith("has_")]
        if man_has_cols:
            df_u["uuid"] = df_u["uuid"].astype(str).str.strip()
            df_u = df_u.merge(df_man[["uuid"] + man_has_cols],
                              on="uuid", how="left", suffixes=("", "_manifest"))
            for c in man_has_cols:
                if c in df_u.columns:
                    df_u[c] = df_u[c].fillna(False).astype(bool)

    _, _seg_status = _load_seg_status(input_dir)
    _failed_pairs: dict[str, set[str]] = {}
    for _key, _rec in _seg_status.items():
        if _rec.get("status") == "complete":
            continue
        _uuid_part, _, _mod_part = _key.partition("__")
        if _uuid_part and _mod_part:
            _failed_pairs.setdefault(_uuid_part, set()).add(_mod_part)
    if _failed_pairs:
        _n_demoted = 0
        for _uid, _failed_mods in _failed_pairs.items():
            mask = df_u["uuid"] == _uid
            if not mask.any():
                continue
            for _mod in _failed_mods:
                _col = f"has_{_mod}"
                if _col in df_u.columns:
                    _prev = bool(df_u.loc[mask, _col].iloc[0])
                    if _prev:
                        df_u.loc[mask, _col] = False
                        _n_demoted += 1
        if _n_demoted:
            print(f"Step-2 status overlay: demoted {_n_demoted} has_* flag(s) "
                  f"to False for patients whose modality processing failed.")

    study_layout = bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))
    if study_layout:
        def _study_flags(uuid):
            mp = os.path.join(proc_dir, str(uuid).strip(), "patient_meta.json")
            if not os.path.exists(mp):
                return 0, False, False
            try:
                m = json.load(open(mp))
            except Exception:
                return 0, False, False
            cts = [s for s in m.get("sequence", []) if s.get("modality") != "pet"]
            n = 0 if m.get("excluded") else int(m.get("n_studies", 0))
            if m.get("excluded") == "multi_study":
                n = int(m.get("n_studies", 0))
            has_pet = bool(m.get("has_pet"))
            has_dvh = any(s.get("lung_MLD_cGy") is not None for s in cts)
            return n, has_pet, has_dvh
        _fl = df_u["uuid"].map(_study_flags)
        df_u["n_meaningful_studies"] = _fl.map(lambda x: x[0]).astype(int)
        df_u["has_pet_reg"] = _fl.map(lambda x: x[1])
        df_u["has_dvh"] = _fl.map(lambda x: x[2])

    required_imaging = (split_cfg.get("REQUIRED_IMAGING")
                        or ["ct_plan", "contours", "dose", "pet_plan"])
    req = [m for m in required_imaging if f"has_{m}" in df_u.columns]

    _min_ct = int(cfg.get("PREPROCESS", {}).get("MIN_MEANINGFUL_CT", 1) or 0)
    _max_ct = int(cfg.get("PREPROCESS", {}).get("MAX_MEANINGFUL_CT", 1) or 0)
    _excl_uuids = {str(u).strip() for u in
                   (cfg.get("PREPROCESS", {}).get("EXCLUDE_UUIDS") or [])}

    def _missing_imaging(r):
        if str(r.get("uuid", "")).strip() in _excl_uuids:
            return ["major_replan(manual)"]
        out = [m for m in req if not bool(r.get(f"has_{m}", False))]
        if study_layout:
            if "has_pet_plan" in df_u.columns and bool(r.get("has_pet_plan")) \
                    and not bool(r.get("has_pet_reg", False)):
                out.append("pet_unregistered")
            if not bool(r.get("has_dvh", False)):
                out.append("dvh")
            n = int(r.get("n_meaningful_studies", 0) or 0)
            if n < _min_ct:
                out.append("no_built_study" if n == 0 else f"too_few_study(n={n})")
            elif _max_ct and n > _max_ct:
                out.append(f"multi_study(n={n})")
        if not bool(r.get("has_clinical", False)):
            out.append("clinical")
        return out

    _miss = df_u.apply(_missing_imaging, axis=1)
    df_u["imaging_valid"] = _miss.map(lambda x: len(x) == 0)
    df_u["exclude_reason"] = _miss.map(
        lambda x: "" if not x else "missing: " + ",".join(x))
    n_valid = int(df_u["imaging_valid"].sum())
    print(f"\nCohort validation (require {req} + PET-registered + DVH + clinical "
          f"+ exactly 1 reliable study):")
    print(f"  valid cohort: {n_valid} / {len(df_u)}  "
          f"(excluded {len(df_u) - n_valid})")
    if study_layout and "n_meaningful_studies" in df_u.columns:
        n_multi = int((df_u["n_meaningful_studies"] > 1).sum())
        n_single = int((df_u["n_meaningful_studies"] == 1).sum())
        print(f"  studies/patient: single={n_single}  multi(excluded)={n_multi}")
    from collections import Counter as _Counter
    for reason, c in _Counter(
            df_u.loc[~df_u["imaging_valid"], "exclude_reason"]).most_common():
        print(f"    {reason}: {c}")
    _excl = df_u.loc[~df_u["imaging_valid"],
                     ["uuid", "patient_name", "rp_status", "exclude_reason"]]
    if len(_excl):
        _ep = os.path.join(input_dir, "excluded_patients.csv")
        _excl.to_csv(_ep, index=False)
        _to_excel(_excl, _ep.replace(".csv", ".xlsx"))
        print(f"  excluded list -> {_ep.replace('.csv', '.xlsx')}")
    df_u = df_u[df_u["imaging_valid"]].reset_index(drop=True)

    complete_case = split_cfg.get("COMPLETE_CASE_MODALITIES", []) or []
    if complete_case:
        available = sorted(c[4:] for c in df_u.columns if c.startswith("has_"))
        missing = [mk for mk in complete_case if f"has_{mk}" not in df_u.columns]
        if missing:
            print(f"Warning: Unknown modality keys in COMPLETE_CASE_MODALITIES: {missing}")
            print(f"   Available keys: {available}")
            print(f"   Filter skipped.")
        else:
            n_before = len(df_u)
            mask = pd.Series(True, index=df_u.index)
            for mk in complete_case:
                mask &= df_u[f"has_{mk}"].astype(bool)
            df_u = df_u[mask].reset_index(drop=True)
            print(f"Complete-case filter {complete_case}: "
                  f"{n_before} -> {len(df_u)} patients")
            if len(df_u) == 0:
                print("Warning: Complete-case cohort is empty - no patient has all "
                      "requested modalities. unified_split.csv will be empty.")

    n_outer = int(split_cfg.get("N_OUTER_FOLDS", 5))
    if len(df_u) < n_outer:
        print(f"Warning: Only {len(df_u)} patients - fewer than N_OUTER_FOLDS={n_outer}. "
              f"OuterFold assignment skipped (all set to -1).")
        df_u["OuterFold"] = -1
    else:
        y = df_u["rp_label"].values
        idx_all = np.arange(len(df_u))
        df_u["OuterFold"] = -1
        skf_outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for fi, (_, test_rel) in enumerate(skf_outer.split(idx_all, y)):
            df_u.loc[idx_all[test_rel], "OuterFold"] = fi
        print(f"OuterFold: {n_outer} stratified folds; "
              f"sizes = {df_u.groupby('OuterFold').size().to_list()}")

    unified_path = os.path.join(input_dir, split_cfg["UNIFIED_FILE"])
    df_u.to_csv(unified_path, index=False)
    _to_excel(df_u, unified_path.replace(".csv", ".xlsx"))
    print(f"Unified split: {unified_path.replace('.csv', '.xlsx')}")

    df_ct = df_u[df_u["has_valid_ct"]][["uuid", "rp_status", "OuterFold"]].copy()
    df_ct.columns = ["UUID", "Group", "OuterFold"]
    ct_path = os.path.join(input_dir, "clean_cohort_manifest_cv.csv")
    df_ct.to_csv(ct_path, index=False)
    _to_excel(df_ct, ct_path.replace(".csv", ".xlsx"))
    print(f"CT manifest:   {ct_path.replace('.csv', '.xlsx')}")

    print(f"\nTotal: {len(df_u)}"
          f"  |  Valid CT: {int(df_u['has_valid_ct'].sum())}"
          f"  |  RP+: {int((df_u['rp_label'] == 1).sum())}"
          f"  |  RP-: {int((df_u['rp_label'] == 0).sum())}")
    print(df_u.groupby(["OuterFold", "rp_status"]).size().to_string())

    _modalities_cfg = (cfg.get("PREPROCESS", {}).get("MODALITIES", {}) or {})
    _disabled_imaging = {mk for mk, mcfg in _modalities_cfg.items()
                         if mcfg.get("enabled", True) is False}
    print(f"\nModality availability in unified_split ({len(df_u)} patients):")
    has_cols = [c for c in df_u.columns if c.startswith("has_")]
    _all_zero = {c for c in has_cols
                 if int(df_u[c].astype(bool).sum()) == 0}
    active_cols = [c for c in has_cols
                   if c[len("has_"):] not in _disabled_imaging
                   and c not in _all_zero]
    for c in sorted(has_cols):
        n = int(df_u[c].astype(bool).sum())
        tags = []
        if c[len("has_"):] in _disabled_imaging:
            tags.append("disabled")
        if c in _all_zero:
            tags.append("all-zero, excluded from intersection")
        tag = f"  [{', '.join(tags)}]" if tags else ""
        print(f"    {c}: {n}{tag}")
    if active_cols:
        all_mask = df_u[active_cols].astype(bool).all(axis=1)
        print(f"    {'ALL modalities':<20s}: {int(all_mask.sum())} "
              f"(intersection over {len(active_cols)} active flags: "
              f"{[c[len('has_'):] for c in active_cols]})")


_STATUS_FILENAME = "preprocess_segment_status.csv"
_STATUS_COLS = ["uuid", "group", "modality", "status", "error", "timestamp"]


def _load_seg_status(inputs_dir: str) -> tuple[str, dict]:
    path = os.path.join(inputs_dir, _STATUS_FILENAME)
    if not os.path.exists(path):
        return path, {}
    df = read_table(path, dtype=str).fillna("")
    if "modality" not in df.columns:
        df["modality"] = "ct_x"
    return path, {
        f"{row['uuid']}__{row['modality']}": row.to_dict()
        for _, row in df.iterrows()
    }


def _mark_seg_status(records: dict, path: str,
                     uuid: str, group: str, modality: str,
                     status: str, error: str = "") -> None:
    key = f"{uuid}__{modality}"
    records[key] = {
        "uuid": uuid,
        "group": group,
        "modality": modality,
        "status": status,
        "error": error,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    rows = list(records.values())
    _df_s = pd.DataFrame(rows, columns=_STATUS_COLS)
    _df_s.to_csv(path, index=False)
    _to_excel(_df_s, path.replace(".csv", ".xlsx"), delete_csv=False)


def _get_modalities(cfg: dict) -> dict:
    return cfg.get("PREPROCESS", {}).get("MODALITIES", {}) or {}


def _filter_manifest_only(records: list, cfg: dict) -> list:
    only = cfg.get("PREPROCESS", {}).get("ONLY_PATIENTS", []) or []
    if not only:
        return records
    wanted_names = {_norm_name(str(x)) for x in only}
    wanted_ids = {str(x).strip().lower() for x in only}
    out = []
    for r in records:
        nm = _norm_name(str(r.get("patient_name", "")))
        hn = str(r.get("hasta_no", "")).strip().lower()
        if nm in wanted_names or hn in wanted_ids:
            out.append(r)
    print(f"  [ONLY_PATIENTS={only}] restricted to {len(out)}/{len(records)} "
          f"patient(s).")
    if not out:
        print("  [!] ONLY_PATIENTS matched no manifest rows - check the "
              "name/hasta_no spelling.")
    return out


def _resolve_source(template: str, row: dict, raw_dir: str) -> str:
    rel = template.replace("{hasta_no}", str(row.get("hasta_no", "")))
    return os.path.join(raw_dir, rel)


def _run_per_patient(manifest: list, fn, workers: int, desc: str = "") -> None:
    rows = list(manifest)

    def _safe(row):
        try:
            return fn(row)
        except Exception as e:
            print(f"  [!] {desc} failed for "
                  f"{str(row.get('uuid', '')).strip()}: {e}")
            return None

    if workers and workers > 1 and len(rows) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_safe, rows))
    else:
        for row in rows:
            _safe(row)


def _parse_contournames(path: str) -> dict:
    idx_to_name: dict[int, str] = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = [l.rstrip("\n") for l in f]
        if len(lines) < 2:
            return idx_to_name
        try:
            n_structs = int(lines[1].strip())
        except (ValueError, IndexError):
            n_structs = 50

        i = 2
        while i < len(lines):
            name_line = lines[i].strip()
            if not name_line or name_line[0].isdigit() or name_line[0] in (",", "-"):
                i += 1
                continue
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                meta = lines[j].strip()
                parts = meta.split(",")
                try:
                    struct_idx = int(parts[0])
                    if struct_idx > 0:
                        idx_to_name[struct_idx] = name_line.lower()
                        i = j + 1
                        continue
                except (ValueError, IndexError):
                    pass
            i += 1
    except Exception:
        pass
    return idx_to_name


# Structure names that contain "ptv" or "boost" but are not treatment targets:
# lung-derived structures and planning helpers (expanded body, margins, physics).
_PTV_EXCLUDE = ("lun", "akc", "+6", "2cm", "fizik")
# A single target structure larger than this is a body outline, not a PTV.
PTV_MAX_CC = 2000.0


def _is_ptv_name(nl: str) -> bool:
    """True for a folded structure name that denotes a treatment target."""
    return (("ptv" in nl) or ("boost" in nl)) and not any(t in nl for t in _PTV_EXCLUDE)


def _parse_wc_file(wc_path: str) -> dict:
    result = {"z_mm": None, "origin": None, "orientation": None, "contours": []}
    try:
        with open(wc_path, encoding="utf-8", errors="ignore") as f:
            lines = [l.rstrip("\n") for l in f]
        if len(lines) < 8:
            return result

        origin_parts = [float(v.strip()) for v in lines[5].split(",")]
        if len(origin_parts) >= 3:
            result["origin"] = tuple(origin_parts[:3])
            result["z_mm"] = origin_parts[2]

        # Some exports write z = 0 in every slice header; the slice position is
        # then only in the file name (T.<z>.WC).
        if result["z_mm"] == 0.0:
            try:
                z_name = float(os.path.basename(wc_path)[2:-3])
                result["z_mm"] = z_name
                result["origin"] = (result["origin"][0], result["origin"][1], z_name)
            except ValueError:
                pass

        orient_parts = [float(v.strip()) for v in lines[6].split(",")]
        if len(orient_parts) >= 6:
            result["orientation"] = orient_parts[:6]

        i = 7
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                n_points = int(line)
            except ValueError:
                i += 1
                continue
            i += 1
            if i >= len(lines):
                break
            try:
                struct_idx = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            i += 1
            xy_pairs = []
            remaining = n_points
            while remaining > 0 and i < len(lines):
                coord_line = lines[i].strip()
                if not coord_line:
                    i += 1
                    continue
                coords = [float(v.strip()) for v in coord_line.split(",")]
                for k in range(0, len(coords) - 1, 2):
                    xy_pairs.append((coords[k], coords[k + 1]))
                    remaining -= 1
                    if remaining <= 0:
                        break
                i += 1
            if xy_pairs:
                result["contours"].append((struct_idx, xy_pairs))
    except Exception:
        pass
    return result


def _fill_polygon_2d(points_xy, grid_shape, origin_xy, spacing_xy):
    if len(points_xy) < 3:
        return np.zeros(grid_shape, dtype=bool)
    try:
        from matplotlib.path import Path
        px = [(p[0] - origin_xy[0]) / spacing_xy[0] for p in points_xy]
        py = [(p[1] - origin_xy[1]) / spacing_xy[1] for p in points_xy]
        ny_g, nx_g = grid_shape
        yy, xx = np.mgrid[0:ny_g, 0:nx_g]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        path = Path(list(zip(px, py)))
        mask = path.contains_points(pts).reshape(grid_shape)
        return mask
    except ImportError:
        ny_g, nx_g = grid_shape
        mask = np.zeros(grid_shape, dtype=bool)
        px = np.array([(p[0] - origin_xy[0]) / spacing_xy[0] for p in points_xy])
        py = np.array([(p[1] - origin_xy[1]) / spacing_xy[1] for p in points_xy])
        px = np.append(px, px[0])
        py = np.append(py, py[0])
        ymin = max(0, int(np.floor(py.min())))
        ymax = min(ny_g - 1, int(np.ceil(py.max())))
        for y in range(ymin, ymax + 1):
            xs = []
            n = len(px) - 1
            for k in range(n):
                y0_, y1_ = py[k], py[k + 1]
                if (y0_ <= y < y1_) or (y1_ <= y < y0_):
                    t = (y - y0_) / (y1_ - y0_)
                    xs.append(px[k] + t * (px[k + 1] - px[k]))
            xs.sort()
            for m in range(0, len(xs) - 1, 2):
                x0i = max(0, int(np.ceil(xs[m])))
                x1i = min(nx_g - 1, int(np.floor(xs[m + 1])))
                if x0i <= x1i:
                    mask[y, x0i:x1i + 1] = True
        return mask


def _rasterize_contours(wc_dir: str, contournames_path: str,
                        info_path: str, ct_grid=None,
                        lung_outlier_threshold_cc: float | None = 10000.0,
                        drop_lung_outliers: bool = False) -> dict:
    fallback_shape = (1, 1, 1)
    _ALL_GROUPS = ("gtv", "ctv", "lung", "ptv",
                   "heart", "esophagus", "spinal_cord", "liver", "breast")

    def _zeros(shape):
        out = {g: np.zeros(shape, dtype=bool) for g in _ALL_GROUPS}
        out["_per_struct_cc"] = {}
        out["_lung_outlier_idxs"] = set()
        return out

    if not os.path.isfile(contournames_path):
        return _zeros(fallback_shape)
    idx_to_name = _parse_contournames(contournames_path)

    wc_files = sorted(glob.glob(os.path.join(wc_dir, "*.WC")))
    if not wc_files:
        return _zeros(fallback_shape)

    slices = []
    for wf in wc_files:
        parsed = _parse_wc_file(wf)
        if parsed["z_mm"] is not None:
            slices.append(parsed)
    if not slices:
        return _zeros(fallback_shape)

    slices.sort(key=lambda s: s["z_mm"])

    if ct_grid is not None:
        origin, spacing, shape = ct_grid
        x_origin, y_origin, z_origin_ct = origin
        x_spacing, y_spacing, z_spacing = spacing[2], spacing[1], spacing[0]
        nz = shape[0]
        ny, nx = shape[1], shape[2]
        grid_shape_3d = (nz, ny, nx)

        def _slice_to_zi(s):
            zi = int(round((s["z_mm"] - z_origin_ct) / z_spacing))
            return max(0, min(nz - 1, zi))

    else:
        z_vals = [s["z_mm"] for s in slices]
        nz = len(z_vals)

        if nz > 1:
            diffs = [abs(z_vals[k + 1] - z_vals[k]) for k in range(nz - 1)]
            z_spacing = float(np.median(diffs))
        else:
            z_spacing = 2.5

        x_origin, y_origin = None, None
        x_spacing, y_spacing = None, None

        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8", errors="ignore") as f:
                    info_lines = [l.strip() for l in f]
                bbox_parts = [float(v.strip()) for v in info_lines[3].split(",")]
                if len(bbox_parts) >= 6:
                    xmin, ymin, zmin, xmax, ymax, zmax = bbox_parts[:6]
                    x_extent = xmax - xmin
                    y_extent = ymax - ymin
            except Exception:
                pass

        ref_slice = slices[0]
        if ref_slice["origin"] is not None:
            x_origin = ref_slice["origin"][0]
            y_origin = ref_slice["origin"][1]

        all_x = []
        all_y = []
        for s in slices[:5]:
            for _, pts in s["contours"]:
                for x, y in pts:
                    all_x.append(x)
                    all_y.append(y)

        if all_x and x_origin is not None:
            coord_x_max = max(all_x)
            coord_x_min = min(all_x)
            coord_y_max = max(all_y)
            coord_y_min = min(all_y)
            try:
                x_extent
                y_extent
            except NameError:
                x_extent = coord_x_max - coord_x_min + 20
                y_extent = coord_y_max - coord_y_min + 20
            x_spacing = 1.0
            y_spacing = 1.0
            nx = max(64, int(round(x_extent / x_spacing)) + 1)
            ny = max(64, int(round(y_extent / y_spacing)) + 1)
        else:
            if x_origin is None:
                x_origin = -250.0
                y_origin = -250.0
            x_spacing = 1.0
            y_spacing = 1.0
            nx = 512
            ny = 512

        grid_shape_3d = (nz, ny, nx)

        def _slice_to_zi(s):
            return slices.index(s)

    def _ascii_fold(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s)
        nfkd = nfkd.replace("ı", "i").replace("İ", "i")
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    _tm_re = re.compile(r"\btm\b")
    _ac_re = re.compile(r"\bac\b")
    _sc_re = re.compile(r"\bsc\b")
    _sc_end_re = re.compile(r"sc$")
    _kit_re = re.compile(r"\bkit")
    _ms_dot_re = re.compile(r"\bm\.s\b")

    _gtv_kws = ("gtv", "tumor", "tumör", "target vol",
                "kitle", "lezyon", "nodul")
    _eso_kws = ("ozefag", "ozafag", "osefag", "esofag", "esophag", "ozfag")
    _liver_kws = ("karaciger", "karaci", "katraci", "liver", "hepar")
    _breast_kws = ("meme", "breast")

    def _classify(name: str) -> str:
        nl = _ascii_fold(name)
        nl_s = nl.strip()
        if _is_ptv_name(nl):
            return "ptv"
        if ("ptv" in nl) or ("boost" in nl):
            return ""
        if "lun" in nl:
            return "lung"
        if "ctv" in nl:
            return "ctv"
        if any(kw in nl for kw in ("gtv", "tumor", "tumör", "target vol")):
            return "gtv"
        if (any(kw in nl for kw in ("kitle", "lezyon", "nodul"))
                or _tm_re.search(nl) or _kit_re.search(nl)):
            return "gtv"
        if ("akc" in nl) or _ac_re.search(nl):
            return "lung"
        if ("kalp" in nl) or ("heart" in nl):
            return "heart"
        if any(kw in nl for kw in _eso_kws) or nl_s.startswith("oze") or nl_s.startswith("oz"):
            return "esophagus"
        if (("ms" in nl) or ("medulla" in nl) or ("spi" in nl)
                or _sc_re.search(nl) or _sc_end_re.search(nl_s)
                or _ms_dot_re.search(nl)):
            return "spinal_cord"
        if any(kw in nl for kw in _liver_kws):
            return "liver"
        if any(kw in nl for kw in _breast_kws):
            return "breast"
        return ""

    _groups = ("gtv", "ctv", "lung", "ptv",
               "heart", "esophagus", "spinal_cord", "liver", "breast")
    group_indices: dict[str, set[int]] = {g: set() for g in _groups}
    for idx, name in idx_to_name.items():
        cls = _classify(name)
        if cls in group_indices:
            group_indices[cls].add(idx)

    gtv_indices = group_indices["gtv"]
    lung_indices = group_indices["lung"]
    ptv_indices = group_indices["ptv"]

    group_vols: dict[str, np.ndarray] = {
        g: np.zeros(grid_shape_3d, dtype=bool) for g in _groups}
    gtv_vol = group_vols["gtv"]
    lung_vol = group_vols["lung"]
    ptv_vol = group_vols["ptv"]

    origin_xy = (x_origin, y_origin)
    spacing_xy = (x_spacing, y_spacing)
    slice_shape_2d = (ny, nx)

    _idx_to_group: dict[int, str] = {}
    for g, idxs in group_indices.items():
        for i in idxs:
            _idx_to_group[i] = g

    per_struct_vols: dict[int, np.ndarray] = {}
    use_ras_flip = ct_grid is not None
    for sl in slices:
        zi = _slice_to_zi(sl) if use_ras_flip else slices.index(sl)
        for struct_idx, pts in sl["contours"]:
            if not pts or struct_idx not in _idx_to_group:
                continue
            if use_ras_flip:
                pts_xy = [(p[0], -p[1]) for p in pts]
                fill = _fill_polygon_2d(pts_xy, slice_shape_2d,
                                        (x_origin, y_origin), spacing_xy)
            else:
                fill = _fill_polygon_2d(pts, slice_shape_2d, origin_xy,
                                        spacing_xy)
            if struct_idx not in per_struct_vols:
                per_struct_vols[struct_idx] = np.zeros(grid_shape_3d, dtype=bool)
            per_struct_vols[struct_idx][zi] |= fill

    vox_mm3 = 1.0
    if ct_grid is not None:
        vox_mm3 = (ct_grid[1][0] * ct_grid[1][1] * ct_grid[1][2])
    per_struct_cc: dict[int, tuple] = {}
    lung_outlier_idxs: set[int] = set()
    for struct_idx, vol in per_struct_vols.items():
        n = int(vol.sum())
        cc = n * vox_mm3 / 1000.0
        per_struct_cc[struct_idx] = (
            _idx_to_group[struct_idx],
            idx_to_name.get(struct_idx, "?"), cc)
        if (_idx_to_group[struct_idx] == "lung"
                and lung_outlier_threshold_cc is not None
                and cc > lung_outlier_threshold_cc):
            lung_outlier_idxs.add(struct_idx)

    for struct_idx, vol in per_struct_vols.items():
        g = _idx_to_group[struct_idx]
        if g == "lung" and drop_lung_outliers and struct_idx in lung_outlier_idxs:
            continue
        if g == "ptv" and per_struct_cc[struct_idx][2] >= PTV_MAX_CC:
            continue
        group_vols[g] |= vol

    group_vols["_per_struct_cc"] = per_struct_cc
    group_vols["_lung_outlier_idxs"] = lung_outlier_idxs
    return group_vols


def _parse_dose_volume(dose_path: str):
    with open(dose_path, "rb") as f:
        raw = f.read()

    lines = []
    pos = 0
    for _ in range(50):
        nl = raw.find(b"\n", pos)
        if nl == -1:
            break
        line_bytes = raw[pos:nl]
        try:
            lines.append(line_bytes.decode("ascii").strip())
            pos = nl + 1
        except UnicodeDecodeError:
            break

    parts = lines[8].split(",")
    nx, ny, nz = int(parts[-3]), int(parts[-2]), int(parts[-1])
    x_ext = float(parts[1])
    y_ext = float(parts[2])
    z_ext = float(parts[3])
    x_start = float(parts[4])
    y_start = float(parts[5])
    z_start = float(parts[6])
    spacing = (x_ext / nx, y_ext / ny, z_ext / nz)
    origin = (x_start, y_start, z_start)

    # Header line 4 holds the unit scale and, as its second field, a dose
    # multiplier that must be applied as well.
    dose_scale = 1e-5
    dose_mult = 1.0
    if len(lines) > 4 and lines[4]:
        fields = lines[4].split(",")
        try:
            dose_scale = float(fields[0])
        except (ValueError, IndexError):
            pass
        if len(fields) > 1:
            try:
                dose_mult = float(fields[1])
            except ValueError:
                pass
    norm_factor = 1.0
    if len(lines) > 5 and lines[5]:
        try:
            norm_factor = float(lines[5])
        except ValueError:
            pass

    n_voxels = nx * ny * nz
    byte_count = n_voxels * 4
    if len(raw) < byte_count:
        raise ValueError(
            f"{dose_path}: file is smaller ({len(raw)} bytes) than the "
            f"expected dose grid ({byte_count} bytes for {nx}x{ny}x{nz}).")
    dose_bytes = raw[-byte_count:]
    dose_flat = np.frombuffer(dose_bytes, dtype=">u4").astype(np.float64)
    dose = dose_flat.reshape((nz, ny, nx)) * dose_scale * dose_mult * norm_factor

    DOSE_MAX_PLAUSIBLE_CGY = 15000.0
    bad = ~np.isfinite(dose) | (dose < 0) | (dose > DOSE_MAX_PLAUSIBLE_CGY)
    if bad.any():
        n_bad = int(bad.sum())
        dose[bad] = 0.0
    dose = dose.astype(np.float32)
    return dose, spacing, origin


def _parse_plan_file(plan_path: str) -> dict:
    result = {"prescribed_dose_cgy": None, "n_fractions": None,
              "dose_per_fraction_cgy": None, "plan_id": None}
    try:
        with open(plan_path, encoding="utf-8", errors="ignore") as fh:
            lines = [l.strip() for l in fh if l.strip()]
    except Exception:
        return result

    for line in lines:
        if not re.match(r"^[\d\s.,+-]+$", line):
            result["plan_id"] = line
            break

    all_floats = []
    for line in lines:
        for tok in line.replace(",", " ").split():
            try:
                v = float(tok)
                all_floats.append(v)
            except ValueError:
                pass

    def _is_plausible_dose_cgy(v):
        if not (500.0 <= v <= 20000.0):
            return False
        if 1900.0 <= v <= 2100.0 and float(int(v)) == v:
            return False
        return True

    for v in all_floats:
        if _is_plausible_dose_cgy(v):
            result["prescribed_dose_cgy"] = v
            break

    fract_next = False
    for line in lines:
        if "fract" in line.lower():
            fract_next = True
            for tok in line.replace(",", " ").split():
                try:
                    v = int(tok)
                    if 1 <= v <= 60:
                        result["n_fractions"] = v
                        fract_next = False
                        break
                except ValueError:
                    pass
            if result["n_fractions"] is not None:
                break
            continue
        if fract_next:
            for tok in line.replace(",", " ").split():
                try:
                    v = int(tok)
                    if 1 <= v <= 60:
                        result["n_fractions"] = v
                        break
                except ValueError:
                    pass
            if result["n_fractions"] is not None:
                break
            fract_next = False

    if result["prescribed_dose_cgy"] is not None and result["n_fractions"] is not None:
        result["dose_per_fraction_cgy"] = (
                result["prescribed_dose_cgy"] / result["n_fractions"])

    return result


def _parse_tel1(tel1_path: str) -> list:
    try:
        with open(tel1_path, encoding="utf-8", errors="ignore") as fh:
            raw_lines = [l.strip() for l in fh if l.strip()]
    except Exception:
        return []

    if not raw_lines:
        return []

    beams = []
    current_beam: dict = {}
    all_floats_in_file = []

    _field_re = {
        "gantry": re.compile(r"gantry", re.I),
        "collimator": re.compile(r"collimator|coll", re.I),
        "couch": re.compile(r"couch|table", re.I),
        "mu": re.compile(r"\bmu\b|monitor.unit", re.I),
        "energy": re.compile(r"energy|mev|beam.energy", re.I),
        "arc": re.compile(r"\barc\b|vmat|dyna", re.I),
        "static": re.compile(r"\bstatic\b|\bimrt\b|\bconformal\b", re.I),
    }

    def _try_float(tok):
        try:
            return float(tok)
        except ValueError:
            return None

    def _floats_in_line(line):
        return [v for tok in line.replace(",", " ").split()
                if (v := _try_float(tok)) is not None]

    for line in raw_lines:
        floats = _floats_in_line(line)
        all_floats_in_file.extend(floats)
        line_lo = line.lower()

        if _field_re["arc"].search(line_lo):
            if current_beam:
                beams.append(current_beam)
            current_beam = {"beam_type": "ARC"}
        elif _field_re["static"].search(line_lo):
            if current_beam:
                beams.append(current_beam)
            current_beam = {"beam_type": "STATIC"}

        for field, pat in _field_re.items():
            if field in ("arc", "static"):
                continue
            if pat.search(line_lo) and floats:
                key_map = {
                    "gantry": "gantry_deg",
                    "collimator": "collimator_deg",
                    "couch": "couch_deg",
                    "mu": "mu",
                    "energy": "energy_mv",
                }
                dest = key_map.get(field)
                if dest and dest not in current_beam:
                    current_beam[dest] = floats[0]

    if current_beam:
        beams.append(current_beam)

    if not any("gantry_deg" in b or "mu" in b for b in beams):
        return [{"raw_line_count": len(raw_lines),
                 "raw_float_values": all_floats_in_file}]

    return beams


def _parse_dosenormsettings_xml(xml_path: str) -> dict:
    result = {"norm_type": None, "isodose_pct": None, "ref_point": None}
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return result

    def _iter_text(element):
        if element.text and element.text.strip():
            yield element.tag, element.text.strip()
        for child in element:
            yield from _iter_text(child)

    for tag, text in _iter_text(root):
        tag_lo = tag.lower()
        if "normtype" in tag_lo or "normalization" in tag_lo:
            result["norm_type"] = text
        elif "isodose" in tag_lo or "percent" in tag_lo:
            try:
                result["isodose_pct"] = float(text)
            except ValueError:
                pass
        elif "refpoint" in tag_lo or "reference" in tag_lo or "point" in tag_lo:
            if result["ref_point"] is None:
                result["ref_point"] = text

    for elem in root.iter():
        for attr_name, attr_val in elem.attrib.items():
            attr_lo = attr_name.lower()
            if "normtype" in attr_lo or "normalization" in attr_lo:
                if result["norm_type"] is None:
                    result["norm_type"] = attr_val
            elif "isodose" in attr_lo or "percent" in attr_lo:
                if result["isodose_pct"] is None:
                    try:
                        result["isodose_pct"] = float(attr_val)
                    except ValueError:
                        pass

    return result


def _extract_plan_params_features(plan_params_dir: str) -> dict:
    nan = float("nan")

    features = {
        "n_beams": nan,
        "total_mu": nan,
        "gantry_mean": nan,
        "gantry_std": nan,
        "gantry_range": nan,
        "collimator_mean": nan,
        "collimator_std": nan,
        "couch_nonzero_fraction": nan,
        "energy_mv_mode": nan,
        "has_arc": nan,
        "n_fractions": nan,
        "prescribed_dose_cgy": nan,
        "dose_per_fraction_cgy": nan,
        "isodose_pct": nan,
        "norm_type_code": nan,
    }

    import re as _re
    def _plan_ct(pdir):
        try:
            L = open(os.path.join(pdir, "plan"), encoding="ascii",
                     errors="ignore").read().splitlines()
            return (L[6].strip() if len(L) > 6 else "").upper()
        except Exception:
            return ""

    def _refvalue(pdir):
        f = os.path.join(pdir, "dosenormsettings.xml")
        if not os.path.exists(f):
            return None
        m = _re.search(r"<RefValue>([\d.]+)</RefValue>",
                       open(f, encoding="utf-8", errors="ignore").read())
        return float(m.group(1)) if m else None

    plan_dirs = [os.path.dirname(p)
                 for p in glob.glob(os.path.join(plan_params_dir, "*/plan"))]
    ct1_plans = [d for d in plan_dirs if _plan_ct(d) == "CT1"]
    pool = ct1_plans or plan_dirs
    sel = max(pool, key=lambda d: (_refvalue(d) or -1.0)) if pool else None

    if sel is not None:
        rv = _refvalue(sel)
        if rv is not None and rv > 0:
            features["prescribed_dose_cgy"] = float(rv)

    tel = os.path.join(sel, "tel.1") if sel else None
    if tel and os.path.exists(tel):
        try:
            tl = open(tel, encoding="latin-1", errors="ignore").read().splitlines()
            en_idx = [i for i, l in enumerate(tl) if _re.search(r"\d+\s*MV", l)]
            _excl = ("kv", "setup", "port", "cbct", "drr")
            names = [tl[i - 1].strip().lower() for i in en_idx if i > 0]
            tx = [n for n in names if not any(x in n for x in _excl)]
            features["n_beams"] = float(len(tx))
            features["has_arc"] = float(1 if any("arc" in n for n in tx) else 0)
        except Exception:
            pass

    xml_candidates = ([os.path.join(sel, "dosenormsettings.xml")] if sel else [])
    xml_candidates += glob.glob(os.path.join(plan_params_dir, "*/dosenormsettings.xml"))
    xml_candidates += glob.glob(os.path.join(plan_params_dir, "dosenormsettings.xml"))
    xml_candidates = [x for x in xml_candidates if os.path.exists(x)]
    if xml_candidates:
        xml_info = _parse_dosenormsettings_xml(xml_candidates[0])
        if xml_info["isodose_pct"] is not None:
            features["isodose_pct"] = float(xml_info["isodose_pct"])
        nt = str(xml_info.get("norm_type") or "").lower()
        if "point" in nt:
            features["norm_type_code"] = 1.0
        elif "volume" in nt or "vol" in nt:
            features["norm_type_code"] = 2.0
        elif "max" in nt:
            features["norm_type_code"] = 3.0
        elif nt:
            features["norm_type_code"] = 0.0

    return features


def step_extract_plan_params(cfg):
    print("\n" + "=" * 60)
    print("  STEP 3 - Extract Plan Parameter Features")
    print("=" * 60)

    input_dir = cfg["INPUTS_DIR"]
    processed_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])

    if not os.path.isdir(processed_dir):
        print(f"Processed/ not found at {processed_dir}. Run step 2 first.")
        return

    import json as _json

    all_rows = []
    uuids = sorted(d for d in os.listdir(processed_dir)
                   if os.path.isdir(os.path.join(processed_dir, d)))

    if cfg.get("PREPROCESS", {}).get("ONLY_PATIENTS", []):
        _mpath = os.path.join(input_dir, "patient_manifest.csv")
        if table_exists(_mpath):
            _recs = read_table(_mpath).fillna("").to_dict("records")
            _wanted = {str(r.get("uuid", "")).strip()
                       for r in _filter_manifest_only(_recs, cfg)}
            uuids = [u for u in uuids if u in _wanted]

    for uuid in uuids:
        plan_params_dir = os.path.join(processed_dir, uuid, "PlanParams")
        if not os.path.isdir(plan_params_dir):
            continue

        features = _extract_plan_params_features(plan_params_dir)
        features["uuid"] = uuid

        json_path = os.path.join(plan_params_dir, "plan_features.json")
        with open(json_path, "w") as fh:
            serializable = {k: (None if (isinstance(v, float) and v != v) else v)
                            for k, v in features.items()}
            _json.dump(serializable, fh, indent=2)

        all_rows.append(features)
        print(f"  {uuid}: n_beams={features.get('n_beams')}, "
              f"n_fractions={features.get('n_fractions')}, "
              f"prescribed_dose={features.get('prescribed_dose_cgy')}")

    if not all_rows:
        print("No PlanParams directories found.")
        return

    df = pd.DataFrame(all_rows)
    cols = ["uuid"] + [c for c in df.columns if c != "uuid"]
    df = df[cols]
    out_path = os.path.join(input_dir, "plan_params_features.csv")
    df.to_csv(out_path, index=False, float_format="%.6f")
    _to_excel(df, out_path.replace(".csv", ".xlsx"))
    print(f"\nPlan features CSV: {out_path}  ({len(df)} patients)")


def _load_dicom_grid_from_dir(dicom_dir: str):
    try:
        dcm_files = [os.path.join(dicom_dir, f)
                     for f in os.listdir(dicom_dir)
                     if f.lower().endswith(".dcm")]
        if not dcm_files:
            return None

        headers = []
        for fp in dcm_files:
            ds = _read_dicom(fp, stop_before_pixels=True)
            headers.append((float(ds.ImagePositionPatient[2]), ds))
        headers.sort(key=lambda x: x[0])

        ds0 = headers[0][1]
        try:
            z_sp = abs(headers[0][0] - headers[1][0])
        except (IndexError, AttributeError):
            z_sp = float(getattr(ds0, "SliceThickness", 2.5))

        ps = ds0.PixelSpacing
        nz = len(headers)
        origin = (float(ds0.ImagePositionPatient[0]),
                  float(ds0.ImagePositionPatient[1]),
                  float(headers[0][0]))
        spacing = (float(z_sp), float(ps[0]), float(ps[1]))
        shape = (nz, int(ds0.Rows), int(ds0.Columns))
        return origin, spacing, shape
    except Exception:
        return None


def _lazy_import_heavy(with_totalsegmentator: bool = True):
    import cv2
    import scipy.ndimage
    import torch
    import nibabel as nib
    import SimpleITK as sitk
    if with_totalsegmentator:
        from totalsegmentator.python_api import totalsegmentator
    else:
        totalsegmentator = None
    return cv2, scipy.ndimage, torch, nib, sitk, totalsegmentator


def _register_pet_to_ct(uuid: str, row: dict, cfg: dict) -> dict | None:
    try:
        import SimpleITK as sitk
    except Exception as e:
        print(f"  [pet-reg] SimpleITK unavailable ({e}); skipping {uuid}.")
        return None

    input_dir = cfg["INPUTS_DIR"]
    P = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"], str(uuid))
    ct_p = os.path.join(P, "CT-Plan", "whole_body.npy")
    pet_p = os.path.join(P, "PET-Plan", "whole_body.npy")
    meta_p = os.path.join(P, "Contours", "contours_meta.txt")
    if not (os.path.exists(ct_p) and os.path.exists(pet_p) and os.path.exists(meta_p)):
        return None

    import json as _json
    ct = np.load(ct_p).astype(np.float32)
    pet = np.load(pet_p).astype(np.float32)
    cm = _json.load(open(meta_p))
    ox, oy, oz = cm["origin"]; sz, sy, sx = cm["spacing"]
    gtv_p = os.path.join(P, "Contours", "gtv_mask.npy")
    gtv = np.load(gtv_p) if os.path.exists(gtv_p) else None
    if gtv is not None and gtv.sum() == 0:
        gtv = None

    pet_cfg = _get_modalities(cfg).get("pet_plan", {})
    pet_src = _resolve_source(pet_cfg.get("source", ""), row, cfg["RAW_DIR"])
    try:
        import pydicom
        files = glob.glob(os.path.join(pet_src, "*"))
        hd = [pydicom.dcmread(f, force=True, stop_before_pixels=True) for f in files]
        hd = [d for d in hd if hasattr(d, "ImagePositionPatient")]
        hd.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        pz = [float(d.ImagePositionPatient[2]) for d in hd]
        ps = [float(v) for v in hd[0].PixelSpacing]
        px0, py0, pz0 = [float(v) for v in hd[0].ImagePositionPatient]
        psz = pz[1] - pz[0] if len(pz) > 1 else float(getattr(hd[0], "SliceThickness", 3.0))
    except Exception as e:
        print(f"  [pet-reg] {uuid}: cannot read PET geometry ({e}); skipping.")
        return None

    def mk(a, o, s):
        im = sitk.GetImageFromArray(a)
        im.SetSpacing([float(s[0]), float(s[1]), float(s[2])])
        im.SetOrigin([float(o[0]), float(o[1]), float(o[2])])
        return im

    fixed = sitk.Cast(mk(np.clip(ct, -1000, 200), (ox, oy, oz), (sx, sy, sz)), sitk.sitkFloat32)
    ctbody = sitk.Cast(mk((ct > -500).astype(np.uint8), (ox, oy, oz), (sx, sy, sz)), sitk.sitkUInt8)
    moving = sitk.Cast(mk(pet, (px0, py0, pz0), (ps[1], ps[0], psz)), sitk.sitkFloat32)

    def _reg(tx, sh, sm):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.15, sitk.sitkWallClock)
        R.SetMetricFixedMask(ctbody)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(1.0, 300, 1e-6, 10)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(sh); R.SetSmoothingSigmasPerLevel(sm)
        R.SetInitialTransform(tx, inPlace=False)
        return R.Execute(fixed, moving)

    inits = []
    mb = mk((pet > 0.15 * pet.max()).astype(np.float32), (px0, py0, pz0), (ps[1], ps[0], psz))
    inits.append(("moments", sitk.CenteredTransformInitializer(
        fixed, mb, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)))
    if gtv is not None:
        gc = np.argwhere(gtv > 0).mean(0)
        g = np.array([ox + gc[2] * sx, oy + gc[1] * sy, oz + gc[0] * sz])
        pc = np.argwhere(pet > 0.5 * pet.max()).mean(0)
        p = np.array([px0 + pc[2] * ps[1], py0 + pc[1] * ps[0], pz0 + pc[0] * psz])
        t = sitk.Euler3DTransform()
        t.SetTranslation([float(p[0] - g[0]), float(p[1] - g[1]), float(p[2] - g[2])])
        inits.append(("suvmax2gtv", t))

    gmir = gtv[:, :, ::-1] if gtv is not None else None

    def _qa(arr):
        if gtv is None:
            return float("nan")
        return float(arr[gtv > 0].mean() / max(1.0, arr[gmir > 0].mean()))

    best = None
    for nm, tx in inits:
        try:
            T = _reg(_reg(tx, [4, 2], [2, 1]), [2, 1], [1, 0])
            arr = sitk.GetArrayFromImage(
                sitk.Resample(moving, fixed, T, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        except Exception as e:
            print(f"  [pet-reg] {uuid}/{nm}: failed ({e})")
            continue
        score = _qa(arr) if gtv is not None else 0.0
        if best is None or (score == score and score > best[0]):
            best = (score, arr.astype(np.float32), nm)

    if best is None:
        print(f"  [pet-reg] {uuid}: all registrations failed.")
        return None

    qa_min = float(cfg.get("PREPROCESS", {}).get("PET_REG_QA_MIN_RATIO", 1.3))
    qa_ratio, arr, init_used = best
    qa_pass = (gtv is not None) and (qa_ratio >= qa_min)
    np.save(os.path.join(P, "PET-Plan", "whole_body_regCT.npy"), arr)
    meta = {"uuid": uuid, "qa_ratio": qa_ratio, "qa_pass": bool(qa_pass),
            "qa_min": qa_min, "init": init_used, "has_gtv": gtv is not None}
    with open(os.path.join(P, "PET-Plan", "pet_reg_meta.json"), "w") as f:
        _json.dump(meta, f, indent=2)
    flag = "OK" if qa_pass else ("NO-GTV" if gtv is None else "QA-FAIL review")
    print(f"  [pet-reg] {uuid}: GTV/mirror ratio={qa_ratio:.2f} via {init_used} -> {flag}")
    return meta


DOSE_AXIS_SIGNS = (-1, 1, -1)


def _select_dose_file(src: str, contour_ct: str = "CT1"):
    cands = glob.glob(os.path.join(src, "*/dose.1"))
    if not cands:
        return None, False

    def _ct_ref(dose_path: str) -> str:
        plan = os.path.join(os.path.dirname(dose_path), "plan")
        try:
            L = open(plan, encoding="ascii", errors="ignore").read().splitlines()
            return L[6].strip() if len(L) > 6 else ""
        except Exception:
            return ""

    matched = [c for c in cands if _ct_ref(c).upper() == contour_ct.upper()]
    pool = matched or cands

    def _dmax(dose_path: str) -> float:
        try:
            a, _, _ = _parse_dose_volume(dose_path)
            return float(np.nanmax(a))
        except Exception:
            return -1.0

    return max(pool, key=_dmax), bool(matched)


def _register_dose_to_ct(uuid: str, row: dict, cfg: dict) -> dict | None:
    from scipy.ndimage import map_coordinates
    import json as _j
    input_dir = cfg["INPUTS_DIR"]
    Pd = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"], str(uuid))
    ct_p = os.path.join(Pd, "CT-Plan", "whole_body.npy")
    meta_p = os.path.join(Pd, "Contours", "contours_meta.txt")
    if not (os.path.exists(ct_p) and os.path.exists(meta_p)):
        return None
    cm = _j.load(open(meta_p))
    if cm.get("origin") is None:
        return None
    ox, oy, oz = cm["origin"]; sz, sy, sx = cm["spacing"]
    ct = np.load(ct_p).astype(np.float32)

    tgt_mask = None
    for nm in ("ptv", "gtv"):
        f = os.path.join(Pd, "Contours", f"{nm}_mask.npy")
        if os.path.exists(f):
            m = np.load(f)
            if m.sum() > 0:
                tgt_mask = m
                break

    dsrc = _resolve_source(_get_modalities(cfg).get("dose", {}).get("source", ""),
                           row, cfg["RAW_DIR"])
    chosen, on_ct = _select_dose_file(dsrc, "CT1")
    if chosen is None:
        return None
    try:
        d, sp, o = _parse_dose_volume(chosen)
    except Exception as e:
        print(f"  [dose-reg] {uuid}: parse failed ({e})")
        return None
    dsx, dsy, dsz = sp; dox, doy, doz = o
    S = DOSE_AXIS_SIGNS
    hiv = np.argwhere(d > 0.5 * d.max())
    if len(hiv) == 0:
        return None
    hic = hiv.mean(0)
    hw = np.array([S[0] * (dox + hic[2] * dsx),
                   S[1] * (doy + hic[1] * dsy),
                   S[2] * (doz + hic[0] * dsz)])
    if tgt_mask is not None:
        tc = np.argwhere(tgt_mask > 0).mean(0)
        TW = np.array([ox + tc[2] * sx, oy + tc[1] * sy, oz + tc[0] * sz])
    else:
        TW = np.array([ox + ct.shape[2] / 2 * sx, oy + ct.shape[1] / 2 * sy,
                       oz + ct.shape[0] / 2 * sz])
    t = TW - hw

    zs = oz + np.arange(ct.shape[0]) * sz
    ys = oy + np.arange(ct.shape[1]) * sy
    xs = ox + np.arange(ct.shape[2]) * sx
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    ii = (S[0] * (X - t[0]) - dox) / dsx
    ji = (S[1] * (Y - t[1]) - doy) / dsy
    ki = (S[2] * (Z - t[2]) - doz) / dsz
    dc = map_coordinates(d, [ki, ji, ii], order=1, mode="constant",
                         cval=0.0).reshape(ct.shape).astype(np.float32)
    os.makedirs(os.path.join(Pd, "Dose"), exist_ok=True)
    np.save(os.path.join(Pd, "Dose", "dose_volume_regCT.npy"), dc)

    qa = {"uuid": uuid, "source": chosen, "on_contour_ct": bool(on_ct),
          "signs": list(S), "max_cGy": float(dc.max())}
    if tgt_mask is not None and dc.max() > 0:
        cov = ((dc >= 0.5 * dc.max()) & (tgt_mask > 0)).sum() / (tgt_mask > 0).sum()
        ratio = dc[tgt_mask > 0].mean() / max(1.0, dc[ct > -500].mean())
        qa["ptv_cov50"] = float(cov)
        qa["dose_in_target_ratio"] = float(ratio)
        qa["qa_pass"] = bool(cov >= 0.8 and ratio >= 2.0 and on_ct)
    else:
        qa["qa_pass"] = False
    with open(os.path.join(Pd, "Dose", "dose_reg_meta.json"), "w") as f:
        _j.dump(qa, f, indent=2)
    flag = "OK" if qa.get("qa_pass") else ("no-CT1-plan" if not on_ct else "review")
    print(f"  [dose-reg] {uuid}: max={dc.max():.0f}cGy "
          f"PTVcov50={qa.get('ptv_cov50', 0)*100:.0f}% "
          f"ratio={qa.get('dose_in_target_ratio', 0):.1f} on_CT1={on_ct} -> {flag}")
    return qa


_DVH_GRID_CGY = list(range(0, 8001, 100))


def _compute_dvh(uuid: str, cfg: dict) -> dict | None:
    import json as _j
    input_dir = cfg["INPUTS_DIR"]
    P = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"], str(uuid))
    dose_p = os.path.join(P, "Dose", "dose_volume_regCT.npy")
    meta_p = os.path.join(P, "Contours", "contours_meta.txt")
    if not (os.path.exists(dose_p) and os.path.exists(meta_p)):
        return None
    dose = np.load(dose_p).astype(np.float32)
    cm = _j.load(open(meta_p))
    if cm.get("spacing") is None:
        return None
    sz, sy, sx = cm["spacing"]
    vox_cc = (sz * sy * sx) / 1000.0
    grid = _DVH_GRID_CGY

    def _metrics(dvals):
        return {
            "Dmean_cGy": round(float(dvals.mean()), 1),
            "Dmax_cGy": round(float(np.percentile(dvals, 99.5)), 1),
            "V5": round(float((dvals >= 500).mean() * 100), 2),
            "V10": round(float((dvals >= 1000).mean() * 100), 2),
            "V20": round(float((dvals >= 2000).mean() * 100), 2),
            "V30": round(float((dvals >= 3000).mean() * 100), 2),
            "dvh_pct": [round(float((dvals >= lvl).mean() * 100), 2) for lvl in grid],
        }

    out = {"uuid": uuid, "dose_max_cGy": round(float(dose.max()), 1),
           "vox_cc": round(vox_cc, 5), "dose_grid_cGy": grid, "structures": {}}
    masks = {}
    for s in ("lung", "gtv", "ctv", "ptv", "heart", "esophagus",
              "spinal_cord", "liver", "breast"):
        f = os.path.join(P, "Contours", f"{s}_mask.npy")
        if not os.path.exists(f):
            continue
        m = np.load(f) > 0
        if m.sum() == 0:
            continue
        masks[s] = m
        rec = {"volume_cc": round(float(m.sum() * vox_cc), 1)}
        rec.update(_metrics(dose[m]))
        out["structures"][s] = rec

    if "lung" in masks and "gtv" in masks:
        lmg = masks["lung"] & ~masks["gtv"]
        if lmg.sum() > 0:
            rec = {"volume_cc": round(float(lmg.sum() * vox_cc), 1)}
            rec.update(_metrics(dose[lmg]))
            out["structures"]["lung_minus_gtv"] = rec

    os.makedirs(os.path.join(P, "DVH"), exist_ok=True)
    with open(os.path.join(P, "DVH", "dvh.json"), "w") as fh:
        _j.dump(out, fh, indent=2)
    lung = out["structures"].get("lung", {})
    print(f"  [dvh] {uuid}: lung MLD={lung.get('Dmean_cGy', '?')}cGy "
          f"V20={lung.get('V20', '?')}%  ({len(out['structures'])} structures)")
    return out


def _generate_alignment_video(uuid: str, cfg: dict, subdir: str = "") -> None:
    try:
        import cv2
    except Exception as e:
        print(f"  [video] OpenCV unavailable ({e}); skipping video for {uuid}.")
        return
    import subprocess as _sp, tempfile as _tf

    input_dir = cfg["INPUTS_DIR"]
    proc_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    P = os.path.join(proc_dir, str(uuid))
    ct_path = os.path.join(P, "CT-Plan", "whole_body.npy")
    if not os.path.exists(ct_path):
        print(f"  [video] no CT-Plan volume for {uuid}; skipping video.")
        return
    ct = np.load(ct_path).astype(np.float32)

    colors = dict(gtv=(0, 0, 255), ctv=(0, 128, 255), ptv=(0, 255, 255),
                  lung=(255, 255, 0), heart=(255, 0, 255),
                  esophagus=(0, 165, 255), spinal_cord=(0, 255, 0),
                  liver=(128, 255, 128), breast=(255, 128, 0))
    masks = {}
    for nm in colors:
        f = os.path.join(P, "Contours", f"{nm}_mask.npy")
        if os.path.exists(f):
            m = np.load(f)
            if m.sum() > 0:
                masks[nm] = m

    import json as _json
    pet_reg = pet_native = pet_qa = None
    _pet_hi = 1.0
    reg_path = os.path.join(P, "PET-Plan", "whole_body_regCT.npy")
    nat_path = os.path.join(P, "PET-Plan", "whole_body.npy")
    meta_path = os.path.join(P, "PET-Plan", "pet_reg_meta.json")
    if os.path.exists(reg_path):
        pet_reg = np.load(reg_path).astype(np.float32)
        _nz = pet_reg[pet_reg > 0]
        _pet_hi = float(np.percentile(_nz, 99.5)) if _nz.size else 1.0
        if os.path.exists(meta_path):
            try:
                pet_qa = _json.load(open(meta_path))
            except Exception:
                pet_qa = None
    elif os.path.exists(nat_path):
        pet_native = np.load(nat_path).astype(np.float32)
        _pet_hi = float(np.percentile(pet_native, 99.5)) or 1.0

    dose_reg = dose_qa = None
    dreg_path = os.path.join(P, "Dose", "dose_volume_regCT.npy")
    dmeta_path = os.path.join(P, "Dose", "dose_reg_meta.json")
    if os.path.exists(dreg_path):
        dose_reg = np.load(dreg_path).astype(np.float32)
        _dose_hi = float(dose_reg.max()) or 1.0
        if os.path.exists(dmeta_path):
            try:
                dose_qa = _json.load(open(dmeta_path))
            except Exception:
                dose_qa = None

    wc = float(cfg.get("PREPROCESS", {}).get("WINDOW_CENTER", -600))
    ww = float(cfg.get("PREPROCESS", {}).get("WINDOW_WIDTH", 1500))
    lo, hi = wc - ww / 2, wc + ww / 2

    def _win(a):
        return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    nz, ny, nx = ct.shape
    cor_y = ny // 2
    fps = int(cfg.get("PREPROCESS", {}).get("VIDEO_FPS", 8))
    videos_dir = (cfg.get("PREPROCESS", {}).get("VIDEO_DIR") or "").strip()
    if not videos_dir:
        videos_dir = os.path.join(input_dir, cfg["INPUTS"].get("VIDEOS", "Videos"))
    if subdir:
        videos_dir = os.path.join(videos_dir, subdir)
    os.makedirs(videos_dir, exist_ok=True)
    tmp = _tf.mkdtemp()

    for _fi, z in enumerate(range(nz - 1, -1, -1)):
        ax = cv2.cvtColor(_win(ct[z]), cv2.COLOR_GRAY2BGR)
        for nm, m in masks.items():
            cnts, _ = cv2.findContours(m[z], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ax, cnts, -1, colors[nm], 1)

        cor = cv2.cvtColor(_win(ct[::-1, cor_y, :]), cv2.COLOR_GRAY2BGR)
        for nm, m in masks.items():
            cm = m[::-1, cor_y, :].astype(np.uint8)
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cor, cnts, -1, colors[nm], 1)
        cor = cv2.resize(cor, (nx, ny), interpolation=cv2.INTER_NEAREST)
        yline = int((nz - 1 - z) / max(1, nz - 1) * (ny - 1))
        cv2.line(cor, (0, yline), (nx, yline), (255, 255, 255), 1)

        panels = [ax, cor]
        _pet_tag = ""
        if pet_reg is not None:
            ph = cv2.cvtColor(_win(ct[z]), cv2.COLOR_GRAY2BGR)
            pnorm = np.clip(pet_reg[z] / _pet_hi, 0, 1)
            heat = cv2.applyColorMap((pnorm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            amask = (pnorm > 0.1)[..., None]
            ph = np.where(amask, (0.45 * ph + 0.55 * heat).astype(np.uint8), ph)
            if "gtv" in masks:
                cnts, _ = cv2.findContours(masks["gtv"][z], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ph, cnts, -1, (0, 255, 0), 1)
            if pet_qa:
                lbl = (f"PET->CT  QA={pet_qa.get('qa_ratio', float('nan')):.1f} "
                       + ("OK" if pet_qa.get("qa_pass") else "REVIEW"))
            else:
                lbl = "PET->CT registered"
            cv2.putText(ph, lbl, (6, ny - 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            panels.append(ph)
            _pet_tag = "  | PET->CT fused"
        elif pet_native is not None:
            pidx = int(round(z / max(1, nz - 1) * (pet_native.shape[0] - 1)))
            pnorm = np.clip(pet_native[pidx] / _pet_hi, 0, 1)
            ph = cv2.applyColorMap((pnorm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            ph = cv2.resize(ph, (nx, ny), interpolation=cv2.INTER_NEAREST)
            cv2.putText(ph, "PET native - NOT registered", (6, ny - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            panels.append(ph)
            _pet_tag = "  | PET (own frame)"

        _dose_tag = ""
        if dose_reg is not None:
            dp = cv2.cvtColor(_win(ct[z]), cv2.COLOR_GRAY2BGR)
            dn = np.clip(dose_reg[z] / _dose_hi, 0, 1)
            heat = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            amask = (dn > 0.1)[..., None]
            dp = np.where(amask, (0.5 * dp + 0.5 * heat).astype(np.uint8), dp)
            for nm in ("ptv", "gtv"):
                if nm in masks:
                    cnts, _ = cv2.findContours(masks[nm][z], cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(dp, cnts, -1, colors[nm], 1)
            if dose_qa:
                lbl = (f"Dose->CT  cov50={dose_qa.get('ptv_cov50', 0)*100:.0f}% "
                       + ("OK" if dose_qa.get("qa_pass") else "review"))
            else:
                lbl = "Dose->CT"
            cv2.putText(dp, lbl, (6, ny - 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            panels.append(dp)
            _dose_tag = "  | Dose->CT(JET)+PTV"

        montage = np.hstack(panels)
        cv2.putText(montage,
                    f"{uuid}  z={z}/{nz}   axial | coronal (sup=top){_pet_tag}{_dose_tag}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        x0 = 6
        for nm in masks:
            cv2.putText(montage, nm, (x0, ny - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, colors[nm], 1)
            x0 += 9 * len(nm) + 10
        cv2.imwrite(os.path.join(tmp, f"f{_fi:04d}.png"), montage)

    out = os.path.join(videos_dir, f"{uuid}_alignment.mp4")
    try:
        _sp.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                 os.path.join(tmp, "f%04d.png"), "-pix_fmt", "yuv420p",
                 "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out],
                check=True, capture_output=True)
        print(f"  [video] wrote {out}  ({nz} frames, {len(masks)} ROIs)")
    except Exception as e:
        print(f"  [video] ffmpeg failed for {uuid}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_STUDY_ROIS = ("gtv", "ctv", "lung", "ptv", "heart", "esophagus",
               "spinal_cord", "liver", "breast")


def _study_load_ct_volume(dcm_dir):
    if not os.path.isdir(dcm_dir):
        return None, None
    files = [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir)
             if f.lower().endswith(".dcm")]
    if not files:
        return None, None
    headers = []
    for fp in files:
        try:
            ds = _read_dicom(fp, stop_before_pixels=True)
            headers.append((float(ds.ImagePositionPatient[2]), fp, ds))
        except Exception:
            continue
    if not headers:
        return None, None
    headers.sort(key=lambda x: x[0])
    slices = []
    for _, fp, ds in headers:
        px = _read_dicom(fp).pixel_array
        slices.append(px * float(getattr(ds, "RescaleSlope", 1.0))
                      + float(getattr(ds, "RescaleIntercept", 0.0)))
    vol = np.ascontiguousarray(np.stack(slices), dtype=np.float32)
    ds0 = headers[0][2]
    z_sp = abs(headers[0][0] - headers[1][0]) if len(headers) > 1 \
        else float(getattr(ds0, "SliceThickness", 2.5))
    ps = ds0.PixelSpacing
    origin = (float(ds0.ImagePositionPatient[0]),
              float(ds0.ImagePositionPatient[1]), float(headers[0][0]))
    spacing = (float(z_sp), float(ps[0]), float(ps[1]))
    return vol, (origin, spacing, vol.shape)


def _study_plans_for_ct(patient_dir, ct_folder):
    ref = ct_folder.replace("1~", "").upper()
    out = []
    for pdir in sorted(glob.glob(os.path.join(patient_dir, "plan", "*"))):
        plan_f = os.path.join(pdir, "plan")
        dose_f = os.path.join(pdir, "dose.1")
        if not (os.path.isfile(plan_f) and os.path.isfile(dose_f)):
            continue
        try:
            L = open(plan_f, encoding="ascii", errors="ignore").read().splitlines()
            ctref = (L[6].strip() if len(L) > 6 else "").upper()
        except Exception:
            ctref = ""
        if ctref == ref:
            name = os.path.basename(pdir)
            out.append((name, dose_f, "+" in name))
    return out


def _study_select_dose(plans):
    if not plans:
        return None
    combined = [p for p in plans if p[2]]
    pool = combined or plans
    return max(pool, key=lambda p: _parse_dose_volume(p[1])[0].max())


def _study_register_dose(dose_path, grid, tgt_mask):
    from scipy.ndimage import map_coordinates
    (ox, oy, oz), (sz, sy, sx), shape = grid
    d, sp, o = _parse_dose_volume(dose_path)
    dsx, dsy, dsz = sp
    dox, doy, doz = o
    S = DOSE_AXIS_SIGNS
    hiv = np.argwhere(d > 0.5 * d.max())
    hic = hiv.mean(0)
    hw = np.array([S[0] * (dox + hic[2] * dsx),
                   S[1] * (doy + hic[1] * dsy),
                   S[2] * (doz + hic[0] * dsz)])
    if tgt_mask is not None and tgt_mask.sum() > 0:
        tc = np.argwhere(tgt_mask > 0).mean(0)
        TW = np.array([ox + tc[2] * sx, oy + tc[1] * sy, oz + tc[0] * sz])
    else:
        nz, ny, nx = shape
        TW = np.array([ox + nx / 2 * sx, oy + ny / 2 * sy, oz + nz / 2 * sz])
    t = TW - hw
    nz, ny, nx = shape
    Z, Y, X = np.meshgrid(oz + np.arange(nz) * sz, oy + np.arange(ny) * sy,
                          ox + np.arange(nx) * sx, indexing="ij")
    ii = (S[0] * (X - t[0]) - dox) / dsx
    ji = (S[1] * (Y - t[1]) - doy) / dsy
    ki = (S[2] * (Z - t[2]) - doz) / dsz
    dc = map_coordinates(d, [ki, ji, ii], order=1, mode="constant",
                         cval=0.0).reshape(shape).astype(np.float32)
    qa = {"max_cGy": float(dc.max())}
    if tgt_mask is not None and tgt_mask.sum() > 0 and dc.max() > 0:
        cov = ((dc >= 0.5 * dc.max()) & (tgt_mask > 0)).sum() / (tgt_mask > 0).sum()
        qa["ptv_cov50"] = round(float(cov), 3)
    return dc, qa


def _study_dvh(dose, masks, vox_cc):
    grid = _DVH_GRID_CGY

    def metrics(dv):
        return {"Dmean_cGy": round(float(dv.mean()), 1),
                "V5": round(float((dv >= 500).mean() * 100), 2),
                "V20": round(float((dv >= 2000).mean() * 100), 2),
                "V30": round(float((dv >= 3000).mean() * 100), 2),
                "dvh_pct": [round(float((dv >= l).mean() * 100), 2) for l in grid]}
    out = {"dose_max_cGy": round(float(dose.max()), 1), "vox_cc": round(vox_cc, 5),
           "dose_grid_cGy": grid, "structures": {}}
    for s, m in masks.items():
        rec = {"volume_cc": round(float(m.sum() * vox_cc), 1)}
        rec.update(metrics(dose[m]))
        out["structures"][s] = rec
    if "lung" in masks and "gtv" in masks:
        lmg = masks["lung"] & ~masks["gtv"]
        if lmg.sum() > 0:
            rec = {"volume_cc": round(float(lmg.sum() * vox_cc), 1)}
            rec.update(metrics(dose[lmg]))
            out["structures"]["lung_minus_gtv"] = rec
    return out


def _study_pick_pet(patient_dir, ref_date):
    cands = []
    for pt in glob.glob(os.path.join(patient_dir, "1~PT*")):
        dcm = os.path.join(pt, "DCMData")
        files = glob.glob(os.path.join(dcm, "*")) if os.path.isdir(dcm) else []
        if not files:
            continue
        try:
            d = _read_dicom(files[0], stop_before_pixels=True)
            date = (getattr(d, "SeriesDate", "") or getattr(d, "AcquisitionDate", "")
                    or getattr(d, "StudyDate", ""))
        except Exception:
            date = ""
        cands.append((date, len(files), pt))
    if not cands:
        return None, None, 0
    cands.sort(key=lambda c: (c[0], c[1]))
    rd = (ref_date or "").replace("-", "")
    pre = [c for c in cands if rd and c[0] and c[0] <= rd]
    date, n, pt = (pre[-1] if pre else cands[-1])
    return pt, date, n


def _study_load_pet(pt_dir):
    dcm = os.path.join(pt_dir, "DCMData")
    hd = []
    for f in glob.glob(os.path.join(dcm, "*")):
        try:
            d = _read_dicom(f)
            if hasattr(d, "ImagePositionPatient"):
                hd.append(d)
        except Exception:
            continue
    hd.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([d.pixel_array * float(getattr(d, "RescaleSlope", 1.0))
                    + float(getattr(d, "RescaleIntercept", 0.0)) for d in hd]
                   ).astype(np.float32)
    ps = [float(v) for v in hd[0].PixelSpacing]
    px0, py0, pz0 = [float(v) for v in hd[0].ImagePositionPatient]
    pz = [float(d.ImagePositionPatient[2]) for d in hd]
    psz = pz[1] - pz[0] if len(pz) > 1 else float(getattr(hd[0], "SliceThickness", 3.0))
    return vol, (px0, py0, pz0), (ps[1], ps[0], psz)


def _study_register_pet(ct, grid, gtv, pt_dir):
    import SimpleITK as sitk
    (ox, oy, oz), (sz, sy, sx), shape = grid
    pet, (px0, py0, pz0), (psx, psy, psz) = _study_load_pet(pt_dir)

    def mk(a, o, s):
        im = sitk.GetImageFromArray(a)
        im.SetSpacing([float(s[0]), float(s[1]), float(s[2])])
        im.SetOrigin([float(o[0]), float(o[1]), float(o[2])])
        return im

    fixed = sitk.Cast(mk(np.clip(ct, -1000, 200), (ox, oy, oz), (sx, sy, sz)), sitk.sitkFloat32)
    ctbody = sitk.Cast(mk((ct > -500).astype(np.uint8), (ox, oy, oz), (sx, sy, sz)), sitk.sitkUInt8)
    moving = sitk.Cast(mk(pet, (px0, py0, pz0), (psx, psy, psz)), sitk.sitkFloat32)

    def _reg(tx, sh, sm):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.15, sitk.sitkWallClock)
        R.SetMetricFixedMask(ctbody)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(1.0, 300, 1e-6, 10)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(sh); R.SetSmoothingSigmasPerLevel(sm)
        R.SetInitialTransform(tx, inPlace=False)
        return R.Execute(fixed, moving)

    inits = []
    mb = mk((pet > 0.15 * pet.max()).astype(np.float32), (px0, py0, pz0), (psx, psy, psz))
    inits.append(("moments", sitk.CenteredTransformInitializer(
        fixed, mb, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)))
    has_gtv = gtv is not None and gtv.sum() > 0
    if has_gtv:
        gc_ = np.argwhere(gtv > 0).mean(0)
        g = np.array([ox + gc_[2] * sx, oy + gc_[1] * sy, oz + gc_[0] * sz])
        pc = np.argwhere(pet > 0.5 * pet.max()).mean(0)
        p = np.array([px0 + pc[2] * psx, py0 + pc[1] * psy, pz0 + pc[0] * psz])
        t = sitk.Euler3DTransform()
        t.SetTranslation([float(p[0] - g[0]), float(p[1] - g[1]), float(p[2] - g[2])])
        inits.append(("suvmax2gtv", t))
    gmir = gtv[:, :, ::-1] if has_gtv else None

    def _qa(arr):
        if not has_gtv:
            return float("nan")
        return float(arr[gtv > 0].mean() / max(1.0, arr[gmir > 0].mean()))

    best = None
    for nm, tx in inits:
        try:
            T = _reg(_reg(tx, [4, 2], [2, 1]), [2, 1], [1, 0])
            arr = sitk.GetArrayFromImage(
                sitk.Resample(moving, fixed, T, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        except Exception:
            continue
        score = _qa(arr)
        if best is None or (score == score and score > best[0]):
            best = (score, arr.astype(np.float32), nm)
    if best is None:
        return None, {}
    return best[1], {"qa_ratio": (round(best[0], 3) if best[0] == best[0] else None),
                     "init": best[2], "has_gtv": bool(has_gtv)}


def _study_video(ct, masks, dose, out_path, title, fps=8):
    import cv2
    nz, ny, nx = ct.shape
    colors = dict(gtv=(0, 0, 255), ctv=(0, 128, 255), ptv=(0, 255, 255),
                  lung=(255, 255, 0), heart=(255, 0, 255),
                  esophagus=(0, 165, 255), spinal_cord=(0, 255, 0),
                  liver=(128, 255, 128), breast=(255, 128, 0))

    def w(a, wc=-600, ww=1500):
        lo, hi = wc - ww / 2, wc + ww / 2
        return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    dmax = float(dose.max()) if dose is not None and dose.max() > 0 else 1.0
    tmp = tempfile.mkdtemp()
    cor_y = ny // 2
    for z in range(nz):
        ax = cv2.cvtColor(w(ct[z]), cv2.COLOR_GRAY2BGR)
        for nm, m in masks.items():
            cnts, _ = cv2.findContours(m[z].astype("uint8"), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ax, cnts, -1, colors[nm], 1)
        df = cv2.cvtColor(w(ct[z]), cv2.COLOR_GRAY2BGR)
        if dose is not None:
            heat = cv2.applyColorMap(
                (np.clip(dose[z] / dmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (dose[z] > 0.1 * dmax)[..., None]
            df = np.where(mask, cv2.addWeighted(df, 0.5, heat, 0.5, 0), df)
            if "ptv" in masks:
                cnts, _ = cv2.findContours(masks["ptv"][z].astype("uint8"),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(df, cnts, -1, (255, 255, 255), 1)
        cor = cv2.cvtColor(w(ct[::-1, cor_y, :]), cv2.COLOR_GRAY2BGR)
        cor = cv2.resize(cor, (nx, ny), interpolation=cv2.INTER_NEAREST)
        yline = int((nz - 1 - z) / max(1, nz - 1) * (ny - 1))
        cv2.line(cor, (0, yline), (nx, yline), (255, 255, 255), 1)
        montage = np.hstack([ax, df, cor])
        cv2.putText(montage, f"{title}  z={z}/{nz}  axial+roi | dose | coronal",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        x0 = 6
        for nm in masks:
            cv2.putText(montage, nm, (x0, ny - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, colors[nm], 1)
            x0 += 9 * len(nm) + 10
        cv2.imwrite(os.path.join(tmp, f"f{z:04d}.png"), montage)
    try:
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                        os.path.join(tmp, "f%04d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_path],
                       check=True, capture_output=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _study_pet_video(ct, gtv, pet, out_path, title, fps=8):
    import cv2
    nz, ny, nx = ct.shape

    def w(a, wc=-600, ww=1500):
        lo, hi = wc - ww / 2, wc + ww / 2
        return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    nzp = pet[pet > 0]
    phi = float(np.percentile(nzp, 99.5)) if nzp.size else 1.0
    tmp = tempfile.mkdtemp()
    cor_y = ny // 2
    for z in range(nz):
        base = cv2.cvtColor(w(ct[z]), cv2.COLOR_GRAY2BGR)
        fused = cv2.cvtColor(w(ct[z]), cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap((np.clip(pet[z] / phi, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_HOT)
        m = (pet[z] > 0.2 * phi)[..., None]
        fused = np.where(m, cv2.addWeighted(fused, 0.4, heat, 0.6, 0), fused)
        if gtv is not None:
            for img in (base, fused):
                cnts, _ = cv2.findContours(gtv[z].astype("uint8"),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1,
                                 (0, 0, 255) if img is base else (255, 255, 255), 1)
        cor = cv2.cvtColor(w(ct[::-1, cor_y, :]), cv2.COLOR_GRAY2BGR)
        cor = cv2.resize(cor, (nx, ny), interpolation=cv2.INTER_NEAREST)
        yline = int((nz - 1 - z) / max(1, nz - 1) * (ny - 1))
        cv2.line(cor, (0, yline), (nx, yline), (255, 255, 255), 1)
        montage = np.hstack([base, fused, cor])
        cv2.putText(montage, f"{title}  z={z}/{nz}  CT+GTV | PET->CT | coronal",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(tmp, f"f{z:04d}.png"), montage)
    try:
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                        os.path.join(tmp, "f%04d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_path],
                       check=True, capture_output=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _build_patient_studies(row, cfg, make_video=False, force=False):
    pre = cfg.get("PREPROCESS", {})
    hasta_no = str(row.get("hasta_no", "")).strip()
    uuid = str(row.get("uuid", "")).strip()
    if not hasta_no or not uuid:
        return {"uuid": uuid, "ok": False, "error": "missing hasta_no/uuid"}
    patient_dir = os.path.join(cfg["RAW_DIR"], "RTs", f"1~{hasta_no}")
    if not os.path.isdir(patient_dir):
        return {"uuid": uuid, "ok": False, "error": f"raw folder missing: {patient_dir}"}
    out_root = os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"]["PROCESSED"], uuid)
    meta_path = os.path.join(out_root, "patient_meta.json")
    if not force and os.path.exists(meta_path):
        print(f"  [studies] {uuid}: patient_meta.json exists, skip.")
        return {"uuid": uuid, "ok": True, "skipped": True}

    inv = _scan_patient_imaging(patient_dir)
    studies = []
    for i in range(1, 11):
        mn = inv.get(f"ct{i}_meaningful")
        if isinstance(mn, str) and mn.startswith("Y"):
            studies.append({"ct_folder": inv[f"ct{i}_name"],
                            "date": inv.get(f"ct{i}_date", ""),
                            "is_total": "(total)" in mn})
    if not studies:
        print(f"  [studies] {uuid} ({hasta_no}): no meaningful CT - skipped.")
        return {"uuid": uuid, "ok": False, "error": "no meaningful CT"}

    studies_dir = os.path.join(out_root, "studies")

    def _has_dcm(ct_folder):
        d = os.path.join(patient_dir, ct_folder, "DCMData")
        return os.path.isdir(d) and any(
            f.lower().endswith(".dcm") for f in os.listdir(d))

    usable = [s for s in studies if _has_dcm(s["ct_folder"])]
    min_ct = int(pre.get("MIN_MEANINGFUL_CT", 1) or 0)
    max_ct = int(pre.get("MAX_MEANINGFUL_CT", 1) or 0)

    def _write_stub(reason):
        if os.path.isdir(studies_dir):
            shutil.rmtree(studies_dir, ignore_errors=True)
        os.makedirs(out_root, exist_ok=True)
        json.dump({"hasta_no": hasta_no, "uuid": uuid,
                   "n_studies": len(usable), "has_pet": False, "sequence": [],
                   "excluded": reason,
                   "meaningful_cts": [s["ct_folder"] for s in usable]},
                  open(meta_path, "w"), indent=2)

    if len(usable) < min_ct:
        _write_stub("too_few_meaningful_ct")
        print(f"  [studies] {uuid} ({hasta_no}): {len(usable)} reliable studies "
              f"(<{min_ct}) - EXCLUDED (no usable CT); stub meta written.")
        return {"uuid": uuid, "ok": False, "excluded": "too_few_meaningful_ct",
                "n_studies": len(usable)}

    if max_ct and len(usable) > max_ct:
        _write_stub("multi_study")
        print(f"  [studies] {uuid} ({hasta_no}): {len(usable)} reliable studies "
              f"(>{max_ct}) - EXCLUDED (multi-phase); stub meta written.")
        return {"uuid": uuid, "ok": False, "excluded": "multi_study",
                "n_studies": len(usable)}

    if force and os.path.isdir(studies_dir):
        shutil.rmtree(studies_dir, ignore_errors=True)

    vdir = None
    if make_video:
        vsub = pre.get("VIDEO_SUBDIR_STEP2", "preprocessing")
        vdir = os.path.join(pre["VIDEO_DIR"], vsub, uuid)
        os.makedirs(vdir, exist_ok=True)

    drop_outliers = bool(pre.get("CONTOURS_DROP_LUNG_OUTLIERS", True))
    outlier_cc = float(pre.get("CONTOURS_LUNG_OUTLIER_CC", 10000.0))

    timeline, ref, built = [], None, 0
    for st in studies:
        ctf = st["ct_folder"]
        cdir = os.path.join(patient_dir, ctf)
        ct, grid = _study_load_ct_volume(os.path.join(cdir, "DCMData"))
        if ct is None:
            print(f"  [studies] {uuid}: {ctf} contoured+planned but no exported "
                  f"DICOM - skipped (no gap).")
            continue
        (origin, spacing, shape) = grid
        built += 1
        sname = f"{built:02d}_{st['date'] or 'nodate'}_{ctf.replace('1~', '')}"
        sdir = os.path.join(studies_dir, sname)

        rois = _rasterize_contours(
            cdir, os.path.join(cdir, "contournames"), os.path.join(cdir, "info"),
            ct_grid=grid, lung_outlier_threshold_cc=outlier_cc,
            drop_lung_outliers=drop_outliers)
        masks = {r: rois[r] for r in _STUDY_ROIS
                 if isinstance(rois.get(r), np.ndarray) and rois[r].sum() > 0}
        if ref is None or (ref.get("gtv") is None and masks.get("gtv") is not None):
            ref = {"ct": ct, "grid": grid, "gtv": masks.get("gtv"), "ct_folder": ctf}

        plans = _study_plans_for_ct(patient_dir, ctf)
        sel = _study_select_dose(plans)
        tgt = masks.get("ptv", masks.get("gtv"))
        dose, dqa = (None, {})
        if sel is not None:
            try:
                dose, dqa = _study_register_dose(sel[1], grid, tgt)
            except Exception as e:
                print(f"  [studies] {uuid}/{sname}: dose reg failed ({e})")
        vox_cc = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        dvh = _study_dvh(dose, masks, vox_cc) if dose is not None else None

        for sub in ("CT", "Contours", "Dose", "DVH"):
            os.makedirs(os.path.join(sdir, sub), exist_ok=True)
        np.save(os.path.join(sdir, "CT", "whole_body.npy"), ct)
        json.dump({"origin": origin, "spacing": spacing, "shape": list(shape)},
                  open(os.path.join(sdir, "CT", "grid.json"), "w"), indent=2)
        for nm, m in masks.items():
            np.save(os.path.join(sdir, "Contours", f"{nm}_mask.npy"), m)
        if dose is not None:
            np.save(os.path.join(sdir, "Dose", "dose_regCT.npy"), dose)
            json.dump({"plan": sel[0], "is_total": bool(sel[2]),
                       "all_plans_on_ct": [p[0] for p in plans], **dqa},
                      open(os.path.join(sdir, "Dose", "dose_meta.json"), "w"), indent=2)
        if dvh is not None:
            json.dump(dvh, open(os.path.join(sdir, "DVH", "dvh.json"), "w"), indent=2)
        lung = (dvh or {}).get("structures", {}).get(
            "lung_minus_gtv", (dvh or {}).get("structures", {}).get("lung", {}))
        smeta = {"order": built, "ct_folder": ctf, "date": st["date"],
                 "is_total": bool(st["is_total"]), "plan": sel[0] if sel else None,
                 "rois": list(masks), "shape": list(shape),
                 "dose_max_cGy": dqa.get("max_cGy"), "ptv_cov50": dqa.get("ptv_cov50"),
                 "lung_MLD_cGy": lung.get("Dmean_cGy"), "lung_V20": lung.get("V20")}
        json.dump(smeta, open(os.path.join(sdir, "study_meta.json"), "w"), indent=2)
        timeline.append(smeta)
        if make_video:
            try:
                _study_video(ct, masks, dose, os.path.join(vdir, f"{sname}.mp4"),
                             f"{hasta_no} {sname}")
            except Exception as e:
                print(f"  [studies] {uuid}/{sname}: video failed ({e})")

    pet_meta = None
    ref_date = timeline[0]["date"] if timeline else ""
    pt_dir, pdate, pslices = _study_pick_pet(patient_dir, ref_date)
    if pt_dir is not None and ref is not None:
        try:
            arr, pqa = _study_register_pet(ref["ct"], ref["grid"], ref["gtv"], pt_dir)
        except Exception as e:
            arr, pqa = None, {}
            print(f"  [studies] {uuid}: PET reg failed ({e})")
        if arr is not None:
            npt = len(glob.glob(os.path.join(patient_dir, "1~PT*")))
            pname = f"00_{pdate or 'nodate'}_PET"
            pdir = os.path.join(studies_dir, pname, "PET")
            os.makedirs(pdir, exist_ok=True)
            np.save(os.path.join(pdir, "whole_body_regCT.npy"), arr)
            pet_meta = {"order": 0, "modality": "pet",
                        "pet_folder": os.path.basename(pt_dir), "date": pdate,
                        "n_pet_series": npt, "anchored_to": ref.get("ct_folder"), **pqa}
            json.dump(pet_meta, open(os.path.join(studies_dir, pname,
                                                  "study_meta.json"), "w"), indent=2)
            if make_video:
                try:
                    _study_pet_video(ref["ct"], ref["gtv"], arr,
                                     os.path.join(vdir, f"{pname}.mp4"),
                                     f"{hasta_no} {pname}")
                except Exception as e:
                    print(f"  [studies] {uuid}: PET video failed ({e})")

    for mkey, mcfg in _get_modalities(cfg).items():
        if mcfg.get("type") != "passthrough" or not mcfg.get("enabled", True):
            continue
        msrc = _resolve_source(mcfg.get("source", ""), row, cfg["RAW_DIR"])
        mdst = os.path.join(out_root, mcfg.get("dest_subdir", mkey))
        if not os.path.isdir(msrc):
            continue
        preserve = bool(mcfg.get("preserve_relative", False))
        os.makedirs(mdst, exist_ok=True)
        for pat in mcfg.get("patterns", []):
            for path in glob.glob(os.path.join(msrc, pat)):
                if not os.path.isfile(path):
                    continue
                if preserve:
                    dst = os.path.join(mdst, os.path.relpath(path, msrc))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                else:
                    dst = os.path.join(mdst, os.path.basename(path))
                try:
                    shutil.copy2(path, dst)
                except Exception:
                    pass

    seq = ([pet_meta] if pet_meta else []) + timeline
    static = {}
    for key, rel in (("clinical", os.path.join("Clinical", "clinical_features.json")),
                     ("rt_tabular", os.path.join("RT-Tabular", "rt_tabular_features.json")),
                     ("plan_params", os.path.join("PlanParams", "plan_features.json"))):
        static[key] = {"path": rel, "present": os.path.exists(os.path.join(out_root, rel))}
    os.makedirs(out_root, exist_ok=True)
    json.dump({"hasta_no": hasta_no, "uuid": uuid, "n_studies": len(timeline),
               "has_pet": pet_meta is not None, "sequence": seq, "static": static},
              open(meta_path, "w"), indent=2)
    pr = (pet_meta or {}).get("qa_ratio")
    print(f"  [studies] {uuid} ({hasta_no}): {len(timeline)} CT studies"
          f"{f' + PET(qa={pr})' if pet_meta else ', no PET'}.")
    return {"uuid": uuid, "ok": True, "n_studies": len(timeline),
            "has_pet": pet_meta is not None}


def _step_2_build_studies(cfg):
    input_dir = cfg["INPUTS_DIR"]
    pre = cfg.get("PREPROCESS", {})
    print("\n" + "=" * 60)
    print("  STEP 2 - Build per-study timeline (Cohort/<uuid>/studies/ + meta)")
    print("=" * 60)
    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    if not table_exists(manifest_path):
        print(f"Error: patient manifest not found ({manifest_path}). Run step 0 first.")
        return
    manifest = _filter_manifest_only(
        read_table(manifest_path).fillna("").to_dict("records"), cfg)
    force = bool(pre.get("FORCE_REPROCESS", False))
    make_video = bool(pre.get("GENERATE_ALIGNMENT_VIDEO", False))
    workers = int(pre.get("WORKERS", 1) or 1)
    print(f"Patients: {len(manifest)}  workers={workers}  "
          f"force={force}  video={make_video}")
    _run_per_patient(
        manifest,
        lambda r: _build_patient_studies(r, cfg, make_video=make_video, force=force),
        workers, "studies")
    n_meta = n_pet = 0
    proc = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    for r in manifest:
        mp = os.path.join(proc, str(r.get("uuid", "")), "patient_meta.json")
        if os.path.exists(mp):
            n_meta += 1
            try:
                if json.load(open(mp)).get("has_pet"):
                    n_pet += 1
            except Exception:
                pass
    print(f"\nStep 2 done. patient_meta.json: {n_meta}/{len(manifest)}  "
          f"(with PET t0: {n_pet}).")


def _render_study_videos(uuid, cfg, sub):
    pre = cfg.get("PREPROCESS", {})
    proc = os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"]["PROCESSED"], uuid)
    meta_p = os.path.join(proc, "patient_meta.json")
    if not os.path.exists(meta_p):
        return
    meta = json.load(open(meta_p))
    base = (pre.get("VIDEO_DIR") or "").strip() or \
        os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"].get("VIDEOS", "Videos"))
    vdir = os.path.join(base, sub, uuid) if sub else os.path.join(base, uuid)
    os.makedirs(vdir, exist_ok=True)
    studies_dir = os.path.join(proc, "studies")
    hasta = meta.get("hasta_no", uuid)

    ref_ct = ref_gtv = None
    for s in meta.get("sequence", []):
        if s.get("modality") == "pet":
            continue
        sname = f"{s['order']:02d}_{s.get('date') or 'nodate'}_{s['ct_folder'].replace('1~', '')}"
        sdir = os.path.join(studies_dir, sname)
        ctf = os.path.join(sdir, "CT", "whole_body.npy")
        if not os.path.exists(ctf):
            continue
        ct = np.load(ctf).astype(np.float32)
        masks = {}
        for nm in _STUDY_ROIS:
            mf = os.path.join(sdir, "Contours", f"{nm}_mask.npy")
            if os.path.exists(mf):
                m = np.load(mf)
                if m.sum() > 0:
                    masks[nm] = m
        df = os.path.join(sdir, "Dose", "dose_regCT.npy")
        dose = np.load(df).astype(np.float32) if os.path.exists(df) else None
        if ref_ct is None or (ref_gtv is None and masks.get("gtv") is not None):
            ref_ct, ref_gtv = ct, masks.get("gtv")
        try:
            _study_video(ct, masks, dose, os.path.join(vdir, f"{sname}.mp4"),
                         f"{hasta} {sname}")
        except Exception as e:
            print(f"  [video] {uuid}/{sname} failed ({e})")

    for s in meta.get("sequence", []):
        if s.get("modality") != "pet" or ref_ct is None:
            continue
        pname = f"00_{s.get('date') or 'nodate'}_PET"
        pf = os.path.join(studies_dir, pname, "PET", "whole_body_regCT.npy")
        if os.path.exists(pf):
            try:
                _study_pet_video(ref_ct, ref_gtv, np.load(pf).astype(np.float32),
                                 os.path.join(vdir, f"{pname}.mp4"), f"{hasta} {pname}")
            except Exception as e:
                print(f"  [video] {uuid}/{pname} failed ({e})")


def step_3_segment(cfg):
    if bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True)):
        return _step_2_build_studies(cfg)
    ct_seg_enabled = bool(cfg.get("PREPROCESS", {}).get(
        "CT_TOTALSEGMENTATOR_ENABLED", False))
    cv2, scipy_ndimage, torch, nib, sitk, totalsegmentator = _lazy_import_heavy(
        with_totalsegmentator=ct_seg_enabled)

    print("\n" + "=" * 60)
    print("  STEP 2 - Segment (read Raw/ -> write Processed/<uuid>/<Modality>/)")
    print("=" * 60)
    _seg_label = ("ENABLED" if ct_seg_enabled else
                  "disabled - CT volumes saved as raw whole_body.npy only")
    print(f"CT TotalSegmentator: {_seg_label}")

    raw_dir = cfg["RAW_DIR"]
    input_dir = cfg["INPUTS_DIR"]
    processed_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    wc = cfg["PREPROCESS"]["WINDOW_CENTER"]
    ww = cfg["PREPROCESS"]["WINDOW_WIDTH"]

    if torch.cuda.is_available():
        device = "gpu"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    def _free_gpu():
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_dicom_volume(folder_path):
        if not folder_path or not os.path.exists(folder_path):
            return None, None
        files = [os.path.join(folder_path, s)
                 for s in os.listdir(folder_path) if s.lower().endswith(".dcm")]
        if not files:
            return None, None

        headers = []
        for fp in files:
            try:
                ds = _read_dicom(fp, stop_before_pixels=True)
                z = float(ds.ImagePositionPatient[2])
            except Exception as e:
                print(f"      [!] Skipping unreadable header {os.path.basename(fp)}: {e}")
                continue
            headers.append((z, fp, ds))
        if not headers:
            print(f"      [!!!] No readable DICOM headers in {folder_path}")
            return None, None
        headers.sort(key=lambda x: x[0])

        ds0 = headers[0][2]
        try:
            z_sp = abs(headers[0][0] - headers[1][0])
        except (IndexError, AttributeError):
            z_sp = ds0.SliceThickness
        ps = ds0.PixelSpacing
        spacing = (float(z_sp), float(ps[0]), float(ps[1]))

        slices = []
        for _, fp, ds in headers:
            try:
                px = _read_dicom(fp).pixel_array
            except Exception as e:
                print(f"      [!!!] Corrupt pixel data in "
                      f"{os.path.basename(fp)}: {e}")
                return None, None
            slc = (px * float(getattr(ds, "RescaleSlope", 1.0))
                   + float(getattr(ds, "RescaleIntercept", 0.0)))
            slices.append(slc)
        try:
            vol = np.stack(slices)
        except Exception as e:
            print(f"      [!!!] Stack failed (inconsistent slice shapes?): {e}")
            return None, None
        return np.ascontiguousarray(vol, dtype=np.float32), spacing

    def save_npy(volume, spacing, path):
        np.save(path, volume)
        mp = path.replace(".npy", "_spacing.txt")
        with open(mp, "w") as f:
            f.write(f"{spacing[0]},{spacing[1]},{spacing[2]}")

    def segment_ct(volume, spacing, save_dir):
        try:
            save_npy(volume, spacing,
                     os.path.join(save_dir, "whole_body.npy"))

            if not ct_seg_enabled:
                return None, None

            affine = np.eye(4)
            affine[0, 0], affine[1, 1], affine[2, 2] = spacing[2], spacing[1], spacing[0]
            vol_t = volume.transpose(2, 1, 0)
            nifti = nib.Nifti1Image(vol_t, affine)

            print("      ... Lungs + Vessels")
            lobes = ['lung_upper_lobe_left', 'lung_lower_lobe_left',
                     'lung_upper_lobe_right', 'lung_middle_lobe_right',
                     'lung_lower_lobe_right']
            seg_lung = totalsegmentator(nifti, output_type="nifti_img",
                                        roi_subset=lobes, fast=False,
                                        device=device, quiet=True)
            mask = seg_lung.get_fdata().transpose(2, 1, 0) > 0
            del seg_lung
            _free_gpu()
            if not np.any(mask):
                print(f"      [!!!] No lungs found")
                return None, "no lungs detected"

            vessel_mask_full = None
            try:
                seg_v = totalsegmentator(nifti, output_type="nifti_img",
                                         task="lung_vessels", fast=False,
                                         device=device, quiet=True)
                vessel_mask_full = seg_v.get_fdata().transpose(2, 1, 0) > 0
                del seg_v
                _free_gpu()
            except Exception as e:
                print(f"      [!] Vessel subtraction skipped: {e}")

            print("      ... Lung nodules (tumor)")
            tumor_mask_full = None
            try:
                seg_t = totalsegmentator(nifti, output_type="nifti_img",
                                         task="lung_nodules", fast=False,
                                         device=device, quiet=True)
                tumor_mask_full = seg_t.get_fdata().transpose(2, 1, 0) == 2
                del seg_t
                _free_gpu()
            except Exception as e:
                print(f"      [!] Tumor segmentation skipped: {e}")

            del nifti, vol_t

            coords = np.argwhere(mask)
            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0)
            y0, x0 = max(0, y0 - 10), max(0, x0 - 10)
            y1 = min(volume.shape[1], y1 + 10)
            x1 = min(volume.shape[2], x1 + 10)

            vc = volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
            mc = mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
            MIN_DIM = 16
            if vc.size == 0 or any(d < MIN_DIM for d in vc.shape):
                print(f"      [!!!] Degenerate lung crop shape={vc.shape}")
                return None, f"degenerate lung crop shape={vc.shape}"

            vc[~mc] = -1000.0
            save_npy(vc, spacing, os.path.join(save_dir, "lung_only.npy"))

            if vessel_mask_full is not None:
                mvc = vessel_mask_full[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
                vp = vc.copy()
                vp[mvc] = -1000.0
                save_npy(vp, spacing,
                         os.path.join(save_dir, "lung_only_no_vessels.npy"))

            if tumor_mask_full is not None and np.any(tumor_mask_full):
                lung_dims = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
                pad = tuple(max(1, int(round(0.10 * d))) for d in lung_dims)
                t_coords = np.argwhere(tumor_mask_full)
                tz0, ty0, tx0 = t_coords.min(axis=0)
                tz1, ty1, tx1 = t_coords.max(axis=0)
                tz0 = max(0, tz0 - pad[0])
                ty0 = max(0, ty0 - pad[1])
                tx0 = max(0, tx0 - pad[2])
                tz1 = min(volume.shape[0] - 1, tz1 + pad[0])
                ty1 = min(volume.shape[1] - 1, ty1 + pad[1])
                tx1 = min(volume.shape[2] - 1, tx1 + pad[2])
                vt = volume[tz0:tz1 + 1, ty0:ty1 + 1, tx0:tx1 + 1].copy()
                mt = tumor_mask_full[tz0:tz1 + 1, ty0:ty1 + 1, tx0:tx1 + 1]
                if vt.size > 0 and all(d >= 4 for d in vt.shape):
                    vt[~mt] = -1000.0
                    save_npy(vt, spacing,
                             os.path.join(save_dir, "tumor_only.npy"))
                else:
                    print(f"      [!] Degenerate tumor crop shape={vt.shape}; skipped")
            else:
                print(f"      [!] No tumor detected by lung_nodules; skipped")

            return vc, None
        except Exception as e:
            print(f"      [!!!] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)

    def _modality_dir_is_complete(tgt, mtype="ct"):
        if mtype == "ct":
            need = (("whole_body.npy", "lung_only.npy") if ct_seg_enabled
                    else ("whole_body.npy",))
        elif mtype == "pet":
            need = ("whole_body.npy",)
        elif mtype == "contours":
            gtv_path = os.path.join(tgt, "gtv_mask.npy")
            if not os.path.exists(gtv_path):
                return False
            try:
                return bool(np.load(gtv_path, mmap_mode="r").any())
            except Exception:
                return False
        elif mtype == "dose":
            need = ("dose_volume.npy",)
        else:
            return os.path.isdir(tgt) and bool(os.listdir(tgt))
        return all(os.path.exists(os.path.join(tgt, fn)) for fn in need)

    def segment_pet(volume, spacing, save_dir):
        try:
            save_npy(volume, spacing, os.path.join(save_dir, "whole_body.npy"))
            return None
        except Exception as e:
            return str(e)

    def passthrough_copy(modality_cfg, source_dir, dest_dir):
        if not os.path.isdir(source_dir):
            return 0, "source dir missing"
        patterns = modality_cfg.get("patterns", [])
        preserve = bool(modality_cfg.get("preserve_relative", False))
        os.makedirs(dest_dir, exist_ok=True)
        n = 0
        for pat in patterns:
            for path in glob.glob(os.path.join(source_dir, pat)):
                if not os.path.isfile(path):
                    continue
                if preserve:
                    rel = os.path.relpath(path, source_dir)
                    dst = os.path.join(dest_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                else:
                    dst = os.path.join(dest_dir, os.path.basename(path))
                shutil.copy2(path, dst)
                n += 1
        return n, (None if n > 0 else "no files matched any pattern")

    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    if not table_exists(manifest_path):
        print(f"Error: patient manifest not found ({manifest_path}). "
              f"Run step 0 first.")
        return
    manifest = read_table(manifest_path).fillna("").to_dict("records")
    manifest = _filter_manifest_only(manifest, cfg)
    print(f"Manifest: {len(manifest)} patients.")

    st_path, patient_status = _load_seg_status(input_dir)
    n_done = sum(1 for v in patient_status.values() if v["status"] == "complete")
    n_fail = sum(1 for v in patient_status.values() if v["status"] != "complete")
    if patient_status:
        print(f"Status: {n_done} complete, {n_fail} failed (both skipped - no retry).")
    force_reprocess = bool(cfg.get("PREPROCESS", {}).get("FORCE_REPROCESS", False))
    if force_reprocess:
        print("FORCE_REPROCESS=true - ignoring prior status; re-segmenting all "
              "selected patients.")

    modalities = _get_modalities(cfg)
    enabled = {k: v for k, v in modalities.items() if v.get("enabled", True)}
    if not enabled:
        print("No modalities enabled - check PREPROCESS.MODALITIES in config.")
        return
    print(f"Enabled modalities: {list(enabled.keys())}")

    for modality_key, modality_cfg in enabled.items():
        mtype = modality_cfg.get("type", "ct")
        print(f"\n  --- Modality: {modality_cfg.get('label', modality_key)} "
              f"({modality_key}, type={mtype}) ---")

        has_flag = f"has_{modality_key}"
        avail = [r for r in manifest if r.get(has_flag, False)]
        print(f"  Source template: {modality_cfg['source']}")
        print(f"  Dest subdir    : {modality_cfg['dest_subdir']}")
        print(f"  Patients       : {len(avail)} (with this modality available)")

        for i, row in enumerate(avail, 1):
            uuid = str(row["uuid"])
            tgt = os.path.join(processed_dir, uuid, modality_cfg["dest_subdir"])
            status_key = f"{uuid}__{modality_key}"

            if not force_reprocess and status_key in patient_status:
                rec = patient_status[status_key]
                if rec["status"] == "complete":
                    print(f"  [{i}] {uuid}/{modality_key}: complete, skip")
                else:
                    print(f"  [{i}] {uuid}/{modality_key}: previously failed "
                          f"({rec['error']}), skip")
                continue

            if not force_reprocess and _modality_dir_is_complete(tgt, mtype):
                print(f"  [{i}] {uuid}/{modality_key}: complete on disk, backfilling")
                _mark_seg_status(patient_status, st_path, uuid,
                                 row.get("rp_status", ""), modality_key, "complete")
                continue

            src = _resolve_source(modality_cfg["source"], row, raw_dir)
            if not os.path.exists(src):
                _mark_seg_status(patient_status, st_path, uuid,
                                 row.get("rp_status", ""), modality_key,
                                 "failed", f"source missing: {src}")
                print(f"  [{i}] {uuid}/{modality_key}: source missing {src}")
                continue

            err: str | None = None
            if mtype in ("ct", "pet"):
                volume, spacing = load_dicom_volume(src)
                if volume is None:
                    _mark_seg_status(patient_status, st_path, uuid,
                                     row.get("rp_status", ""), modality_key,
                                     "failed",
                                     f"DICOM load failed (missing/unreadable/"
                                     f"corrupt pixel data) in {src}")
                    print(f"  [{i}] {uuid}/{modality_key}: DICOM load failed "
                          f"- see warnings above; marked failed and skipped.")
                    continue
                print(f"  [{i}] {uuid}/{modality_key}: processing "
                      f"(shape={volume.shape})...")
                os.makedirs(tgt, exist_ok=True)
                if mtype == "ct":
                    _, err = segment_ct(volume, spacing, tgt)
                else:
                    err = segment_pet(volume, spacing, tgt)
                del volume
            elif mtype == "passthrough":
                print(f"  [{i}] {uuid}/{modality_key}: copying...")
                n_copied, err = passthrough_copy(modality_cfg, src, tgt)
                print(f"       -> {n_copied} file(s) copied")
            elif mtype == "contours":
                print(f"  [{i}] {uuid}/{modality_key}: rasterizing contours...")
                try:
                    os.makedirs(tgt, exist_ok=True)
                    cn_path = os.path.join(src, "contournames")
                    info_path = os.path.join(src, "info")
                    ct_dicom_dir = os.path.join(src, "DCMData")
                    ct_grid = _load_dicom_grid_from_dir(ct_dicom_dir)
                    if ct_grid is None:
                        print(f"       [!] CT grid not found at {ct_dicom_dir}; "
                              f"using estimated grid")
                    _outlier_cc = float(cfg.get("PREPROCESS", {}).get(
                        "CONTOURS_LUNG_OUTLIER_CC", 10000.0))
                    _drop_outliers = bool(cfg.get("PREPROCESS", {}).get(
                        "CONTOURS_DROP_LUNG_OUTLIERS", False))
                    masks = _rasterize_contours(
                        src, cn_path, info_path, ct_grid=ct_grid,
                        lung_outlier_threshold_cc=_outlier_cc,
                        drop_lung_outliers=_drop_outliers)

                    try:
                        _idx_to_name_log = _parse_contournames(cn_path)
                        import unicodedata as _ud2
                        def _fold2(s):
                            k = _ud2.normalize("NFKD", s).replace(
                                "ı", "i").replace("İ", "i")
                            return "".join(c for c in k
                                           if not _ud2.combining(c)).lower()

                        _tm_re2 = re.compile(r"\btm\b")
                        _ac_re2 = re.compile(r"\bac\b")
                        _sc_re2 = re.compile(r"\bsc\b")
                        _sc_end_re2 = re.compile(r"sc$")
                        _kit_re2 = re.compile(r"\bkit")
                        _ms_dot_re2 = re.compile(r"\bm\.s\b")
                        _gtv_kws2 = ("gtv", "tumor", "tumör", "target vol")
                        _eso_kws2 = ("ozefag", "ozafag", "osefag",
                                     "esofag", "esophag", "ozfag")
                        _liver_kws2 = ("karaciger", "karaci", "katraci",
                                       "liver", "hepar")
                        _breast_kws2 = ("meme", "breast")

                        def _classify2(name: str) -> str:
                            nl = _fold2(name);
                            s = nl.strip()
                            if _is_ptv_name(nl): return "ptv"
                            if ("ptv" in nl) or ("boost" in nl): return ""
                            if "lun" in nl: return "lung"
                            if "ctv" in nl: return "ctv"
                            if any(k in nl for k in _gtv_kws2): return "gtv"
                            if (any(k in nl for k in ("kitle", "lezyon", "nodul"))
                                    or _tm_re2.search(nl) or _kit_re2.search(nl)): return "gtv"
                            if ("akc" in nl) or _ac_re2.search(nl): return "lung"
                            if ("kalp" in nl) or ("heart" in nl): return "heart"
                            if any(k in nl for k in _eso_kws2) or s.startswith("oz"): return "esophagus"
                            if (("ms" in nl) or ("medulla" in nl) or ("spi" in nl)
                                    or _sc_re2.search(nl) or _sc_end_re2.search(s)
                                    or _ms_dot_re2.search(nl)): return "spinal_cord"
                            if any(k in nl for k in _liver_kws2): return "liver"
                            if any(k in nl for k in _breast_kws2): return "breast"
                            return ""

                        _names_log = list(_idx_to_name_log.items())
                        _lung_match = [(i, n) for i, n in _names_log if _classify2(n) == "lung"]
                        _gtv_match = [(i, n) for i, n in _names_log if _classify2(n) == "gtv"]
                        _ctv_match = [(i, n) for i, n in _names_log if _classify2(n) == "ctv"]
                        _ptv_match = [(i, n) for i, n in _names_log if _classify2(n) == "ptv"]
                        _heart_match = [(i, n) for i, n in _names_log if _classify2(n) == "heart"]
                        _eso_match = [(i, n) for i, n in _names_log if _classify2(n) == "esophagus"]
                        _sc_match = [(i, n) for i, n in _names_log if _classify2(n) == "spinal_cord"]
                        _liver_match = [(i, n) for i, n in _names_log if _classify2(n) == "liver"]
                        _breast_match = [(i, n) for i, n in _names_log if _classify2(n) == "breast"]
                        _unmatched = [(i, n) for i, n in _names_log
                                      if _classify2(n) == ""
                                      and n.strip().lower() not in ("", "general")]
                    except Exception:
                        _names_log = []
                        _lung_match = _gtv_match = _ctv_match = _ptv_match = []
                        _heart_match = _eso_match = _sc_match = _liver_match = []
                        _breast_match = []
                        _unmatched = []

                    def _safe_save(fname, arr):
                        try:
                            os.makedirs(tgt, exist_ok=True)
                            np.save(os.path.join(tgt, fname), arr)
                            return None
                        except Exception as _se:
                            return f"{fname}: {_se}"

                    _save_errs = []
                    _mask_specs = [
                        ("gtv_mask.npy", "gtv"),
                        ("ctv_mask.npy", "ctv"),
                        ("lung_mask.npy", "lung"),
                        ("ptv_mask.npy", "ptv"),
                        ("heart_mask.npy", "heart"),
                        ("esophagus_mask.npy", "esophagus"),
                        ("spinal_cord_mask.npy", "spinal_cord"),
                        ("liver_mask.npy", "liver"),
                        ("breast_mask.npy", "breast"),
                    ]
                    for _fn, _key in _mask_specs:
                        _arr = masks.get(_key)
                        if _arr is None:
                            continue
                        e = _safe_save(_fn, _arr.astype(np.uint8))
                        if e:
                            _save_errs.append(e)
                    if _save_errs:
                        print(f"       [!] save failures: {_save_errs}")
                    import json as _json
                    if ct_grid is not None:
                        _origin, _spacing, _shape = ct_grid
                        _meta = {
                            "origin": list(_origin),
                            "spacing": list(_spacing),
                            "shape": list(_shape),
                            "grid_source": ct_dicom_dir,
                        }
                    else:
                        _meta = {
                            "origin": None,
                            "spacing": None,
                            "shape": list(masks["gtv"].shape),
                            "grid_source": "estimated",
                        }
                    _gtv_hu = None; _align_ok = None
                    _ct_npy = os.path.join(processed_dir, uuid, "CT-Plan",
                                           "whole_body.npy")
                    if (os.path.exists(_ct_npy) and masks.get("gtv") is not None
                            and masks["gtv"].any()):
                        try:
                            _ctvol = np.load(_ct_npy)
                            if _ctvol.shape == masks["gtv"].shape:
                                _gtv_hu = float(_ctvol[masks["gtv"] > 0].mean())
                                _align_ok = bool(_gtv_hu > -300)
                        except Exception:
                            pass
                    _meta["gtv_mean_hu"] = _gtv_hu
                    _meta["align_ok"] = _align_ok
                    with open(os.path.join(tgt, "contours_meta.txt"), "w") as _mf:
                        _json.dump(_meta, _mf, indent=2)
                    _gtv_n = int(masks['gtv'].sum())
                    _ctv_n = int(masks.get('ctv', np.zeros((1,))).sum())
                    _lung_n = int(masks['lung'].sum())
                    _ptv_n = int(masks['ptv'].sum())
                    _heart_n = int(masks.get('heart', np.zeros((1,))).sum())
                    _eso_n = int(masks.get('esophagus', np.zeros((1,))).sum())
                    _sc_n = int(masks.get('spinal_cord', np.zeros((1,))).sum())
                    _liver_n = int(masks.get('liver', np.zeros((1,))).sum())
                    _breast_n = int(masks.get('breast', np.zeros((1,))).sum())
                    print(f"       -> GTV: {_gtv_n}, CTV: {_ctv_n}, "
                          f"Lung: {_lung_n}, PTV: {_ptv_n}  "
                          f"| Heart: {_heart_n}, Eso: {_eso_n}, "
                          f"SC: {_sc_n}, Liver: {_liver_n}, Breast: {_breast_n}")

                    _vox_mm3 = 1.0
                    if ct_grid is not None:
                        _vox_mm3 = (ct_grid[1][0] * ct_grid[1][1]
                                    * ct_grid[1][2])
                    _cc = lambda n: n * _vox_mm3 / 1000.0
                    _gtv_cc, _lung_cc, _ptv_cc = _cc(_gtv_n), _cc(_lung_n), _cc(_ptv_n)
                    _ctv_cc = _cc(_ctv_n)
                    _heart_cc, _eso_cc = _cc(_heart_n), _cc(_eso_n)
                    _sc_cc, _liver_cc, _breast_cc = _cc(_sc_n), _cc(_liver_n), _cc(_breast_n)
                    _per_struct_cc = masks.get("_per_struct_cc", {})
                    _lung_outliers = masks.get("_lung_outlier_idxs", set())

                    _flags = []
                    if _gtv_n == 0:  _flags.append("no_gtv")
                    if _lung_n == 0: _flags.append("no_lung")
                    if _ptv_n == 0:  _flags.append("no_ptv")
                    if 0 < _lung_cc < 1000:  _flags.append("lung_small")
                    if _lung_cc > 8000:      _flags.append("lung_large")
                    if _gtv_cc > 1000:       _flags.append("gtv_huge")
                    if 0 < _ptv_cc < 20:     _flags.append("ptv_small")
                    if _ptv_cc > 3000:       _flags.append("ptv_huge")
                    if _lung_outliers:
                        _flags.append("lung_outlier")
                    if not _flags:
                        _flags = ["ok"]

                    _log_path = os.path.join(
                        input_dir, "contours_annotation_log.csv")
                    _log_new = not os.path.exists(_log_path)
                    try:
                        import csv as _csv2
                        with open(_log_path, "a", newline="",
                                  encoding="utf-8") as _lf:
                            _w = _csv2.writer(_lf)
                            _ps_str = "; ".join(
                                f"{i}:{g}:{n}={cc:.1f}cc"
                                for i, (g, n, cc) in sorted(
                                    _per_struct_cc.items()))
                            _lo_str = "; ".join(
                                f"{i}:{_per_struct_cc[i][1]}={_per_struct_cc[i][2]:.1f}cc"
                                for i in sorted(_lung_outliers))
                            if _log_new:
                                _w.writerow([
                                    "uuid", "patient_name", "rp_status",
                                    "n_structs", "all_struct_names",
                                    "matched_gtv", "matched_ctv",
                                    "matched_lung", "matched_ptv",
                                    "matched_heart", "matched_esophagus",
                                    "matched_spinal_cord", "matched_liver",
                                    "matched_breast",
                                    "unmatched",
                                    "per_struct_cc", "lung_outliers",
                                    "gtv_voxels", "ctv_voxels",
                                    "lung_voxels", "ptv_voxels",
                                    "heart_voxels", "esophagus_voxels",
                                    "spinal_cord_voxels", "liver_voxels",
                                    "breast_voxels",
                                    "gtv_cc", "ctv_cc", "lung_cc", "ptv_cc",
                                    "heart_cc", "esophagus_cc",
                                    "spinal_cord_cc", "liver_cc", "breast_cc",
                                    "ct_grid_source",
                                    "voxel_mm3", "flags",
                                ])
                            _w.writerow([
                                uuid,
                                row.get("patient_name", ""),
                                row.get("rp_status", ""),
                                len(_names_log),
                                "; ".join(n for _, n in _names_log),
                                "; ".join(f"{i}:{n}" for i, n in _gtv_match),
                                "; ".join(f"{i}:{n}" for i, n in _ctv_match),
                                "; ".join(f"{i}:{n}" for i, n in _lung_match),
                                "; ".join(f"{i}:{n}" for i, n in _ptv_match),
                                "; ".join(f"{i}:{n}" for i, n in _heart_match),
                                "; ".join(f"{i}:{n}" for i, n in _eso_match),
                                "; ".join(f"{i}:{n}" for i, n in _sc_match),
                                "; ".join(f"{i}:{n}" for i, n in _liver_match),
                                "; ".join(f"{i}:{n}" for i, n in _breast_match),
                                "; ".join(f"{i}:{n}" for i, n in _unmatched),
                                _ps_str, _lo_str,
                                _gtv_n, _ctv_n, _lung_n, _ptv_n,
                                _heart_n, _eso_n, _sc_n, _liver_n, _breast_n,
                                f"{_gtv_cc:.1f}", f"{_ctv_cc:.1f}",
                                f"{_lung_cc:.1f}", f"{_ptv_cc:.1f}",
                                f"{_heart_cc:.1f}", f"{_eso_cc:.1f}",
                                f"{_sc_cc:.1f}", f"{_liver_cc:.1f}",
                                f"{_breast_cc:.1f}",
                                "CT-DICOM" if ct_grid is not None else "estimated",
                                f"{_vox_mm3:.3f}",
                                "|".join(_flags),
                            ])
                    except Exception as _le:
                        print(f"       [!] annotation-log write failed: {_le}")

                    def _disp(ms):
                        return ", ".join(n for _, n in ms) or "-"

                    print(f"       structures -> "
                          f"lung: [{_disp(_lung_match)}]  "
                          f"gtv: [{_disp(_gtv_match)}]  "
                          f"ctv: [{_disp(_ctv_match)}]  "
                          f"ptv: [{_disp(_ptv_match)}]")
                    if any((_heart_match, _eso_match, _sc_match,
                            _liver_match, _breast_match)):
                        print(f"       OARs       -> "
                              f"heart: [{_disp(_heart_match)}]  "
                              f"eso: [{_disp(_eso_match)}]  "
                              f"sc: [{_disp(_sc_match)}]  "
                              f"liver: [{_disp(_liver_match)}]  "
                              f"breast: [{_disp(_breast_match)}]")
                    print(f"       unmatched: "
                          f"[{', '.join(n for _, n in _unmatched) or '-'}]")
                    if _lung_outliers:
                        _lo_lines = ", ".join(
                            f"idx {i} ({_per_struct_cc[i][1]!r}) = "
                            f"{_per_struct_cc[i][2]:.0f}cc"
                            for i in sorted(_lung_outliers))
                        _drop_lbl = ("DROPPED from union" if _drop_outliers
                                     else "kept in union - set "
                                          "PREPROCESS.CONTOURS_DROP_LUNG_OUTLIERS"
                                          "=true to exclude")
                        print(f"       [!] lung outliers ({_drop_lbl}): "
                              f"{_lo_lines}")
                    if _flags != ["ok"]:
                        print(f"       [!] annotation flags: {_flags}")
                    if _gtv_n == 0 and _lung_n == 0 and _ptv_n == 0:
                        try:
                            import unicodedata as _ud
                            def _fold(s):
                                k = _ud.normalize("NFKD", s).replace(
                                    "ı", "i").replace("İ", "i")
                                return "".join(c for c in k
                                               if not _ud.combining(c)).lower()

                            _idx_to_name = _parse_contournames(cn_path)
                            _names = list(_idx_to_name.items())
                            _tm_re3 = re.compile(r"\btm\b")
                            _ac_re3 = re.compile(r"\bac\b")
                            _sc_re3 = re.compile(r"\bsc\b")
                            _sc_end_re3 = re.compile(r"sc$")
                            _kit_re3 = re.compile(r"\bkit")
                            _ms_dot_re3 = re.compile(r"\bm\.s\b")
                            _eso_kws3 = ("ozefag", "ozafag", "osefag",
                                         "esofag", "esophag", "ozfag")
                            _liver_kws3 = ("karaciger", "karaci", "katraci",
                                           "liver", "hepar")
                            _breast_kws3 = ("meme", "breast")

                            def _cls3(name):
                                nl = _fold(name);
                                s = nl.strip()
                                if _is_ptv_name(nl): return "ptv"
                                if ("ptv" in nl) or ("boost" in nl): return ""
                                if "lun" in nl: return "lung"
                                if "ctv" in nl: return "ctv"
                                if any(k in nl for k in ("gtv", "tumor", "tumör", "target vol")): return "gtv"
                                if (any(k in nl for k in ("kitle", "lezyon", "nodul"))
                                        or _tm_re3.search(nl) or _kit_re3.search(nl)): return "gtv"
                                if ("akc" in nl) or _ac_re3.search(nl): return "lung"
                                if ("kalp" in nl) or ("heart" in nl): return "heart"
                                if any(k in nl for k in _eso_kws3) or s.startswith("oz"): return "esophagus"
                                if (("ms" in nl) or ("medulla" in nl) or ("spi" in nl)
                                        or _sc_re3.search(nl) or _sc_end_re3.search(s)
                                        or _ms_dot_re3.search(nl)): return "spinal_cord"
                                if any(k in nl for k in _liver_kws3): return "liver"
                                if any(k in nl for k in _breast_kws3): return "breast"
                                return ""

                            _lung_idx = [i for i, n in _names if _cls3(n) == "lung"]
                            _gtv_idx = [i for i, n in _names if _cls3(n) == "gtv"]
                            _ptv_idx = [i for i, n in _names if _cls3(n) == "ptv"]
                            print(f"       [!] All three masks empty. "
                                  f"contournames: {len(_idx_to_name)} structs")
                            print(f"            structure names: "
                                  f"{[n for _, n in _names]}")
                            print(f"            matched gtv idx: {_gtv_idx}  "
                                  f"lung idx: {_lung_idx}  ptv idx: {_ptv_idx}")
                            import glob as _glb
                            _wcs = sorted(_glb.glob(os.path.join(src, "*.WC")))
                            print(f"            WC files: {len(_wcs)}")
                            if _wcs:
                                _z_vals = []
                                _idx_counts: dict[int, int] = {}
                                _xy_samples: list = []
                                for _wf in _wcs:
                                    _p = _parse_wc_file(_wf)
                                    if _p["z_mm"] is not None:
                                        _z_vals.append(_p["z_mm"])
                                    for si, pts in _p["contours"]:
                                        _idx_counts[si] = _idx_counts.get(si, 0) + 1
                                        if pts and len(_xy_samples) < 2:
                                            _xy_samples.append((si, pts[0]))
                                if _z_vals:
                                    print(f"            WC z_mm range: "
                                          f"[{min(_z_vals):.2f}, {max(_z_vals):.2f}]")
                                print(f"            WC contour-index counts: "
                                      f"{dict(sorted(_idx_counts.items()))}")
                                print(f"            sample WC (struct_idx, first xy): "
                                      f"{_xy_samples}")
                            if ct_grid is not None:
                                _o, _sp, _sh = ct_grid
                                _z_lo = _o[2]
                                _z_hi = _o[2] + _sp[0] * (_sh[0] - 1)
                                print(f"            CT grid origin={_o}  "
                                      f"spacing={_sp}  shape={_sh}")
                                print(f"            CT z range: "
                                      f"[{min(_z_lo, _z_hi):.2f}, "
                                      f"{max(_z_lo, _z_hi):.2f}]")
                                if _wcs:
                                    _sample_wc = _parse_wc_file(_wcs[0])
                                    print(f"            WC slice[0] origin: "
                                          f"{_sample_wc.get('origin')}  "
                                          f"orient: {_sample_wc.get('orientation')}")
                        except Exception as _e:
                            print(f"       [!] diagnostic dump failed: {_e}")
                except Exception as e:
                    err = str(e)
                    import traceback
                    traceback.print_exc()
            elif mtype == "dose":
                print(f"  [{i}] {uuid}/{modality_key}: parsing dose volume...")
                try:
                    os.makedirs(tgt, exist_ok=True)
                    chosen_dose, _on_ct1 = _select_dose_file(src, "CT1")
                    if chosen_dose is None:
                        err = f"no dose.1 found under {src}"
                    else:
                        if not _on_ct1:
                            print(f"       [!] no plan on contour CT (CT1); "
                                  f"using {os.path.basename(os.path.dirname(chosen_dose))}")
                        dose_arr, spacing, origin = _parse_dose_volume(chosen_dose)
                        np.save(os.path.join(tgt, "dose_volume.npy"), dose_arr)
                        import json
                        meta = {
                            "spacing": list(spacing),
                            "origin": list(origin),
                            "shape": list(dose_arr.shape),
                            "source": chosen_dose,
                        }
                        with open(os.path.join(tgt, "dose_volume_meta.txt"), "w") as mf:
                            json.dump(meta, mf, indent=2)
                        _flat = dose_arr.reshape(-1)
                        _nan = int(np.isnan(_flat).sum())
                        _tot = int(_flat.size)
                        _dmax = float(np.nanmax(_flat)) if _tot - _nan > 0 else float("nan")
                        print(f"       -> dose shape={dose_arr.shape}, "
                              f"spacing={[round(s, 2) for s in spacing]}, "
                              f"max={_dmax:.1f} cGy, "
                              f"NaN={_nan}/{_tot}")
                except Exception as e:
                    err = str(e)
                    import traceback
                    traceback.print_exc()
            else:
                err = f"unknown modality type: {mtype}"

            if _modality_dir_is_complete(tgt, mtype):
                _mark_seg_status(patient_status, st_path, uuid,
                                 row.get("rp_status", ""), modality_key, "complete")
            else:
                _mark_seg_status(patient_status, st_path, uuid,
                                 row.get("rp_status", ""), modality_key,
                                 "failed", err or "incomplete output")

    workers = int(cfg.get("PREPROCESS", {}).get("WORKERS", 1) or 1)

    if bool(cfg.get("PREPROCESS", {}).get("REGISTER_PET", True)):
        print(f"\n  --- Registering PET to CT (rigid MI, GTV-QA) [{workers}w] ---")
        _run_per_patient(
            manifest,
            lambda r: _register_pet_to_ct(str(r.get("uuid", "")).strip(), r, cfg),
            workers, "pet-reg")

    if bool(cfg.get("PREPROCESS", {}).get("REGISTER_DOSE", True)):
        print(f"\n  --- Mapping dose to CT (Monaco signs + target-anchor) [{workers}w] ---")
        _run_per_patient(
            manifest,
            lambda r: _register_dose_to_ct(str(r.get("uuid", "")).strip(), r, cfg),
            workers, "dose-reg")

    if bool(cfg.get("PREPROCESS", {}).get("COMPUTE_DVH", True)):
        print(f"\n  --- Computing DVHs (dose x contour masks) [{workers}w] ---")
        _run_per_patient(
            manifest,
            lambda r: _compute_dvh(str(r.get("uuid", "")).strip(), cfg),
            workers, "dvh")

    if bool(cfg.get("PREPROCESS", {}).get("GENERATE_ALIGNMENT_VIDEO", False)):
        _sub = cfg.get("PREPROCESS", {}).get("VIDEO_SUBDIR_STEP2", "step2")
        print(f"\n  --- Generating alignment videos (-> {_sub or '.'}) [{workers}w] ---")
        _run_per_patient(
            manifest,
            lambda r: _generate_alignment_video(str(r.get("uuid", "")).strip(),
                                                cfg, subdir=_sub),
            workers, "video")

    _ann_log = os.path.join(input_dir, "contours_annotation_log.csv")
    if os.path.exists(_ann_log):
        try:
            import csv as _csv3
            import unicodedata as _ud4
            _tm_re4 = re.compile(r"\btm\b")
            _ac_re4 = re.compile(r"\bac\b")
            _sc_re4 = re.compile(r"\bsc\b")
            _sc_end_re4 = re.compile(r"sc$")
            _kit_re4 = re.compile(r"\bkit")
            _ms_dot_re4 = re.compile(r"\bm\.s\b")
            _eso_kws4 = ("ozefag", "ozafag", "osefag", "esofag", "esophag", "ozfag")
            _liver_kws4 = ("karaciger", "karaci", "katraci", "liver", "hepar")
            _breast_kws4 = ("meme", "breast")

            def _fold4(s):
                k = _ud4.normalize("NFKD", s).replace("ı", "i").replace("İ", "i")
                return "".join(c for c in k if not _ud4.combining(c)).lower()

            def _classify4(name):
                nl = _fold4(name);
                s = nl.strip()
                if _is_ptv_name(nl): return "PTV"
                if ("ptv" in nl) or ("boost" in nl): return ""
                if "lun" in nl: return "LUNG"
                if "ctv" in nl: return "CTV"
                if any(k in nl for k in ("gtv", "tumor", "tumör", "target vol")): return "GTV"
                if (any(k in nl for k in ("kitle", "lezyon", "nodul"))
                        or _tm_re4.search(nl) or _kit_re4.search(nl)): return "GTV"
                if ("akc" in nl) or _ac_re4.search(nl): return "LUNG"
                if ("kalp" in nl) or ("heart" in nl): return "HEART"
                if any(k in nl for k in _eso_kws4) or s.startswith("oz"): return "ESOPHAGUS"
                if (("ms" in nl) or ("medulla" in nl) or ("spi" in nl)
                        or _sc_re4.search(nl) or _sc_end_re4.search(s)
                        or _ms_dot_re4.search(nl)): return "SPINAL_CORD"
                if any(k in nl for k in _liver_kws4): return "LIVER"
                if any(k in nl for k in _breast_kws4): return "BREAST"
                return "-"

            _name_to_uuids: dict[str, list[str]] = {}
            with open(_ann_log, encoding="utf-8") as _f:
                _r = _csv3.DictReader(_f)
                for _row in _r:
                    _all = _row.get("all_struct_names", "")
                    _uid = _row.get("uuid", "")
                    for _n in (n.strip() for n in _all.split(";")):
                        if _n:
                            _name_to_uuids.setdefault(_n, []).append(_uid)
            _dict_path = os.path.join(input_dir,
                                      "contours_structure_name_dictionary.csv")
            _cls_counts: dict[str, int] = {k: 0 for k in
                                           ("GTV", "CTV", "PTV", "LUNG", "HEART", "ESOPHAGUS",
                                            "SPINAL_CORD", "LIVER", "BREAST", "-")}
            with open(_dict_path, "w", newline="", encoding="utf-8") as _f:
                _w = _csv3.writer(_f)
                _w.writerow(["structure_name", "classification",
                             "n_patients", "sample_uuids_first5"])
                _order = {"GTV": 0, "CTV": 1, "PTV": 2, "LUNG": 3,
                          "HEART": 4, "ESOPHAGUS": 5,
                          "SPINAL_CORD": 6, "LIVER": 7, "BREAST": 8, "-": 9}
                for _n, _uids in sorted(
                        _name_to_uuids.items(),
                        key=lambda kv: (_order.get(_classify4(kv[0]), 9),
                                        -len(kv[1]), kv[0])):
                    _cls = _classify4(_n)
                    _cls_counts[_cls] = _cls_counts.get(_cls, 0) + 1
                    _w.writerow([_n, _cls, len(_uids), ";".join(_uids[:5])])
            _to_excel(pd.read_csv(_dict_path), _dict_path.replace(".csv", ".xlsx"))
            print(f"\nStructure-name dictionary: {_dict_path} "
                  f"({len(_name_to_uuids)} unique names)")
            print(f"  Classification breakdown:")
            for _k in ("GTV", "CTV", "PTV", "LUNG", "HEART", "ESOPHAGUS",
                       "SPINAL_CORD", "LIVER", "BREAST", "-"):
                print(f"    {_k:<12s}: {_cls_counts.get(_k, 0)}")
            _unmatched_names = [(_n, _uids) for _n, _uids in _name_to_uuids.items()
                                if _classify4(_n) == "-"
                                and _n.strip().lower() not in ("", "general")]
            _unm_path = os.path.join(input_dir,
                                     "contours_unmatched_names.csv")
            try:
                with open(_unm_path, "w", newline="", encoding="utf-8") as _uf:
                    _uw = _csv3.writer(_uf)
                    _uw.writerow(["structure_name", "n_patients",
                                  "sample_uuids_first5"])
                    for _n, _uids in sorted(_unmatched_names,
                                            key=lambda kv: (-len(kv[1]), kv[0])):
                        _uw.writerow([_n, len(_uids), ";".join(_uids[:5])])
                _to_excel(pd.read_csv(_unm_path), _unm_path.replace(".csv", ".xlsx"))
            except Exception as _ue:
                print(f"  [!] failed to write unmatched-names CSV: {_ue}")

            print(f"\nUnmatched structure names: {len(_unmatched_names)} unique "
                  f"(see {_unm_path})")
            if not _unmatched_names:
                print("  (none - every non-'general' name was classified)")
            else:
                _unmatched_names.sort(key=lambda kv: -len(kv[1]))
                for _n, _uids in _unmatched_names:
                    print(f"    {len(_uids):>3d}x {_n!r}")
        except Exception as _de:
            print(f"\n[!] Failed to build structure-name dictionary: {_de}")

    _ann_log_csv = os.path.join(input_dir, "contours_annotation_log.csv")
    if os.path.exists(_ann_log_csv):
        _to_excel(pd.read_csv(_ann_log_csv), _ann_log_csv.replace(".csv", ".xlsx"))

    print("\nStep 2 done.")


def step_4_verify(cfg):
    print("\n" + "=" * 60)
    print("  STEP 4 - Verify & Stats")
    print("=" * 60)
    input_dir = cfg["INPUTS_DIR"]
    processed_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    gen_previews = bool(cfg["PREPROCESS"].get("GENERATE_PREVIEWS", False))
    required = {k: v for k, v in cfg["PREPROCESS"]["GENERATED_FILES"].items()
                if gen_previews or not v.endswith(".avi")}
    npy_keys = [k for k, fn in required.items() if fn.endswith(".npy")]

    modalities = _get_modalities(cfg)
    audit_modalities = {k: v for k, v in modalities.items()
                        if v.get("enabled", True)}

    ct_seg_enabled = bool(cfg.get("PREPROCESS", {}).get(
        "CT_TOTALSEGMENTATOR_ENABLED", False))
    ct_required = (required if ct_seg_enabled else
                   {k: v for k, v in required.items() if v.startswith("whole_body")})
    type_required = {
        "ct": ct_required,
        "pet": {k: v for k, v in required.items()
                if v.startswith("whole_body")},
        "contours": {k: v for k, v in required.items()
                     if k in ("Has_GTV_Mask", "Has_Lung_Mask", "Has_PTV_Mask")},
        "dose": {k: v for k, v in required.items()
                 if k in ("Has_Dose_Volume",)},
        "passthrough": {},
        "plan_params": {k: v for k, v in required.items()
                        if k in ("Has_Plan_Features",)},
    }

    rows = []
    bad_uuids: set[str] = set()
    if not os.path.isdir(processed_dir):
        print(f"No Processed/ found at {processed_dir}.")
        return

    for uid in sorted(os.listdir(processed_dir)):
        upath = os.path.join(processed_dir, uid)
        if not os.path.isdir(upath):
            continue
        for modality_key, modality_cfg in audit_modalities.items():
            mdir = os.path.join(upath, modality_cfg["dest_subdir"])
            if not os.path.isdir(mdir):
                continue
            mtype = modality_cfg.get("type", "ct")
            req_for_type = type_required.get(mtype, {})
            file_count = sum(len(fs) for _, _, fs in os.walk(mdir))
            row = {"Patient_UUID": uid, "Modality": modality_key,
                   "Type": mtype, "File_Count": file_count}
            ok = True
            if mtype == "passthrough":
                ok = file_count > 0
            else:
                contents = set(os.listdir(mdir))
                for col, fn in req_for_type.items():
                    present = fn in contents
                    row[col] = present
                    if not present:
                        ok = False
                        continue
                    if col in npy_keys:
                        try:
                            arr = np.load(os.path.join(mdir, fn), mmap_mode="r")
                            if arr.size == 0 or 0 in arr.shape:
                                ok = False
                                row[col] = False
                        except Exception:
                            ok = False
                            row[col] = False
            row["OK"] = ok
            if not ok and modality_key == "ct_plan":
                bad_uuids.add(uid)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        cols = (["Patient_UUID", "Modality", "OK", "File_Count"]
                + list(required.keys()))
        df = df.reindex(columns=cols, fill_value=False)
        df = df.sort_values(["Modality", "OK"])
        print()
        for mk in audit_modalities:
            sub = df[df["Modality"] == mk]
            if not len(sub):
                continue
            print(f"  {mk:>10s}: {sub['OK'].sum()}/{len(sub)} usable")
        out = os.path.join(input_dir, "preprocess_sanity_check.csv")
        df.to_csv(out, index=False)
        _to_excel(df, out.replace(".csv", ".xlsx"))
        print(f"Report: {out}")
    else:
        print("No processed patients found yet.")

    if bad_uuids:
        unified_path = os.path.join(input_dir, cfg["SPLIT"]["UNIFIED_FILE"])
        ct_manifest_path = os.path.join(input_dir, "clean_cohort_manifest_cv.csv")
        for path, uuid_col in [(unified_path, "uuid"), (ct_manifest_path, "UUID")]:
            if not os.path.exists(path):
                continue
            df_m = read_table(path)
            n_before = len(df_m)
            df_m = df_m[~df_m[uuid_col].isin(bad_uuids)].copy()
            n_after = len(df_m)
            if n_after < n_before:
                df_m.to_csv(path, index=False)
                print(f"Filtered {os.path.basename(path)}: "
                      f"{n_before} -> {n_after} (dropped {n_before - n_after})")
        print(f"Excluded UUIDs: {sorted(bad_uuids)}")

    unified_path = os.path.join(input_dir, "unified_split.csv")
    if os.path.exists(unified_path):
        df_u = read_table(unified_path)
        total = len(df_u)
        rp_pos = (df_u["rp_label"] == 1).sum()
        rp_neg = (df_u["rp_label"] == 0).sum()
        ct_valid = df_u["has_valid_ct"].sum() if "has_valid_ct" in df_u.columns else "?"
        has_tab = (df_u["excel_row_idx"] >= 0).sum() if "excel_row_idx" in df_u.columns else "?"

        print(f"\n--- Dataset Stats ---")
        print(f"Total patients:     {total}")
        print(f"  RP+:              {rp_pos} ({100 * rp_pos / total:.1f}%)")
        print(f"  RP-:              {rp_neg} ({100 * rp_neg / total:.1f}%)")
        print(f"  Valid CT:         {ct_valid}")
        print(f"  Tabular matched:  {has_tab}")
        if "Split" in df_u.columns:
            print(f"  Test set:         {(df_u['Split'] == 'Test').sum()}")
            print(f"  Train/CV set:     {(df_u['Split'] == 'Train_CV').sum()}")
        elif "OuterFold" in df_u.columns:
            sizes = df_u.groupby("OuterFold").size().to_dict()
            print(f"  OuterFold sizes:  {sizes}")
        print("---------------------")


def _write_availability_excel(df: "pd.DataFrame", path: str, modality_cols: list) -> None:
    import openpyxl
    from openpyxl.styles import (Alignment, Border, Font, PatternFill, Side)
    from openpyxl.utils import get_column_letter

    HEADER_BG = "1F3864"
    HEADER_FG = "FFFFFF"
    YES_BG = "C6EFCE"
    YES_FG = "375623"
    NO_BG = "FCE4D6"
    NO_FG = "843C0C"
    RP1_BG = "FFCCCC"
    RP0_BG = "DDEEFF"
    ROW_ODD = "F2F2F2"
    ROW_EVEN = "FFFFFF"

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def fill(hex_color):
        return PatternFill("solid", fgColor=hex_color)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Availability"

    cols = list(df.columns)
    modality_set = set(modality_cols)

    for ci, col in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.font = Font(bold=True, color=HEADER_FG, size=14)
        cell.fill = fill(HEADER_BG)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    for ri, (_, row) in enumerate(df.iterrows(), start=2):
        row_bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
        for ci, col in enumerate(cols, start=1):
            val = row[col]
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

            if col in modality_set:
                if val == "Y":
                    cell.fill = fill(YES_BG)
                    cell.font = Font(bold=True, color=YES_FG, size=14)
                else:
                    cell.fill = fill(NO_BG)
                    cell.font = Font(color=NO_FG, size=14)
            elif col == "rp_label":
                if str(val) == "1":
                    cell.fill = fill(RP1_BG)
                    cell.font = Font(bold=True, size=14)
                elif str(val) == "0":
                    cell.fill = fill(RP0_BG)
                    cell.font = Font(size=14)
                else:
                    cell.fill = fill(row_bg)
                    cell.font = Font(color="999999", size=14)
            else:
                cell.fill = fill(row_bg)
                cell.font = Font(size=14)
                if col == "patient_name":
                    cell.alignment = Alignment(horizontal="left", vertical="center")

    for ci, col in enumerate(cols, start=1):
        max_len = max(len(str(col)), *(len(str(df.iloc[r][col])) for r in range(len(df))))
        ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 3, 22)

    summary_row = len(df) + 2
    ws.cell(row=summary_row, column=1, value="TOTAL Y").font = Font(bold=True, size=14)
    ws.cell(row=summary_row, column=1).fill = fill(HEADER_BG)
    ws.cell(row=summary_row, column=1).font = Font(bold=True, color=HEADER_FG, size=14)
    ws.cell(row=summary_row, column=1).alignment = Alignment(horizontal="center")
    for ci, col in enumerate(cols, start=1):
        if col in modality_set:
            n_yes = (df[col] == "Y").sum()
            cell = ws.cell(row=summary_row, column=ci, value=f"{n_yes}/{len(df)}")
            cell.font = Font(bold=True, color=HEADER_FG, size=14)
            cell.fill = fill(HEADER_BG)
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        elif col not in ("patient_name", "uuid"):
            ws.cell(row=summary_row, column=ci).fill = fill(HEADER_BG)

    wb.save(path)


def _to_excel(df: "pd.DataFrame", path: str, *, delete_csv: bool = True) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    H_BG, H_FG = "1F3864", "FFFFFF"
    ODD, EVEN = "F2F2F2", "FFFFFF"
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill = lambda h: PatternFill("solid", fgColor=h)

    wb = openpyxl.Workbook();
    ws = wb.active
    cols = list(df.columns)

    for ci, col in enumerate(cols, 1):
        c = ws.cell(row=1, column=ci, value=col)
        c.font = Font(bold=True, color=H_FG, size=14)
        c.fill = fill(H_BG)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
    ws.row_dimensions[1].height = 26
    ws.freeze_panes = "A2"

    for ri, (_, row) in enumerate(df.iterrows(), 2):
        row_bg = ODD if ri % 2 == 0 else EVEN
        for ci, col in enumerate(cols, 1):
            val = row[col]
            c = ws.cell(row=ri, column=ci, value=val)
            c.font = Font(size=14)
            c.fill = fill(row_bg)
            c.border = border
            c.alignment = Alignment(horizontal="left" if ci == 1 else "center",
                                    vertical="center")

    for ci, col in enumerate(cols, 1):
        try:
            max_len = max(len(str(col)),
                          max(len(str(v)) for v in df[col].fillna("")))
        except Exception:
            max_len = len(str(col))
        ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 3, 42)

    wb.save(path)

    if delete_csv:
        csv_path = path.replace(".xlsx", ".csv")
        if csv_path != path and os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except OSError:
                pass


def step_build_data_matrix(cfg):
    print("\n" + "=" * 60)
    print("  STEP 8 - Data Availability Matrix")
    print("=" * 60)
    input_dir = cfg["INPUTS_DIR"]
    processed_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])

    map_path = os.path.join(input_dir, "patient_name_to_uuid.csv")
    if not table_exists(map_path):
        print(f"patient_name_to_uuid.csv not found. Run step 0 first.")
        return
    df_map = read_table(map_path, dtype=str).fillna("")
    print(f"Patient universe : {len(df_map)} patients")

    uuid_to_name = dict(zip(df_map["uuid"], df_map["patient_name"]))
    uuid_to_group = dict(zip(df_map["uuid"], df_map["rp_status"]))

    master_name = cfg.get("PREPROCESS", {}).get("MASTER_OUTPUT",
                                                "master_clinical_data.xlsx")
    master_path = os.path.join(input_dir, master_name)
    df_master = None
    if table_exists(master_path):
        df_master = pd.read_excel(master_path)
        print(f"Master clinical  : {len(df_master)} rows")
    else:
        print(f"Master clinical  : not found (run step 4)")

    split_path = os.path.join(input_dir, cfg["SPLIT"]["UNIFIED_FILE"])
    uuid_to_row_idx: dict[str, int] = {}
    uuid_to_label: dict[str, str] = {}
    if table_exists(split_path):
        df_split = read_table(split_path, dtype=str).fillna("")
        for _, r in df_split.iterrows():
            uid = r["uuid"]
            idx = r.get("excel_row_idx", "")
            if idx not in ("", "-1"):
                uuid_to_row_idx[uid] = int(idx)
            uuid_to_label[uid] = r.get("rp_label", "")
        print(f"Unified split    : {len(df_split)} patients linked")
    else:
        print(f"Unified split    : not found (run step 5)")

    _, seg_status = _load_seg_status(input_dir)
    print(f"Segment statuses : {len(seg_status)} entries")

    ct_modalities = _get_modalities(cfg)

    clinical_cols = cfg.get("TRAIN_CLINICAL", {}).get("DATA", {}).get("BASE_COLS", [])
    rt_dose_cols = cfg.get("TRAIN_RT", {}).get("DATA", {}).get("BASE_COLS", [])

    def _tabular_status(uuid: str) -> tuple[str, str]:
        row_idx = uuid_to_row_idx.get(uuid)
        if row_idx is None or df_master is None:
            return "absent", "absent"
        if row_idx < 0 or row_idx >= len(df_master):
            return "absent", "absent"
        row = df_master.iloc[row_idx]

        def _check(cols):
            if not cols:
                return "OK"
            n_missing = sum(
                1 for c in cols
                if c not in row.index or pd.isna(row[c])
            )
            return "OK" if n_missing == 0 else f"partial ({n_missing}/{len(cols)})"

        return _check(clinical_cols), _check(rt_dose_cols)

    def _ct_status(uuid: str, modality_key: str, modality_cfg: dict) -> str:
        if not modality_cfg.get("enabled", True):
            return "disabled"

        if bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True)):
            root = os.path.join(processed_dir, uuid)
            mtype = modality_cfg.get("type", "ct")
            mp = os.path.join(root, "patient_meta.json")
            if os.path.exists(mp):
                try:
                    _m = json.load(open(mp))
                    if _m.get("excluded"):
                        return f"excluded:{_m['excluded']}"
                except Exception:
                    pass
            sd = os.path.join(root, "studies")
            pat = {
                "ct": os.path.join(sd, "[0-9]*", "CT", "whole_body.npy"),
                "contours": os.path.join(sd, "[0-9]*", "Contours", "*_mask.npy"),
                "dose": os.path.join(sd, "[0-9]*", "Dose", "dose_regCT.npy"),
                "pet": os.path.join(sd, "00_*", "PET", "whole_body_regCT.npy"),
            }.get(mtype)
            if pat is not None:
                return "complete" if glob.glob(pat) else "not_processed"
            if mtype == "passthrough":
                tgt = os.path.join(root, modality_cfg.get("dest_subdir", modality_key))
                return "complete" if (os.path.isdir(tgt) and os.listdir(tgt)) \
                    else "not_processed"

        status_key = f"{uuid}__{modality_key}"
        if status_key in seg_status:
            rec = seg_status[status_key]
            if rec["status"] == "complete":
                return "complete"
            return f"failed: {rec.get('error', '?')}"

        tgt = os.path.join(processed_dir, uuid,
                           modality_cfg.get("dest_subdir", modality_key))
        mtype = modality_cfg.get("type", "ct")
        ct_sentinel = ("lung_only.npy"
                       if cfg.get("PREPROCESS", {}).get(
            "CT_TOTALSEGMENTATOR_ENABLED", False)
                       else "whole_body.npy")
        sentinel = {
            "ct": ct_sentinel,
            "pet": "whole_body.npy",
            "contours": "gtv_mask.npy",
            "dose": "dose_volume.npy",
            "plan_params": "plan_features.json",
            "passthrough": None,
        }.get(mtype)
        if os.path.isdir(tgt) and (
                sentinel is None or os.path.exists(os.path.join(tgt, sentinel))):
            return "complete (legacy)"

        return "not_processed"

    import json as _qjson
    _study_layout = bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))
    _pet_qa_min = float(cfg.get("PREPROCESS", {}).get("PET_REG_QA_MIN_RATIO", 1.3))

    def _study_seq(uuid: str):
        mp = os.path.join(processed_dir, uuid, "patient_meta.json")
        if not os.path.exists(mp):
            return None, None
        try:
            meta = _qjson.load(open(mp))
        except Exception:
            return None, None
        cts = [s for s in meta.get("sequence", []) if s.get("modality") != "pet"]
        if not cts:
            return meta, None
        totals = [s for s in cts if s.get("is_total")]
        pool = totals or cts
        primary = max(pool, key=lambda s: (s.get("dose_max_cGy") or 0,
                                           s.get("lung_MLD_cGy") or 0))
        return meta, primary

    def _pet_align_status(uuid: str) -> str:
        if _study_layout:
            meta, _ = _study_seq(uuid)
            if meta is None:
                return ""
            pet = next((s for s in meta.get("sequence", [])
                        if s.get("modality") == "pet"), None)
            if pet is None:
                return "no_pet"
            r = pet.get("qa_ratio")
            rtxt = f"{r:.1f}" if isinstance(r, (int, float)) and r == r else "?"
            if not pet.get("has_gtv"):
                return f"no_gtv (r={rtxt})"
            ok = isinstance(r, (int, float)) and r == r and r >= _pet_qa_min
            return f"OK (r={rtxt})" if ok else f"review (r={rtxt})"
        d = os.path.join(processed_dir, uuid, "PET-Plan")
        meta = os.path.join(d, "pet_reg_meta.json")
        if not os.path.exists(meta):
            return ("not_registered"
                    if os.path.exists(os.path.join(d, "whole_body.npy")) else "")
        try:
            m = _qjson.load(open(meta))
        except Exception:
            return "error"
        r = m.get("qa_ratio")
        rtxt = f"{r:.1f}" if isinstance(r, (int, float)) and r == r else "?"
        if not m.get("has_gtv"):
            return f"no_gtv (r={rtxt})"
        return f"OK (r={rtxt})" if m.get("qa_pass") else f"review (r={rtxt})"

    def _contours_align_status(uuid: str) -> str:
        if _study_layout:
            meta, _ = _study_seq(uuid)
            if meta is None:
                return ""
            n = meta.get("n_studies", 0)
            return f"{n} study(s)" if n else "no_study"
        meta = os.path.join(processed_dir, uuid, "Contours", "contours_meta.txt")
        if not os.path.exists(meta):
            return ""
        try:
            m = _qjson.load(open(meta))
        except Exception:
            return "error"
        ok = m.get("align_ok"); hu = m.get("gtv_mean_hu")
        if ok is None:
            return "unknown"
        hutxt = f"{hu:.0f}" if isinstance(hu, (int, float)) and hu == hu else "?"
        return f"OK (HU={hutxt})" if ok else f"review (HU={hutxt})"

    def _dvh_status(uuid: str) -> str:
        if _study_layout:
            _, prim = _study_seq(uuid)
            if prim is None:
                return ""
            mld = prim.get("lung_MLD_cGy"); v20 = prim.get("lung_V20")
            if mld is None:
                return "no_lung"
            return f"MLD={mld/100:.1f}Gy V20={v20:.0f}% ({prim.get('ct_folder')})"
        f = os.path.join(processed_dir, uuid, "DVH", "dvh.json")
        if not os.path.exists(f):
            return ""
        try:
            m = _qjson.load(open(f))
        except Exception:
            return "error"
        lung = m.get("structures", {}).get("lung", {})
        mld = lung.get("Dmean_cGy"); v20 = lung.get("V20")
        if mld is None:
            return "no_lung"
        return f"MLD={mld/100:.1f}Gy V20={v20:.0f}%"

    def _dose_align_status(uuid: str) -> str:
        if _study_layout:
            _, prim = _study_seq(uuid)
            if prim is None:
                return ""
            cov = prim.get("ptv_cov50")
            if cov is None:
                return "no_dose"
            return f"OK (cov50={cov*100:.0f}%)" if cov >= 0.8 \
                else f"review (cov50={cov*100:.0f}%)"
        meta = os.path.join(processed_dir, uuid, "Dose", "dose_reg_meta.json")
        if not os.path.exists(meta):
            return ""
        try:
            m = _qjson.load(open(meta))
        except Exception:
            return "error"
        cov = m.get("ptv_cov50")
        covtxt = f"{cov*100:.0f}%" if isinstance(cov, (int, float)) and cov == cov else "?"
        if not m.get("on_contour_ct", True):
            return "no_CT1_plan"
        return f"OK (cov50={covtxt})" if m.get("qa_pass") else f"review (cov50={covtxt})"

    _v20_tol = float(cfg.get("PREPROCESS", {}).get("DOSE_ALIGN_V20_TOL", 8.0))
    _mld_tol = float(cfg.get("PREPROCESS", {}).get("DOSE_ALIGN_MLD_TOL_GY", 3.0))

    def _dose_align_check(uuid: str) -> str:
        if _study_layout:
            _, prim = _study_seq(uuid)
            if prim is None:
                return ""
            cv20 = prim.get("lung_V20"); cmld = prim.get("lung_MLD_cGy")
        else:
            f = os.path.join(processed_dir, uuid, "DVH", "dvh.json")
            if not os.path.exists(f):
                return ""
            try:
                lung = _qjson.load(open(f)).get("structures", {}).get("lung", {})
            except Exception:
                return "error"
            cv20 = lung.get("V20"); cmld = lung.get("Dmean_cGy")
        if cv20 is None or cmld is None:
            return "no_lung"
        ridx = uuid_to_row_idx.get(uuid)
        if ridx is None or df_master is None or not (0 <= ridx < len(df_master)):
            return "no_excel"
        mrow = df_master.iloc[ridx]
        ev20 = mrow.get("lung_V20"); emld = mrow.get("MLD")
        if pd.isna(ev20) or pd.isna(emld):
            return "no_excel"
        dv20 = abs(cv20 - float(ev20))
        dmld = abs(cmld / 100.0 - float(emld))
        ok = (dv20 <= _v20_tol) and (dmld <= _mld_tol)
        return (f"{'OK' if ok else 'review'} "
                f"(dV20={dv20:.0f}%,dMLD={dmld:.1f}Gy)")

    records = []
    for _, mr in df_map.iterrows():
        uuid = mr["uuid"]
        clin_s, rt_s = _tabular_status(uuid)

        row: dict = {
            "uuid": uuid,
            "patient_name": uuid_to_name.get(uuid, ""),
            "rp_label": uuid_to_label.get(uuid, ""),
            "clinical": clin_s,
            "rt_doses": rt_s,
        }
        for mk, mcfg in ct_modalities.items():
            row[mk] = _ct_status(uuid, mk, mcfg)
        row["contours_ct_align"] = _contours_align_status(uuid)
        row["pet_ct_align"] = _pet_align_status(uuid)
        row["dose_ct_align"] = _dose_align_status(uuid)
        row["dose_align_check"] = _dose_align_check(uuid)
        row["dvh_lung"] = _dvh_status(uuid)

        records.append(row)

    df_out = pd.DataFrame(records)

    modality_cols = ["clinical", "rt_doses"] + list(ct_modalities.keys())
    print(f"\nAvailability ({len(df_out)} patients):")
    for col in modality_cols:
        if col not in df_out.columns:
            continue
        counts = df_out[col].value_counts()
        col_label = f"  {col}"
        for val, cnt in counts.items():
            pct = 100 * cnt / len(df_out)
            print(f"{col_label:<18s}  {val:<40s} {cnt:>4d}  ({pct:.0f}%)")
            col_label = ""

    print("\nAlignment QA (contours->CT, PET->CT, dose->CT, dose-vs-Excel):")
    for col in ["contours_ct_align", "pet_ct_align", "dose_ct_align",
                "dose_align_check"]:
        if col not in df_out.columns:
            continue
        vals = df_out[df_out[col].astype(str).str.strip() != ""][col].astype(str)
        vals = vals.map(lambda s: s.split(" ")[0])
        counts = vals.value_counts()
        col_label = f"  {col}"
        if counts.empty:
            print(f"{col_label:<20s}  (none)")
            continue
        for val, cnt in counts.items():
            print(f"{col_label:<20s}  {val:<28s} {cnt:>4d}")
            col_label = ""

    if df_master is not None:
        pairs_v20, pairs_mld = [], []
        for uuid, ridx in uuid_to_row_idx.items():
            if not (0 <= ridx < len(df_master)):
                continue
            if _study_layout:
                _, prim = _study_seq(uuid)
                if prim is None:
                    continue
                cv20 = prim.get("lung_V20")
                cmld = prim.get("lung_MLD_cGy")
            else:
                f = os.path.join(processed_dir, uuid, "DVH", "dvh.json")
                if not os.path.exists(f):
                    continue
                try:
                    lung = _qjson.load(open(f)).get("structures", {}).get("lung", {})
                except Exception:
                    continue
                cv20 = lung.get("V20"); cmld = lung.get("Dmean_cGy")
            mrow = df_master.iloc[ridx]
            if cv20 is not None and pd.notna(mrow.get("lung_V20")):
                pairs_v20.append((cv20, float(mrow["lung_V20"])))
            if cmld is not None and pd.notna(mrow.get("MLD")):
                pairs_mld.append((cmld / 100.0, float(mrow["MLD"])))

        def _report(name, pairs, unit):
            if len(pairs) < 2:
                print(f"  {name}: {len(pairs)} patient(s) - too few to validate")
                return
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            mad = float(np.mean(np.abs(a - b)))
            r = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
            print(f"  {name}: n={len(pairs)}  mean|computed-excel|={mad:.1f}{unit}  r={r:.2f}")

        print("\nDVH validation (computed vs clinical Excel):")
        _report("lung V20", pairs_v20, "%")
        _report("lung MLD", pairs_mld, "Gy")

    inv_path = os.path.join(input_dir, "patient_imaging_inventory.csv")
    if table_exists(inv_path):
        inv_df = read_table(inv_path, dtype=str).fillna("")
        inv_cols = [c for c in inv_df.columns
                    if c not in ("patient_name", "hasta_no")]
        df_out = df_out.merge(inv_df[inv_cols], on="uuid", how="left")
        print(f"Merged imaging inventory ({len(inv_cols)-1} columns: CT dates, "
              f"plans, phantom...) into the availability matrix.")

    out_path = os.path.join(input_dir, "data_availability_matrix.csv")
    df_out.to_csv(out_path, index=False)
    _to_excel(df_out, out_path.replace(".csv", ".xlsx"))
    print(f"\nStatus matrix: {out_path}  "
          f"({len(df_out)} rows x {len(df_out.columns)} cols)")

    def _is_yes(status: str) -> str:
        s = str(status)
        return "Y" if (s == "OK" or s.startswith("complete")) else "N"

    bin_records = []
    for r in records:
        brow = {"patient_name": r["patient_name"], "uuid": r["uuid"],
                "rp_label": r["rp_label"]}
        for col in modality_cols:
            brow[col] = _is_yes(r.get(col, ""))
        bin_records.append(brow)
    df_bin = pd.DataFrame(bin_records)
    bin_path = os.path.join(input_dir, "availability_matrix_binary.xlsx")
    _write_availability_excel(df_bin, bin_path, modality_cols)
    print(f"Binary matrix: {bin_path}  ({len(df_bin)} rows x {len(df_bin.columns)} cols)")

    enabled_imaging = [mk for mk, mcfg in ct_modalities.items()
                       if mcfg.get("enabled", True)]
    _all_zero_cols = {col for col in modality_cols
                      if (df_bin[col] == "Y").sum() == 0}
    active_cols = [c for c in (["clinical", "rt_doses"] + enabled_imaging)
                   if c not in _all_zero_cols]
    disabled_cols = [c for c in modality_cols
                     if c not in (["clinical", "rt_doses"] + enabled_imaging)]

    print("\nBinary availability (Y count per modality):")
    for col in modality_cols:
        n_yes = (df_bin[col] == "Y").sum()
        tags = []
        if col in disabled_cols:
            tags.append("disabled")
        if col in _all_zero_cols and col not in disabled_cols:
            tags.append("all-zero, excluded from intersection")
        tag = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {col:<16s} {n_yes:>4d} / {len(df_bin)}{tag}")
    if active_cols:
        all_y = (df_bin[active_cols] == "Y").all(axis=1).sum()
        print(f"  {'ALL modalities':<16s} {all_y:>4d} / {len(df_bin)}  "
              f"<- complete-case cohort size "
              f"(over {len(active_cols)} active: {active_cols})")


def step_generate_videos(cfg):
    print("\n" + "=" * 60)
    print("  STEP 9 - Generate Alignment Videos")
    print("=" * 60)
    input_dir = cfg["INPUTS_DIR"]
    manifest_path = os.path.join(input_dir, "patient_manifest.csv")
    if not table_exists(manifest_path):
        print(f"Error: manifest not found ({manifest_path}). Run step 0 first.")
        return
    manifest = read_table(manifest_path).fillna("").to_dict("records")
    manifest = _filter_manifest_only(manifest, cfg)
    sub = cfg.get("PREPROCESS", {}).get("VIDEO_SUBDIR_STEP9", "step9")
    base = (cfg.get("PREPROCESS", {}).get("VIDEO_DIR") or "").strip() or \
        os.path.join(input_dir, cfg["INPUTS"].get("VIDEOS", "Videos"))
    vdir = os.path.join(base, sub) if sub else base
    proc_dir = os.path.join(input_dir, cfg["INPUTS"]["PROCESSED"])
    workers = int(cfg.get("PREPROCESS", {}).get("WORKERS", 1) or 1)
    study_layout = bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))

    if study_layout:
        todo = [r for r in manifest if str(r.get("uuid", "")).strip()
                and os.path.exists(os.path.join(proc_dir,
                    str(r.get("uuid", "")).strip(), "patient_meta.json"))]
        print(f"Video output dir: {os.path.join(base, sub) if sub else base}   "
              f"({len(todo)} patients, {workers}w)  [study layout]")
        _run_per_patient(
            todo,
            lambda r: _render_study_videos(str(r.get("uuid", "")).strip(), cfg, sub),
            workers, "video")
        print(f"\nDone. Rendered study videos for {len(todo)} patient(s).")
        return

    todo = [r for r in manifest
            if str(r.get("uuid", "")).strip()
            and os.path.exists(os.path.join(proc_dir,
                                            str(r.get("uuid", "")).strip(),
                                            "CT-Plan", "whole_body.npy"))]
    print(f"Video output dir: {vdir}   ({len(todo)} patients, {workers}w)")
    _run_per_patient(
        todo,
        lambda r: _generate_alignment_video(str(r.get("uuid", "")).strip(),
                                            cfg, subdir=sub),
        workers, "video")
    print(f"\nDone. Rendered videos for {len(todo)} patient(s) -> {vdir}")


STEPS = {
    0: ("Organize", step_0_organize),
    1: ("Validate", step_1_validate),
    2: ("Segment", step_3_segment),
    3: ("ExtractPlanParams", step_extract_plan_params),
    4: ("Verify", step_4_verify),
    5: ("BuildMaster", step_build_master),
    6: ("ExportPatientTabular", step_export_patient_tabular),
    7: ("GenerateSplit", step_generate_split),
    8: ("DataMatrix", step_build_data_matrix),
    9: ("GenerateVideos", step_generate_videos),
}


def parse_step_range(s: str) -> list[int]:
    result = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._streams[0].isatty()
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="Preprocessing Pipeline")
    parser.add_argument("--step", type=str, default=None,
                        help="Step(s) to run: 0, 0-2, 1,3,4  (default: all)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to write a copy of all console output to "
                             "(default: <INPUTS_DIR>/logs/preprocess_<ts>.log).")
    args = parser.parse_args()

    if args.step is not None:
        steps = parse_step_range(args.step)
    else:
        steps = list(STEPS.keys())

    cfg = load_config()

    log_path = args.log_file
    if log_path is None:
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(cfg["INPUTS_DIR"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"preprocess_{_ts}.log")
    log_fh = None
    try:
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        log_fh.write(f"\n{'=' * 60}\nPreprocessing run: "
                     f"{datetime.datetime.now().isoformat()}\n"
                     f"Steps: {steps}\n"
                     f"Command: {' '.join(sys.argv)}\n{'=' * 60}\n")
        sys.stdout = _TeeStream(sys.__stdout__, log_fh)
        sys.stderr = _TeeStream(sys.__stderr__, log_fh)
        print(f"[log] mirroring all output to: {log_path}")
    except Exception as _le:
        print(f"[log] failed to open log file {log_path}: {_le} - "
              f"continuing console-only.")

    try:
        for s in steps:
            if s not in STEPS:
                print(f"Unknown step: {s}")
                continue
            name, fn = STEPS[s]
            fn(cfg)

        print("\n" + "=" * 60)
        print("  Pipeline complete.")
        print("=" * 60)
    finally:
        if log_fh is not None:
            try:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                log_fh.close()
                print(f"[log] log saved: {log_path}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
