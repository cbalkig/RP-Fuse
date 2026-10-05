import glob
import json
import logging
import os
import sys

import yaml


def table_exists(path) -> bool:
    if os.path.exists(path):
        return True
    if path.lower().endswith(".csv"):
        return os.path.exists(path[:-4] + ".xlsx")
    if path.lower().endswith((".xlsx", ".xls")):
        return os.path.exists(os.path.splitext(path)[0] + ".csv")
    return False


def read_table(path, **kwargs):
    import pandas as pd
    if os.path.exists(path):
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path, **kwargs)
        return pd.read_csv(path, **kwargs)
    if path.lower().endswith(".csv"):
        alt = path[:-4] + ".xlsx"
        if os.path.exists(alt):
            return pd.read_excel(alt, **kwargs)
    elif path.lower().endswith((".xlsx", ".xls")):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            return pd.read_csv(alt, **kwargs)
    raise FileNotFoundError(f"No tabular file at {path} (nor sibling .xlsx/.csv).")


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('RPFUSE_CONFIG', 'config.yaml')

    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        candidate = os.path.join('configs', config_path)
        if os.path.exists(candidate):
            config_path = candidate

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    base_name = config.pop('EXTENDS', None)
    while base_name:
        base_path = base_name if os.path.isabs(base_name) else \
            os.path.join(config_dir, base_name)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"EXTENDS base config not found: {base_path}")
        with open(base_path, 'r') as bf:
            base_cfg = yaml.safe_load(bf)
        next_base = base_cfg.pop('EXTENDS', None)
        config = _deep_merge(base_cfg, config)
        config_dir = os.path.dirname(os.path.abspath(base_path))
        base_name = next_base

    return config


def _apply_outer_fold_view(df, outer_k: int, seed: int = 42, n_inner: int = 5,
                             inner_val_frac: float = 0.15, repeat: int = 0):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split

    if "OuterFold" not in df.columns:
        raise ValueError("DataFrame has no 'OuterFold' column - re-run "
                         "preprocess_pipeline.py --step 2 to add it.")

    df = df.copy().reset_index(drop=True)
    df["Split"] = np.where(df["OuterFold"] == outer_k, "Test", "Train_CV")
    df["Fold"] = -1
    df["IsInnerVal"] = False

    dev_mask = (df["Split"] == "Train_CV").values
    dev_idx = np.where(dev_mask)[0]
    y_dev = df.loc[dev_mask, "rp_label"].values

    inner_seed = seed + outer_k + repeat * 1000
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
    fold_col = df.columns.get_loc("Fold")
    for fi, (_, val_rel) in enumerate(skf.split(dev_idx, y_dev)):
        df.iloc[dev_idx[val_rel], fold_col] = fi

    if inner_val_frac > 0 and len(dev_idx) > 0:
        _, inner_val_rel = train_test_split(
            np.arange(len(dev_idx)), test_size=inner_val_frac,
            stratify=y_dev, random_state=inner_seed)
        iv_col = df.columns.get_loc("IsInnerVal")
        df.iloc[dev_idx[inner_val_rel], iv_col] = True

    return df


def load_unified_split(input_dir: str):
    import pandas as pd
    path = os.path.join(input_dir, "unified_split.csv")
    if not table_exists(path):
        raise FileNotFoundError(
            f"Unified split not found: {path} (nor its .xlsx sibling)\n"
            "Run 'preprocess_pipeline.py --step 5' first."
        )
    df = read_table(path)

    try:
        cfg = load_config()
        proc_dir = os.path.join(cfg["INPUTS_DIR"], cfg["INPUTS"]["PROCESSED"])
        modalities = cfg.get("PREPROCESS", {}).get("MODALITIES", {}) or {}
        ct_subdir = modalities.get("ct_plan", {}).get("dest_subdir", "CT-Plan")
        req_file = "whole_body.npy"

        study_layout = bool(cfg.get("PREPROCESS", {}).get("STUDY_LAYOUT", True))

        def _ct_present(root: str) -> bool:
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

        if os.path.isdir(proc_dir):
            demoted: list[str] = []
            ct_rows = df.index[df["has_valid_ct"].astype(bool)]
            for idx in ct_rows:
                uid = str(df.at[idx, "uuid"]).strip()
                if not _ct_present(os.path.join(proc_dir, uid)):
                    df.at[idx, "has_valid_ct"] = False
                    demoted.append(uid)

            if demoted:
                log = logging.getLogger(__name__)
                log.warning(
                    "load_unified_split: %d patient(s) had has_valid_ct=True "
                    "but Processed/<uuid>/%s/%s is missing on disk - excluded "
                    "from CT splits. Re-run preprocess step 2 to include them: %s",
                    len(demoted), ct_subdir, req_file, demoted)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.getLogger(__name__).error(
            "load_unified_split: existence check failed (%s) - "
            "has_valid_ct flags may be stale.", exc)

    outer_env = os.environ.get("NESTED_OUTER_FOLD")
    if outer_env is not None and outer_env != "":
        try:
            outer_k = int(outer_env)
        except ValueError:
            return df
        repeat = int(os.environ.get("NESTED_CV_REPEAT", "0"))
        df = _apply_outer_fold_view(df, outer_k=outer_k, repeat=repeat)
    elif "Split" not in df.columns and "OuterFold" in df.columns:
        try:
            cfg = load_config()
            default_k = int(cfg.get("SPLIT", {}).get("DEFAULT_TEST_FOLD", 0))
        except Exception:
            default_k = 0
        logging.getLogger(__name__).info(
            "load_unified_split: no NESTED_OUTER_FOLD env var -> applying "
            "outer-fold view with k=%d (SPLIT.DEFAULT_TEST_FOLD).", default_k)
        df = _apply_outer_fold_view(df, outer_k=default_k)
    return df


def load_ct_fusion_predictions(cfg: dict, outputs_dir: str,
                                excel_row_idxs, unified_split_df):
    import os
    import numpy as np
    import pandas as pd

    fusion_cfg = (cfg.get("FUSION", {}) or {}).get("CT_FUSION", {}) or {}
    if not fusion_cfg.get("ENABLED", False):
        return None, None

    ct_root_raw = fusion_cfg.get("CT_RESULTS_DIR", "results_ct")
    ct_root = ct_root_raw if os.path.isabs(ct_root_raw) else os.path.join(outputs_dir, ct_root_raw)
    mode = fusion_cfg.get("MODE", "single")

    if mode == "single":
        single = fusion_cfg.get("SINGLE", {}) or {}
        members = [{"SUBMODALITY": single.get("SUBMODALITY"),
                    "MODEL":       single.get("MODEL")}]
    elif mode == "ensemble":
        members = (fusion_cfg.get("ENSEMBLE", {}) or {}).get("MEMBERS", []) or []
    else:
        return None, None
    if not members or any((not m.get("SUBMODALITY") or not m.get("MODEL"))
                          for m in members):
        return None, None

    try:
        split = unified_split_df.set_index("excel_row_idx")
    except KeyError:
        return None, None
    if split.index.has_duplicates:
        logging.getLogger(__name__).error(
            "load_ct_fusion_predictions: duplicated excel_row_idx in "
            "unified_split - cannot resolve uuids reliably.")
        return None, None
    try:
        uuids_target = [str(split.loc[int(eid), "uuid"]).strip()
                        for eid in excel_row_idxs]
    except KeyError:
        return None, None

    member_preds: list = []
    for m in members:
        sub = m["SUBMODALITY"]
        model = m["MODEL"]
        uuid_to_p: dict = {}
        for K in range(5):
            xlsx = os.path.join(ct_root, sub, f"outer_{K}",
                                "patient_predictions",
                                f"{model}_test_predictions.xlsx")
            if not os.path.exists(xlsx):
                return None, None
            df = pd.read_excel(xlsx)
            uuid_col = "UUID" if "UUID" in df.columns else "PatientID"
            if uuid_col not in df.columns or "RP_Probability" not in df.columns:
                return None, None
            for _, row in df.iterrows():
                uuid_to_p[str(row[uuid_col]).strip()] = float(row["RP_Probability"])
        try:
            member_preds.append(np.array([uuid_to_p[u] for u in uuids_target],
                                         dtype=np.float32))
        except KeyError:
            return None, None

    preds = (np.mean(member_preds, axis=0).astype(np.float32)
             if len(member_preds) > 1 else member_preds[0])
    info = {"mode": mode, "members": members, "n_loaded": len(uuids_target)}
    return preds, info


def setup_logging(log_dir: str, filename: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_h = logging.FileHandler(log_path, mode='a')
    file_h.setFormatter(formatter)
    logger.addHandler(file_h)

    return logger


def print_array_metrics(all_results: dict, test_results: dict, log) -> None:
    import numpy as np
    log.info("\n" + "=" * 70)
    log.info("  OUTER FOLD METRICS ARRAYS & AVERAGES")
    log.info("=" * 70)

    METRICS = ["auc", "f1", "sensitivity", "specificity", "pr_auc", "precision", "brier"]
    for model_name, rs in all_results.items():
        log.info(f"\n[{model_name}]")
        for metric in METRICS:
            if not any(metric in r.get("train", {}) for r in rs):
                continue
            tr_vals = [r["train"].get(metric, float('nan')) for r in rs]
            vl_vals = [r["val"].get(metric, float('nan')) for r in rs]
            te_vals = [r.get("test", {}).get(metric, float('nan')) for r in rs]

            tr_str = f"[{', '.join(f'{v:.3f}' for v in tr_vals)}] ({np.mean(tr_vals):.3f} ± {np.std(tr_vals):.3f})"
            vl_str = f"[{', '.join(f'{v:.3f}' for v in vl_vals)}] ({np.mean(vl_vals):.3f} ± {np.std(vl_vals):.3f})"

            if not np.isnan(te_vals[0]):
                te_str = f"[{', '.join(f'{v:.3f}' for v in te_vals)}] ({np.mean(te_vals):.3f} ± {np.std(te_vals):.3f})"
            else:
                te_str = "N/A"

            ens_test = test_results.get(model_name, {}).get(metric, float('nan'))
            ens_str = f" | Ens+Platt: {ens_test:.3f}" if not np.isnan(ens_test) else ""

            log.info(f"  {metric.upper():<11} | Train: {tr_str}")
            log.info(f"  {'':<11} | Val  : {vl_str}")
            log.info(f"  {'':<11} | Test : {te_str}{ens_str}")
