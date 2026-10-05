from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_FOLD_VAL_METRIC_KEYS = ("auc", "f1", "sensitivity", "specificity", "acc")


def _fold_has_stale_variant_cache(outer_dir: str) -> tuple[bool, str | None]:
    pred_dir = os.path.join(outer_dir, "patient_predictions")
    if not os.path.isdir(pred_dir):
        return False, None
    metas = sorted(glob.glob(os.path.join(pred_dir, "*_cv_meta.json")))
    if not metas:
        return False, None
    for path in metas:
        variant = os.path.basename(path).replace("_cv_meta.json", "")
        try:
            with open(path) as fp:
                meta = json.load(fp)
        except Exception as exc:
            return True, f"{variant}: meta unreadable ({exc})"
        fold_val = meta.get("fold_val") or []
        if not fold_val:
            return True, f"{variant}: meta missing per-fold val metrics (old schema)"
        for i, split in enumerate(fold_val):
            vals = [split.get(k, 0.0) for k in _FOLD_VAL_METRIC_KEYS]
            if all(v == 0.0 for v in vals):
                return True, f"{variant}: fold {i} all val metrics == 0"
    return False, None


def _bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                       n_boot: int = 1000, seed: int = 42
                       ) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score as _auc
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(_auc(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _max_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import f1_score as _f1
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(y_prob):
        f1 = _f1(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _cm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    from sklearn.metrics import f1_score, confusion_matrix
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        acc = float((y_pred == y_true).mean())
    except Exception:
        sens = spec = acc = float("nan")
    return f1, sens, spec, acc


def _log_pooled_stream_competence(orig_results_dir: str, n_outer: int,
                                  n_repeats: int, multi_repeat: bool,
                                  logger: logging.Logger) -> None:
    from sklearn.metrics import roc_auc_score
    aux_frames, ce_frames = [], []
    for rp in range(n_repeats):
        repeat_dir = (os.path.join(orig_results_dir, f"repeat_{rp}")
                      if multi_repeat else orig_results_dir)
        for k in range(n_outer):
            ap = os.path.join(repeat_dir, f"outer_{k}", "aux_oof.csv")
            cp = os.path.join(repeat_dir, f"outer_{k}", "component_effects.csv")
            if os.path.isfile(ap):
                aux_frames.append(pd.read_csv(ap))
            if os.path.isfile(cp):
                ce_frames.append(pd.read_csv(cp))
    if not aux_frames or not ce_frames:
        return
    pool = pd.concat(aux_frames, ignore_index=True)
    if "label" not in pool.columns or pool["label"].nunique() < 2:
        return
    weights = (pd.concat(ce_frames).groupby("component")["mean_fusion_weight"]
               .mean().to_dict())
    y = pool["label"].values
    rows = []
    for c in pool.columns:
        if not c.startswith("aux_"):
            continue
        s = c[4:]
        if pool[c].notna().sum() != len(pool):
            continue
        rows.append((s, weights.get(s, float("nan")), roc_auc_score(y, pool[c].values)))
    if not rows:
        return
    rows.sort(key=lambda r: -(r[1] if r[1] == r[1] else -1.0))
    logger.info("\n  Per-stream pooled-OOF competence vs learned fusion weight "
                "(n=%d, outer-test only):", len(pool))
    logger.info("    %-8s %8s   %s", "stream", "weight", "pooled-OOF AUC")
    for s, w, a in rows:
        logger.info("    %-8s %8s   %.3f", s, (f"{w:.3f}" if w == w else "  n/a"), a)


def aggregate_nested_results(orig_results_dir: str, n_outer: int,
                              logger: logging.Logger,
                              n_repeats: int = 1) -> pd.DataFrame:
    from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                                 average_precision_score, brier_score_loss)

    multi_repeat = n_repeats > 1
    logger.info("\n" + "=" * 70)
    logger.info("  Nested-CV aggregation: %d repeat(s) x %d outer folds",
                n_repeats, n_outer)
    logger.info("=" * 70)

    variant_data: dict[str, dict[int, list[pd.DataFrame]]] = {}
    key_col: str | None = None

    for rp in range(n_repeats):
        repeat_dir = (os.path.join(orig_results_dir, f"repeat_{rp}")
                      if multi_repeat else orig_results_dir)
        for k in range(n_outer):
            outer_dir = os.path.join(repeat_dir, f"outer_{k}", "patient_predictions")
            if not os.path.isdir(outer_dir):
                logger.warning("  repeat_%d/outer_%d: missing %s - skipping",
                               rp, k, outer_dir)
                continue
            for path in glob.glob(os.path.join(outer_dir, "*_test_predictions.xlsx")):
                variant = os.path.basename(path).replace("_test_predictions.xlsx", "")
                if variant == "ALL_MODELS":
                    continue
                try:
                    df_fold = pd.read_excel(path)
                except Exception as e:
                    logger.warning("  failed to read %s: %s", path, e)
                    continue
                df_fold["__outer_fold"] = k
                variant_data.setdefault(variant, {}).setdefault(rp, []).append(df_fold)

    for rp_frames in variant_data.values():
        for frames in rp_frames.values():
            for df in frames:
                for cand in ("PatientName", "patient_name", "UUID", "Excel_idx"):
                    if cand in df.columns:
                        key_col = cand
                        break
                if key_col:
                    break
            if key_col:
                break
        if key_col:
            break

    rows = []
    for variant, rp_frames in sorted(variant_data.items()):
        per_repeat_aucs: list[float] = []
        pooled_y_true: list[int] = []
        pooled_y_prob: list[float] = []
        pooled_y_pred: list[int] = []
        pooled_y_pred_y: list[int] = []
        foldwise_aucs: list[float] = []

        for rp in range(n_repeats):
            fold_frames = rp_frames.get(rp, [])
            if not fold_frames:
                continue
            for fr in fold_frames:
                if "TrueLabel" in fr.columns and "RP_Probability" in fr.columns:
                    yt = fr["TrueLabel"].values.astype(int)
                    yp = fr["RP_Probability"].values.astype(float)
                    if len(np.unique(yt)) > 1:
                        try:
                            foldwise_aucs.append(float(roc_auc_score(yt, yp)))
                        except Exception:
                            pass
            df_rep = pd.concat(fold_frames, ignore_index=True, sort=False)
            if key_col and key_col in df_rep.columns:
                agg_dict: dict = {"TrueLabel": "first", "RP_Probability": "mean"}
                if "PredictedLabel" in df_rep.columns:
                    agg_dict["PredictedLabel"] = "first"
                if "PredictedLabel_Youden" in df_rep.columns:
                    agg_dict["PredictedLabel_Youden"] = "first"
                df_rep = df_rep.groupby(key_col, as_index=False).agg(agg_dict)
            if "TrueLabel" not in df_rep.columns or "RP_Probability" not in df_rep.columns:
                logger.warning("  %s repeat_%d: missing cols - skipping", variant, rp)
                continue
            y_true = df_rep["TrueLabel"].values.astype(int)
            y_prob = df_rep["RP_Probability"].values.astype(float)
            if len(np.unique(y_true)) > 1:
                try:
                    per_repeat_aucs.append(float(roc_auc_score(y_true, y_prob)))
                except Exception:
                    pass
            pooled_y_true.extend(y_true.tolist())
            pooled_y_prob.extend(y_prob.tolist())
            if "PredictedLabel" in df_rep.columns:
                pooled_y_pred.extend(df_rep["PredictedLabel"].values.astype(int).tolist())
            if "PredictedLabel_Youden" in df_rep.columns:
                pooled_y_pred_y.extend(df_rep["PredictedLabel_Youden"].values.astype(int).tolist())

        if not per_repeat_aucs:
            logger.warning("  %s: no valid AUC computed - skipping", variant)
            continue

        y_true_arr = np.array(pooled_y_true, dtype=int)
        y_prob_arr = np.array(pooled_y_prob, dtype=float)
        if len(pooled_y_pred) == len(y_true_arr):
            y_pred_arr = np.array(pooled_y_pred, dtype=int)
            oof_thr = float("nan")
        else:
            oof_thr = _max_f1_threshold(y_true_arr, y_prob_arr)
            y_pred_arr = (y_prob_arr >= oof_thr).astype(int)
            logger.warning("  %s: PredictedLabel missing/incomplete - falling back to "
                           "pooled max-F1 threshold (mildly optimistic F1/Sens/Spec).", variant)

        auc_mean = float(np.mean(per_repeat_aucs))
        auc_std  = float(np.std(per_repeat_aucs)) if len(per_repeat_aucs) > 1 else 0.0
        ci_low, ci_high = _bootstrap_auc_ci(y_true_arr, y_prob_arr)

        if foldwise_aucs:
            fw_mean = float(np.mean(foldwise_aucs))
            fw_std  = float(np.std(foldwise_aucs, ddof=1)) if len(foldwise_aucs) > 1 else 0.0
        else:
            fw_mean = fw_std = float("nan")

        f1, sens, spec, acc = _cm_metrics(y_true_arr, y_pred_arr)

        n_pts = int(len(y_true_arr) // max(n_repeats, 1))
        row = {
            "variant":            variant,
            "n_patients_oof":     n_pts,
            "n_outer_folds":      n_outer,
            "n_repeats":          n_repeats,
            "OOF_AUC":            auc_mean,
            "OOF_AUC_std":        auc_std,
            "OOF_AUC_CI95_low":   ci_low,
            "OOF_AUC_CI95_high":  ci_high,
            "OOF_AUC_foldwise_mean": fw_mean,
            "OOF_AUC_foldwise_std":  fw_std,
            "n_folds_scored":        len(foldwise_aucs),
            "OOF_threshold":      oof_thr,
            "OOF_F1":             f1,
            "OOF_Sens":           sens,
            "OOF_Spec":           spec,
            "OOF_Acc":            acc,
        }
        try:
            row["OOF_PR_AUC"] = float(average_precision_score(y_true_arr, y_prob_arr))
        except Exception:
            row["OOF_PR_AUC"] = float("nan")
        try:
            row["OOF_Brier"] = float(brier_score_loss(y_true_arr, y_prob_arr))
        except Exception:
            row["OOF_Brier"] = float("nan")
        _tn, _fp, _fn, _tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
        row["OOF_F2"] = ((5 * _tp) / (5 * _tp + 4 * _fn + _fp)
                         if (5 * _tp + 4 * _fn + _fp) > 0 else 0.0)
        if len(pooled_y_pred_y) == len(y_true_arr):
            f1y, sy, spy, accy = _cm_metrics(y_true_arr, np.array(pooled_y_pred_y, dtype=int))
            row.update({"OOF_F1_Youden": f1y, "OOF_Sens_Youden": sy,
                        "OOF_Spec_Youden": spy, "OOF_Acc_Youden": accy})
        rows.append(row)

    if not rows:
        logger.error("Nested-CV aggregation: no completed outer folds to "
                     "aggregate (every fold failed - e.g. an OOM-killed "
                     "DataLoader worker). Nothing written; fix the per-fold "
                     "failure above and re-run.")
        return None

    df_summary = pd.DataFrame(rows).sort_values("OOF_AUC", ascending=False)
    summary_path = os.path.join(orig_results_dir, "nested_cv_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    logger.info("\n  %-22s  %-6s  %-28s  %-16s  %-18s  %-18s",
                "Model", "n_pts", "AUC (mean±std [95% CI])", "Fold-wise (m±SD)",
                "F1/Sens/Spec @F1", "F1/Sens/Spec @Youden")
    logger.info("  " + "-" * 116)
    for _, r in df_summary.iterrows():
        auc_str = (f"{r['OOF_AUC']:.3f}±{r['OOF_AUC_std']:.3f} "
                   f"[{r['OOF_AUC_CI95_low']:.3f}-{r['OOF_AUC_CI95_high']:.3f}]")
        fw_str = (f"{r['OOF_AUC_foldwise_mean']:.3f}±{r['OOF_AUC_foldwise_std']:.3f}"
                  if pd.notna(r.get("OOF_AUC_foldwise_mean")) else "-")
        f1_str = f"{r['OOF_F1']:.2f}/{r['OOF_Sens']:.2f}/{r['OOF_Spec']:.2f}"
        y_str = (f"{r['OOF_F1_Youden']:.2f}/{r['OOF_Sens_Youden']:.2f}/{r['OOF_Spec_Youden']:.2f}"
                 if "OOF_F1_Youden" in r and pd.notna(r.get("OOF_F1_Youden")) else "-")
        logger.info("  %-22s  %-6d  %-28s  %-16s  %-18s  %-18s",
                    r["variant"], r["n_patients_oof"], auc_str, fw_str, f1_str, y_str)
    logger.info("  " + "-" * 116)
    logger.info("  Saved -> %s", summary_path)
    try:
        _log_pooled_stream_competence(orig_results_dir, n_outer, n_repeats,
                                      multi_repeat, logger)
    except Exception as _e:
        logger.info("  [stream-competence] skipped: %s", _e)
    return df_summary


def run_nested_cv(main_fn,
                   n_outer: int,
                   module_globals: dict,
                   results_dir_key: str = "RESULTS_DIR",
                   pred_dir_key: str = "PRED_DIR",
                   log: logging.Logger | None = None,
                   n_repeats: int = 1,
                   quick: bool = False) -> pd.DataFrame:
    if log is None:
        log = logging.getLogger(__name__)

    if results_dir_key not in module_globals:
        raise KeyError(f"{results_dir_key} not in module globals")
    orig_results = module_globals[results_dir_key]
    orig_pred = module_globals.get(pred_dir_key, os.path.join(orig_results, "patient_predictions"))

    if quick:
        n_outer = 1
        log.info("[QUICK] Quick mode: running only outer fold 0 of each repeat.")

    multi_repeat = n_repeats > 1

    log.info("\n" + "#" * 70)
    log.info("  NESTED CV: %d repeat(s) x %d outer folds%s",
             n_repeats, n_outer, "  [QUICK]" if quick else "")
    log.info("  Results root: %s", orig_results)
    log.info("#" * 70 + "\n")

    for rp in range(n_repeats):
        repeat_dir = (os.path.join(orig_results, f"repeat_{rp}")
                      if multi_repeat else orig_results)
        if multi_repeat:
            os.makedirs(repeat_dir, exist_ok=True)

        checkpoint_path = os.path.join(repeat_dir, "nested_cv_checkpoint.json")
        completed_folds: set[int] = set()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path) as fp:
                    data = json.load(fp)
                completed_folds = set(data.get("completed_folds", []))
                log.info("[RESUME] repeat_%d checkpoint: folds done = %s",
                         rp, sorted(completed_folds))
            except Exception as exc:
                log.warning("[RESUME] Could not read checkpoint: %s", exc)

        stale_folds: list[tuple[int, str]] = []
        for k in sorted(completed_folds):
            is_stale, reason = _fold_has_stale_variant_cache(
                os.path.join(repeat_dir, f"outer_{k}"))
            if is_stale:
                stale_folds.append((k, reason or "unspecified"))
        for k, reason in stale_folds:
            completed_folds.discard(k)
            log.info("[RESUME] repeat_%d outer fold %d stale (%s) - re-executing.",
                     rp, k, reason)
        if stale_folds:
            try:
                with open(checkpoint_path, "w") as fp:
                    json.dump({"completed_folds": sorted(completed_folds)}, fp)
            except Exception as exc:
                log.warning("[RESUME] Could not save trimmed checkpoint: %s", exc)

        if multi_repeat:
            log.info("\n" + "#" * 70)
            log.info("#  REPEAT %d / %d", rp + 1, n_repeats)
            log.info("#" * 70)

        for k in range(n_outer):
            if k in completed_folds:
                log.info("\n[RESUME] repeat_%d / outer_%d already complete - skipping.\n",
                         rp, k)
                continue

            log.info("\n" + "#" * 70)
            log.info("#  REPEAT %d/%d  OUTER FOLD %d/%d", rp + 1, n_repeats, k + 1, n_outer)
            log.info("#" * 70 + "\n")

            outer_dir = os.path.join(repeat_dir, f"outer_{k}")
            outer_pred = os.path.join(outer_dir, "patient_predictions")
            os.makedirs(outer_dir, exist_ok=True)
            os.makedirs(outer_pred, exist_ok=True)
            module_globals[results_dir_key] = outer_dir
            if pred_dir_key in module_globals:
                module_globals[pred_dir_key] = outer_pred

            os.environ["NESTED_OUTER_FOLD"] = str(k)
            os.environ["NESTED_CV_REPEAT"]  = str(rp)
            fold_ok = False
            try:
                main_fn()
                fold_ok = True
            except Exception as e:
                log.error("repeat_%d outer fold %d failed: %s", rp, k, e, exc_info=True)
            finally:
                os.environ.pop("NESTED_OUTER_FOLD", None)
                os.environ.pop("NESTED_CV_REPEAT", None)

            if fold_ok:
                completed_folds.add(k)
                try:
                    with open(checkpoint_path, "w") as fp:
                        json.dump({"completed_folds": sorted(completed_folds)}, fp)
                    log.info("[RESUME] repeat_%d outer fold %d marked complete.", rp, k)
                except Exception as exc:
                    log.warning("[RESUME] Could not save checkpoint: %s", exc)

    module_globals[results_dir_key] = orig_results
    if pred_dir_key in module_globals:
        module_globals[pred_dir_key] = orig_pred

    return aggregate_nested_results(orig_results, n_outer, log, n_repeats=n_repeats)
