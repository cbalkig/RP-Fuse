#!/usr/bin/env python3
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if _os.uname().sysname == "Darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier, PassiveAggressiveClassifier)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                          QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              BaggingClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False
import shap

warnings.filterwarnings("ignore")
import logging as _logging
_logging.getLogger("shap").setLevel(_logging.WARNING)

from util import load_config, load_unified_split, read_table
import train_rpfuse as dg
from fusion_utils import build_clinical_features, build_rt_features
from tab_models import TorchMLPClassifier, ARCHS

_cfg = load_config()


_OUT = os.path.join(_cfg["OUTPUTS_DIR"], "results_tabular_compare")
RESULTS_DIR = _OUT

if os.environ.get("NESTED_OUTER_FOLD") is not None:
    _outer_k = os.environ["NESTED_OUTER_FOLD"]
    RESULTS_DIR = os.path.join(RESULTS_DIR, f"outer_{_outer_k}")


PRED_DIR = os.path.join(RESULTS_DIR, "patient_predictions")
_PETPLUS_NAMES = dg.pet_feature_names()
SEED = 42


def _models():
    bal = dict(class_weight="balanced")
    m = {
        "LogReg-L2":     lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=5000, **bal),
        "LogReg-L1":     lambda: LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000, **bal),
        "LogReg-Elastic":lambda: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                                    C=1.0, max_iter=8000, **bal),
        "Ridge":         lambda: RidgeClassifier(**bal),
        "SGD-log":       lambda: SGDClassifier(loss="log_loss", max_iter=5000, random_state=SEED, **bal),
        "PassiveAggr":   lambda: PassiveAggressiveClassifier(max_iter=5000, random_state=SEED, **bal),
        "LDA":           lambda: LinearDiscriminantAnalysis(),
        "QDA":           lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
        "GaussianNB":    lambda: GaussianNB(),
        "BernoulliNB":   lambda: BernoulliNB(),
        "KNN":           lambda: KNeighborsClassifier(n_neighbors=7),
        "SVM-RBF":       lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED, **bal),
        "SVM-Linear":    lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=SEED, **bal),
        "SVM-Poly":      lambda: SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=SEED, **bal),
        "DecisionTree":  lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED, **bal),
        "RandomForest":  lambda: RandomForestClassifier(n_estimators=400, max_depth=4, random_state=SEED, **bal),
        "ExtraTrees":    lambda: ExtraTreesClassifier(n_estimators=400, max_depth=4, random_state=SEED, **bal),
        "Bagging":       lambda: BaggingClassifier(n_estimators=200, random_state=SEED),
        "AdaBoost":      lambda: AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED),
        "GradBoost":     lambda: GradientBoostingClassifier(random_state=SEED),
        "HistGB":        lambda: HistGradientBoostingClassifier(max_depth=3, random_state=SEED),
        "XGBoost":       lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                              random_state=SEED, verbosity=0),
        "LightGBM":      lambda: LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "GaussianProc":  lambda: GaussianProcessClassifier(random_state=SEED),
        "MLP-64-32":     lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000, random_state=SEED),
        "MLP-128-64-32": lambda: MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, max_iter=3000, random_state=SEED),
        "MLP-wide":      lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=3000, random_state=SEED),
    }
    if _HAS_CAT:
        m["CatBoost"] = lambda: CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                                                   verbose=0, random_seed=SEED)
    for _a in ARCHS:
        m[f"NN-{_a}"] = (lambda a=_a: TorchMLPClassifier(arch=a, seed=SEED))
    return m
_TREE = {"DecisionTree", "RandomForest", "ExtraTrees", "GradBoost", "XGBoost",
         "LightGBM", "CatBoost"}
_LINEAR = {"LogReg-L2", "LogReg-L1", "LogReg-Elastic", "Ridge", "SGD-log", "LDA", "SVM-Linear"}


def _cohort():
    split = load_unified_split(_cfg["INPUTS_DIR"])
    allowed = {"single"}
    split = split.copy()
    split["uuid"] = split["uuid"].astype(str).str.strip()
    cohort_dir = os.path.join(_cfg["INPUTS_DIR"], _cfg["INPUTS"]["PROCESSED"])
    keep = []
    for _, r in split.iterrows():
        ud = os.path.join(cohort_dir, r["uuid"])
        ok_cls = os.path.isdir(ud) and dg.study_class(ud) in allowed
        ofld = r.get("OuterFold", -1)
        keep.append(bool(ok_cls) and pd.notna(ofld) and int(ofld) >= 0)
    split = split[pd.Series(keep, index=split.index)].reset_index(drop=True)
    print(f"Cohort: {len(split)} single-course patients with a valid OuterFold "
          f"(RP+ {int(split['rp_label'].sum())}).")
    import endpoint as _endpoint
    _ep = _endpoint.EndpointConfig(dg._full_cfg)
    if _ep.enabled:
        split = _endpoint.apply_to_split(split, dg.read_table(dg.EXCEL_PATH), _ep)
        print(f"  endpoint applied -> {len(split)} patients, "
              f"RP+ {int(split['rp_label'].sum())}.")
    return split


_DOSE_MODALITIES = ("dvh_curve", "dosimetrics", "dosimetrics_all")
_RADIOMICS_MODALITIES = ("radiomics_lung", "radiomics_v20", "radiomics_v20_nogtv", "radiomics_gate")
_PETOMICS_MODALITIES = ("petomics_gtv", "petomics_lung", "petomics_lung_si")
_TABLE_MODALITIES = _DOSE_MODALITIES + _RADIOMICS_MODALITIES + _PETOMICS_MODALITIES


def _load_dose_modality(kind: str):
    base = os.path.join(dg._input_dir, "Derivated_Modals")
    npz = os.path.join(base, f"{kind}.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        names, X, uu = list(z["names"]), z["X"].astype(np.float32), z["uuids"]
        return {str(u).strip(): X[i] for i, u in enumerate(uu)}, names
    df = pd.read_csv(os.path.join(base, f"{kind}.csv"))
    names = [c for c in df.columns if c != "uuid"]
    return {str(r["uuid"]).strip(): r[names].to_numpy(np.float32) for _, r in df.iterrows()}, names


def build_modality(modality, split):
    y = split["rp_label"].astype(int).to_numpy()
    folds = split["OuterFold"].astype(int).to_numpy()
    if modality in _TABLE_MODALITIES + ("petplus",):
        if modality == "petplus":
            fmap, names = dg.build_pet_plus_map()[0], _PETPLUS_NAMES
        else:
            fmap, names = _load_dose_modality(modality)
        dim = len(names)
        rows, mask = [], []
        for u in split["uuid"]:
            v = fmap.get(u)
            mask.append(v is not None)
            rows.append(v if v is not None else np.zeros(dim, np.float32))
        X = np.vstack(rows).astype(np.float32)
        m = np.array(mask)
        return X[m], y[m], folds[m], names, m
    master = dg.read_table(dg.EXCEL_PATH)
    Xb, names = (build_clinical_features(master) if modality == "clinical"
                 else build_rt_features(master))
    eid = split["excel_row_idx"].astype(int).to_numpy()
    ok = (eid >= 0) & (eid < len(Xb))
    return Xb[eid[ok]].astype(np.float32), y[ok], folds[ok], list(names), ok


def get_metrics_suite(y_true, y_prob):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, average_precision_score, brier_score_loss
    import numpy as np
    try:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
    except:
        auc, pr_auc, brier = float("nan"), float("nan"), float("nan")
    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    yp = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, yp)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f2 = (5 * tp) / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) > 0 else 0.0
    else:
        sens, spec, prec, f2 = 0.0, 0.0, 0.0, 0.0
    return auc, best_f1, sens, spec, pr_auc, prec, brier, f2


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.ravel(model.decision_function(X))
    return model.predict(X)


def oof_auc(make, X, y, folds):
    oof = np.zeros(len(y)); train_aucs = []
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        sc = StandardScaler().fit(X[tr])
        m = make(); m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = _scores(m, sc.transform(X[te]))
        if len(set(y[tr])) > 1:
            train_aucs.append(roc_auc_score(y[tr], _scores(m, sc.transform(X[tr]))))
    return roc_auc_score(y, oof), float(np.mean(train_aucs)) if train_aucs else float("nan")


def nested_oof_auc(make, X, y, folds, n_inner=5):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    oof_preds = np.zeros(len(y))
    train_aucs, train_f1s, train_sens, train_specs, train_pr_aucs, train_precs, train_briers, train_f2s = [], [], [], [], [], [], [], []
    val_preds_all = np.zeros(len(y))

    outer_k = int(os.environ.get("NESTED_OUTER_FOLD", -1))
    target_folds = [outer_k] if outer_k >= 0 else np.unique(folds)

    for k in target_folds:
        if k not in folds: continue
        te, dev = folds == k, folds != k
        Xdev, ydev = X[dev], y[dev]
        sc = StandardScaler().fit(Xdev)
        Xdev_s, Xte_s = sc.transform(Xdev), sc.transform(X[te])
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED + int(k))
        fold_preds = []

        for tr_i, va_i in skf.split(Xdev_s, ydev):
            m = make()
            try:
                m.fit(Xdev_s[tr_i], ydev[tr_i], X_val=Xdev_s[va_i], y_val=ydev[va_i])
            except TypeError:
                m.fit(Xdev_s[tr_i], ydev[tr_i])

            pv = _scores(m, Xdev_s[va_i])
            val_preds_all[dev.nonzero()[0][va_i]] = pv

            ptr = _scores(m, Xdev_s[tr_i])
            if len(set(ydev[tr_i])) > 1:
                t_auc, t_f1, t_sen, t_spe, t_pra, t_prc, t_br, t_f2 = get_metrics_suite(ydev[tr_i], ptr)
                train_aucs.append(t_auc); train_f1s.append(t_f1); train_sens.append(t_sen); train_specs.append(t_spe); train_pr_aucs.append(t_pra); train_precs.append(t_prc); train_briers.append(t_br); train_f2s.append(t_f2)

            ab = dg.fit_platt(pv, ydev[va_i])
            pte = _scores(m, Xte_s)
            fold_preds.append(dg.apply_platt(pte, ab))

        oof_preds[te] = np.mean(np.stack(fold_preds, 0), 0)

    te_mask = folds == outer_k if outer_k >= 0 else np.ones(len(y), dtype=bool)
    test_auc, test_f1, test_sens, test_spec, test_pra, test_prc, test_br, test_f2 = get_metrics_suite(y[te_mask], oof_preds[te_mask])
    val_auc, val_f1, val_sens, val_spec, val_pra, val_prc, val_br, val_f2 = get_metrics_suite(y, val_preds_all)

    tr_metrics = (np.mean(train_aucs), np.mean(train_f1s), np.mean(train_sens), np.mean(train_specs), np.mean(train_pr_aucs), np.mean(train_precs), np.mean(train_briers), np.mean(train_f2s) if train_f2s else 0.0)
    val_metrics = (val_auc, val_f1, val_sens, val_spec, val_pra, val_prc, val_br, val_f2)
    te_metrics = (test_auc, test_f1, test_sens, test_spec, test_pra, test_prc, test_br, test_f2)

    return oof_preds, te_mask, tr_metrics, val_metrics, te_metrics


def shap_best(name, make, X, y, names, modality):
    try:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        model = make(); model.fit(Xs, y)
        if name in _TREE:
            sv = shap.TreeExplainer(model).shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
        elif name in _LINEAR:
            sv = shap.LinearExplainer(model, Xs).shap_values(Xs)
        else:
            bg = shap.sample(Xs, min(50, len(Xs)), random_state=SEED)
            sv = shap.KernelExplainer(lambda d: _scores(model, d), bg).shap_values(Xs, nsamples=100)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[-1]
        if sv.ndim == 2 and sv.shape[1] == len(names):
            imp = pd.DataFrame({
                "feature": names,
                "mean_abs_shap": np.abs(sv).mean(0),
                "mean_shap": sv.mean(0),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            imp.to_csv(os.path.join(_OUT, f"{modality}_feature_importance.csv"), index=False)
            print(f"  feature effects -> {modality}_feature_importance.csv "
                  f"({len(names)} features; top: {imp.iloc[0]['feature']})")
        import matplotlib as _mpl
        with _mpl.rc_context({"font.size": 15, "axes.labelsize": 16,
                              "xtick.labelsize": 14, "ytick.labelsize": 14}):
            shap.summary_plot(sv, Xs, feature_names=names, show=False,
                              max_display=8, plot_size=(10, 6.5))
            plt.title(f"{modality}: feature importance - fusion network {name}", fontsize=16)
            plt.tight_layout()
            dst = os.path.join(_OUT, f"{modality}_shap.png")
            plt.savefig(dst, dpi=200, bbox_inches="tight"); plt.close()
        print(f"  SHAP -> {dst}")
    except Exception as exc:
        print(f"  SHAP skipped for {modality}/{name}: {exc}")


def run_modality(modality, split):
    X, y, folds, names, rows = build_modality(modality, split)
    split = split[rows].reset_index(drop=True)
    print(f"\n=== {modality}: X={X.shape}, RP+={int(y.sum())}/{len(y)} ===", flush=True)

    outer_k = int(os.environ.get("NESTED_OUTER_FOLD", -1))
    pred_dir = globals().get("PRED_DIR", os.path.join(_OUT, "patient_predictions"))
    if outer_k >= 0:
        os.makedirs(pred_dir, exist_ok=True)

    rows = []
    print(f"  {'Model':15s} | {'Train (AUC/F1/Sn/Sp/PR/Prc/Br)':40s} | {'Val (AUC/F1/Sn/Sp/PR/Prc/Br)':40s} | {'Test (AUC/F1/Sn/Sp/PR/Prc/Br)':40s}")
    for name, make in _models().items():
        try:
            preds, te_mask, tr_m, val_m, te_m = nested_oof_auc(make, X, y, folds)
            if outer_k >= 0:
                pd.DataFrame({
                    "UUID": split.loc[te_mask, "uuid"].values,
                    "TrueLabel": y[te_mask],
                    "RP_Probability": preds[te_mask]
                }).to_excel(os.path.join(pred_dir, f"{modality}_{name}_test_predictions.xlsx"), index=False)
        except Exception as exc:
            print(f"  {name}: failed ({exc})"); continue

        rows.append({"model": name, "oof_auc": round(te_m[0], 4), "oof_f2": round(te_m[7], 4), "train_auc": round(tr_m[0], 4), "val_auc": round(val_m[0], 4)})
        tr_s = f"{tr_m[0]:.2f}/{tr_m[1]:.2f}/{tr_m[2]:.2f}/{tr_m[3]:.2f}/{tr_m[4]:.2f}/{tr_m[5]:.2f}/{tr_m[6]:.2f}"
        val_s = f"{val_m[0]:.2f}/{val_m[1]:.2f}/{val_m[2]:.2f}/{val_m[3]:.2f}/{val_m[4]:.2f}/{val_m[5]:.2f}/{val_m[6]:.2f}"
        te_s = f"{te_m[0]:.2f}/{te_m[1]:.2f}/{te_m[2]:.2f}/{te_m[3]:.2f}/{te_m[4]:.2f}/{te_m[5]:.2f}/{te_m[6]:.2f}"
        print(f"  {name:15s} | {tr_s:40s} | {val_s:40s} | {te_s:40s}")

    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    out_dir = globals().get("RESULTS_DIR", _OUT)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{modality}_model_comparison.csv"), index=False)
    best = df.iloc[0]

    def _top(pred):
        sub = df[df.model.apply(pred)]
        return (sub.iloc[0]["model"], float(sub.iloc[0]["oof_auc"])) if len(sub) else (None, float("nan"))
    nn_model,  nn_auc  = _top(lambda m: m.startswith("NN-"))
    mlp_model, mlp_auc = _top(lambda m: m.startswith("MLP"))
    cls_model, cls_auc = _top(lambda m: not m.startswith(("NN-", "MLP")))
    if nn_model is not None and outer_k < 0:
        shap_best(nn_model, _models()[nn_model], X, y, names, modality)
    return {"modality": modality, "best_overall_model": best["model"],
            "best_overall_auc": round(float(best["oof_auc"]), 4),
            "best_nn_model": nn_model, "best_nn_auc": round(float(nn_auc), 4) if nn_model else float("nan")}


def shap_pass(split):
    out_dir = globals().get("RESULTS_DIR", _OUT)
    summ = os.path.join(out_dir, "nested_cv_summary.csv")
    if not os.path.exists(summ):
        print(f"  shap_pass: {summ} not found; skipping deterministic SHAP."); return
    s = pd.read_csv(summ)
    s["mod"] = s["variant"].str.rsplit("_", n=1).str[0]
    s["algo"] = s["variant"].str.rsplit("_", n=1).str[1]
    print("\n########## deterministic SHAP on pooled-best NN per modality ##########", flush=True)
    for m in ["clinical", "rt", "dosimetrics", "dosimetrics_all", "dvh_curve", "radiomics_lung", "radiomics_v20", "radiomics_v20_nogtv", "radiomics_gate", "petplus"]:
        nn = s[(s["mod"] == m) & (s["algo"].str.startswith("NN-"))].sort_values(
            "OOF_AUC", ascending=False)
        if nn.empty:
            print(f"  shap_pass[{m}]: no NN-* in summary; skipping"); continue
        best_nn = nn.iloc[0]["algo"]
        try:
            X, y, _, names, _rows = build_modality(m, split)
            print(f"  shap_pass[{m}] pooled-best NN = {best_nn} (OOF AUC {nn.iloc[0]['OOF_AUC']:.3f})")
            shap_best(best_nn, _models()[best_nn], X, y, names, m)
        except Exception as exc:
            print(f"  shap_pass[{m}] failed: {exc}")


def val_selected_table():
    out_dir = globals().get("RESULTS_DIR", _OUT)
    summ_path = os.path.join(out_dir, "nested_cv_summary.csv")
    if not os.path.exists(summ_path):
        print("  val_selected_table: no nested_cv_summary.csv; skipping."); return
    summ = pd.read_csv(summ_path)
    summ["mod"] = summ["variant"].str.rsplit("_", n=1).str[0]
    summ["algo"] = summ["variant"].str.rsplit("_", n=1).str[1]
    rows = []
    for mod in sorted(summ["mod"].unique()):
        mcs = glob.glob(os.path.join(out_dir, "outer_*", f"{mod}_model_comparison.csv"))
        if not mcs:
            continue
        val = pd.concat([pd.read_csv(f) for f in mcs]).groupby("model")["val_auc"].mean()
        cand = summ[summ["mod"] == mod].set_index("algo")
        val = val[[m for m in val.index if m in cand.index]]
        if val.empty:
            continue
        best = val.idxmax()
        r = cand.loc[best]

        oof_all = pd.to_numeric(cand["OOF_AUC"], errors="coerce").dropna()
        sel_oof = float(r["OOF_AUC"])
        both = pd.DataFrame({"val": val, "oof": cand.loc[val.index, "OOF_AUC"]}).astype(float)
        rho = float(both["val"].corr(both["oof"], method="spearman")) if len(both) > 2 else float("nan")

        rows.append({"modality": mod, "best_model": best, "val_auc": round(float(val.max()), 4),
                     "OOF_AUC": r["OOF_AUC"], "CI_low": r["OOF_AUC_CI95_low"],
                     "CI_high": r["OOF_AUC_CI95_high"], "foldwise_mean": r["OOF_AUC_foldwise_mean"],
                     "foldwise_std": r["OOF_AUC_foldwise_std"], "OOF_PR_AUC": r.get("OOF_PR_AUC"),
                     "OOF_F1": r["OOF_F1"], "OOF_F2": r.get("OOF_F2"), "OOF_Sens": r["OOF_Sens"],
                     "OOF_Spec": r["OOF_Spec"], "OOF_Brier": r.get("OOF_Brier"),
                     "roster_n": int(len(oof_all)),
                     "roster_max": round(float(oof_all.max()), 4),
                     "roster_q75": round(float(oof_all.quantile(0.75)), 4),
                     "roster_median": round(float(oof_all.median()), 4),
                     "roster_q25": round(float(oof_all.quantile(0.25)), 4),
                     "roster_min": round(float(oof_all.min()), 4),
                     "selection_gap": round(float(oof_all.max() - oof_all.median()), 4),
                     "selected_oof_rank": int((oof_all > sel_oof).sum() + 1),
                     "val_oof_spearman": round(rho, 3)})
    if not rows:
        print("  val_selected_table: no modality had both val + OOF; skipping."); return
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "best_per_modality_val.csv"), index=False)
    print("\n########## VAL-SELECTED best per modality (report each model's OOF) ##########")
    for _, r in out.iterrows():
        print(f"  {r['modality']:16s} {r['best_model']:16s} val={r['val_auc']:.3f} -> "
              f"OOF={r['OOF_AUC']:.3f} [{r['CI_low']:.2f}-{r['CI_high']:.2f}] "
              f"PR={r['OOF_PR_AUC']:.3f} F1={r['OOF_F1']:.2f} F2={r['OOF_F2']:.2f} "
              f"Sn={r['OOF_Sens']:.2f} Sp={r['OOF_Spec']:.2f} BS={r['OOF_Brier']:.3f}")

    print("\n########## ROSTER DISTRIBUTION (what the modality really carries) ##########")
    print(f"  {'modality':16s} {'n':>3} {'min':>6} {'q25':>6} {'med':>6} {'q75':>6} {'max':>6}"
          f" {'gap':>6} {'rank':>5} {'val~OOF':>8}")
    for _, r in out.iterrows():
        print(f"  {r['modality']:16s} {int(r['roster_n']):>3} {r['roster_min']:>6.3f} "
              f"{r['roster_q25']:>6.3f} {r['roster_median']:>6.3f} {r['roster_q75']:>6.3f} "
              f"{r['roster_max']:>6.3f} {r['selection_gap']:>6.3f} "
              f"{int(r['selected_oof_rank']):>4d}/{int(r['roster_n']):<2d} "
              f"{r['val_oof_spearman']:>7.2f}")
    print("  gap     = roster max - median: the part of the headline that is model choice,")
    print("            not modality. Large gap => quote the range, not the point estimate.")
    print("  rank    = where the VAL-selected model lands in the roster's OOF ranking.")
    print("  val~OOF = Spearman(inner-val AUC, pooled-OOF AUC) across the roster. Near 0 or")
    print("            negative means selection is picking noise, so read the reported OOF")
    print("            as a draw from the distribution rather than as 'the best model'.")
    print(f"\n  -> {os.path.join(out_dir, 'best_per_modality_val.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["clinical", "rt", "dosimetrics", "dosimetrics_all", "dvh_curve", "radiomics_lung", "radiomics_v20", "radiomics_v20_nogtv", "radiomics_gate", "petplus", "all"], default="all")
    args = ap.parse_args()
    split = _cohort()
    mods = ["clinical", "rt", "dosimetrics", "dosimetrics_all", "dvh_curve", "radiomics_lung", "radiomics_v20", "radiomics_v20_nogtv", "radiomics_gate", "petplus"] if args.modality == "all" else [args.modality]
    summary = []
    for i, m in enumerate(mods, 1):
        print(f"\n########## modality {i}/{len(mods)}: {m} "
              f"(building features --- PET+ loads volumes, may take a minute) ##########",
              flush=True)
        try:
            summary.append(run_modality(m, split))
        except Exception as exc:
            print(f"  !! {m} FAILED: {exc}")
    out_dir = globals().get("RESULTS_DIR", _OUT)
    os.makedirs(out_dir, exist_ok=True)
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(out_dir, "best_per_modality.csv"), index=False)
        print(f"\nWrote summary ({len(summary)}/{len(mods)} modalities) -> "
              f"{os.path.join(out_dir, 'best_per_modality.csv')}")


if __name__ == "__main__":
    if bool(_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False)) and os.environ.get("NESTED_OUTER_FOLD") is None:
        from nested_cv import run_nested_cv
        run_nested_cv(main_fn=main, n_outer=int(_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5)),
                      module_globals=globals(), results_dir_key="RESULTS_DIR",
                      pred_dir_key="PRED_DIR", log=_logging.getLogger(__name__))
        val_selected_table()
        shap_pass(_cohort())
    else:
        main()
