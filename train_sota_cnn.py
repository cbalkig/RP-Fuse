#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import torch.multiprocessing as _torch_mp
import train_dose as td

_torch_mp.set_sharing_strategy("file_system")
import train_rpfuse as dg
import dataclasses
import json
import torch.nn.functional as F
import gate as _gate

log = logging.getLogger("train_sota")

SEED = getattr(td, "SEED", 42)
TARGET_SHAPE = getattr(td, "TARGET_SHAPE", (128, 128, 128))
DOSE_NORM_CGY = 6000.0


def _device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def _resize(vol: np.ndarray) -> np.ndarray:
    if tuple(vol.shape) == tuple(TARGET_SHAPE):
        return vol
    z = [t / c for t, c in zip(TARGET_SHAPE, vol.shape)]
    return scipy.ndimage.zoom(vol, z, order=1, prefilter=False)


def _load_ct(path: str) -> np.ndarray:
    v = np.load(path).astype(np.float32)
    lo, hi = -600 - 750, -600 + 750
    v = np.clip(v, lo, hi)
    return (v - lo) / (hi - lo)


def _load_dose(path: str) -> np.ndarray:
    v = np.load(path).astype(np.float32)
    return np.clip(v / DOSE_NORM_CGY, 0.0, 1.0)


def _build_liang(paths: dict) -> np.ndarray | None:
    if not paths.get("dose"):
        return None
    d = _resize(_load_dose(paths["dose"]))
    return d[None].astype(np.float32)


def _build_kapoor(paths: dict) -> np.ndarray | None:
    if not (paths.get("ct") and paths.get("dose")):
        return None
    ct = _resize(_load_ct(paths["ct"]))
    ds = _resize(_load_dose(paths["dose"]))
    return np.stack([ct, ds], 0).astype(np.float32)


class C3D3D(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 2, p: float = 0.5):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool3d(2))
        self.features = nn.Sequential(
            blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 256),
            blk(256, 256), nn.AdaptiveAvgPool3d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def _make_liang():
    return C3D3D(in_ch=1, n_classes=2, p=0.5)


def _make_kapoor():
    from monai.networks.nets import DenseNet121
    return DenseNet121(spatial_dims=3, in_channels=2, out_channels=2,
                       dropout_prob=0.2)


class ViTWrapper(nn.Module):
    def __init__(self, img_size=TARGET_SHAPE, patch: int = 16):
        super().__init__()
        from monai.networks.nets import ViT
        self.vit = ViT(in_channels=1, img_size=tuple(img_size),
                       patch_size=(patch, patch, patch), spatial_dims=3,
                       hidden_size=384, mlp_dim=1536, num_layers=6, num_heads=6,
                       classification=True, num_classes=2, post_activation=None,
                       dropout_rate=0.1)

    def forward(self, x):
        out = self.vit(x)
        return out[0] if isinstance(out, (list, tuple)) else out


def _build_midroni(paths: dict) -> np.ndarray | None:
    if not paths.get("ct"):
        return None
    return _resize(_load_ct(paths["ct"]))[None].astype(np.float32)


def _make_midroni():
    return ViTWrapper(img_size=TARGET_SHAPE, patch=16)


GATE_LR = 1e-3


def _build_kapoor_gated(paths: dict) -> np.ndarray | None:
    if not (paths.get("ct") and paths.get("dose") and paths.get("lung")):
        return None
    ct = _resize(_load_ct(paths["ct"]))
    ds = _resize(_load_dose(paths["dose"]))
    lung = np.load(paths["lung"]) > 0
    gtv = (np.load(paths["gtv"]) > 0) if paths.get("gtv") else np.zeros_like(lung)
    healthy = _resize((lung & ~gtv).astype(np.float32)) >= 0.5
    gtv_r = _resize(gtv.astype(np.float32)) >= 0.5
    if gtv_r.any():
        sp = dg.ct_voxel_spacing(paths["ct"])
        sp_r = [s0 * n0 / n1 for s0, n0, n1 in zip(sp, lung.shape, TARGET_SHAPE)]
        dist = scipy.ndimage.distance_transform_edt(~gtv_r, sampling=sp_r)
        peri = np.exp(-dist / dg.PERI_TAU_MM).astype(np.float32)
    else:
        peri = np.zeros(TARGET_SHAPE, np.float32)
    return np.stack([ct, ds, healthy.astype(np.float32), peri], 0).astype(np.float32)


class _DoseGate(nn.Module):
    def __init__(self, cfg: "_gate.GateConfig"):
        super().__init__()
        self.cfg = cfg
        self.b_raw = nn.Parameter(_gate.init_control_points(cfg))
        self.a_raw = nn.Parameter(_gate.init_slope(cfg))
        self.w_raw = nn.Parameter(torch.zeros(cfg.n_points))
        self.built = _gate.build_terms(self, cfg)

    def forward(self, dose, anat):
        dose, shift = _gate.apply_dose_terms(self, self.cfg, dose, anat, None)
        g = _gate.phi(dose, self.a_raw, self.b_raw, self.w_raw, shift)
        return _gate.apply_anatomy_terms(self, self.cfg, g, anat, mask_to_lung=True)


class GatedDenseNet(nn.Module):
    instances: list = []

    def __init__(self):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.net = DenseNet121(spatial_dims=3, in_channels=2, out_channels=2,
                               dropout_prob=0.2)
        base = dg._gate_config(K=4)
        self.cfg = dataclasses.replace(
            base, use_anatomy=True, use_lobe=False, use_region=False, use_vuln_ntcp=False,
            use_dose_gradient=False, use_conv=False, use_cross=False, idx_cc=-1, idx_vuln=-1)
        self.gate = _DoseGate(self.cfg)
        GatedDenseNet.instances.append(self)

    def optimizer_param_groups(self, lr: float, wd: float):
        gate_ids = {id(p) for p in self.gate.parameters()}
        return [{"params": [p for p in self.parameters() if id(p) not in gate_ids],
                 "lr": lr, "weight_decay": wd},
                {"params": list(self.gate.parameters()), "lr": GATE_LR, "weight_decay": 0.0}]

    def forward(self, x):
        f = F.relu(self.net.features(x[:, :2]))
        g = self.gate(x[:, 1:2], x[:, 2:4])
        gw = F.adaptive_avg_pool3d(g, f.shape[2:]).to(f.dtype)
        pooled = (f * gw).sum(dim=(2, 3, 4)) / (gw.sum(dim=(2, 3, 4)) + 1e-6)
        return self.net.class_layers.out(pooled)


def _make_kapoor_gated():
    return GatedDenseNet()


def _build_liang_gated(paths: dict) -> np.ndarray | None:
    if not (paths.get("dose") and paths.get("lung")):
        return None
    ds = _resize(_load_dose(paths["dose"]))
    lung = np.load(paths["lung"]) > 0
    gtv = (np.load(paths["gtv"]) > 0) if paths.get("gtv") else np.zeros_like(lung)
    healthy = _resize((lung & ~gtv).astype(np.float32)) >= 0.5
    gtv_r = _resize(gtv.astype(np.float32)) >= 0.5
    if gtv_r.any():
        sp = dg.ct_voxel_spacing(paths["ct"]) if paths.get("ct") else (3.0, 1.0, 1.0)
        sp_r = [s0 * n0 / n1 for s0, n0, n1 in zip(sp, lung.shape, TARGET_SHAPE)]
        dist = scipy.ndimage.distance_transform_edt(~gtv_r, sampling=sp_r)
        peri = np.exp(-dist / dg.PERI_TAU_MM).astype(np.float32)
    else:
        peri = np.zeros(TARGET_SHAPE, np.float32)
    return np.stack([ds, healthy.astype(np.float32), peri], 0).astype(np.float32)


class GatedC3D(nn.Module):
    instances: list = []

    def __init__(self):
        super().__init__()
        self.net = C3D3D(in_ch=1, n_classes=2, p=0.5)
        self.trunk = self.net.features[:-1]
        base = dg._gate_config(K=4)
        self.cfg = dataclasses.replace(
            base, use_anatomy=True, use_lobe=False, use_region=False, use_vuln_ntcp=False,
            use_dose_gradient=False, use_conv=False, use_cross=False, idx_cc=-1, idx_vuln=-1)
        self.gate = _DoseGate(self.cfg)
        GatedC3D.instances.append(self)

    def optimizer_param_groups(self, lr: float, wd: float):
        gate_ids = {id(p) for p in self.gate.parameters()}
        return [{"params": [p for p in self.parameters() if id(p) not in gate_ids],
                 "lr": lr, "weight_decay": wd},
                {"params": list(self.gate.parameters()), "lr": GATE_LR, "weight_decay": 0.0}]

    def forward(self, x):
        f = self.trunk(x[:, 0:1])
        g = self.gate(x[:, 0:1], x[:, 1:3])
        gw = F.adaptive_avg_pool3d(g, f.shape[2:]).to(f.dtype)
        pooled = (f * gw).sum(dim=(2, 3, 4)) / (gw.sum(dim=(2, 3, 4)) + 1e-6)
        return self.net.classifier(pooled.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


def _make_liang_gated():
    return GatedC3D()


def _dump_gate_curves(name: str, pred_dir: str, outer_k: int) -> None:
    out_dir = os.path.join(os.path.dirname(pred_dir.rstrip("/")), "gate_curves")
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(GatedDenseNet.instances + GatedC3D.instances):
        try:
            d = _gate.curve_dict(m.gate, m.cfg)
            with open(os.path.join(out_dir, f"{name}_outer{outer_k}_inner{i}.json"), "w") as fp:
                json.dump(d, fp, indent=1)
            log.info("  %s: inner %d gate d50=%.1f Gy steepness=%.2f", name, i,
                     float(d.get("d50_Gy", float("nan"))), float(d.get("steepness", float("nan"))))
        except Exception as e:
            log.warning("  %s: gate curve export failed for inner %d (%s)", name, i, e)
    GatedDenseNet.instances.clear()
    GatedC3D.instances.clear()


CNN_METHODS = {
    "liang2020":   (_build_liang,   _make_liang,   ("dose",)),
    "kapoor2023":  (_build_kapoor,  _make_kapoor,  ("ct", "dose")),
    "midroni2026": (_build_midroni, _make_midroni, ("ct",)),
    "kapoor2023_gated": (_build_kapoor_gated, _make_kapoor_gated, ("ct", "dose", "lung")),
    "liang2020_gated": (_build_liang_gated, _make_liang_gated, ("dose", "lung")),
}


class SotaCNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, builder, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.builder = builder
        self.augment = augment
        self._c = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        uuid = str(row["UUID"]).strip()
        label = int(row.get("Group") == "RP")
        try:
            paths = dg.resolve_modality_paths(os.path.join(dg._cohort_dir, uuid))
            arr = self.builder(paths)
            if arr is None:
                raise FileNotFoundError(f"{uuid}: missing modality")
            self._c = arr.shape[0]
            if self.augment:
                for ax in (1, 2, 3):
                    if np.random.rand() < 0.5:
                        arr = np.flip(arr, ax)
                arr = np.ascontiguousarray(arr)
        except Exception as e:
            log.warning("  cnn dataset: %s failed (%s) - zeros", uuid, e)
            c = self._c or 1
            arr = np.zeros((c, *TARGET_SHAPE), np.float32)
        return torch.from_numpy(arr.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def _filter_cohort(split: pd.DataFrame, need: tuple[str, ...]) -> pd.DataFrame:
    keep = []
    for u in split["uuid"].astype(str):
        paths = dg.resolve_modality_paths(os.path.join(dg._cohort_dir, u))
        keep.append(all(paths.get(k) for k in need))
    out = split[pd.Series(keep, index=split.index)].reset_index(drop=True)
    return out


def cnn_run_method(name: str, split: pd.DataFrame, pred_dir: str) -> dict | None:
    builder, factory, need = CNN_METHODS[name]
    outer_k = int(os.environ.get("NESTED_OUTER_FOLD", -1))
    coh = _filter_cohort(split, need)
    if len(coh) < 20:
        log.error("  %s: only %d patients with %s - skipping", name, len(coh), need)
        return None

    manifest = pd.DataFrame({
        "UUID":  coh["uuid"].astype(str).values,
        "Group": np.where(coh["rp_label"].astype(int).values == 1, "RP", "noRP"),
    })
    folds_arr = coh["OuterFold"].astype(int).values
    labels = coh["rp_label"].astype(int).values

    if outer_k < 0:
        log.error("  %s: NESTED_OUTER_FOLD unset (run under nested CV) - skipping", name)
        return None
    dev_pos = [i for i, f in enumerate(folds_arr) if f != outer_k]
    test_pos = [i for i, f in enumerate(folds_arr) if f == outer_k]
    if not test_pos or len(set(labels[dev_pos])) < 2:
        log.error("  %s: degenerate fold %d - skipping", name, outer_k)
        return None
    dev_labels = [int(labels[i]) for i in dev_pos]
    test_labels = [int(labels[i]) for i in test_pos]
    test_uuids = [manifest.iloc[i]["UUID"] for i in test_pos]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + outer_k)
    inner = [(tr, va) for tr, va in skf.split(dev_pos, dev_labels)]

    ds_clean = SotaCNNDataset(manifest, builder, augment=False)
    ds_noisy = SotaCNNDataset(manifest, builder, augment=True)
    device = _device()
    log.info("  %s: dev=%d test=%d (fold %d)", name, len(dev_pos), len(test_pos), outer_k)

    _, tp_ens, oof_vp, oof_vl = td.run_cv_cnn(
        name, factory, ds_clean, ds_noisy,
        dev_pos, dev_labels, inner, device,
        test_pos, test_labels,
        batch_size=td.BATCH_SIZE, accumulation=td.ACCUMULATION)

    if tp_ens is None:
        log.error("  %s: no ensemble prediction - skipping", name)
        return None
    if name.endswith("_gated"):
        _dump_gate_curves(name, pred_dir, outer_k)

    thr = td.optimal_f1_threshold(oof_vp, oof_vl) if len(oof_vp) else 0.5
    thr_y = td.youden_threshold(oof_vp, oof_vl) if len(oof_vp) else 0.5
    os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame({
        "UUID":                    test_uuids,
        "TrueLabel":               np.array(test_labels, int),
        "RP_Probability":          tp_ens,
        "PredictedLabel":          (tp_ens >= thr).astype(int),
        "PredictedLabel_Youden":   (tp_ens >= thr_y).astype(int),
    }).to_excel(os.path.join(pred_dir, f"{name}_test_predictions.xlsx"), index=False)

    from sklearn.metrics import roc_auc_score, average_precision_score
    _y = np.array(test_labels, int)
    te_auc = roc_auc_score(_y, tp_ens) if len(set(test_labels)) > 1 else float("nan")
    te_pr = average_precision_score(_y, tp_ens) if len(set(test_labels)) > 1 else float("nan")
    _prev = float(_y.mean()); _bs_ref = _prev * (1.0 - _prev)
    _bs = float(np.mean((tp_ens - _y) ** 2))
    _bss = 1.0 - _bs / _bs_ref if _bs_ref > 0 else float("nan")
    log.info("  %s: fold %d test AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  BrierSkill=%+.3f "
             "(n_test=%d)", name, outer_k, te_auc, te_pr, _bs, _bss, len(test_pos))
    return {"method": name, "fold": outer_k, "test_auc": float(te_auc),
            "test_pr_auc": float(te_pr), "test_brier": _bs, "test_brier_skill": _bss}


_DEEP_CACHE = os.path.join(dg._input_dir, "sota_deep_cache")
_ENCODERS: dict = {}


def _get_encoder(device, arch: str = "r18"):
    if arch not in _ENCODERS:
        if arch == "r50":
            enc = td.MedicalNet3D_R50(getattr(td, "_MEDICALNET_R50_PATH", ""),
                                      dropout_rate=0.0, in_chans=1)
        else:
            enc = td.MedicalNet3D_R18(getattr(td, "_MEDICALNET_R18_PATH", ""),
                                      dropout_rate=0.0, in_chans=1)
        enc.eval().to(device)
        for p in enc.parameters():
            p.requires_grad = False
        _ENCODERS[arch] = enc
    return _ENCODERS[arch]


def _embed(enc, vol: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(vol[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        f = enc.backbone(enc.input_adapter(x))
        if isinstance(f, (list, tuple)):
            f = f[-1]
        if f.dim() > 2:
            f = f.mean(dim=tuple(range(2, f.dim())))
    return f.squeeze(0).float().cpu().numpy()


def _masked_volume(paths: dict, modality: str, roi: str) -> np.ndarray | None:
    from train_sota import _roi_mask
    vol_path = paths.get(modality)
    mask = _roi_mask(paths, roi)
    if not vol_path or mask is None:
        return None
    if modality == "ct":
        v = _load_ct(vol_path)
    elif modality == "dose":
        v = _load_dose(vol_path)
    else:
        v = np.load(vol_path).astype(np.float32)
    v = v * (np.asarray(mask) > 0)
    return _resize(v).astype(np.float32)


def build_deep_feature_map(split: pd.DataFrame, modality: str, roi: str,
                           arch: str = "r18") -> tuple[dict[str, np.ndarray], list[str]]:
    device = _device()
    enc = _get_encoder(device, arch)
    tag = f"deep_{modality}_{roi}_{arch}"
    os.makedirs(_DEEP_CACHE, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for u in split["uuid"].astype(str):
        cache = os.path.join(_DEEP_CACHE, f"{u}__{tag}.npy")
        if os.path.exists(cache):
            try:
                out[u] = np.load(cache).astype(np.float32); continue
            except Exception:
                pass
        paths = dg.resolve_modality_paths(os.path.join(dg._cohort_dir, u))
        v = _masked_volume(paths, modality, roi)
        if v is None:
            continue
        try:
            feat = _embed(enc, v, device).astype(np.float32)
            np.save(cache, feat)
            out[u] = feat
        except Exception as e:
            log.warning("  deep[%s/%s] failed for %s: %s: %s",
                        modality, roi, u, type(e).__name__, e)
    dim = len(next(iter(out.values()))) if out else 0
    names = [f"{tag}_{i}" for i in range(dim)]
    log.info("  deep[%s/%s]: %d/%d patients, %d features",
             modality, roi, len(out), len(split), dim)
    return out, names
