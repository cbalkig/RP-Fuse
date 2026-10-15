#!/usr/bin/env python3
from __future__ import annotations
import os, sys, csv, argparse
import dataclasses
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
import wandb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import gate as _gate
import train_rpfuse as dg


def _gate_cfg(K: int):
    base = dg._gate_config(K=K)
    _abl = _gcfg.get("ABLATE", {}) or {}
    _grad = bool(_abl.get("DOSE_GRADIENT", False))
    return dataclasses.replace(base, use_dose_gradient=_grad, use_conv=False,
                               use_cross=False,
                               use_vuln_ntcp=bool(_abl.get("VULN_SHIFT", _abl.get("VULN_NTCP", False))),
                               init_d50_gy=GATE_DOSE_RESPONSE_INIT_GY,
                               init_transition_gy=GATE_DOSE_RESPONSE_TRANSITION_GY)
from focal_loss import FocalLoss
from util import read_table

log = dg.log

if not dg._IMAGING_TOKENS:
    dg._IMAGING_TOKENS = ["ct"]
    dg.log.info("gate_ct: forced dg._IMAGING_TOKENS=['ct'] so the dataset loads real volumes "
                "(fusion token flags are off for the gate).")


_gcfg = dg._full_cfg.get("TRAIN_GATE_CT", {})
def _gflag(n, d): return bool(_gcfg.get(n, d))
TOKEN           = str(_gcfg.get("TOKEN", "early"))
TRAIN_ONLY_GATE = _gflag("TRAIN_ONLY_GATE", True)
GATE_AS_CHANNEL = _gflag("AS_CHANNEL", True)
GATE_EARLY_CT_ONLY = _gflag("EARLY_CT_ONLY", False)
GATE_REG        = float(_gcfg.get("REG", 1e-3))
_clampb         = _gcfg.get("CLAMP_B", [0.20, 0.50])
CLAMP_B         = tuple(float(x) for x in _clampb) if _clampb else None
_init_b         = _gcfg.get("INIT_B", None)
GATE_INIT_B     = float(_init_b) if _init_b is not None else dg.DG_INIT_B
GATE_PROX_NONNEG= _gflag("PROX_NONNEG", True)
GATE_EMBED      = int(_gcfg.get("EMBED", 64))
GATE_AUX        = float(_gcfg.get("AUX", 0.0))
GATE_PROGRESSIVE_UNFREEZE = _gflag("PROGRESSIVE_UNFREEZE", False)
GATE_UNFREEZE_PATIENCE    = int(_gcfg.get("UNFREEZE_PATIENCE", 6))
GATE_WARMUP_UNFREEZE      = _gflag("WARMUP_UNFREEZE", False)
GATE_MAX_EPOCHS           = int(_gcfg.get("MAX_EPOCHS", 0)) or None
GATE_FOCAL_GAMMA          = float(_gcfg.get("FOCAL_GAMMA", 2.0))
GATE_FOCAL_GAMMA_FINAL    = float(_gcfg.get("FOCAL_GAMMA_FINAL", GATE_FOCAL_GAMMA))
GATE_FOCAL_GAMMA_ANNEAL_FRAC = float(_gcfg.get("FOCAL_GAMMA_ANNEAL_FRAC", 1.0))
GATE_CLASS_WEIGHT_POWER   = float(_gcfg.get("CLASS_WEIGHT_POWER", 1.0))


def _gate_alpha(tr_lbl, device):
    a = dg._class_weights(tr_lbl, device)
    if GATE_CLASS_WEIGHT_POWER != 1.0 and a is not None:
        a = a ** GATE_CLASS_WEIGHT_POWER
    return a


def _gate_focal_gamma_at(epoch: int, max_epochs: int) -> float:
    if GATE_FOCAL_GAMMA_FINAL == GATE_FOCAL_GAMMA or max_epochs <= 1:
        return GATE_FOCAL_GAMMA
    span = max(1, int(round(GATE_FOCAL_GAMMA_ANNEAL_FRAC * max_epochs)))
    frac = min(1.0, max(0.0, (epoch - 1) / span))
    cos = 0.5 * (1.0 + float(np.cos(np.pi * frac)))
    return GATE_FOCAL_GAMMA_FINAL + (GATE_FOCAL_GAMMA - GATE_FOCAL_GAMMA_FINAL) * cos
GATE_DOSE_RESPONSE   = bool(dg._cfg_alias(_gcfg, "DOSE_RESPONSE", "NTCP", False))
GATE_DOSE_RESPONSE_K = int(dg._cfg_alias(_gcfg, "DOSE_RESPONSE_K", "NTCP_K", 4))
GATE_DOSE_RESPONSE_INIT_GY = float(dg._cfg_alias(_gcfg, "DOSE_RESPONSE_INIT_GY", "NTCP_INIT_GY", 20.0))
GATE_DOSE_RESPONSE_TRANSITION_GY = float(dg._cfg_alias(_gcfg, "DOSE_RESPONSE_TRANSITION_GY", "NTCP_TRANSITION_GY", dg.DOSE_RESPONSE_TRANSITION_GY))
# THRESH_LR gives the curve thresholds (b_raw) their own optimiser group and learning rate;
# GATE_LR_CONSTANT keeps the gate groups out of the plateau decay. Both default off.
_thr_lr = _gcfg.get("THRESH_LR", None)
GATE_THRESH_LR   = float(_thr_lr) if _thr_lr is not None else None
GATE_LR_CONSTANT = _gflag("GATE_LR_CONSTANT", False)
# VOLUME_FRACTION appends V_phi = sum(phi)/sum(healthy lung), the share of the lung the curve
# admits, to the pooled feature and adds it to the logit through a zero-initialised linear
# term, so the loss sees how much lung the curve admits and not only which voxels it
# favours. VOLUME_FRACTION_SCALE sets its unit (10 = per 10 % of the lung). Default off.
GATE_VOLUME_FRACTION = _gflag("VOLUME_FRACTION", False)
GATE_VOLFRAC_SCALE   = float(_gcfg.get("VOLUME_FRACTION_SCALE", 10.0))
GATE_SPOTLIGHT      = _gflag("SPOTLIGHT", False)
GATE_SPOTLIGHT_INIT = float(_gcfg.get("SPOTLIGHT_INIT", 0.4))
GATE_DISPERSION = _gflag("DISPERSION", False)
GATE_BAND_POOLING = _gflag("BAND_POOLING", False)
GATE_LATERALITY = _gflag("LATERALITY", False)
GATE_CONTENT_ATTN = _gflag("CONTENT_ATTN", False)
if GATE_BAND_POOLING and GATE_DISPERSION:
    raise ValueError("BAND_POOLING and DISPERSION both change the pooled width; enable one")
if GATE_BAND_POOLING and GATE_CONTENT_ATTN:
    raise ValueError("BAND_POOLING and CONTENT_ATTN define different poolings; enable one")
HARD_MASK    = _gflag("HARD_MASK", False)
NO_GATE      = _gflag("NO_GATE", False)
if NO_GATE:
    GATE_SPOTLIGHT = False
GATE_INPUT_VOL = str(_gcfg.get("INPUT_VOL", "ct") or "ct").lower()
if GATE_INPUT_VOL not in ("ct", "dose"):
    raise SystemExit(f"TRAIN_GATE_CT.INPUT_VOL must be 'ct' or 'dose', got {GATE_INPUT_VOL!r}")
if GATE_INPUT_VOL == "dose" and not NO_GATE:
    raise SystemExit("TRAIN_GATE_CT.INPUT_VOL='dose' needs NO_GATE=true: gating the dose "
                     "map with itself is not an experiment anyone asked for.")
HARD_MASK_GY = float(_gcfg.get("HARD_MASK_GY", 20.0))
_HARD_MASK_THR = HARD_MASK_GY / (dg.DOSE_NORM_CGY / 100.0)
EXPORT_EMB = _gflag("EXPORT_EMB", False)


RESULTS_DIR = os.path.join(dg._output_dir, "results_gate_ct" + dg._RESULTS_SUFFIX)
PRED_DIR    = os.path.join(RESULTS_DIR, "patient_predictions")


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


RUN_CONFIG = {
    "git_sha": _git_sha(),
    "results_dir": RESULTS_DIR,
    "results_suffix": dg._RESULTS_SUFFIX,
    "backbone_depth": dg.BACKBONE_DEPTH,
    "backbone_weights": os.path.basename(dg._resolve_backbone_weights(dg.BACKBONE_DEPTH) or ""),
    "single_study_only": dg.SINGLE_STUDY_ONLY,
    "exclude_major_replan": dg.EXCLUDE_MAJOR_REPLAN,
    "token": TOKEN, "train_only_gate": TRAIN_ONLY_GATE, "gate_as_channel": GATE_AS_CHANNEL,
    "gate_reg": GATE_REG, "clamp_b": CLAMP_B, "init_b": GATE_INIT_B,
    "prox_nonneg": GATE_PROX_NONNEG, "embed": GATE_EMBED, "aux": GATE_AUX,
    "progressive_unfreeze": GATE_PROGRESSIVE_UNFREEZE,
    "unfreeze_patience": GATE_UNFREEZE_PATIENCE,
    "max_epochs": (GATE_MAX_EPOCHS or dg.MAX_EPOCHS),
    "ntcp": GATE_DOSE_RESPONSE, "ntcp_k": GATE_DOSE_RESPONSE_K, "ntcp_init_gy": GATE_DOSE_RESPONSE_INIT_GY,
    "volume_fraction": GATE_VOLUME_FRACTION, "volume_fraction_scale": GATE_VOLFRAC_SCALE,
    "spotlight": GATE_SPOTLIGHT, "spotlight_init": GATE_SPOTLIGHT_INIT,
    "hard_mask": HARD_MASK, "hard_mask_gy": HARD_MASK_GY, "no_gate": NO_GATE,
    "aug_p_flip": dg.P_FLIP, "aug_p_noise": dg.P_NOISE, "aug_noise_std": dg.NOISE_STD,
    "aug_p_mixup": dg.P_MIXUP, "aug_p_rot": dg.P_ROT, "aug_p_scale": dg.P_SCALE,
    "aug_p_elastic": dg.P_ELASTIC, "aug_p_gamma": dg.P_GAMMA, "aug_p_blur": dg.P_BLUR,
    "batch_size": dg.BATCH_SIZE, "lr": dg.LR, "gate_lr": dg.GATE_LR,
    "thresh_lr": GATE_THRESH_LR, "gate_lr_constant": GATE_LR_CONSTANT,
    "weight_decay": dg.WEIGHT_DECAY, "dropout": dg.DROPOUT,
    "warmup_epochs": dg.WARMUP_EPOCHS,
    "early_stop_patience": dg.EARLY_STOP_PATIENCE,
    "early_stop_min_epochs": dg.EARLY_STOP_MIN_EPOCHS,
    "amp_dtype": str(dg.AMP_DTYPE),
    "focal_gamma": GATE_FOCAL_GAMMA, "focal_gamma_final": GATE_FOCAL_GAMMA_FINAL,
    "focal_gamma_anneal_frac": GATE_FOCAL_GAMMA_ANNEAL_FRAC,
    "focal_alpha": f"class_weights^{GATE_CLASS_WEIGHT_POWER}",
}

_RUN_CONFIG_LOGGED = False


def _log_run_config():
    global _RUN_CONFIG_LOGGED
    if _RUN_CONFIG_LOGGED:
        return
    _RUN_CONFIG_LOGGED = True
    log.info("======== RUN CONFIG (gate_ct @ git %s) ========", RUN_CONFIG["git_sha"])
    for k, v in RUN_CONFIG.items():
        log.info("    %-22s = %s", k, v)
    log.info("===============================================")


_use_amp = None


class GateDose(nn.Module):
    def __init__(self):
        super().__init__()
        self.steepness   = nn.Parameter(torch.tensor(float(dg.DG_INIT_A)))
        self.dose_thresh = nn.Parameter(torch.tensor(float(GATE_INIT_B)))
        self.amplify     = nn.Parameter(torch.tensor(float(dg.DG_INIT_LAMBDA)))
        self.prox_weight = nn.Parameter(torch.tensor(0.0))
        _, _ic, _iv = dg._anat_layout()
        if dg.USE_LOBE_GATE and _ic >= 0:
            self.lobe_raw = nn.Parameter(torch.tensor(dg.GATE_OFF_INIT))
        if _iv >= 0:
            self.vuln_raw = nn.Parameter(torch.tensor(dg.GATE_OFF_INIT))
        self.baseline    = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("_init", torch.tensor(
            [float(dg.DG_INIT_A), float(GATE_INIT_B), float(dg.DG_INIT_LAMBDA), 0.0, 1.0]))

    def _thr(self):
        return self.dose_thresh.clamp(CLAMP_B[0], CLAMP_B[1]) if CLAMP_B else self.dose_thresh

    def gate(self, dose, anat=None, ct=None):
        logit = self.steepness * (dose - self._thr())
        if dg.USE_ANATOMY_GATE and anat is not None:
            pw = self.prox_weight.clamp(min=0.0) if GATE_PROX_NONNEG else self.prox_weight
            extra = pw * anat[:, 1:2]
            _, ic, iv = dg._anat_layout()
            if dg.USE_LOBE_GATE and ic >= 0:
                extra = extra + F.softplus(self.lobe_raw) * anat[:, ic:ic + 1]
            if iv >= 0:
                extra = extra + F.softplus(self.vuln_raw) * anat[:, iv:iv + 1]
            g = torch.sigmoid(logit + extra)
            return g * anat[:, 0:1]
        return torch.sigmoid(logit)

    def forward(self, vol, dose, anat=None):
        return vol * (self.baseline + self.amplify * self.gate(dose, anat, ct=vol))

    @property
    def lobe_weight(self):
        with torch.no_grad():
            return float(F.softplus(self.lobe_raw)) if hasattr(self, "lobe_raw") else 0.0
    @property
    def vuln_weight(self):
        with torch.no_grad():
            return float(F.softplus(self.vuln_raw)) if hasattr(self, "vuln_raw") else 0.0

    def reg(self):
        cur = torch.stack([self.steepness, self.dose_thresh, self.amplify,
                           self.prox_weight, self.baseline])
        return ((cur - self._init) ** 2).sum()


class IdentityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.steepness = torch.tensor(0.0)
        self.dose_thresh = torch.tensor(0.0)
        self.amplify = torch.tensor(0.0)
        self.prox_weight = torch.tensor(0.0)
        self.baseline = torch.tensor(1.0)
    def gate(self, dose, anat=None, ct=None): return torch.zeros_like(dose)
    def forward(self, vol, dose, anat=None, ct=None):
        if anat is not None:
            return vol * anat[:, 0:1]
        return vol
    def reg(self): return torch.tensor(0.0, device=self.steepness.device)


class HardMaskGate(nn.Module):
    def __init__(self, thr_norm):
        super().__init__()
        self.register_buffer("_thr_n", torch.tensor(float(thr_norm)))

    def gate(self, dose, anat=None, ct=None):
        m = (dose >= self._thr_n).to(dose.dtype)
        if dg.USE_ANATOMY_GATE and anat is not None:
            m = m * anat[:, 0:1]
        return m

    def forward(self, vol, dose, anat=None):
        return vol * self.gate(dose, anat, ct=vol)

    def reg(self):
        return torch.zeros((), device=self._thr_n.device)

    @property
    def steepness(self):   return torch.zeros((), device=self._thr_n.device)
    @property
    def dose_thresh(self): return self._thr_n
    @property
    def amplify(self):     return torch.ones((), device=self._thr_n.device)
    @property
    def prox_weight(self): return torch.zeros((), device=self._thr_n.device)
    @property
    def baseline(self):    return torch.zeros((), device=self._thr_n.device)


class DoseResponseGate(nn.Module):
    def __init__(self, K: int = 4):
        super().__init__()
        self.K = K
        self.cfg = _gate_cfg(K)
        self.b_raw = nn.Parameter(_gate.init_control_points(self.cfg))
        self.a_raw = nn.Parameter(_gate.init_slope(self.cfg))
        self.w_raw = nn.Parameter(torch.zeros(K))
        self._optional = _gate.build_terms(self, self.cfg)
        with torch.no_grad():
            self.init_steepness = float(self.steepness)
            self.init_dose_thresh = float(self.dose_thresh)

    def _phi(self, d, shift=None):
        return _gate.phi(d, self.a_raw, self.b_raw, self.w_raw, shift)

    def forward(self, dose, anat=None, ct=None):
        dose, shift = _gate.apply_dose_terms(self, self.cfg, dose, anat, ct)
        g = self._phi(dose, shift)
        return _gate.apply_anatomy_terms(self, self.cfg, g, anat, mask_to_lung=True)

    def reg(self):
        return _gate.curvature_reg(self)

    @property
    def dose_thresh(self):
        d, _, i = _gate.curve_d50(self)
        return float(d[i])

    @property
    def steepness(self):
        return _gate.curve_steepness(self)

    @property
    def amplify(self):  return 1.0
    @property
    def baseline(self): return 1.0
    @property
    def prox_weight(self):
        with torch.no_grad():
            return float(F.softplus(self.prox_raw)) if hasattr(self, "prox_raw") else 0.0
    @property
    def lobe_weight(self):
        with torch.no_grad():
            return float(F.softplus(self.lobe_raw)) if hasattr(self, "lobe_raw") else 0.0
    @property
    def vuln_weight(self):
        with torch.no_grad():
            return float(F.softplus(self.vuln_raw)) if hasattr(self, "vuln_raw") else 0.0
    @property
    def region_weight(self):
        with torch.no_grad():
            return float(self.region_delta) if hasattr(self, "region_delta") else 0.0

    def curve_dict(self):
        with torch.no_grad():
            a = F.softplus(self.a_raw); w = torch.softmax(self.w_raw, 0)
            b = torch.sigmoid(self.b_raw)
            d = torch.linspace(0, 1, 101, device=self.b_raw.device)
            phi = self._phi(d.unsqueeze(-1))
        return {"K": self.K,
                "b_norm": [float(x) for x in b],
                "b_Gy": [float(x) * dg.DOSE_NORM_CGY / 100.0 for x in b],
                "a": [float(x) for x in a], "w": [float(x) for x in w],
                "prox_weight": self.prox_weight,
                "d_grid": [float(x) for x in d], "phi": [float(x) for x in phi]}


def _gate_vars(gp, model=None):
    d = dict(gp.curve_dict()) if hasattr(gp, "curve_dict") else {}
    for _k, _attr, _scale in (("steepness", "steepness", 1.0),
                              ("dose_thresh_norm", "dose_thresh", 1.0),
                              ("d50_Gy", "dose_thresh", dg.DOSE_NORM_CGY / 100.0),
                              ("lobe_weight", "lobe_weight", 1.0),
                              ("vuln_weight", "vuln_weight", 1.0),
                              ("region_weight", "region_weight", 1.0)):
        _v = getattr(gp, _attr, None)
        if _v is not None:
            d[_k] = round(float(_v) * _scale, 4)
    if hasattr(gp, "lat_raw"):
        with torch.no_grad():
            d["lat_weight"] = round(float(F.softplus(gp.lat_raw)), 4)
    d["f"] = (round(float(torch.sigmoid(model.f_raw)), 4)
              if (model is not None and GATE_SPOTLIGHT and hasattr(model, "f_raw")) else 1.0)
    if model is not None and getattr(model, "volfrac", False):
        with torch.no_grad():
            _vw = model.gate.vol_head.weight.float()
            d["vol_beta"] = round(float(_vw[1, 0] - _vw[0, 0]), 4)
            d["vol_scale"] = GATE_VOLFRAC_SCALE
    return d


class GateNet(nn.Module):
    def __init__(self, path=None, dropout=dg.DROPOUT):
        super().__init__()
        depth = dg.BACKBONE_DEPTH
        if path is None:
            path = dg._resolve_backbone_weights(depth) or dg._MEDICALNET_R18_PATH
        if dg.BACKBONE_KIND == "handcrafted":
            self.backbone = dg.HandcraftedBank3D()
        else:
            self.backbone = dg.Backbone3D(path, depth)
        fdim = self.backbone.feat_dim
        self.ntcp = (GATE_DOSE_RESPONSE and not HARD_MASK and not NO_GATE)
        if self.ntcp:
            self.gate = DoseResponseGate(GATE_DOSE_RESPONSE_K)
            nbase = 1 if GATE_EARLY_CT_ONLY else (2 if dg.PET_ROBUST else 3)
            self.early_adapter = nn.Conv3d(nbase, 1, 1, bias=False)
            nn.init.constant_(self.early_adapter.weight, 1.0 / nbase)
            self.tokens = ["early"]
        else:
            if NO_GATE:
                self.gate = IdentityGate()
            else:
                self.gate = HardMaskGate(_HARD_MASK_THR) if HARD_MASK else GateDose()
            self.tokens = [t for t in TOKEN.split("+") if t in ("ct", "early")]
            if not self.tokens:
                raise ValueError(f"TRAIN_GATE_CT.TOKEN={TOKEN!r} has no valid token (ct/early)")
            nbase = 2 if dg.PET_ROBUST else 3
            if "early" in self.tokens:
                ein = nbase + (1 if GATE_AS_CHANNEL else 0)
                self.early_adapter = nn.Conv3d(ein, 1, 1, bias=False)
                nn.init.constant_(self.early_adapter.weight, 1.0 / ein)
            if "ct" in self.tokens and GATE_AS_CHANNEL:
                self.ct_adapter = nn.Conv3d(3, 1, 1, bias=False)
                nn.init.constant_(self.ct_adapter.weight, 1.0 / 3)
        if GATE_SPOTLIGHT:
            self.f_raw = nn.Parameter(torch.logit(torch.tensor(float(GATE_SPOTLIGHT_INIT))))
        if self.ntcp and GATE_LATERALITY:
            self.gate.lat_raw = nn.Parameter(torch.tensor(float(dg.GATE_OFF_INIT)))
        if self.ntcp and GATE_CONTENT_ATTN:
            self.att_score = nn.Conv3d(fdim, 1, 1)
            nn.init.zeros_(self.att_score.weight)
            nn.init.zeros_(self.att_score.bias)
        E = GATE_EMBED
        _pdim = (fdim * 3 if (self.ntcp and GATE_BAND_POOLING)
                 else fdim * 2 if (self.ntcp and GATE_DISPERSION) else fdim)
        self.volfrac = bool(self.ntcp and GATE_VOLUME_FRACTION)
        if self.volfrac:
            _pdim += 1
            # on the gate module, so it trains at the gate learning rate without decay
            self.gate.vol_head = nn.Linear(1, 2, bias=False)
            nn.init.zeros_(self.gate.vol_head.weight)
        self.proj = nn.Sequential(nn.Linear(_pdim, E), nn.GELU())
        if len(self.tokens) > 1:
            self.scorer = nn.Sequential(nn.Linear(E, E // 2), nn.Tanh(), nn.Linear(E // 2, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(E, 2))
        self.aux = nn.ModuleDict({t: nn.Sequential(nn.Dropout(dropout), nn.Linear(E, 2))
                                  for t in self.tokens})
        self.dose_gate = self.gate
        if (TRAIN_ONLY_GATE or GATE_WARMUP_UNFREEZE) and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _spotlight(self, ct, g):
        if not GATE_SPOTLIGHT:
            return ct
        f = torch.sigmoid(self.f_raw)
        return ct * (f + (1.0 - f) * g)

    def _lat_boost(self, g, anat):
        if not (GATE_LATERALITY and anat is not None and hasattr(self.gate, "lat_raw")):
            return g
        prox = anat[:, 1:2]
        B, _, D, H, W = prox.shape
        xs = torch.arange(W, device=g.device, dtype=g.dtype).view(1, 1, 1, 1, W)
        mass = prox.sum(dim=(1, 2, 3, 4))
        xc = (prox * xs).sum(dim=(1, 2, 3, 4)) / mass.clamp_min(1e-6)
        right = (xs >= W / 2.0).expand(B, 1, D, H, W)
        ipsi = torch.where((xc >= W / 2.0).view(B, 1, 1, 1, 1), right, ~right).to(g.dtype)
        ipsi = ipsi * (mass > 0).to(g.dtype).view(B, 1, 1, 1, 1)
        return g * (1.0 + F.softplus(self.gate.lat_raw) * ipsi)

    def _vol_frac(self, gw, lungw):
        """V_phi per patient in units of VOLUME_FRACTION_SCALE; both maps are average-pooled
        to the feature grid, which keeps their means, so the ratio is the full-resolution one."""
        num = gw.flatten(2).sum(-1)
        if lungw is None:
            den = torch.full_like(num, float(gw[0].numel()))
        else:
            den = lungw.flatten(2).sum(-1).clamp_min(1e-6)
        return GATE_VOLFRAC_SCALE * num / den

    def _pool(self, fmap, gw, lungw=None):
        out = self._pool_core(fmap, gw, lungw)
        if self.volfrac:
            out = torch.cat([out, self._vol_frac(gw, lungw)], dim=1)
        return out

    def _pool_core(self, fmap, gw, lungw=None):
        if GATE_CONTENT_ATTN:
            scores = self.att_score(fmap) + (gw + 1e-6).log()
            w = torch.softmax(scores.flatten(2), dim=-1)
            return (fmap.flatten(2) * w).sum(-1)
        if GATE_BAND_POOLING:
            def _wmean(w):
                d_ = w.flatten(2).sum(-1).clamp_min(1e-6)
                return (fmap * w).flatten(2).sum(-1) / d_
            pen = gw * (1.0 - gw)
            ctx = lungw if lungw is not None else torch.ones_like(gw)
            return torch.cat([_wmean(gw), _wmean(pen), _wmean(ctx)], dim=1)
        den = gw.flatten(2).sum(-1).clamp_min(1e-6)
        mean = (fmap * gw).flatten(2).sum(-1) / den
        if not GATE_DISPERSION:
            return mean
        m = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        var = ((fmap - m) ** 2 * gw).flatten(2).sum(-1) / den
        std = (var + 1e-6).sqrt()
        return torch.cat([mean, std], dim=1)

    def _emb(self, t, ct, ctg, pet, dose, g):
        bb, proj = self.backbone, self.proj
        g_map = g if getattr(bb, "use_deep_gate", False) else None
        if t == "ct":
            x = self.ct_adapter(torch.cat([ctg, dose, g], 1)) if GATE_AS_CHANNEL else ctg
            return proj(bb(x, gate_map=g_map))
        chans = [ctg, dose] if dg.PET_ROBUST else [ctg, pet, dose]
        if GATE_AS_CHANNEL:
            chans = chans + [g]
        return proj(bb(self.early_adapter(torch.cat(chans, 1)), gate_map=g_map))

    def forward(self, b):
        ct, dose = b["ct"], b["dose"]
        if GATE_INPUT_VOL == "dose":
            ct = dose
        pet = b.get("pet")
        anat = b.get("anat")
        if self.ntcp:
            g = self.gate(dose, anat, ct=ct)
            g = self._lat_boost(g, anat)
            ct = self._spotlight(ct, g)
            chans = [ct] if GATE_EARLY_CT_ONLY else ([ct, dose] if dg.PET_ROBUST else [ct, pet, dose])
            x = self.early_adapter(torch.cat(chans, 1))
            fmap = self.backbone.forward_map(x)
            gw = F.adaptive_avg_pool3d(g, fmap.shape[2:])
            lungw = (F.adaptive_avg_pool3d(anat[:, 0:1], fmap.shape[2:])
                     if ((GATE_BAND_POOLING or self.volfrac) and anat is not None) else None)
            pooled = self._pool(fmap, gw, lungw)
            emb = self.proj(pooled)
            logit = self.head(emb)
            if self.volfrac:
                logit = logit + self.gate.vol_head(pooled[:, -1:])
            return {"logit_final": logit, "logit_early": self.aux["early"](emb)}
        g = self.gate.gate(dose, anat, ct=ct)
        ctg = self._spotlight(ct, g) if GATE_SPOTLIGHT else self.gate(ct, dose, anat)
        emb = {t: self._emb(t, ct, ctg, pet, dose, g) for t in self.tokens}
        if len(self.tokens) == 1:
            fused = emb[self.tokens[0]]
        else:
            toks = torch.stack([emb[t] for t in self.tokens], 1)
            w = torch.softmax(self.scorer(toks).squeeze(-1), 1)
            fused = (toks * w.unsqueeze(-1)).sum(1)
        out = {"logit_final": self.head(fused)}
        for t in self.tokens:
            out[f"logit_{t}"] = self.aux[t](emb[t])
        return out

    @torch.no_grad()
    def pooled_feature(self, b):
        self.eval()
        ct, dose = b["ct"], b["dose"]
        if GATE_INPUT_VOL == "dose":
            ct = dose
        pet = b.get("pet")
        anat = b.get("anat")
        if self.ntcp:
            g = self.gate(dose, anat, ct=ct)
            g = self._lat_boost(g, anat)
            ct = self._spotlight(ct, g)
            chans = [ct] if GATE_EARLY_CT_ONLY else ([ct, dose] if dg.PET_ROBUST else [ct, pet, dose])
            x = self.early_adapter(torch.cat(chans, 1))
            fmap = self.backbone.forward_map(x)
            gw = F.adaptive_avg_pool3d(g, fmap.shape[2:])
            lungw = (F.adaptive_avg_pool3d(anat[:, 0:1], fmap.shape[2:])
                     if ((GATE_BAND_POOLING or self.volfrac) and anat is not None) else None)
            return self._pool(fmap, gw, lungw)
        g = self.gate.gate(dose, anat, ct=ct)
        ctg = self._spotlight(ct, g) if GATE_SPOTLIGHT else self.gate(ct, dose, anat)
        t = self.tokens[0]
        bb = self.backbone
        g_map = g if getattr(bb, "use_deep_gate", False) else None
        if t == "ct":
            x = self.ct_adapter(torch.cat([ctg, dose, g], 1)) if GATE_AS_CHANNEL else ctg
            return bb(x, gate_map=g_map)
        chans = [ctg, dose] if dg.PET_ROBUST else [ctg, pet, dose]
        if GATE_AS_CHANNEL:
            chans = chans + [g]
        return bb(self.early_adapter(torch.cat(chans, 1)), gate_map=g_map)


def _export_fold_embeddings(model, ds_clean, mdf, device, outer_k, inner_i,
                            results_dir, folds, dev_idx, test_idx):
    import numpy as _np, json as _json
    all_idx = list(range(len(mdf)))
    uuids = [str(mdf.iloc[i]["UUID"]).strip() for i in all_idx]
    bs = max(dg.BATCH_SIZE, 2)
    loader = DataLoader(Subset(ds_clean, all_idx), shuffle=False, **dg._loader_kwargs(bs))
    feats = []
    model.eval()
    with torch.no_grad():
        _b0 = next(iter(loader))
        _b0 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _b0.items()}
        _ct, _dose, _anat = _b0["ct"], _b0.get("dose"), _b0.get("anat")
        log.info("    [diag] inputs: ct[std=%.4g mean=%.4g] dose[std=%.4g] anat[sum=%.4g] "
                 "(std~0 => degenerate volumes)", float(_ct.std()), float(_ct.mean()),
                 float(_dose.std()) if _dose is not None else -1.0,
                 float(_anat.sum()) if _anat is not None else -1.0)
        if getattr(model, "ntcp", False) and _dose is not None:
            _g = model.gate(_dose, _anat, ct=_ct)
            log.info("    [diag] gate g: mean=%.4g max=%.4g per-sample mass=%s "
                     "(mass~0 => gate kills all voxels -> constant features)",
                     float(_g.mean()), float(_g.max()),
                     [round(float(x), 2) for x in _g.flatten(1).sum(1)[:4]])
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            feats.append(model.pooled_feature(b).detach().cpu().float().numpy())
    feats = _np.concatenate(feats, 0)
    log.info("    [diag] per-patient feature variation: mean-across-features std=%.4g, "
             "global range=%.4g (near-0 => identical features for every patient -> AUC 0.5)",
             float(_np.mean(_np.std(feats, axis=0))), float(_np.ptp(feats)))
    tr_rel, val_rel = folds[inner_i]
    tr_uuids  = [str(mdf.iloc[dev_idx[i]]["UUID"]).strip() for i in tr_rel]
    val_uuids = [str(mdf.iloc[dev_idx[i]]["UUID"]).strip() for i in val_rel]
    tst_uuids = [str(mdf.iloc[i]["UUID"]).strip() for i in test_idx]
    leak = (set(tr_uuids) & set(val_uuids)) | (set(tr_uuids) & set(tst_uuids))
    if leak:
        raise RuntimeError(f"[export] fold(k={outer_k},i={inner_i}) train set overlaps "
                           f"val/test - leakage: {sorted(leak)[:5]}")
    tag = f"fold{outer_k}_fold{inner_i}"
    _np.savez_compressed(os.path.join(results_dir, f"imaging_emb_{tag}.npz"),
                         uuids=_np.array(uuids), feats=feats)
    with open(os.path.join(results_dir, f"fold_partition_{tag}.json"), "w") as fp:
        _json.dump({"outer": int(outer_k), "inner": int(inner_i),
                    "feat_dim": int(feats.shape[1]), "n_patients": int(feats.shape[0]),
                    "train": tr_uuids, "val": val_uuids, "test": tst_uuids}, fp, indent=2)
    log.info("    [export] fold(k=%d,i=%d) -> imaging_emb_%s.npz  (n=%d, fdim=%d; "
             "train=%d val=%d test=%d)", outer_k, inner_i, tag, feats.shape[0],
             feats.shape[1], len(tr_uuids), len(val_uuids), len(tst_uuids))


def _append_live_history(path, rec, fold_tag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"inner_fold": fold_tag, **rec}
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _unfreeze_module(module, opt, lr, wd):
    ps = [p for p in module.parameters() if not p.requires_grad]
    for p in ps:
        p.requires_grad = True
    if ps:
        opt.add_param_group({"params": ps, "lr": lr, "weight_decay": wd})
    return sum(p.numel() for p in ps)


def _gate_param_groups(model):
    """Optimiser groups (readout | gate curve | thresholds if THRESH_LR) and the scheduler
    floor: per group when GATE_LR_CONSTANT pins the gate groups at their own rate."""
    gate_all = list(model.gate.parameters()) + ([model.f_raw] if GATE_SPOTLIGHT else [])
    thr_p = []
    if GATE_THRESH_LR is not None:
        b_raw = getattr(model.gate, "b_raw", None)
        if isinstance(b_raw, nn.Parameter):
            thr_p = [b_raw]
        else:
            log.warning("THRESH_LR=%s set but the gate has no b_raw control points; ignored.", GATE_THRESH_LR)
    gate_p = [p for p in gate_all if not any(p is q for q in thr_p)]
    other_p = [p for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith("gate.") and n != "f_raw"]
    groups = [{"params": other_p, "lr": dg.LR, "weight_decay": dg.WEIGHT_DECAY, "floor": 1e-7},
              {"params": gate_p, "lr": dg.GATE_LR, "weight_decay": 0.0,
               "floor": dg.GATE_LR if GATE_LR_CONSTANT else 1e-7},
              {"params": thr_p, "lr": GATE_THRESH_LR or 0.0, "weight_decay": 0.0,
               "floor": GATE_THRESH_LR if (GATE_LR_CONSTANT and GATE_THRESH_LR) else 1e-7}]
    groups = [g for g in groups if g["params"]]
    floors = [g.pop("floor") for g in groups]
    if thr_p:
        log.info("  gate thresholds (b_raw) in their own group: lr=%g%s", GATE_THRESH_LR,
                 " (constant)" if GATE_LR_CONSTANT else "")
    return groups, (floors if GATE_LR_CONSTANT else 1e-7)


def train_one_gate(tl, vl, device, tr_lbl, seed_offset, train_eval_loader=None,
                   max_epochs=None, live_csv=None, fold_tag=None, ext_outer=0, ext_inner=0,
                   test_loader=None):
    dg.set_seed(dg.SEED + seed_offset)
    max_epochs = max_epochs or dg.MAX_EPOCHS
    model = GateNet().to(device)

    wandb.init(
        project="rp-fuse",
        group="gate_ct",
        name=f"fold_O{ext_outer}_I{ext_inner}",
        config=RUN_CONFIG,
        reinit=True
    )
    groups, min_lrs = _gate_param_groups(model)
    opt = torch.optim.AdamW(groups)
    crit = FocalLoss(alpha=_gate_alpha(tr_lbl, device), gamma=GATE_FOCAL_GAMMA)
    if GATE_FOCAL_GAMMA_FINAL != GATE_FOCAL_GAMMA or GATE_CLASS_WEIGHT_POWER != 1.0:
        log.info("  Focal: gamma %.2f -> %.2f over %.0f%%; class-weight power %.2f",
                 GATE_FOCAL_GAMMA, GATE_FOCAL_GAMMA_FINAL,
                 100 * GATE_FOCAL_GAMMA_ANNEAL_FRAC, GATE_CLASS_WEIGHT_POWER)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and dg.AMP_DTYPE == torch.float16
    scaler = GradScaler("cuda") if use_scaler else None
    checkpoint_metric = str(_gcfg.get("CHECKPOINT_METRIC", getattr(dg, "CHECKPOINT_METRIC", "loss"))).lower()
    sched_mode = "min" if checkpoint_metric == "loss" else "max"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=sched_mode, factor=0.5,
                                                       patience=7, min_lr=min_lrs)
    best_auc, best_loss, best_state, best_thr, ep_count, no_imp, best_epoch = -1.0, float("inf"), None, 0.5, 0, 0, 0
    min_epochs = dg.EARLY_STOP_MIN_EPOCHS
    stages, stage_i = [], 0
    if GATE_PROGRESSIVE_UNFREEZE:
        bb = model.backbone.net
        stages = [getattr(bb, nm) for nm in ("layer4", "layer3", "layer2", "layer1") if hasattr(bb, nm)]
    hist: list[dict] = []
    auc_hist: list[float] = []
    loss_hist: list[float] = []
    warmup_unfrozen = False
    _gate_checked = False
    _gate_init = dg.gate_param_snapshot(model.gate)
    log.info("[gate] %s init: %s", type(model.gate).__name__,
             {k: round(v, 4) for k, v in _gate_init.items()})
    if GATE_WARMUP_UNFREEZE and not GATE_PROGRESSIVE_UNFREEZE:
        log.info("  Fusion-matched unfreeze: backbone frozen for %d warmup epochs, "
                 "then fine-tuned at 0.1x LR.", dg.WARMUP_EPOCHS)
    for epoch in range(1, max_epochs + 1):
        crit.gamma = _gate_focal_gamma_at(epoch, max_epochs)
        if (GATE_WARMUP_UNFREEZE and not GATE_PROGRESSIVE_UNFREEZE
                and epoch == dg.WARMUP_EPOCHS + 1 and not warmup_unfrozen):
            n_new = _unfreeze_module(model.backbone, opt, dg.LR * 0.1, dg.WEIGHT_DECAY)
            log.info("  [epoch %d] unfroze backbone (%d params) at 0.1x LR", epoch, n_new)
            warmup_unfrozen = True
        model.train()
        opt.zero_grad()
        n = len(tl)
        epoch_loss = 0.0
        _bar = tqdm(tl, desc=f"epoch {epoch}/{max_epochs}", unit="batch", leave=False,
                    dynamic_ncols=True, disable=not dg.SHOW_PROGRESS)
        for step, batch in enumerate(_bar, 1):
            b = dg._to_device(batch, device)
            lbl = b["label"]
            if dg.P_MIXUP > 0 and dg.MIXUP_ALPHA > 0 and lbl.size(0) > 1 and np.random.rand() < dg.P_MIXUP:
                b, ya, yb, lam = dg._mixup(b, dg.MIXUP_ALPHA)
            else:
                ya, yb, lam = lbl, lbl, 1.0
            with autocast("cuda", dtype=dg.AMP_DTYPE, enabled=use_amp):
                out = model(b)
                main = lam * crit(out["logit_final"], ya) + (1 - lam) * crit(out["logit_final"], yb)
                aux = sum(lam * crit(out[f"logit_{t}"], ya) + (1 - lam) * crit(out[f"logit_{t}"], yb)
                          for t in model.tokens) / len(model.tokens)
                loss = main + GATE_AUX * aux
                if GATE_REG > 0:
                    loss = loss + GATE_REG * model.gate.reg()
                loss = loss / dg.ACCUMULATION
            (scaler.scale(loss).backward() if scaler else loss.backward())
            if not _gate_checked:
                _gate_checked = True
                dg.assert_gate_params_learnable(model.gate)
                log.info("[gate] dose-response span over 0-%.0f Gy: %.3f",
                         dg.DOSE_NORM_CGY / 100.0,
                         dg.assert_gate_curve_active(model.gate))
                log.info("[gate] first-step |grad|: %s",
                         {k: f"{v:.2e}" for k, v in
                          dg.gate_grad_snapshot(model.gate).items()})
            epoch_loss += loss.item() * dg.ACCUMULATION
            _bar.set_postfix(loss=f"{loss.item() * dg.ACCUMULATION:.4f}", refresh=False)
            if step % dg.ACCUMULATION == 0 or step == n:
                if scaler:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                opt.zero_grad()
        val_loss, vp, vl_y = dg.evaluate(model, vl, crit, device)
        val_auc = roc_auc_score(vl_y, vp) if len(set(vl_y.tolist())) > 1 else 0.5
        auc_hist.append(val_auc)
        loss_hist.append(val_loss)
        sm_auc = float(np.mean(auc_hist[-3:]))
        sm_loss = float(np.mean(loss_hist[-3:]))
        sched.step(sm_loss if checkpoint_metric == "loss" else sm_auc); ep_count = epoch
        thr_e = dg.optimal_f1_threshold(vp, vl_y)
        vm_e = dg.metrics_at_threshold(vp, np.asarray(vl_y), thr_e)
        thr_y_e = dg.youden_threshold(vp, np.asarray(vl_y))
        vm_y = dg.metrics_at_threshold(vp, np.asarray(vl_y), thr_y_e)
        rec = {"epoch": epoch, "val_auc": float(val_auc),
               "val_f1": float(vm_e["f1"]),
               "val_sensitivity": float(vm_e["sensitivity"]),
               "val_specificity": float(vm_e["specificity"]),
               "val_sensitivity_youden": float(vm_y["sensitivity"]),
               "val_specificity_youden": float(vm_y["specificity"]),
               "train_auc": float("nan"), "train_f1": float("nan")}
        train_eval_loss = None
        if train_eval_loader is not None:
            train_eval_loss, tpe, tle = dg.evaluate(model, train_eval_loader, crit, device)
            tmv = dg.metrics_at_threshold(tpe, np.asarray(tle), thr_e)
            rec["train_auc"], rec["train_f1"] = float(tmv["auc"]), float(tmv["f1"])
            rec["train_sensitivity"] = float(tmv["sensitivity"])
            rec["train_specificity"] = float(tmv["specificity"])
        if test_loader is not None:
            _, tte, tty = dg.evaluate(model, test_loader, crit, device)
            ttm = dg.metrics_at_threshold(tte, np.asarray(tty), thr_e)
            ttm_y = dg.metrics_at_threshold(tte, np.asarray(tty), thr_y_e)
            rec["test_auc"] = float(ttm["auc"])
            rec["test_f1"] = float(ttm["f1"])
            rec["test_sensitivity"] = float(ttm["sensitivity"])
            rec["test_specificity"] = float(ttm["specificity"])
            rec["test_sensitivity_youden"] = float(ttm_y["sensitivity"])
            rec["test_specificity_youden"] = float(ttm_y["specificity"])
        gp = model.gate
        rec.update({"steepness": float(gp.steepness), "dose_thresh": float(gp.dose_thresh),
                    "amplify": float(gp.amplify), "prox_weight": float(gp.prox_weight),
                    "lobe_weight": float(getattr(gp, "lobe_weight", 0.0)),
                    "vuln_weight": float(getattr(gp, "vuln_weight", 0.0)),
                    "region_weight": float(getattr(gp, "region_weight", 0.0)),
                    "baseline": float(gp.baseline),
                    "threshold_Gy": float(gp.dose_thresh) * dg.DOSE_NORM_CGY / 100.0,
                    "unfrozen_stage": stage_i})
        if GATE_SPOTLIGHT:
            rec["spotlight_f"] = float(torch.sigmoid(model.f_raw))
        _snap = dg.gate_param_snapshot(gp)
        for k, v in _snap.items():
            rec[f"gate.{k}"] = v
            rec[f"gate.d_{k}"] = v - _gate_init.get(k, v)
        log.info("[gate] epoch %d drift-from-init: %s", epoch,
                 {k: round(rec[f"gate.d_{k}"], 5) for k in _snap})
        hist.append(rec)

        avg_train_loss = epoch_loss / max(n, 1)
        log_dict = {
            "Loss/train_batch_avg": avg_train_loss,
            "Loss/val": val_loss,
            "AUC/val": val_auc,
            "F1/val": rec["val_f1"],
            "Sensitivity/val": rec["val_sensitivity"],
            "Specificity/val": rec["val_specificity"],
            "Sensitivity_Youden/val": rec["val_sensitivity_youden"],
            "Specificity_Youden/val": rec["val_specificity_youden"],
        }
        if train_eval_loss is not None:
            log_dict["Loss/train_eval"] = train_eval_loss
        if not np.isnan(rec["train_auc"]):
            log_dict["AUC/train"] = rec["train_auc"]
            log_dict["F1/train"] = rec["train_f1"]
            log_dict["Sensitivity/train"] = rec["train_sensitivity"]
            log_dict["Specificity/train"] = rec["train_specificity"]
        if "test_auc" in rec:
            log_dict["AUC/test"] = rec["test_auc"]
            log_dict["F1/test"] = rec["test_f1"]
            log_dict["Sensitivity/test"] = rec["test_sensitivity"]
            log_dict["Specificity/test"] = rec["test_specificity"]
            log_dict["Sensitivity_Youden/test"] = rec["test_sensitivity_youden"]
            log_dict["Specificity_Youden/test"] = rec["test_specificity_youden"]

        for k in ("steepness", "dose_thresh", "prox_weight", "lobe_weight", "vuln_weight",
                  "region_weight", "threshold_Gy", "unfrozen_stage", "spotlight_f"):
            if k in rec:
                log_dict[f"Gate/{k}"] = rec[k]
        for _k, _v in rec.items():
            if _k.startswith("gate."):
                log_dict[f"GateRaw/{_k[5:]}"] = _v
        wandb.log(log_dict, step=epoch)

        if live_csv is not None:
            _append_live_history(live_csv, rec, fold_tag)
        _sel_floor = min(min_epochs, max_epochs)
        is_better = epoch >= _sel_floor and (
            (sm_loss < best_loss - 1e-4) if checkpoint_metric == "loss"
            else (sm_auc > best_auc + 1e-4))
        if is_better:
            best_loss, best_auc, best_thr, best_epoch = sm_loss, sm_auc, thr_e, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if stages and stage_i < len(stages) and no_imp >= GATE_UNFREEZE_PATIENCE:
                nadd = _unfreeze_module(stages[stage_i], opt, dg.LR * 0.1, dg.WEIGHT_DECAY)
                stage_i += 1; no_imp = 0
                log.info("    progressive unfreeze -> stage %d/%d (+%s params) @ epoch %d (val plateau)",
                         stage_i, len(stages), f"{nadd:,}", epoch)
        fully_unfrozen = (not stages) or (stage_i >= len(stages))
        if epoch >= min_epochs and no_imp >= dg.EARLY_STOP_PATIENCE and fully_unfrozen:
            break
    if best_state:
        model.load_state_dict(best_state)
    model._gate_history = hist
    model._deployed_epoch = best_epoch
    wandb.finish()
    return model, best_thr, ep_count


def _save_xlsx(uuids, y_true, probs, thr, unc=None, thr_y=None):
    os.makedirs(PRED_DIR, exist_ok=True)
    preds = (probs >= thr).astype(int)
    cols = {"UUID": uuids, "TrueLabel": y_true,
            "RP_Probability": np.round(probs, 4), "PredictedLabel": preds,
            "Correct": (preds == y_true).astype(int)}
    if thr_y is not None:
        cols["PredictedLabel_Youden"] = (probs >= thr_y).astype(int)
    if unc is not None:
        cols["Uncertainty_MC_std"] = np.round(unc, 4)
    pd.DataFrame(cols).to_excel(os.path.join(PRED_DIR, "GATE-CT_test_predictions.xlsx"),
                                index=False)


def gate_main(smoke: bool = False):
    device = dg.resolve_device()
    _log_run_config()
    log.info("GATE-CT | backbone=R%d  token=%s  train_only_gate=%s  gate_as_channel=%s  reg=%.3g  clamp_b=%s  "
             "init_b=%.2f  prox_nonneg=%s  embed=%d  aux=%.2g  prog_unfreeze=%s(pat=%d)  max_ep=%s  "
             "ntcp=%s(K=%d)",
             dg.BACKBONE_DEPTH, TOKEN, TRAIN_ONLY_GATE, GATE_AS_CHANNEL, GATE_REG, CLAMP_B,
             GATE_INIT_B, GATE_PROX_NONNEG, GATE_EMBED, GATE_AUX,
             GATE_PROGRESSIVE_UNFREEZE, GATE_UNFREEZE_PATIENCE, (GATE_MAX_EPOCHS or dg.MAX_EPOCHS),
             GATE_DOSE_RESPONSE, GATE_DOSE_RESPONSE_K)
    if GATE_SPOTLIGHT:
        log.info("           spotlight=ON  background_floor_init=%.2f (CT*(f+(1-f)g); darkens non-gated)",
                 GATE_SPOTLIGHT_INIT)
    if NO_GATE:
        log.info("GATE-CT | NO GATE (control): IdentityGate, CT restricted to the "
                 "healthy-lung mask and pooled uniformly%s",
                 "" if dg.USE_ANATOMY_GATE else " -- ANATOMY GATE OFF, so the CT is not "
                 "even lung-masked and this is not the control the gate arm is compared with")
    elif HARD_MASK:
        log.info("           HARD-MASK baseline=ON  CT'=CT*1[dose>=%.0f Gy]*healthy-lung (rigid, non-learnable; overrides NTCP)",
                 HARD_MASK_GY)
    manifest = dg.read_manifest()
    cache = dg.CACHE_VOLUMES and not smoke
    ds_clean = dg.MultiModalDataset(manifest, dg._cohort_dir, {}, 0, mode="clean", cache=cache)
    ds_noisy = dg.MultiModalDataset(manifest, dg._cohort_dir, {}, 0, mode="heavy", cache=cache)
    mdf = ds_clean.df
    log.info("Patients on disk (CT+Dose): %d", len(mdf))
    dev_idx, test_idx, folds = dg.resolve_split(mdf)
    labels = [1 if str(g).strip() == "RP" else 0 for g in mdf["Group"].values]
    test_labels = [labels[i] for i in test_idx]
    test_uuids = [str(mdf.iloc[i]["UUID"]) for i in test_idx]
    bs = max(dg.BATCH_SIZE, 2)
    me = 1 if smoke else (GATE_MAX_EPOCHS or dg.MAX_EPOCHS)
    if smoke:
        folds = folds[:1]

    te = (DataLoader(Subset(ds_clean, test_idx), shuffle=False, **dg._loader_kwargs(bs))
          if (test_idx and not smoke) else None)
    live_csv = os.path.join(RESULTS_DIR, "gate_history_live.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(live_csv):
        os.remove(live_csv)
    cv, best = [], None
    all_hist, inner_rows = [], []
    fold_probs, fold_aucs, oof_vp, oof_vl = [], [], [], []
    for fi, (tr_rel, val_rel) in enumerate(folds, 1):
        tr = [dev_idx[i] for i in tr_rel]; va = [dev_idx[i] for i in val_rel]
        tr_lbl, va_lbl = [labels[i] for i in tr], [labels[i] for i in va]
        if smoke:
            tr, va = tr[:3], va[:2]; tr_lbl, va_lbl = [labels[i] for i in tr], [labels[i] for i in va]
        tl = DataLoader(Subset(ds_noisy, tr), shuffle=True, **dg._loader_kwargs(bs))
        vl = DataLoader(Subset(ds_clean, va), shuffle=False, **dg._loader_kwargs(bs))
        to = DataLoader(Subset(ds_clean, tr), shuffle=False, **dg._loader_kwargs(bs))
        ext_outer = int(os.getenv("NESTED_OUTER_FOLD", "0"))
        model, thr, ep = train_one_gate(tl, vl, device, tr_lbl, seed_offset=fi,
                                        train_eval_loader=to, max_epochs=me,
                                        live_csv=live_csv, fold_tag=fi,
                                        ext_outer=ext_outer, ext_inner=fi - 1,
                                        test_loader=te)
        crit = FocalLoss(alpha=dg._class_weights(tr_lbl, device), gamma=2)
        _, vp, _ = dg.evaluate(model, vl, crit, device)
        _, tp, _ = dg.evaluate(model, to, crit, device)
        vm = dg.metrics_at_threshold(vp, np.array(va_lbl), thr)
        tm = dg.metrics_at_threshold(tp, np.array(tr_lbl), thr)
        log.info("  Fold %d | Train AUC=%.3f | Val AUC=%.3f F1=%.3f | Ep=%d",
                 fi, tm["auc"], vm["auc"], vm["f1"], ep)
        if EXPORT_EMB and not smoke:
            _export_fold_embeddings(model, ds_clean, mdf, device, ext_outer, fi - 1,
                                    RESULTS_DIR, folds, dev_idx, test_idx)
        gpf = model.gate
        def _gp(name):
            v = getattr(gpf, name, None)
            return None if v is None else float(v)

        _extra = []
        if dg.USE_ANATOMY_GATE:
            for _nm, _fmt, _on in (("prox_weight", "prox=%.2f", True),
                                   ("lobe_weight", "lobe=%.2f", dg.USE_LOBE_GATE),
                                   ("vuln_weight", "vuln=%.2f", True)):
                _v = _gp(_nm)
                if _on and _v is not None:
                    _extra.append(_fmt % _v)
        if dg.USE_REGION_GATE and _gp("region_weight") is not None:
            _extra.append("region=%+.2f" % _gp("region_weight"))
        if GATE_SPOTLIGHT:
            _extra.append("f=%.2f" % float(torch.sigmoid(model.f_raw)))
        if getattr(model, "volfrac", False):
            _vw = model.gate.vol_head.weight.detach().float()
            _extra.append("betaV=%+.3f" % float(_vw[1, 0] - _vw[0, 0]))
        init_steep_ = float(getattr(gpf, "init_steepness", dg.DG_INIT_A))
        init_thr_ = float(getattr(gpf, "init_dose_thresh", dg.DG_INIT_B))
        log.info("    gate[fold %d] | steepness=%.2f  dose_thresh=%.3f (~%.1f Gy)%s   "
                 "[init: steepness=%.3f dose_thresh=%.2f]",
                 fi, float(gpf.steepness), float(gpf.dose_thresh),
                 float(gpf.dose_thresh) * dg.DOSE_NORM_CGY / 100.0,
                 ("  " + "  ".join(_extra)) if _extra else "",
                 init_steep_, init_thr_)
        if not smoke and GATE_DOSE_RESPONSE and hasattr(gpf, "curve_dict"):
            import json as _json
            with open(os.path.join(RESULTS_DIR, f"gate_fold{ext_outer}_fold{fi - 1}.json"), "w") as _gfp:
                _json.dump(_gate_vars(gpf, model), _gfp, indent=2)
        if not smoke and EXPORT_EMB:
            _tr = {k: v.detach().cpu() for k, v in model.state_dict().items()
                   if not k.startswith("backbone.")}
            _tr["__arch__"] = {"ntcp_k": GATE_DOSE_RESPONSE_K, "gate_as_channel": GATE_AS_CHANNEL,
                               "pet_robust": bool(dg.PET_ROBUST), "backbone_depth": dg.BACKBONE_DEPTH,
                               "ntcp": GATE_DOSE_RESPONSE, "spotlight": GATE_SPOTLIGHT,
                               "early_ct_only": GATE_EARLY_CT_ONLY,
                               "hard_mask": HARD_MASK, "no_gate": NO_GATE,
                               "dispersion": GATE_DISPERSION, "embed": GATE_EMBED,
                               "token": TOKEN, "spotlight_init": GATE_SPOTLIGHT_INIT,
                               "input_vol": GATE_INPUT_VOL,
                               "hard_mask_thr": _HARD_MASK_THR, "dropout": dg.DROPOUT,
                               "backbone_frozen": bool(TRAIN_ONLY_GATE and not GATE_WARMUP_UNFREEZE),
                               "use_anatomy_gate": bool(dg.USE_ANATOMY_GATE),
                               "use_lobe_gate": bool(dg.USE_LOBE_GATE),
                               "use_region_gate": bool(dg.USE_REGION_GATE)}
            torch.save(_tr, os.path.join(RESULTS_DIR, f"gatenet_fold{ext_outer}_fold{fi - 1}.pt"))
        dep_ep = getattr(model, "_deployed_epoch", ep)
        for _r in getattr(model, "_gate_history", []):
            _rr = dict(_r); _rr["inner_fold"] = fi
            _rr["deployed"] = int(_r.get("epoch") == dep_ep)
            all_hist.append(_rr)
        cv.append({"train": tm, "val": vm, "epochs": ep, "threshold": thr})
        if te is not None:
            ab = dg.fit_platt(np.asarray(vp), np.asarray(va_lbl))
            _, tpd, _ = dg.evaluate(model, te, crit, device)
            tpc = dg.apply_platt(tpd, ab)
            fold_probs.append(tpc)
            oof_vp.extend(dg.apply_platt(np.asarray(vp), ab).tolist()); oof_vl.extend(va_lbl)
            a_k = roc_auc_score(test_labels, tpc) if len(set(test_labels)) > 1 else 0.5
            fold_aucs.append(a_k)
            tem = dg.metrics_at_threshold(tpc, np.array(test_labels), thr)
            cv[-1]["test"] = tem
            log.info("    [diag] inner fold %d -> outer-test AUC=%.3f (val AUC=%.3f)", fi, a_k, vm["auc"])
            inner_rows.append({
                "inner_fold": fi, "deployed_epoch": dep_ep,
                "val_auc": round(float(vm["auc"]), 4), "outer_test_auc": round(float(a_k), 4),
                "threshold_Gy": round(float(gpf.dose_thresh) * dg.DOSE_NORM_CGY / 100.0, 1),
                "steepness": round(float(gpf.steepness), 3), "amplify": round(float(gpf.amplify), 3),
                "prox_weight": round(float(gpf.prox_weight), 3), "baseline": round(float(gpf.baseline), 3)})
        if not smoke and (best is None or vm["auc"] > best["auc"]):
            if best is not None: del best["model"]
            model.to("cpu"); best = {"auc": vm["auc"], "model": model, "fold": fi}
        else:
            del model
        dg._shutdown(tl, vl, to)
        try:
            wandb.finish()
        except Exception:
            pass
        if device.type == "cuda": torch.cuda.empty_cache()

    msv = dg._mean_std(cv, "val")
    log.info("CV val: AUC=%.3f±%.3f  F1=%.3f±%.3f", msv["auc"][0], msv["auc"][1],
             msv["f1"][0], msv["f1"][1])
    test_metrics, test_metrics_y = {}, {}
    if te is not None and best is not None and fold_probs:
        thr = dg.optimal_f1_threshold(np.asarray(oof_vp), np.asarray(oof_vl))
        thr_y = dg.youden_threshold(np.asarray(oof_vp), np.asarray(oof_vl))
        tp_ens = np.mean(np.stack(fold_probs, 0), 0)
        test_metrics = dg.metrics_at_threshold(tp_ens, np.array(test_labels), thr)
        test_metrics_y = dg.metrics_at_threshold(tp_ens, np.array(test_labels), thr_y)
        log.info("Inner-model ensemble | %d models  test AUC=%.3f (per-model: %s)  thr=%.3f",
                 len(fold_probs), test_metrics["auc"],
                 ", ".join(f"{a:.3f}" for a in fold_aucs), thr)
        model = best["model"].to(device)
        gp = model.gate
        log.info("Learned gate (deployed): steepness=%.3f dose_thresh=%.3f (thr~%.1f Gy) "
                 "amplify=%.3f prox_weight=%.3f baseline=%.3f",
                 float(gp.steepness), float(gp.dose_thresh),
                 float(gp.dose_thresh) * dg.DOSE_NORM_CGY / 100.0,
                 float(gp.amplify), float(gp.prox_weight), float(gp.baseline))
        if GATE_SPOTLIGHT:
            f = float(torch.sigmoid(model.f_raw))
            log.info("Spotlight (deployed): background floor f=%.3f  (focus/background contrast=%.1fx)",
                     f, 1.0 / max(f, 1e-3))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if GATE_DOSE_RESPONSE and hasattr(gp, "curve_dict"):
            import json as _json
            with open(os.path.join(RESULTS_DIR, "dose_response_gate.json"), "w") as fp:
                _json.dump(_gate_vars(gp, model), fp, indent=2)
            cd = gp.curve_dict()
            log.info("Neural-NTCP curve (deployed): K=%d  thresholds(Gy)=%s  weights=%s  d50~%.1f Gy",
                     cd["K"], [round(x, 1) for x in cd["b_Gy"]], [round(x, 2) for x in cd["w"]],
                     float(gp.dose_thresh) * dg.DOSE_NORM_CGY / 100.0)
        _, tp_std, _ = dg.evaluate_mc(model, te, device)
        log.info("TEST | AUC=%.3f | F1-thr=%.3f: F1=%.3f Sens=%.3f Spec=%.3f | "
                 "Youden-thr=%.3f: Sens=%.3f Spec=%.3f F1=%.3f (ensemble of %d)",
                 test_metrics["auc"], thr, test_metrics["f1"], test_metrics["sensitivity"],
                 test_metrics["specificity"], thr_y, test_metrics_y["sensitivity"],
                 test_metrics_y["specificity"], test_metrics_y["f1"], len(fold_probs))
        try:
            wandb.init(project="rp-fuse", group="gate_ct", name=f"fold_O{ext_outer}_TEST",
                       config=RUN_CONFIG, reinit=True)
            wandb.log({
                "Test/AUC": test_metrics["auc"],
                "Test/F1": test_metrics["f1"],
                "Test/Sensitivity": test_metrics["sensitivity"],
                "Test/Specificity": test_metrics["specificity"],
                "Test/F1_Youden": test_metrics_y["f1"],
                "Test/Sensitivity_Youden": test_metrics_y["sensitivity"],
                "Test/Specificity_Youden": test_metrics_y["specificity"],
                "Test/thr_f1": float(thr), "Test/thr_youden": float(thr_y)})
            wandb.finish()
        except Exception as _e:
            log.warning("wandb test-summary logging failed: %s", _e)
        _save_xlsx(test_uuids, np.array(test_labels), tp_ens, thr, unc=tp_std, thr_y=thr_y)
        del model, best
        dg._shutdown(te)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if all_hist:
        with open(os.path.join(RESULTS_DIR, "gate_history.csv"), "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(all_hist[0].keys()))
            w.writeheader(); w.writerows(all_hist)
    test_members_mean = test_members_std = float("nan")
    if inner_rows:
        keys_num = ("val_auc", "outer_test_auc", "threshold_Gy", "steepness",
                    "amplify", "prox_weight", "baseline")
        ndig = {"threshold_Gy": 1, "val_auc": 4, "outer_test_auc": 4}
        mean_row = {"inner_fold": "mean", "deployed_epoch": ""}
        std_row = {"inner_fold": "std", "deployed_epoch": ""}
        mean_row.update({k: round(float(np.mean([r[k] for r in inner_rows])), ndig.get(k, 3)) for k in keys_num})
        std_row.update({k: round(float(np.std([r[k] for r in inner_rows])), ndig.get(k, 3)) for k in keys_num})
        rows_out = inner_rows + [mean_row, std_row]
        with open(os.path.join(RESULTS_DIR, "gate_inner_folds.csv"), "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        test_members_mean = mean_row["outer_test_auc"]; test_members_std = std_row["outer_test_auc"]
        log.info("Inner-fold outer-test AUC: [%s]  members=%.3f±%.3f  |  ensemble=%.3f",
                 ", ".join(f"{r['outer_test_auc']:.3f}" for r in inner_rows),
                 test_members_mean, test_members_std,
                 float(test_metrics.get("auc", float("nan"))))

    mst = dg._mean_std(cv, "train")
    pd.DataFrame([{"model": "GATE-CT", "token": TOKEN, "train_only_gate": TRAIN_ONLY_GATE,
                   "gate_as_channel": GATE_AS_CHANNEL, "val_auc_mean": msv["auc"][0],
                   "val_auc_std": msv["auc"][1], "train_auc_mean": mst["auc"][0],
                   "test_auc": test_metrics.get("auc", np.nan),
                   "test_auc_members_mean": test_members_mean,
                   "test_auc_members_std": test_members_std,
                   "test_f1": test_metrics.get("f1", np.nan),
                   "test_sensitivity": test_metrics.get("sensitivity", np.nan),
                   "test_specificity": test_metrics.get("specificity", np.nan),
                   "test_sensitivity_youden": test_metrics_y.get("sensitivity", np.nan),
                   "test_specificity_youden": test_metrics_y.get("specificity", np.nan)}]).to_csv(
        os.path.join(RESULTS_DIR, "cv_results.csv"), index=False, float_format="%.4f")
    import util
    util.print_array_metrics({"GATE-CT": cv}, {"GATE-CT": test_metrics}, log)
    log.info("Done. Results -> %s", RESULTS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    use_nested = bool(dg._full_cfg.get("SPLIT", {}).get("USE_NESTED_CV", False))
    if use_nested and not args.smoke and os.environ.get("NESTED_OUTER_FOLD") is None:
        from nested_cv import run_nested_cv
        n_outer = int(dg._full_cfg.get("SPLIT", {}).get("N_OUTER_FOLDS", 5))
        n_repeats = int(dg._full_cfg.get("SPLIT", {}).get("N_REPEATS", 1))
        run_nested_cv(main_fn=lambda: gate_main(False), n_outer=n_outer,
                      n_repeats=n_repeats,
                      module_globals=globals(), results_dir_key="RESULTS_DIR",
                      pred_dir_key="PRED_DIR", log=log)
    else:
        gate_main(args.smoke)
