from __future__ import annotations

import os

import numpy as np
import pandas as pd

__all__ = ["EndpointConfig", "resolve", "apply_to_split", "apply_to_manifest",
           "read_manifest", "summary"]


def _section(cfg: dict | None) -> dict:
    c = cfg or {}
    return c.get("TRAIN_RPFUSE") or {}


class EndpointConfig:
    def __init__(self, cfg: dict | None = None):
        d = _section(cfg).get("DATA", {}) or {}
        self.excel_rel = d.get("EXCEL_PATH", "master_clinical_data.xlsx")
        self.enabled = bool(d.get("RP_ENDPOINT_ENABLED", True))
        self.window_m = float(d.get("RP_ENDPOINT_WINDOW_MONTHS", 6.0))
        self.min_fu_m = float(d.get("RP_MIN_FOLLOWUP_MONTHS", 6.0))

    def __repr__(self):
        return (f"EndpointConfig(enabled={self.enabled}, window={self.window_m}m, "
                f"min_followup={self.min_fu_m}m)")


def _master_times(master: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fu = ((pd.to_datetime(master.get("last_followup_date"), errors="coerce")
           - pd.to_datetime(master.get("RT_date"), errors="coerce")).dt.days / 30.44)
    fu = fu.where(fu >= 0)
    onset = pd.to_numeric(master.get("RP_onset_months"), errors="coerce")
    return fu.to_numpy(dtype=float), onset.to_numpy(dtype=float)


def resolve(excel_row_idx, raw_label, master: pd.DataFrame,
            cfg: EndpointConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    idx = np.asarray(excel_row_idx, dtype=int)
    y = np.asarray(raw_label, dtype=int).copy()
    keep = np.ones(len(y), dtype=bool)
    if not cfg.enabled or (cfg.window_m <= 0 and cfg.min_fu_m <= 0):
        return y, keep, {"late_to_negative": 0, "dropped_unobserved": 0,
                         "unlinked": int((idx < 0).sum())}

    fu_all, on_all = _master_times(master)
    ok = (idx >= 0) & (idx < len(master))
    safe = np.clip(idx, 0, max(len(master) - 1, 0))
    fu = np.where(ok, fu_all[safe], np.nan)
    onset = np.where(ok, on_all[safe], np.nan)

    n_late = 0
    if cfg.window_m > 0:
        late = (y == 1) & np.isfinite(onset) & (onset > cfg.window_m)
        y[late] = 0
        n_late = int(late.sum())
    if cfg.min_fu_m > 0:
        drop = (y == 0) & np.isfinite(fu) & (fu < cfg.min_fu_m)
        keep &= ~drop
    return y, keep, {"late_to_negative": n_late,
                     "dropped_unobserved": int((~keep).sum()),
                     "unlinked": int((~ok).sum())}


def summary(stats: dict, n_before: int, n_after: int, n_pos: int,
            cfg: EndpointConfig) -> str:
    return (f"RP endpoint 'within {cfg.window_m:.0f} months' "
            f"(min follow-up {cfg.min_fu_m:.0f}m): "
            f"{stats['late_to_negative']} late onset(s) -> negative, "
            f"{stats['dropped_unobserved']} unobserved dropped"
            + (f", {stats['unlinked']} unlinked kept" if stats.get("unlinked") else "")
            + f" | {n_before} -> {n_after} patients, RP+ {n_pos} "
              f"({100 * n_pos / max(n_after, 1):.0f}%)")


def apply_to_split(split: pd.DataFrame, master: pd.DataFrame,
                   cfg: EndpointConfig, log=None) -> pd.DataFrame:
    if not cfg.enabled:
        return split
    y, keep, st = resolve(split["excel_row_idx"].to_numpy(),
                          split["rp_label"].to_numpy(), master, cfg)
    out = split.copy()
    out["rp_label"] = y
    if "rp_status" in out.columns:
        out["rp_status"] = np.where(y == 1, "RP", "RP Not Present")
    out = out[keep].reset_index(drop=True)
    if log is not None:
        log.info(summary(st, len(split), len(out), int(out["rp_label"].sum()), cfg))
    return out


def apply_to_manifest(manifest: pd.DataFrame, split: pd.DataFrame,
                      master: pd.DataFrame, cfg: EndpointConfig,
                      log=None) -> pd.DataFrame:
    if not cfg.enabled:
        return manifest
    link = dict(zip(split["uuid"].astype(str).str.strip(),
                    split["excel_row_idx"].astype(int)))
    uu = manifest["UUID"].astype(str).str.strip()
    idx = uu.map(lambda u: link.get(u, -1)).to_numpy()
    raw = (manifest["Group"].astype(str).str.strip() == "RP").astype(int).to_numpy()
    y, keep, st = resolve(idx, raw, master, cfg)
    out = manifest.copy()
    out["Group"] = np.where(y == 1, "RP", "RP Not Present")
    out = out[keep].reset_index(drop=True)
    if log is not None:
        log.info(summary(st, len(manifest), len(out),
                         int((out["Group"] == "RP").sum()), cfg))
    return out


def read_manifest(manifest_path: str, input_dir: str,
                  cfg: EndpointConfig, log=None) -> pd.DataFrame:
    from util import read_table, load_unified_split
    m = read_table(manifest_path)
    if not cfg.enabled:
        return m
    return apply_to_manifest(m, load_unified_split(input_dir),
                             read_table(os.path.join(input_dir, cfg.excel_rel)),
                             cfg, log=log)
