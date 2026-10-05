import glob
import json
import logging
import os
import random

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset

PRIMARY_STUDY_TOKEN = "*_PRIMARY"


def resolve_primary_study(patient_dir: str) -> str | None:
    cands = []
    for mp in sorted(glob.glob(os.path.join(patient_dir, "studies", "*",
                                            "study_meta.json"))):
        sdir = os.path.dirname(mp)
        if not os.path.exists(os.path.join(sdir, "CT", "whole_body.npy")):
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            m = {}
        cands.append((sdir, m))
    if not cands:
        return None
    totals = [c for c in cands if c[1].get("is_total")]
    pool = totals or cands
    return max(pool, key=lambda c: (c[1].get("dose_max_cGy") or 0,
                                    c[1].get("lung_MLD_cGy") or 0))[0]


class MedicalDataset(Dataset):
    def __init__(self, df, root_dir, generated_files_config, transform=None,
                 target_shape=None, window_width=1500, window_center=-600, cache_data=False,
                 tumor_side_map: dict | None = None,
                 dose_weighted_crop: bool = False,
                 crop_keep_fraction: float = 0.65):
        self.root_dir = root_dir
        self.files_map = generated_files_config
        self.transform = transform
        self.target_shape = target_shape
        self.tumor_side_map = tumor_side_map or {}
        self.dose_weighted_crop = dose_weighted_crop
        self.crop_keep_fraction = float(crop_keep_fraction)

        self.win_min = window_center - (window_width / 2)
        self.win_max = window_center + (window_width / 2)

        self._cache: dict = {} if cache_data else None

        self.patient_path_map = {}
        if os.path.exists(root_dir):
            for dir_name in os.listdir(root_dir):
                full = os.path.join(root_dir, dir_name)
                if os.path.isdir(full):
                    self.patient_path_map[dir_name.strip()] = full
        else:
            logging.error(f"Root directory not found: {root_dir}")

        if 'UUID' in df.columns:
            df['UUID'] = df['UUID'].astype(str).str.strip()

        if 'Group' not in df.columns and 'Label' not in df.columns:
            logging.warning("No target labels found. Using random labels.")

        initial_count = len(df)
        self.df = df[df['UUID'].isin(self.patient_path_map.keys())].reset_index(drop=True)
        final_count = len(self.df)

        if initial_count != final_count:
            logging.info(f"[Dataset] Filtered missing files. Kept {final_count} / {initial_count}.")

    def __len__(self):
        return len(self.df)

    def _normalize(self, volume):
        volume = np.clip(volume, self.win_min, self.win_max)
        volume = (volume - self.win_min) / (self.win_max - self.win_min)
        return volume

    def _dose_crop(self, volume, tumor_side: int):
        if not self.dose_weighted_crop or tumor_side not in (1, 2):
            return volume
        W = volume.shape[-1]
        cut = int(round(W * self.crop_keep_fraction))
        if tumor_side == 1:
            return volume[..., :cut]
        return volume[..., W - cut:]

    def _resize_volume(self, volume):
        if self.target_shape is None:
            return volume
        current_shape = volume.shape
        zoom_factors = [t / c for t, c in zip(self.target_shape, current_shape)]
        resized = scipy.ndimage.zoom(volume, zoom_factors, order=1, prefilter=False)
        return resized

    def _load_one(self, patient_dir: str, fname: str, tumor_side: int) -> np.ndarray:
        parts = fname.split(os.sep)
        if PRIMARY_STUDY_TOKEN in parts:
            sdir = resolve_primary_study(patient_dir)
            if sdir is None:
                raise FileNotFoundError(
                    os.path.join(patient_dir, "studies", PRIMARY_STUDY_TOKEN))
            i = parts.index(PRIMARY_STUDY_TOKEN)
            path = os.path.join(sdir, *parts[i + 1:])
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        elif "*" in fname or "?" in fname:
            matches = sorted(glob.glob(os.path.join(patient_dir, fname)))
            if not matches:
                raise FileNotFoundError(os.path.join(patient_dir, fname))
            path = matches[0]
        else:
            path = os.path.join(patient_dir, fname)
        vol = np.load(path).astype(np.float32)
        vol = self._normalize(vol)
        vol = self._dose_crop(vol, tumor_side)
        if self.target_shape:
            vol = self._resize_volume(vol)
        return vol

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['UUID']

        fname_before = self.files_map['Active_Before_File']

        patient_dir = self.patient_path_map[patient_id]
        tumor_side = int(self.tumor_side_map.get(patient_id, 0))

        try:
            if self._cache is not None and idx in self._cache:
                vol = self._cache[idx].copy()
            else:
                vol_before = self._load_one(patient_dir, fname_before, tumor_side)
                vol = np.expand_dims(vol_before, axis=0)

                if self._cache is not None:
                    self._cache[idx] = vol.copy()

            if self.transform:
                vol = self.transform(vol)
            else:
                vol = torch.from_numpy(vol)

        except Exception as e:
            logging.error(f"Error loading {patient_id}: {e}")
            d_shape = self.target_shape if self.target_shape else (128, 128, 128)
            dummy = torch.zeros((1, *d_shape), dtype=torch.float32)
            return dummy, torch.tensor(0, dtype=torch.long)

        if 'Group' in row:
            val = 1 if row['Group'] == 'RP' else 0
            label = torch.tensor(val, dtype=torch.long)
        elif 'Label' in row:
            val = row['Label']
            label = torch.tensor(val, dtype=torch.long)
        else:
            val = random.randint(0, 1)
            label = torch.tensor(val, dtype=torch.long)

        return vol, label
