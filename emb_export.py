from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)


def pooled_features(model, dataset, device, batch_size: int = 4,
                    loader_kwargs: dict | None = None) -> np.ndarray:
    kw = dict(loader_kwargs or {})
    kw.pop("batch_size", None)
    loader = DataLoader(Subset(dataset, list(range(len(dataset)))), shuffle=False,
                        batch_size=batch_size, **kw)
    feats = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            vol = batch[0] if isinstance(batch, (tuple, list)) else batch
            feats.append(model.pooled_feature(vol.to(device)).detach().cpu().float().numpy())
    return np.concatenate(feats, 0)


def export_fold_embeddings(model, dataset_clean, manifest_df, device, outer_k: int,
                           inner_i: int, results_dir: str, folds, dev_indices,
                           test_indices, batch_size: int = 4,
                           loader_kwargs: dict | None = None) -> str:
    feats = pooled_features(model, dataset_clean, device, batch_size, loader_kwargs)
    uuids = [str(u).strip() for u in manifest_df["UUID"].values]
    if len(uuids) != len(feats):
        raise RuntimeError(f"[export] {len(feats)} features for {len(uuids)} manifest rows - "
                           "the dataset and the manifest are not in the same order")

    tr_rel, val_rel = folds[inner_i]
    tr_uuids = [uuids[dev_indices[i]] for i in tr_rel]
    val_uuids = [uuids[dev_indices[i]] for i in val_rel]
    tst_uuids = [uuids[i] for i in (test_indices or [])]
    leak = (set(tr_uuids) & set(val_uuids)) | (set(tr_uuids) & set(tst_uuids))
    if leak:
        raise RuntimeError(f"[export] fold(k={outer_k},i={inner_i}) train set overlaps "
                           f"val/test - leakage: {sorted(leak)[:5]}")

    os.makedirs(results_dir, exist_ok=True)
    tag = f"fold{outer_k}_fold{inner_i}"
    np.savez_compressed(os.path.join(results_dir, f"imaging_emb_{tag}.npz"),
                        uuids=np.array(uuids), feats=feats)
    with open(os.path.join(results_dir, f"fold_partition_{tag}.json"), "w") as fp:
        json.dump({"outer": int(outer_k), "inner": int(inner_i),
                   "feat_dim": int(feats.shape[1]), "n_patients": int(feats.shape[0]),
                   "train": tr_uuids, "val": val_uuids, "test": tst_uuids}, fp, indent=2)
    log.info("    [export] fold(k=%d,i=%d) -> imaging_emb_%s.npz  (n=%d, fdim=%d; "
             "train=%d val=%d test=%d; per-feature std=%.4g)", outer_k, inner_i, tag,
             feats.shape[0], feats.shape[1], len(tr_uuids), len(val_uuids), len(tst_uuids),
             float(np.mean(np.std(feats, axis=0))))
    return tag
