import numpy as np
import scipy.ndimage
import torch


class MedicalTransforms3D:
    def __init__(self, mode='standard', p_flip=0.5, p_rot=0.5, rot_range=20,
                 p_elastic=0.0, elastic_alpha=15, elastic_sigma=5,
                 p_gamma=0.0, gamma_range=(0.7, 1.5),
                 p_cutout=0.0, cutout_size=(32, 32, 32),
                 img_size=224):
        self.mode = mode
        self.img_size = img_size

        self.p_flip = p_flip
        self.p_rot = p_rot
        self.rot_range = rot_range

        if mode == 'heavy':
            self.p_elastic = 0.3
            self.p_gamma = 0.5
            self.p_cutout = 0.3
        else:
            self.p_elastic = p_elastic
            self.p_gamma = p_gamma
            self.p_cutout = p_cutout

        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.gamma_range = gamma_range
        self.cutout_size = cutout_size

    def _elastic_transform(self, volume):
        volume = volume.astype(np.float32)
        c, d, h, w = volume.shape

        rand_d = (np.random.rand(d, h, w).astype(np.float32) * 2 - 1)
        rand_h = (np.random.rand(d, h, w).astype(np.float32) * 2 - 1)
        rand_w = (np.random.rand(d, h, w).astype(np.float32) * 2 - 1)

        dz = scipy.ndimage.gaussian_filter(rand_d, self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dy = scipy.ndimage.gaussian_filter(rand_h, self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dx = scipy.ndimage.gaussian_filter(rand_w, self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha

        z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing='ij')

        indices = np.stack([z + dz, y + dy, x + dx], axis=0).reshape(3, -1)

        distorted = np.zeros_like(volume)
        for i in range(c):
            distorted[i] = scipy.ndimage.map_coordinates(
                volume[i], indices, order=1, mode='nearest'
            ).reshape(d, h, w)
        return distorted

    def _random_gamma(self, volume):
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])

        min_v, max_v = volume.min(), volume.max()
        if max_v - min_v < 1e-7: return volume

        vol_norm = (volume - min_v) / (max_v - min_v)
        vol_aug = np.power(vol_norm, gamma)
        return vol_aug * (max_v - min_v) + min_v

    def _coarse_dropout(self, volume):
        volume = volume.copy()
        c, d, h, w = volume.shape
        cd, ch, cw = self.cutout_size

        cd, ch, cw = min(cd, d), min(ch, h), min(cw, w)

        zd = np.random.randint(0, d - cd + 1)
        yd = np.random.randint(0, h - ch + 1)
        xd = np.random.randint(0, w - cw + 1)

        fill_val = volume.min()
        volume[:, zd:zd + cd, yd:yd + ch, xd:xd + cw] = fill_val
        return volume

    def _get_random_slice(self, volume):
        c, d, h, w = volume.shape

        if self.mode == 'heavy':
            start = int(d * 0.05)
            end = int(d * 0.95)
            slice_idx = np.random.randint(start, end) if end > start else d // 2
        else:
            slice_idx = d // 2

        return volume[:, slice_idx, :, :]

    def __call__(self, volume):
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        volume = volume.astype(np.float32)

        if self.mode == 'heavy':
            if np.random.rand() < self.p_elastic:
                try:
                    volume = self._elastic_transform(volume)
                except Exception as e:
                    print(f"Elastic transform failed: {e}")

            if np.random.rand() < self.p_cutout:
                volume = self._coarse_dropout(volume)

        if np.random.rand() < self.p_flip:
            axis = np.random.choice([2, 3])
            volume = np.flip(volume, axis=axis).copy()

        if np.random.rand() < self.p_rot:
            angle = np.random.uniform(-self.rot_range, self.rot_range)
            volume = scipy.ndimage.rotate(volume, angle, axes=(2, 3),
                                          reshape=False, order=1, mode='nearest')

        if self.mode == 'heavy':
            if np.random.rand() < self.p_gamma:
                volume = self._random_gamma(volume)

        return torch.from_numpy(volume).float()
