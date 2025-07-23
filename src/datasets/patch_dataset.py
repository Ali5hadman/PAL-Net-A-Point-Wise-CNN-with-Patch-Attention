import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.utils import get_patches, get_patches_distance_based

class PatchDataset(Dataset):
    def __init__(self, base_ds, mean_lm, patch_size, cache_dir):
        """
        base_ds    : your original Dataset (LafasDataset or Subset)
        mean_lm    : (50,3) numpy array of the TRAIN split mean landmarks
        patch_size : number of points per patch (e.g. 1000)
        cache_dir  : folder where we'll dump .npy caches
        """
        self.base_ds    = base_ds
        self.mean_lm    = mean_lm
        self.patch_size = patch_size
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir  = cache_dir

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # use idx for a unique cache filename
        cache_fp = os.path.join(self.cache_dir, f"{idx:06d}_patch.npy")

        # 1) try loading it
        if os.path.exists(cache_fp):
            patch = np.load(cache_fp)           # shape (patch_size,3)
        else:
            # 2) compute & save it
            #    base_ds[idx] → (points100k, landmarks50, raw_verts)
            _, _, raw_verts = self.base_ds[idx]
            patch = get_patches(
                np.array([ raw_verts.numpy() ]),
                np.array(self.mean_lm[np.newaxis, ...]),
                self.patch_size
            )[0]
            np.save(cache_fp, patch)

        # finally, load the target landmarks
        x_sampled, lm50, _ = self.base_ds[idx]

        return torch.from_numpy(patch), lm50, x_sampled




class PatchDataset_sampled(Dataset):
    def __init__(self, base_ds, mean_lm, patch_size, cache_dir):
        """
        base_ds    : your original Dataset (LafasDataset or Subset)
        mean_lm    : (50,3) numpy array of the TRAIN split mean landmarks
        patch_size : number of points per patch (e.g. 1000)
        cache_dir  : folder where we'll dump .npy caches
        """
        self.base_ds    = base_ds
        self.mean_lm    = mean_lm
        self.patch_size = patch_size
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir  = cache_dir

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # use idx for a unique cache filename
        cache_fp = os.path.join(self.cache_dir, f"{idx:06d}_patch.npy")

        # 1) try loading it
        if os.path.exists(cache_fp):
            patch = np.load(cache_fp)           # shape (patch_size,3)
        else:
            # 2) compute & save it
            #    base_ds[idx] → (points100k, landmarks50, raw_verts)
            sampled_verts, _, _ = self.base_ds[idx]
            patch = get_patches(
                np.array([ sampled_verts.numpy() ]),
                np.array(self.mean_lm[np.newaxis, ...]),
                self.patch_size
            )[0]
            np.save(cache_fp, patch)

        # finally, load the target landmarks
        x_sampled, lm50, _ = self.base_ds[idx]

        return torch.from_numpy(patch), lm50, x_sampled
    


class PatchDataset_distance_based(Dataset):
    def __init__(self, base_ds, mean_lm, patch_size, cache_dir):
        """
        base_ds    : your original Dataset (LafasDataset or Subset)
        mean_lm    : (50,3) numpy array of the TRAIN split mean landmarks
        patch_size : number of points per patch (e.g. 1000)
        cache_dir  : folder where we'll dump .npy caches
        """
        self.base_ds    = base_ds
        self.mean_lm    = mean_lm
        self.patch_size = patch_size
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir  = cache_dir

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # use idx for a unique cache filename
        cache_fp = os.path.join(self.cache_dir, f"{idx:06d}_patch.npy")

        # 1) try loading it
        if os.path.exists(cache_fp):
            patch = np.load(cache_fp)           # shape (patch_size,3)
        else:
            # 2) compute & save it
            #    base_ds[idx] → (points100k, landmarks50, raw_verts)
            _, _, raw_verts = self.base_ds[idx]
            patch = get_patches_distance_based(
                np.array([ raw_verts.numpy() ]),
                np.array(self.mean_lm[np.newaxis, ...]),
                self.patch_size
            )[0]
            np.save(cache_fp, patch)

        # finally, load the target landmarks
        x_sampled, lm50, _ = self.base_ds[idx]

        return torch.from_numpy(patch), lm50, x_sampled


