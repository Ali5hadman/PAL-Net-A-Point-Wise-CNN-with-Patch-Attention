import os
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import trimesh
# import openmesh


def apply_transformation(pc, T):
    ones = np.ones((pc.shape[0], 1))
    pc_h = np.hstack([pc, ones])
    return (T @ pc_h.T).T[:, :3] / (T @ pc_h.T).T[:, 3][:, None]

def get_obj(dir_path, file_name):
    num_points = 100000
    mesh = trimesh.load(os.path.join(dir_path, file_name), force='mesh')
    transMatrix = np.load(os.path.join(dir_path, "transformation_matrix.npy"))
    mesh.apply_transform(transMatrix)
    vertices_raw = mesh.vertices                   # (Nᵢ, 3) variable Nᵢ
    points100k = trimesh.sample.sample_surface_even(mesh, num_points)[0]
    return vertices_raw.astype(np.float32), points100k.astype(np.float32)

def get_obj_lite(dir_path, file_name):
    num_points = 10000
    mesh = trimesh.load(os.path.join(dir_path, file_name), force='mesh')
    transMatrix = np.load(os.path.join(dir_path, "transformation_matrix.npy"))
    mesh.apply_transform(transMatrix)
    vertices_raw = mesh.vertices                   # (Nᵢ, 3) variable Nᵢ
    points100k = trimesh.sample.sample_surface_even(mesh, num_points)[0]
    return vertices_raw.astype(np.float32), points100k.astype(np.float32)

def get_landmarks(dir_path, file_name):
    coords = []
    with open(os.path.join(dir_path, file_name), 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('landmark names'):
                c = line.strip().split('\t')[1].strip('()').split(',')
                coords.append([float(x) for x in c])
    if len(coords) != 50:
        raise ValueError(f"[BROKEN FILE] {dir_path}/{file_name} has {len(coords)} landmarks")
    
    transMatrix = np.load(os.path.join(dir_path, "transformation_matrix.npy"))
    coords = apply_transformation(np.array(coords), np.array(transMatrix))
    return np.array(coords, dtype=np.float32)


class ThresholdSampler(Sampler):
    def __init__(self, data_source, threshold, ref_idx=0):
        self.data_source = data_source
        self.threshold   = threshold
        # load reference landmarks once
        _, self.ref_lm, _ = data_source[ref_idx]
        # build the list of “good” indices
        self.valid_indices = []
        for idx in range(len(data_source)):
            _, lm, _ = data_source[idx]
            if torch.norm(lm - self.ref_lm) < threshold:
                self.valid_indices.append(idx)

    def __iter__(self):
        # you can shuffle here if you like
        return iter(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)
    

class LafasDataset(Dataset):
    def __init__(self, root_dirs, cache_dir=None, transform=None):
        self.samples = []
        for root in root_dirs:
            for dp, _, files in os.walk(root):
                if any(f.endswith('.obj') for f in files) and any(f.endswith('.txt') for f in files):
                    obj = next(f for f in files if f.endswith('.obj'))
                    txt = next(f for f in files if f.endswith('.txt'))
                    self.samples.append((dp, obj, txt))

        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dirpath, obj_file, txt_file = self.samples[idx]

        # build a unique cache filename per sample
        if self.cache_dir:
            safe = dirpath.strip(os.sep).replace(os.sep, '_')
            cache_path = os.path.join(self.cache_dir, f"{safe}.npz")
        else:
            cache_path = None

        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            vertices_raw = data['vertices']      # shape (Nᵢ,3)
            points100k   = data['points']        # shape (100000,3)
            landmarks50  = data['landmarks']     # shape (50,3)
        else:
            vertices_raw, points100k = get_obj(dirpath, obj_file)
            landmarks50             = get_landmarks(dirpath, txt_file)
            if cache_path:
                np.savez(cache_path,
                         vertices=vertices_raw,
                         points=points100k,
                         landmarks=landmarks50)

        # to torch
        vertices_raw = torch.from_numpy(vertices_raw)   # (Nᵢ,3), variable per sample
        points100k   = torch.from_numpy(points100k)     # (100000,3)
        landmarks50  = torch.from_numpy(landmarks50)    # (50,3)

        if self.transform:
            points100k, landmarks50 = self.transform(points100k, landmarks50)

        # return 3-tuple
        return points100k, landmarks50, vertices_raw


def mesh_collate_fn(batch):
    """
    batch is a list of tuples (points100k, landmarks50, vertices_raw)
    We stack the fixed-size tensors and keep the variable-length list as-is.
    """
    points_batch    = torch.stack([b[0] for b in batch], dim=0)  # (B,100000,3)
    landmarks_batch = torch.stack([b[1] for b in batch], dim=0)  # (B,    50,3)
    vertices_list   = [b[2] for b in batch]                     # list of (Ni,3) tensors
    return points_batch, landmarks_batch, vertices_list





class FaceScapeNeutralDataset(Dataset):
    def __init__(self,
                 root_dirs,
                 landmark_indices_path,
                 cache_dir=None,
                 transform=None):
        """
        Args:
            root_dirs (list of str): paths to your FACESPACE folder(s)
            landmark_indices_path (str): path to npz containing 'v10' indices
            cache_dir (str, optional): where to write cached .npz per mesh
            transform (callable, optional): fn(points100k, landmarks50) -> (...)
        """
        # load the “v10” landmark index array
        data = np.load(landmark_indices_path)
        self.landmark_indices = data['v10']

        # collect only the neutral‐pose .obj files
        self.samples = []
        for root in root_dirs:
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith('_neutral.obj'):
                        self.samples.append((dirpath, f))

        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dirpath, obj_file = self.samples[idx]

        # build a unique cache filename
        if self.cache_dir:
            safe = dirpath.strip(os.sep).replace(os.sep, '_')
            cache_path = os.path.join(self.cache_dir, f"{safe}.npz")
        else:
            cache_path = None

        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            vertices_raw = data['vertices']       # (Ni,3)
            points100k   = data['points100k']     # (100000,3)
            landmarks50  = data['landmarks']      # (50,3)
        else:
            # load mesh via trimesh
            print(f"Loading {obj_file} from {dirpath}")
            mesh = trimesh.load(os.path.join(dirpath, obj_file), force='mesh')
            vertices_raw = mesh.vertices.astype(np.float32)
            points100k, _ = trimesh.sample.sample_surface_even(mesh, 10000)

            # load same mesh into OpenMesh to grab the v10 landmarks
            om = openmesh.read_trimesh(os.path.join(dirpath, obj_file))
            landmarks50 = om.points()[self.landmark_indices].astype(np.float32)

            if cache_path:
                np.savez(cache_path,
                         vertices=vertices_raw,
                         points100k=points100k,
                         landmarks=landmarks50)

        # convert to torch
        vertices_raw = torch.from_numpy(vertices_raw)
        points100k   = torch.from_numpy(points100k)
        landmarks50  = torch.from_numpy(landmarks50)

        if self.transform:
            points100k, landmarks50 = self.transform(points100k, landmarks50)

        return points100k, landmarks50, vertices_raw




class LafasDataset_lite(Dataset):
    def __init__(self, root_dirs, cache_dir=None, transform=None):
        self.samples = []
        for root in root_dirs:
            for dp, _, files in os.walk(root):
                if any(f.endswith('.obj') for f in files) and any(f.endswith('.txt') for f in files):
                    obj = next(f for f in files if f.endswith('.obj'))
                    txt = next(f for f in files if f.endswith('.txt'))
                    self.samples.append((dp, obj, txt))

        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dirpath, obj_file, txt_file = self.samples[idx]

        # build a unique cache filename per sample
        if self.cache_dir:
            safe = dirpath.strip(os.sep).replace(os.sep, '_')
            cache_path = os.path.join(self.cache_dir, f"{safe}.npz")
        else:
            cache_path = None

        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            vertices_raw = data['vertices']      # shape (Nᵢ,3)
            points100k   = data['points']        # shape (100000,3)
            landmarks50  = data['landmarks']     # shape (50,3)
        else:
            vertices_raw, points100k = get_obj_lite(dirpath, obj_file)
            landmarks50             = get_landmarks(dirpath, txt_file)
            if cache_path:
                np.savez(cache_path,
                         vertices=vertices_raw,
                         points=points100k,
                         landmarks=landmarks50)

        # to torch
        vertices_raw = torch.from_numpy(vertices_raw)   # (Nᵢ,3), variable per sample
        points100k   = torch.from_numpy(points100k)     # (100000,3)
        landmarks50  = torch.from_numpy(landmarks50)    # (50,3)

        if self.transform:
            points100k, landmarks50 = self.transform(points100k, landmarks50)

        # return 3-tuple
        return points100k, landmarks50, vertices_raw

