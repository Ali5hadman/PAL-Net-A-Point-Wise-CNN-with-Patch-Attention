# PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization

[![arXiv](https://img.shields.io/badge/arXiv-2510.00910-b31b1b.svg)](https://arxiv.org/abs/2510.00910)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2510.00910-blue.svg)](https://doi.org/10.48550/arXiv.2510.00910)
[![License: Non-Commercial](https://img.shields.io/badge/license-Noncommercial-lightgrey.svg)](#license)

Code for the paper:

**PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization**  
*Ali Shadman Yazdi, Annalisa Cappella, Benedetta Baldini, Riccardo Solazzo, Gianluca Tartaglia, Chiarella Sforza, Giuseppe Baselli*  
arXiv:2510.00910 — https://arxiv.org/abs/2510.00910

---

## ✨ Overview

This repository provides the training & evaluation pipeline for **PAL-Net**, a lightweight point-wise CNN with patch-attention for automatic localization of anatomical facial landmarks on **3D facial scans**.

- `run.py` — pipeline for **LA-FAS–style** data using `LafasDataset`
- `run_facescape.py` — pipeline for **FaceScape (neutral)** using `FaceScapeNeutralDataset`
- `src/` — datasets, patch builders, models (PAL-Net + ablations), losses, utilities

You can run the scripts **as-is** by pointing them to your data, or plug in a **custom dataset dataloader** and give its path to `run.py`.

---

## 📦 Installation

**Requirements**

- Python 3.9–3.11
- PyTorch (CUDA recommended)
- NumPy, Pandas, scikit-learn, tqdm, Matplotlib

**Setup (example)**

```bash
# (optional) create a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# install a torch build that matches your CUDA (or cpu wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas scikit-learn tqdm matplotlib
```

---

## 📂 Datasets

### LA-FAS (for `run.py`)

`run.py` expects an LA-FAS–style dataset via `LafasDataset`. Edit the **paths** inside `run.py` by setting `root_dirs=[...]` (e.g., `"dataset"`, `"validation_set"`), or mirror the same folder names on your machine.

**Example layout**

```
dataset/
└─ <subject_id>/
   ├─ mesh.(npz|ply|xyz|...)      # input points / mesh vertices
   └─ landmarks.(npz|txt|npy)     # L x 3 landmark coordinates
validation_set/
└─ ...
```

During the run, the script may:
- filter outliers with `ThresholdSampler`
- compute a mean landmark template on the train split
- create patch caches (e.g., `patch_cache_train/`, `patch_cache_test/`)

### FaceScape (neutral subset, for `run_facescape.py`)

`run_facescape.py` uses `FaceScapeNeutralDataset` and requires:
- one or more FaceScape root directories
- `landmark_indices.npz` (landmark indexing file)

**Example layout**

```
/data/facial_landmarks/FACESCAPE/
├─ <subject_1>/
├─ <subject_2>/
└─ landmark_indices.npz
```

You can either:
- edit those paths once at the top of `run_facescape.py`, or
- mirror the paths on your system via symlink/mount so no code changes are needed.

---

## ▶️ How to Run

### 1) LA-FAS pipeline (`run.py`)

```bash
python run.py
```

**Outputs**

- `best_model_ref.pth` — best checkpoint  
- `results_.csv` — appended metrics per run  
- NPY error arrays:
  - `point_wise_closes.npy`, `point_wise_average.npy`
  - `distance_wise_closes.npy`, `distance_wise_average.npy`
- temporary caches: `patch_cache_train/`, `patch_cache_test/` (deleted at the end)

### 2) FaceScape (neutral) pipeline (`run_facescape.py`)

```bash
python run_facescape.py
```

**Outputs** mirror those of `run.py`, with FaceScape-specific cache folders (e.g., `facescape_cache(processed)_npz/`).

---

## 🧩 Custom Dataset Dataloader

You can implement your **own** dataset and **use it in `run.py`** (e.g., place your class in `src/datasets/custom_dataset.py` and import it inside `run.py`). The training code expects your dataset to return:

```python
(points:   torch.Tensor[N, 3],  # input point cloud / vertices
 landmarks: torch.Tensor[L, 3],  # target landmark coordinates
 extras:    Any)                 # optional (e.g., ids, masks)
```

**Minimal example** (`src/datasets/custom_dataset.py`)

```python
import os, numpy as np, torch
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, root_dirs, cache_dir="custom_cache_npz"):
        self.root_dirs = root_dirs if isinstance(root_dirs, (list, tuple)) else [root_dirs]
        self.samples = self._index_files(self.root_dirs)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def _index_files(self, roots):
        pairs = []
        for r in roots:
            # discover and return list of (points_path, landmarks_path)
            pass
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pts_path, lmk_path = self.samples[idx]
        points = torch.from_numpy(np.load(pts_path)["points"]).float()
        landmarks = torch.from_numpy(np.load(lmk_path)["landmarks"]).float()
        return points, landmarks, None
```

---

## 🔁 Reproducibility

Both pipelines enforce deterministic settings:

```python
set_seed(12345)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 🧰 Troubleshooting

- **CUDA not found / mismatched build** → install a PyTorch wheel that matches your CUDA driver, or use CPU wheels.  
- **Out of Memory** → lower `batch_size` or `patch_size` in the scripts.  
- **File not found** → verify `root_dirs` and any `landmark_indices.npz`. Ensure symlinks/mounts are visible to Python.  
- **Slow first epoch** → patch caching can take time on the very first run; subsequent runs are faster.

---

## 📚 Citation

If you use this repository, please cite:

> Ali Shadman Yazdi, Annalisa Cappella, Benedetta Baldini, Riccardo Solazzo, Gianluca Tartaglia, Chiarella Sforza, Giuseppe Baselli.  
> “PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization.” arXiv:2510.00910, 2025. https://arxiv.org/abs/2510.00910

```bibtex
@article{shadman2025palnet,
  title   = {PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization},
  author  = {Shadman Yazdi, Ali and Cappella, Annalisa and Baldini, Benedetta and Solazzo, Riccardo and Tartaglia, Gianluca and Sforza, Chiarella and Baselli, Giuseppe},
  journal = {arXiv preprint arXiv:2510.00910},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.00910},
  url     = {https://arxiv.org/abs/2510.00910}
}
```

---

## 📜 License

This repository is released for **non-commercial** use.

- **Code**: **PolyForm Noncommercial 1.0.0**  
  You may use, modify, and share the code for **non-commercial** purposes.  
  **Commercial use requires prior permission** from the authors.

- **Non-code assets** (docs, figures): **CC BY-NC 4.0**  
  You may share and adapt with attribution for **non-commercial** purposes.

**Apply once (repo root):**

1. Create `LICENSE` and paste the full text of **PolyForm Noncommercial 1.0.0**.  
2. Create `LICENSE-NONCODE` with:

```
Docs & figures © 2025 Ali Shadman Yazdi et al. Licensed under CC BY-NC 4.0.  
Full text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
```

3. (Optional) Add a contact for commercial licensing:

```
Commercial licensing: your.email@domain.tld
```

---

## 🙏 Acknowledgements

- FaceScape dataset authors and maintainers  
- PyTorch and the open-source community
