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

This repository provides the full pipeline for training and evaluating **PAL-Net**, a point-wise CNN with patch-attention for localizing anatomical landmarks on **3D facial scans**.

- `run.py`: main script for training/evaluation on LA-FAS-style datasets
- `run_facescape.py`: adapted version for FaceScape (neutral-only)
- `src/`: dataset loaders, patch extraction, models, loss functions, utils

You can plug in a **custom dataset loader** (e.g. your own `.npz` format) and modify `run.py` to use it.

---

## 📦 Installation

**Requirements:**
- Python 3.9–3.11
- PyTorch (CUDA recommended)
- NumPy, Pandas, scikit-learn, tqdm, Matplotlib

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate (Windows)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm matplotlib trimesh
```

---

## 📂 Dataset Format

The model expects **point cloud meshes + 3D landmark coordinates** in the LA-FAS format:

```
dataset/
└── S001/
    ├── mesh.obj
    ├── transformation_matrix.npy   # 4x4 matrix (optional)
    └── landmarks.txt                # 50 rows of 3D landmarks
```

- Mesh is loaded with `trimesh`, and transformed using the `transformation_matrix.npy`
- Landmarks are read from `.txt`, parsed and transformed to match the mesh
- 100,000 surface points are sampled for training

You can also use the `FaceScapeNeutralDataset` loader (see `run_facescape.py`).

---

## 🔁 Custom Datasets

You can create your own dataset class as long as it returns a tuple of:
```python
(points100k: torch.Tensor, landmarks50: torch.Tensor, optional_raw_vertices: torch.Tensor or None)
```

To use your own dataset, modify `run.py`:
```python
from src.datasets.custom_dataset import MyCustomDataset
...
dataset = MyCustomDataset("/path/to/my/data")
```

---

## 🚀 Running the Code

### Preprocess & Train

```bash
python run.py
```

Edit `run.py` to set:
- Dataset paths
- Patch size (default = 1000)
- Caching directory
- Network variant (`PALNET`, `PLNET_noatt`, etc.)

Training uses:
- `CombinedLoss` (localization + distance)
- Patch-based batching via `PatchDataset`
- Early stopping and model checkpointing

### Evaluate

The evaluation section (bottom of `run.py`) includes:
- Landmark prediction
- Point-wise and distance-based errors
- Result CSV saving

---

## 📊 Reproducibility

- Set seeds via `set_seed()`
- All patches are cached deterministically
- Dataloader shuffling can be disabled for testing

---

## 🧪 Troubleshooting

- ⚠️ If you get CUDA OOM errors, reduce `batch_size` or `patch_size`
- ✅ If you get very high errors, check if your landmarks are misaligned with your mesh
- 📌 For custom data: ensure landmark count and proper scaling

---

## 📄 License

This repository is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International License**.

- ✅ Free to use, modify, and build upon for **non-commercial** purposes
- ❌ Not for commercial use or redistribution without explicit permission

For full license text: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 📚 Citation

If you use this code or model, please cite:



---

For questions or collaborations: contact [Ali Shadman]([https://www.linkedin.com/in/alishadman/](https://www.linkedin.com/in/ali-shadman-006a871b1/)) or open an issue.

---

