# PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization

[![arXiv](https://img.shields.io/badge/arXiv-2510.00910-b31b1b.svg)](https://arxiv.org/abs/2510.00910)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2510.00910-blue.svg)](https://doi.org/10.48550/arXiv.2510.00910)
[![License: Non-Commercial](https://img.shields.io/badge/license-Noncommercial-lightgrey.svg)](#license)

Code for:

**PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization**  
*Ali Shadman Yazdi, Annalisa Cappella, Benedetta Baldini, Riccardo Solazzo, Gianluca Tartaglia, Chiarella Sforza, Giuseppe Baselli*  
arXiv:2510.00910 — https://arxiv.org/abs/2510.00910

---

## Overview

This repository contains the training & evaluation pipeline for **PAL-Net**, a lightweight point-wise CNN with patch-attention for automatic localization of anatomical facial landmarks on **3D facial scans**.

- `run.py` — trains/evaluates on **LA-FAS**-style data using `LafasDataset`
- `run_facescape.py` — trains/evaluates on **FaceScape (neutral)** using `FaceScapeNeutralDataset`
- `src/` contains datasets, patch builders, models, losses, and utilities

> ✳️ You can run these scripts **as-is** by pointing them to your data paths, or plug in a **custom dataset dataloader** (see below).

---

## Requirements

- Python 3.9–3.11
- PyTorch (CUDA build recommended)
- NumPy, Pandas, scikit-learn, tqdm, Matplotlib

Install (example):

```bash
# (optional) create a virtual env
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# choose the right torch build for your CUDA/CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas scikit-learn tqdm matplotlib
```

## Data
LA-FAS (for run.py)
run.py expects an LA-FAS-style dataset via LafasDataset. Adjust paths in the script by setting root_dirs=[...] to where your data lives (e.g., "dataset", "validation_set").

``` bash
dataset/
└─ <subject_id>/
   ├─ mesh.(npz|ply|xyz|...)     # input point cloud / mesh representation
   └─ landmarks.(npz|txt|npy)    # Lx3 landmark coordinates
validation_set/
└─ ...
```

FaceScape (for run_facescape.py)
run_facescape.py uses FaceScapeNeutralDataset and needs:

one or more FaceScape root directories a landmark_indices.npz file
``` bash
/data/facial_landmarks/FACESCAPE/
├─ <subject_1>/
├─ <subject_2>/
└─ landmark_indices.npz
```
