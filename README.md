# PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization

[![arXiv](https://img.shields.io/badge/arXiv-2510.00910-b31b1b.svg)](https://arxiv.org/abs/2510.00910)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2510.00910-blue.svg)](https://doi.org/10.48550/arXiv.2510.00910)
[![License: Non-Commercial](https://img.shields.io/badge/license-Noncommercial-lightgrey.svg)](#license)

Code accompanying the paper:

**PAL-Net: A Point-Wise CNN with Patch-Attention for 3D Facial Landmark Localization**  
*Ali Shadman Yazdi, Annalisa Cappella, Benedetta Baldini, Riccardo Solazzo, Gianluca Tartaglia, Chiarella Sforza, Giuseppe Baselli*  
arXiv:2510.00910 — https://arxiv.org/abs/2510.00910

---

## Overview

This repository provides the training & evaluation pipeline for **PAL-Net**, a lightweight point-wise CNN with patch-attention for automatic localization of anatomical facial landmarks on 3D facial scans.

The main entry point is **`run_facescape.py`**, which:
- builds the **FaceScape (neutral)** dataset
- generates patches
- trains with early stopping & LR scheduling
- evaluates and saves metrics/checkpoints

> ✳️ The instructions below let you run the existing code **as-is** (no source edits required).

---

## Requirements

- Python 3.9–3.11  
- PyTorch (CUDA build recommended)  
- NumPy, Pandas, scikit-learn, tqdm, Matplotlib

Install (example):

```bash
# optional: create a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# choose the right torch build for your CUDA/CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas scikit-learn tqdm matplotlib


