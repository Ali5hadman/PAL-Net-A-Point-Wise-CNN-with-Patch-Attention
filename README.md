# PAL‑Net: A Point‑Wise CNN with Patch Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Build Status](https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention/actions)

PAL‑Net is a deep learning framework for **automatic localization of 50 medically significant facial landmarks** on 3D stereophotogrammetric scans. By combining a point‑wise convolutional backbone with a lightweight patch‑attention module, PAL‑Net achieves state‑of‑the‑art accuracy while remaining fast and easy to train.

---

## 📖 Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Dataset Preparation](#dataset-preparation)  
4. [Quick Start](#quick-start)  
   - [Inference Demo](#inference-demo)  
   - [Training from Scratch](#training-from-scratch)  
   - [k‑Fold Cross‑Validation](#k-fold-cross-validation)  
5. [Visualization](#visualization)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [Citation](#citation)  
9. [License](#license)  
10. [Contact](#contact)  

---

## ✨ Features

- **Point‑Wise Convolutions**: Efficiently process 3D facial meshes without voxelization.  
- **Patch‑Attention Module**: Learns to focus on medically relevant regions for improved landmark accuracy.  
- **Modular Design**: Plug‑and‑play backbones, custom loss functions, and easy dataset integration.  
- **Visualization Tools**: Built‑in scripts for plotting predicted vs. ground‑truth landmarks.  
- **Reproducible Benchmarks**: k‑Fold cross‑validation pipeline with automatic metric logging.

---

## 🚀 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention.git
   cd PAL-Net-A-Point-Wise-CNN-with-Patch-Attention
