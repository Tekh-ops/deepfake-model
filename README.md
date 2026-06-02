# Unysis Deepfake Detection System

## Overview
This repository contains the codebase for a deepfake detection system, including:
- Backend (API + inference logic)
- Frontend (UI)
- Machine Learning pipeline (data processing + training scripts)

Due to repository size constraints, large datasets and model artifacts are **intentionally excluded from version control**. This document provides a **complete and transparent description** of those ignored components.

---

## Ignored Files & Directories

The following directories are excluded via `.gitignore` to comply with GitHub size limits and to keep the repository lightweight and code-focused.

---

### 1. `dataset_raw/` (≈ 41.59 GB, 13,904 files)

Raw video datasets used for training and evaluation.

#### Structure:
- **Celeb-DF**
  - Celeb-real: 158 videos
  - Celeb-synthesis: 795 videos
  - YouTube-real: 250 videos

- **Celeb-DF-v2**
  - Celeb-real: 590 videos
  - Celeb-synthesis: 5,639 videos
  - YouTube-real: 300 videos

- **DFDC Dataset**
  - dfdc_train_part_0: 1,335 videos
  - dfdc_train_part_1: 1,700 videos
  - dfdc_train_part_49: 3,135 videos

#### File Types:
- `.mp4`: 13,899
- `.json`: 3
- `.txt`: 2

#### Reason for Exclusion:
- Extremely large size (40+ GB)
- Public datasets that can be re-downloaded
- Not suitable for version control

---

### 2. `base_deepfake/` (≈ 13.35 GB, 4,015 files)

Pre-processed and structured dataset used for training/validation/testing.

#### Structure:
- **train**
  - real: 1,404 videos
  - fake: 1,404 videos

- **test**
  - real: 302 videos
  - fake: 302 videos

- **val**
  - real: 301 videos
  - fake: 301 videos

#### File Types:
- `.mp4`: 4,014
- `.json`: 1

#### Reason for Exclusion:
- Large storage footprint
- Derived from raw datasets (reproducible)
- Not required to understand or review code

---

### 3. `ml_pipeline/processed_data/` (≈ 8.81 GB, 153,540 files)

Fully processed dataset used for model training (frame/face extraction).

#### Structure:

##### faces
- train: real (21,060), fake (21,060)
- test: real (4,530), fake (4,530)

##### frames
- train:
  - real: 21,060 (~2741 MB)
  - fake: 21,060 (~3996 MB)
- test:
  - real: 4,530 (~562 MB)
  - fake: 4,530 (~835 MB)

##### final
- train: real (21,060), fake (21,060)
- test: real (4,530), fake (4,530)

#### File Types:
- `.jpg`: 153,540

#### Reason for Exclusion:
- High file count (150K+ files)
- Generated data (reproducible via pipeline)
- Not suitable for Git tracking

---

### 4. `ml_pipeline/models/` (Model Weights)

Contains trained model weights and checkpoints.

#### Structure:
- `frame_model.pth` (~15.58 MB)
- `model.pth` (~15.56 MB)

- **checkpoints/**
  - `best_model.pth` (~15.58 MB)
  - `checkpoint_epoch_1.pth` (~46.34 MB)

#### Total Size:
≈ 90+ MB

#### Reason for Exclusion:
- Binary files not suitable for version control
- Frequently updated during training
- Can be regenerated via training pipeline

---

## Summary of Exclusions

| Category | Size | Reason |
|--------|------|--------|
| Raw datasets | ~41.6 GB | Public + too large |
| Base dataset | ~13.3 GB | Derived + large |
| Processed data | ~8.8 GB | Generated + high file count |
| Model weights | ~90 MB | Binary + reproducible |

---

## Justification

All excluded files fall into one or more of the following categories:

- **Large-scale datasets** (not suitable for Git)
- **Derived/generated data** (can be recreated)
- **Binary artifacts** (model weights, checkpoints)
- **High file count directories** (performance issues in Git)

The repository therefore contains only:
- Source code
- Configuration
- Scripts required to regenerate all artifacts

This ensures:
- Lightweight repository
- Faster cloning
- Clean version control practices
- Reproducibility of results