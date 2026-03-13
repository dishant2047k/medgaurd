# MedGuard AI — Datasets Guide

## 📦 Recommended Public Datasets

### 1. Fall Detection
| Dataset | Description | Link |
|---------|-------------|------|
| **UR Fall Detection** | RGB-D fall videos, 70 sequences | http://fenix.ur.edu.pl/mkepski/ds/uf.html |
| **Le2i Fall Detection** | 4 environments, 191 videos | https://imvia.u-bourgogne.fr/en/database/fall-detection-database-2.html |
| **Multiple Cameras Fall** | 8 cameras, 24 fall scenarios | http://www.iro.umontreal.ca/~labimage/Dataset/ |
| **FallAllD** | 26 subjects, IMU + video | https://ieee-dataport.org/open-access/fallalld |
| **OCCU** | Older adult falls, occlusion | https://github.com/verlab/OCCU-dataset |

### 2. Seizure / Epilepsy
| Dataset | Description | Link |
|---------|-------------|------|
| **CHB-MIT EEG** | EEG seizure (can pair with video) | https://physionet.org/content/chbmit/1.0.0/ |
| **SienaScalp EEG** | 14 patients, seizure annotations | https://physionet.org/content/siena-scalp-eeg/1.0.0/ |
| **SeizIt** | Wearable seizure detection | https://www.seizeit.nl/ |
| **Custom video** | See synthetic data section below | — |

### 3. Action Recognition (base for anomaly)
| Dataset | Description | Link |
|---------|-------------|------|
| **NTU RGB+D 120** | 120 action classes, 114k videos | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| **Kinetics-700** | 700 action classes, YouTube | https://github.com/google-deepmind/kinetics-dataset |
| **UCF-101** | 101 actions, 13k clips | https://www.crcv.ucf.edu/data/UCF101.php |
| **HMDB51** | 51 actions, 7k clips | https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ |
| **Penn Action** | 2326 sequences, 15 actions | http://dreamdragon.github.io/PennAction/ |

### 4. Facial Expression / Distress
| Dataset | Description | Link |
|---------|-------------|------|
| **AffectNet** | 450k images, 8 expressions | http://mohammadmahoor.com/affectnet/ |
| **RAF-DB** | 30k images, real-world expressions | http://www.whdeng.cn/RAF/model1.html |
| **FER2013** | 35k images, 7 expressions | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Pain Intensity** | UNBC-McMaster Pain Dataset | https://www.paine.unbc.ca/ |

### 5. Medical Behaviour (Specialized)
| Dataset | Description | Link |
|---------|-------------|------|
| **MPOSE2021** | Pose-based action recognition | https://github.com/PIC4SeR/MPOSE2021_Dataset |
| **ShanghaiTech** | Anomaly detection in surveillance | https://svip-lab.github.io/dataset/campus_dataset.html |
| **CUHK Avenue** | Abnormal event detection | http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html |
| **UCSD Anomaly** | Pedestrian abnormal behaviour | http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm |

---

## 🤖 Synthetic Data Generation

When real data is scarce, generate synthetic training data:

### Option 1: Mocap + Blender
```bash
# Install Blender (free): https://blender.org
# Use MoCapAct dataset for motion capture data
# Render synthetic seizure/fall animations
pip install bpy  # Blender Python API
```

### Option 2: Pose Augmentation
```python
# Augment real skeleton sequences
import numpy as np

def augment_sequence(seq):
    """Apply random augmentations to a pose sequence."""
    # Horizontal flip
    if np.random.random() > 0.5:
        seq[:, 0::3] = 1.0 - seq[:, 0::3]  # flip x coords

    # Add Gaussian noise
    seq += np.random.normal(0, 0.01, seq.shape)

    # Random temporal crop/stretch
    factor = np.random.uniform(0.8, 1.2)
    new_len = int(len(seq) * factor)
    indices = np.linspace(0, len(seq)-1, new_len).astype(int)
    seq = seq[indices]

    return seq
```

### Option 3: GAN-based Synthesis
```bash
# MotionDiffuse: text-driven motion generation
# https://github.com/mingyuan-zhang/MotionDiffuse
pip install motion-diffuse
```

---

## 📥 Automated Download Script

```bash
# Run from project root
python datasets/download_datasets.py --datasets ucf101 fer2013 shanghaitech
```

---

## 🏷️ Annotation Tools (for custom data)

| Tool | Best For | Link |
|------|----------|------|
| **CVAT** | Video frame annotation | https://cvat.ai |
| **LabelStudio** | Multi-modal annotation | https://labelstud.io |
| **VOTT** | Video object tagging | https://github.com/microsoft/VoTT |
| **Labelbox** | Team annotation workflow | https://labelbox.com |

---

## 📁 Expected Data Format

After preprocessing, data should be organised as:

```
datasets/processed/
├── normal/
│   ├── seq_001.npy    # shape: (30, 51) — 30 frames × 17 keypoints × 3 (x,y,conf)
│   ├── seq_002.npy
│   └── ...
├── fall/
│   ├── seq_001.npy
│   └── ...
├── seizure/
├── cardiac/
├── unconscious/
└── facial_distress/
```

Run preprocessing:
```bash
python datasets/prepare_data.py \
    --input_dir ./datasets/raw \
    --output_dir ./datasets/processed \
    --sequence_len 30 \
    --fps 15
```
