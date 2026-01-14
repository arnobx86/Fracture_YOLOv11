# Advanced Bone Fracture Detection using YOLO v11

## 📌 Project Overview
This project presents an **advanced, end-to-end deep learning system** for **human bone fracture detection, classification, segmentation, and explainability** using **YOLO v11**.  
The system is designed for **medical imaging research and academic submission**, with a focus on:

- Accurate fracture localization
- Multi-class fracture classification
- Multi-modal learning (X-ray + MRI)
- Pixel-level fracture segmentation
- Model explainability using Grad-CAM

The implementation is **fully functional, reproducible, and extensible**, suitable for:
- BSc / MSc final year projects
- Thesis implementation
- Journal and conference research (Applied Intelligence, ESWA, Sensors, etc.)

---

## ✨ Key Features

### ✅ YOLO v11–based Detection
- Anchor-free, real-time object detection
- Optimized for small and subtle fracture patterns

### ✅ X-ray + MRI Multi-Modal Channel Fusion
- X-ray and MRI images stacked as multi-channel input
- Enables learning complementary anatomical information

### ✅ YOLO v11 Segmentation (YOLO-Seg)
- Pixel-level fracture localization
- Particularly effective for thin fracture lines

### ✅ Explainable AI (Grad-CAM)
- Heatmap visualization showing model attention
- Improves clinical interpretability and trust

### ✅ Research-Grade Dual-Backbone Scaffold
- Clearly documented extension point for true multi-modal feature fusion
- Suitable for novel architecture claims in academic papers

---

## 🗂️ Project Structure

```
Fracture_YOLOv11_Advanced/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── masks/
│   │   ├── train/
│   │   └── val/
│
├── src/
│   ├── train_channel_fusion.py
│   ├── train_segmentation.py
│   ├── models/
│   │   └── dual_backbone_stub.py
│   └── visualization/
│       └── gradcam_yolo.py
│
├── fracture.yaml
├── fracture-seg.yaml
├── requirements.txt
├── README.md
└── weights/
```

---

## 🧪 Dataset Preparation

### Image Format
- Supported formats: `.jpg`, `.png`
- Recommended resolution: **640 × 640**

### YOLO Label Format
```
<class_id> <x_center> <y_center> <width> <height>
```
(All values normalized between 0 and 1)

### Segmentation Masks
- Binary or multi-class masks
- Same filename as image
- Stored in `dataset/masks/`

---

## ⚙️ Installation

```bash
conda create -n fracture python=3.10
conda activate fracture
pip install -r requirements.txt
```

---

## 🚀 Training

### X-ray + MRI Channel Fusion
```bash
python src/train_channel_fusion.py
```

### YOLO v11 Segmentation
```bash
python src/train_segmentation.py
```

---

## 🔍 Explainability (Grad-CAM)

```bash
python src/visualization/gradcam_yolo.py
```

---

## 📊 Outputs
YOLO v11 automatically generates:
- Loss curves
- Precision–Recall curves
- Confusion matrix
- Best model weights

All saved inside the `runs/` directory.
## 📜 License
For academic and research purposes only.
