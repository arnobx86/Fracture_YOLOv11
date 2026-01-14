# Advanced YOLOv11 Bone Fracture Detection (SUBMISSION READY)

## Included Features
1. X-ray + MRI Channel Fusion
2. Grad-CAM Explainability
3. Dual-Backbone Research Scaffold
4. YOLOv11 Segmentation Support

## How to Run
pip install -r requirements.txt

### Channel Fusion Training
python src/train_channel_fusion.py

### Segmentation Training
python src/train_segmentation.py

### Grad-CAM Visualization
python src/visualization/gradcam_yolo.py

## Notes
- Dataset must follow YOLO format
- MRI images are stacked as channels with X-rays
- Segmentation masks go in dataset/masks/

This package is suitable for:
- MSc / BSc final project
- Journal paper implementation
- Research extension
