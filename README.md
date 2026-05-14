AI 681 - Assignments 2 & 3: Feature Extraction + Person Detection
**Effective Object Detection in Low-Light Security Environments Using ViTs**

Submitted by: Amal Al-Shboul & Roaa Shattnawi  
Supervised by: Dr. Nawaf Alsrehin  
Second Semester 2025/2026

AI-681/
├── app.py
├── yolov10_llvip_best.pt     ← الموديل
├── feature_extraction.py
├── templates/
│   └── index.html
├── visualization/
│   └── AI681_visualization.py
├── sample_images/             
│   ├── infrared/
│   └── visible/

└── README.md
## Step 1: Install Requirements
pip install torch torchvision transformers opencv-python numpy flask matplotlib ultralytics scikit-learn tqdm Pillow

---

## Step 2: Download Dataset
Download LLVIP dataset from:
https://huggingface.co/datasets/jsonhash/LLVIP

Extract so the structure looks like:
LLVIP/
├── visible/train/
└── infrared/train/

---

## Step 3: Run Feature Extraction
python feature_extraction.py

The script will auto-detect the dataset path.
Output files: infrared_features.npy, visible_features.npy, fused_features.npy

---

## Step 4: Run Web Demo
python app.py

Then open: http://127.0.0.1:5000

Upload a visible + infrared image pair to see:
- Feature vectors (768 per image, 1536 fused)
- Attention maps (where ViT is looking)
- Pixel-level fusion visualization
- Feature statistics and vector preview

---
## Assignment 3 - Model Training (Google Colab)
Training notebook for YOLOv10 on LLVIP dataset:
[Open in Google Colab](https://colab.research.google.com/drive/1zWUg0ub9YCfb078wxuaH3CJWZCYEh8wk?usp=sharing)

## Model Performance (Assignment 3)
| Metric       | Value  |
|--------------|--------|
| mAP@0.5      | 97.9%  |
| mAP@0.5:0.95 | 63.9%  |
| Precision    | 94.8%  |
| Recall       | 94.8%  |
| F1 Score     | 94.8%  |

Dataset: LLVIP | Model: YOLOv10n | Training: 30 epochs, 480 images
---

## Source Code
https://github.com/rshattnawi/AI-681-Feature-Selection---Extraction
