from flask import Flask, request, jsonify, render_template
import torch
from transformers import ViTModel, ViTImageProcessor, ViTConfig
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ===== Load ViT Model (Assignment 2) =====
print("Loading ViT model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
config    = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config.output_attentions = True
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config)
vit_model.eval()
print("ViT Model ready ✓")

# ===== Load YOLOv10 Model (Assignment 3) =====
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov10_llvip_best.pt")
yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    print("Loading YOLOv10 model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLOv10 Model ready ✓")
else:
    print(f"⚠️  YOLOv10 model not found at {YOLO_MODEL_PATH}")

# ===== Feature Extraction (Assignment 2) =====
def extract_features(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    inputs  = processor(images=img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return features

# ===== Attention Map (Assignment 2) =====
def generate_feature_map(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs  = processor(images=img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)

    if outputs.attentions is None or len(outputs.attentions) == 0:
        return None

    try:
        last_attn = outputs.attentions[-1]
        avg_attn  = last_attn[0].mean(dim=0)
        cls_attn  = avg_attn[0, 1:].numpy()
        cls_attn  = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        att_map   = cls_attn.reshape(14, 14)
        att_map_resized = cv2.resize(att_map, (224, 224))

        heatmap_colored = plt.cm.viridis(att_map_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        img_small = cv2.resize(img_rgb, (224, 224))
        overlay   = cv2.addWeighted(img_small, 0.45, heatmap_colored, 0.55, 0)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.patch.set_facecolor('#0f0f1a')
        axes[0].imshow(att_map_resized, cmap='viridis')
        axes[0].set_title('Attention Heatmap', fontsize=9, color='white', pad=6)
        axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title('Overlay on Image', fontsize=9, color='white', pad=6)
        axes[1].axis('off')
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', facecolor='#0f0f1a', dpi=120)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Attention map error: {e}")
        return None

# ===== Object Detection (Assignment 3) =====
def detect_persons(img_array):
    if yolo_model is None:
        return None, []

    # Save temp image
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "detect_input.jpg")
    cv2.imwrite(tmp_path, img_array)

    results = yolo_model(tmp_path, conf=0.3, verbose=False)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Convert to base64
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Get detections info
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        detections.append({
            "label": "person",
            "confidence": round(conf * 100, 1)
        })

    return img_base64, detections

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file      = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    features = extract_features(img)

    h, w  = img.shape[:2]
    new_w = 300
    new_h = int(h * new_w / w)
    img_resized = cv2.resize(img, (new_w, new_h))

    _, buffer  = cv2.imencode(".jpg", img_resized)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    feature_map_img = generate_feature_map(img)

    return jsonify({
        "features":      features.tolist(),
        "feature_count": len(features),
        "mean":          float(np.mean(features)),
        "std":           float(np.std(features)),
        "min":           float(np.min(features)),
        "max":           float(np.max(features)),
        "image_preview": img_base64,
        "feature_map":   feature_map_img,
        "input_shape":   f"{img.shape[1]} × {img.shape[0]} × {img.shape[2]}"
    })

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if yolo_model is None:
        return jsonify({"error": "YOLOv10 model not loaded. Make sure yolov10_llvip_best.pt is in the project folder."}), 500

    file      = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    detected_img, detections = detect_persons(img)

    return jsonify({
        "detected_image": detected_img,
        "detections":     detections,
        "person_count":   len(detections)
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
