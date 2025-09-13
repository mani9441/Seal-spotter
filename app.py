import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from mmdet.apis import init_detector, inference_detector
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog

# Inject your dataset metadata manually
MetadataCatalog.get("inference").thing_classes = ["Seal", "Tag_White", "Tag_Yellow"]

import os

# Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

# Paths for the models
MODELS = {
    'retinanet': {
        'type': 'mmdet',
        'config': 'configs/retinanet/atss_custom.py',
        'weights': 'models/retinanet/epoch_100.pth'
    },
    'rtmdet': {
        'type': 'mmdet',
        'config': 'configs/rtmdet/test.py',
        'weights': 'models/rtmdet/epoch_100.pth'
    },
    'ppyoloe-localised': {
        'type': 'mmdet',
        'config': 'configs/ppyoloe_localised/ppyoloeloc.py',
        'weights': 'models/ppyoloe_localised/epoch_100.pth'
    },
    'ppyoloe-normal': {
        'type': 'mmdet',
        'config': 'configs/ppyoloe_normal/ppyolonormal.py',
        'weights': 'models/ppyoloe_normal/epoch_100.pth'
    },
    'mask-rcnn': {
        'type': 'detectron2',
        'config': 'configs/maskRCNN/config.yaml',
        'weights': 'models/maskRCNN/model_final.pth'
    }
}

# Cache for loaded models
loaded_models = {}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def preload_models():
    """Load all models at startup so inference is faster."""
    for model_name in MODELS.keys():
        model, error = load_model_by_name(model_name)
        if error:
            print(f"⚠️ Could not preload {model_name}: {error}")
        else:
            print(f"✅ Preloaded {model_name}")


def load_model_by_name(model_name):
    """Load model (MMDetection or Detectron2)."""
    if model_name not in loaded_models:
        if model_name not in MODELS:
            return None, "Model not found"

        model_info = MODELS[model_name]
        config_path = model_info['config']
        weights_path = model_info['weights']
        model_type = model_info['type']

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            return None, f"Model files not found for {model_name}"

        try:
            if model_type == 'mmdet':
                model = init_detector(config_path, weights_path, device=device)
            elif model_type == 'detectron2':
                cfg = get_cfg()
                cfg.merge_from_file(config_path)
                cfg.MODEL.WEIGHTS = weights_path
                cfg.MODEL.DEVICE = device
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                cfg.freeze()
                model = DefaultPredictor(cfg)

                # Force inject metadata for whatever dataset is in cfg.DATASETS.TEST
                if len(cfg.DATASETS.TEST) > 0:
                    dataset_name = cfg.DATASETS.TEST[0]
                else:
                    dataset_name = "inference"

                MetadataCatalog.get(dataset_name).thing_classes = ["Seal", "Tag_White", "Tag_Yellow"]

            else:
                return None, f"Unsupported model type: {model_type}"

            loaded_models[model_name] = model
            print(f"✅ Loaded {model_name} ({model_type})")
        except Exception as e:
            return None, f"Failed to load {model_name}: {str(e)}"

    return loaded_models[model_name], None

def predict_with_mmdetection(image_bytes, model):
    """Inference with MMDetection model."""
    try:
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Failed to decode image"}

        result = inference_detector(model, img)
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        predictions = []
        label_map = model.dataset_meta['classes']
        for bbox, score, label_id in zip(bboxes, scores, labels):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = bbox
            label = label_map[label_id] if label_id < len(label_map) else 'unknown'
            predictions.append({
                "label": label,
                "confidence": float(score),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
            color = (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{label}: {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return {"success": True, "predictions": predictions,
                "image_url": f"data:image/jpeg;base64,{img_base64}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

def predict_with_detectron2(image_bytes, model):
    """Inference with Detectron2 Mask R-CNN (web-safe)."""
    try:
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Failed to decode image"}

        outputs = model(img)

        # Safe metadata
        if len(model.cfg.DATASETS.TEST) > 0:
            metadata = MetadataCatalog.get(model.cfg.DATASETS.TEST[0])

        else:
            # Create a dummy metadata with at least one class
            metadata = MetadataCatalog.get("__unused")
            if not hasattr(metadata, "thing_classes") or len(metadata.thing_classes) == 0:
                metadata.thing_classes = ["object"]

        instances = outputs["instances"].to("cpu")

        # Draw only if there are predictions
        if len(instances) > 0:
            metadata = MetadataCatalog.get("my_custom_val")
            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)

            annotated = v.draw_instance_predictions(instances).get_image()[:, :, ::-1]
        else:
            annotated = img.copy()  # no detections → return original

        predictions = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.numpy()[0]
            score = instances.scores[i].item()
            label_id = instances.pred_classes[i].item()
            label = metadata.thing_classes[label_id] if label_id < len(metadata.thing_classes) else "unknown"
            predictions.append({
                "label": label,
                "confidence": float(score),
                "box": [float(c) for c in bbox]
            })

        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {"success": True, "predictions": predictions,
                "image_url": f"data:image/jpeg;base64,{img_base64}"}
    except Exception as e:
        print(f"❌ Error during Detectron2 prediction: {e}")
        return {"success": False, "error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files['file']
    model_name = request.form.get('model')
    if not model_name:
        return jsonify({"success": False, "error": "Model not specified"}), 400

    model, error = load_model_by_name(model_name)
    if error:
        return jsonify({"success": False, "error": error}), 500

    # Read file once → pass raw bytes
    image_bytes = file.read()

    if MODELS[model_name]['type'] == 'mmdet':
        response = predict_with_mmdetection(image_bytes, model)
    else:
        response = predict_with_detectron2(image_bytes, model)

    if not response.get("success", False):
        return jsonify(response), 500
    return jsonify(response)

if __name__ == "__main__":
    #preload_models()  #Only if used if predloading of models required
    app.run(host="0.0.0.0", port=5000, debug=True)
