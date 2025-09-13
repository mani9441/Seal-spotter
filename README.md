# Seal-Spotter

Seal-Spotter is a Flask-based web application for detecting and classifying seals
and their identification tags in images.  
It provides a simple web UI and a JSON API to run inference with several
pre-trained object-detection models (MMDetection and Detectron2).

---

## ✨ Features

- **Multiple Detectors**:

  - RetinaNet (MMDetection)
  - RTMDet (MMDetection)
  - PP-YOLOE (normal & localised)
  - Mask R-CNN (Detectron2)

- **Web Interface & REST API**  
  Upload an image, choose a model, and receive an annotated image + JSON of detections.

- **GPU / CPU**  
  Automatically uses GPU if available; falls back to CPU.

---

## 🗂 Project Structure

```
.
├── app.py                 # Flask application entry point
├── templates/             # HTML templates (index.html, results.html)
├── static/                # JS/CSS (if any)
├── models/                # Model configs and weights (see below)
├── requirements.txt       # Python dependencies
└── README.md
```

### Models

The application supports the following detectors:

| Name              | Framework   | Config path                                   |
| ----------------- | ----------- | --------------------------------------------- |
| retinanet         | MMDetection | configs/retinanet/atss_custom.py              |
| rtmdet            | MMDetection | configs/rtmdet/test.py                        |
| ppyoloe-localised | MMDetection | configs/ppyolo/ppyoloeloc.py                  |
| ppyoloe-normal    | MMDetection | configs/ppyolo/ppyoloe_normal/ppyolonormal.py |
| mask-rcnn         | Detectron2  | configs/maskRCNN/config.yaml                  |

> **Large weights** are tracked with [Git LFS](https://git-lfs.github.com/) and aviables in `models/`
> or can be downloaded separately.

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mani9441/Seal-spotter.git
cd Seal-spotter
```

### 2️⃣ Install Dependencies

It’s recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Requirements include **Flask**, **torch**, **opencv-python**,
> **mmdet**, **detectron2**, and others.

### 3️⃣ Add Model Weights

Download the `.pth` weight files and place them in the `models/` directory,
matching the paths defined in `app.py`.

### 4️⃣ Run the Web App

```bash
python app.py
```

The app will start at **[http://0.0.0.0:5000/](http://0.0.0.0:5000/)** (or [http://localhost:5000](http://localhost:5000)).

---

## 🌐 API Usage

`POST /predict`

- **file**: image file
- **model**: one of `retinanet`, `rtmdet`, `ppyoloe-localised`, `ppyoloe-normal`, `mask-rcnn`

Example with `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -F "model=mask-rcnn" \
  -F "file=@example.jpg"
```

Response:

```json
{
  "success": true,
  "predictions": [
    {
      "label": "Seal",
      "confidence": 0.93,
      "box": [x1, y1, x2, y2]
    }
  ],
  "image_url": "data:image/jpeg;base64,..."
}
```

---

## 🖥 Deployment

You can deploy on any cloud provider (AWS EC2, GCP VM, Azure, etc.):

1. Provision a VM with GPU (optional but recommended).
2. Install the same dependencies and model weights.
3. Run `gunicorn` or `uwsgi` behind Nginx for production.

---

## 📜 License

MIT License.
See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- OpenCV, PyTorch, Flask

---

---

## 👤 Author

**Manikanta Kalyanam**

- 📧 Email: k.manikanta9441@gmail.com
- 💻 GitHub: [@mani9441](https://github.com/mani9441)
- Portfolio: [Manikanta kalyanam](https://mani9441.github.io/portfolio/)

### My Contribution

I designed and implemented the entire Seal-Spotter application:

- Built the **Flask** backend and REST API.
- Integrated **MMDetection** and **Detectron2** pipelines for multi-model inference.
- Developed the **web interface** for uploading images and displaying results.
- Trained and fine-tuned custom models for detecting seals and their identification tags.
- Deployed the project to the cloud and managed GitHub repository, Git LFS, and large-file handling.

---
