# 🧠 Tumor-Response-Predictor: Breast Histopathology Image Classification with Grad-CAM

This project implements a deep learning pipeline to classify histopathology image patches as **Invasive Ductal Carcinoma (IDC)** or **Non-IDC**, using a transfer learning model trained on the [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) dataset. It includes:

- A **MobileNetV2/ResNet-based classifier**
- **Grad-CAM** for visual interpretability
- An interactive **Streamlit app**
- A **DCGAN module** for synthetic image generation
- Patient-aware, class-balanced sampling logic
- Modular, production-ready code architecture

---

## 📁 Project Structure

```
tumor-response-predictor/
├── data/
│   ├── breast_histopathology_images/
│   └── gan_train/               # Balanced dataset for GAN training
├── generated/                   # GAN outputs (images + models)
├── saved_models/                # Classifier model checkpoints and ROC curves
├── scripts/
│   ├── analyze_dataset_distribution.py
│   ├── download_data.py
│   ├── sample_gan_training_data.py
│   ├── train_model.py
│   ├── infer_image.py
├── src/
│   ├── config.py
│   ├── models/
│   │   └── classifier.py
│   ├── data_loader/
│   │   └── histo_dataset.py
│   ├── preprocessing/
│   │   └── transforms.py
│   ├── training/
│   │   └── train_classifier.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── visualization/
│   │   └── grad_cam.py
│   └── gan/
│       └── generate_images.py
│       ├── models.py
│       ├── dataset.py
│       └── train_gan.py
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- ✅ IDC vs. Non-IDC classification with MobileNetV2
- ✅ Grad-CAM heatmaps for model interpretability
- ✅ Full training pipeline on the breast cancer histopathology dataset
- ✅ DCGAN-based synthetic image generation
- ✅ Streamlit app for interactive predictions
- ✅ Training on M1/MPS or CUDA-enabled GPUs
- ✅ Cloud-friendly structure (Google Drive, Streamlit Cloud ready)

---

## 🧪 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/tumor-response-predictor.git
cd tumor-response-predictor
conda create -n tumor-response-predictor python=3.10
conda activate tumor-response-predictor
pip install -r requirements.txt
```

### 2. Download the dataset from Kaggle

```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
```

### 3. Prepare GAN training dataset

```bash
python scripts/sample_gan_training_data.py
```

---

## 🧠 Train the Classifier

```bash
PYTHONPATH=. /opt/anaconda3/envs/tumor-response-predictor/bin/python scripts/train_model.py
```

Model and ROC curve will be saved in `saved_models/`.

---

## 🔍 Inference with Grad-CAM

```bash
PYTHONPATH=. /opt/anaconda3/envs/tumor-response-predictor/bin/python scripts/infer_image.py \
  --image path/to/test_image.png \
  --model saved_models/model_final_<timestamp>.pt
```

---

## 🖼️ Streamlit App (Interactive UI)

```bash
PYTHONPATH=. streamlit run streamlit_app/app.py
```

Upload an image → get prediction + Grad-CAM.

---

## 🎨 Train GAN for Synthetic Tumor Generation

```bash
PYTHONPATH=. /opt/anaconda3/envs/tumor-response-predictor/bin/python src/gan/train_gan.py
```

Outputs:
- `generated/epoch_*.png` — synthetic image samples
- `netG_epoch*.pth` — generator checkpoints

---

## 📊 Evaluation Tools

- 📈 Accuracy, Precision, Recall, F1-score
- 🧮 Confusion matrix and classification report
- 🔥 ROC curve saved as PNG
- 👀 Grad-CAM overlay for model explainability

---

## 🌐 Deployment (Future-Ready)

- ✅ **Streamlit Cloud** or **Hugging Face Spaces** ready for public sharing
- ✅ **Google Drive** or **AWS S3** for storing datasets/models
- ✅ Easy Docker integration (coming soon)

---

## 🚧 Potential Improvements

| Area             | Description                                                        |
|------------------|---------------------------------------------------------------------|
| 🧪 Data Augmentation | Add real-time augmentation in training pipeline                  |
| 🎭 Conditional GAN   | Generate class-specific synthetic patches                        |
| 📦 Docker Support    | Containerize with a `Dockerfile` for cloud deployment            |
| ☁️ Cloud Storage     | Auto-sync models or data to Google Drive or S3                   |
| 🔁 Batch Inference   | Add CLI tool for full-folder predictions                         |
| 🧬 Multimodal Learning| Combine image and metadata for richer prediction                |

---

## 👨‍💻 Author

**Aryan Maheshwari**  
MS in Applied Data Science, University of Southern California  
Inspired by AI for healthcare, explainability, and synthetic data generation

---

## 📜 License

MIT License
