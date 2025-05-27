import streamlit as st
from PIL import Image
import torch
import numpy as np
from src.models.classifier import build_model
from src.preprocessing.transforms import get_transforms
from src.visualization.gradcam import generate_gradcam
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Title
st.title("Breast Histopathology Classifier (IDC vs. Non-IDC)")

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Upload model path
model_path = st.text_input("Path to trained model (.pt)", "saved_models/model_final_20250525_224103.pt")

@st.cache_resource
def load_model(model_path):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_path)

# Upload image
uploaded_file = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    transform = get_transforms(train=False)
    img_tensor = transform(image)

    # Predict
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    class_names = ["Non-IDC (0)", "IDC (1)"]
    st.markdown(f"### Prediction: **{class_names[pred]}**")
    st.markdown(f"**Confidence:** {confidence:.4f}")

    # Grad-CAM
    st.markdown("### Grad-CAM Visualization")
    cam = generate_gradcam(model, img_tensor, pred, device)
    st.image(cam, caption="Grad-CAM Output", use_container_width=True)
