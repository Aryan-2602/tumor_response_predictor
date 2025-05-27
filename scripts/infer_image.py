import argparse
from PIL import Image
import torch
import matplotlib.pyplot as plt
from src.models.classifier import build_model
from src.preprocessing.transforms import get_transforms
from src.visualization.gradcam import generate_gradcam

def predict(image_path, model_path, device):
    # Load model
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    img_tensor = transform(img)
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()

    # Grad-CAM
    heatmap = generate_gradcam(model, img_tensor, pred_class, device=device)
    return pred_class, probs[0][pred_class].item(), heatmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    label, confidence, cam = predict(args.image, args.model, device)
    print(f"Prediction: {label} | Confidence: {confidence:.4f}")
    plt.imshow(cam)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()
