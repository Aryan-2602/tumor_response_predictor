import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image_tensor, target_class, device, final_conv_layer="features.18"):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into the final conv layer
    final_layer = dict([*model.named_modules()])[final_conv_layer]
    forward_handle = final_layer.register_forward_hook(forward_hook)
    backward_handle = final_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    class_score = output[0, target_class]
    model.zero_grad()
    class_score.backward()

    # Get hooked values
    grads = gradients[0].cpu().detach().numpy()
    acts = activations[0].cpu().detach().numpy()
    weights = np.mean(grads, axis=(2, 3))[0]

    cam = np.zeros(acts.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.resize(image_np, (224, 224))

    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    forward_handle.remove()
    backward_handle.remove()

    return superimposed_img
