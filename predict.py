import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

model_path = 'model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 1)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
size = (128, 128)
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def generate_gradcam_overlay(model, input_tensor, original_image):
    model.eval()
    target_layer = model.features[-1]
    activations = []
    gradients = []

    def forward_hook(_module, _inp, out):
        activations.append(out)

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = int(torch.sigmoid(output).round().item())

    model.zero_grad()
    output.backward()

    forward_handle.remove()
    backward_handle.remove()

    acts = activations[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.nn.functional.relu(cam)
    cam = cam[0, 0].cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    original_image_array = np.array(original_image)
    overlay = original_image_array.copy()
    if (pred_class == 1):
        H, W = original_image_array.shape[:2]
        heatmap = cv2.resize(cam, (W, H))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

        if len(original_image_array.shape) == 2 or original_image_array.shape[2] == 1:
            original_image_array = cv2.cvtColor(original_image_array, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_image_array, 0.6, heatmap, 0.4, 0)
    return overlay


def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        predicted_prob = torch.sigmoid(output).item()
        prediction = round(predicted_prob)
    overlay = generate_gradcam_overlay(model, img_t, img)
    return prediction, predicted_prob, overlay

# Reduced version of predict_image for use of test CSV file.
def predict_image_for_test_suit(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        predicted_prob = torch.sigmoid(output).item()
        prediction = round(predicted_prob)
    return prediction, predicted_prob

if __name__ == "__main__":
    image_path = '/Users/christian/Desktop/CEC_2025/yes/yes__165.png'
    # image_path = '/Users/christian/Desktop/CEC_2025/no/no__594.png'
    prediction, probability, overlay = predict_image(image_path)
    print(f'prediction: {prediction}, probability: {probability}')

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap Overlay")
    plt.axis('off')
    plt.show()
