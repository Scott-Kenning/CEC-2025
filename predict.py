import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision import transforms
from PIL import Image

model_path = 'model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.EfficientNet_B0_Weights.DEFAULT
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

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        predicted_prob = torch.sigmoid(output).item()
        prediction = round(predicted_prob)

    return prediction, predicted_prob

if __name__ == "__main__":
    image_path = '/Users/christian/Desktop/CEC_2025/CEC_test/fish.jpeg'
    prediction, probability = predict_image(image_path)
    print(f'prediction: {prediction}, confidence: {probability}')
