import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from PIL import Image
import seaborn as sns
import os
import time

data_dir = os.getenv('CEC_2025_dataset')
if data_dir is None:
    raise ValueError("env variable not set :(")

print("data directory:", data_dir)

num_epochs = 10
optimizer = None

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# mean and std taken from imagenet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
size = (128, 128)
train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_test_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("loading images...")
images = []
labels = []

max_images_per_class = 25

for label, cls in enumerate(['no', 'yes']):
    cls_dir = os.path.join(data_dir, cls)
    cur_images = 0
    for img_name in os.listdir(cls_dir):
        if cur_images >= max_images_per_class:
            break
        img_path = os.path.join(cls_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
        labels.append(label)
        cur_images += 1

X_train_pil, X_temp_pil, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val_pil, X_test_pil, y_val, y_test = train_test_split(
    X_temp_pil, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_t = [train_transform(img) for img in X_train_pil]
X_val_t   = [val_test_transform(img) for img in X_val_pil]
X_test_t  = [val_test_transform(img) for img in X_test_pil]

X_train = torch.stack(X_train_t)
X_val   = torch.stack(X_val_t)
X_test  = torch.stack(X_test_t)

y_train = torch.tensor(y_train).unsqueeze(1).float()
y_val   = torch.tensor(y_val).unsqueeze(1).float()
y_test  = torch.tensor(y_test).unsqueeze(1).float()

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val),
    batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=batch_size, shuffle=False
)

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 1)
)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available())

print("starting training...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy().round()
            val_preds.extend(preds)
            val_targets.extend(labels_batch.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_recall = recall_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds)

    print(f"values at epoch [{epoch+1}/{num_epochs}]")
    print(f"  train Loss: {avg_loss}")
    print(f"  val loss:   {val_loss}")
    print(f"  val acc:    {val_accuracy}")
    print(f"  val recall: {val_recall}")
    print(f"  val prec:   {val_precision}")

model.eval()
preds, targets = [], []
with torch.no_grad():
    for inputs, labels_batch in test_loader:
        inputs = inputs.to(device)
        outputs = torch.sigmoid(model(inputs)).cpu().numpy().round()
        preds.extend(outputs)
        targets.extend(labels_batch.numpy())

print("====== final results ======:")
print(f"accuracy:   {accuracy_score(targets, preds)}")
print(f"recall:     {recall_score(targets, preds)}")
print(f"precision:  {precision_score(targets, preds)}")
print(f"confusion matrix:\n{confusion_matrix(targets, preds)}")
print(f"total time: {time.time() - start_time:.2f} seconds")

torch.save(model.state_dict(), 'model.pth')
