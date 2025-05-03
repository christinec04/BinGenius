import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os

# ==== CONFIG ====
# DATA_DIR = "/Users/adityavaidya/Desktop/Class folders/Data Analysis and Mining/FinalProject/Recycling"
DATA_DIR = "data"
# MODEL_SAVE_PATH = "/Users/adityavaidya/Desktop/Class folders/Data Analysis and Mining/FinalProject/Recycling/mobilenetv2_trashnet.pth"
MODEL_SAVE_PATH = "models/mobilenetv2_trashnet2.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
BATCH_SIZE = 32
NUM_EPOCHS = 7
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORMS ====
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),  # Lighting robustness
    transforms.RandomHorizontalFlip(p=0.5),  # Help with orientation invariance
    transforms.RandomRotation(degrees=10),   # Small rotations for better generalization
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # MobileNetV2 standard
                         [0.229, 0.224, 0.225])
])

# ==== LOAD DATASET ====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==== LOAD MODEL ====
model = models.mobilenet_v2(weights=True)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== TRAINING ====
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}, Accuracy = {train_acc:.4f}")

    # ==== VALIDATION ====
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total
    print(f"\tValidation Accuracy = {val_acc:.4f}")

# ==== SAVE MODEL ====
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")