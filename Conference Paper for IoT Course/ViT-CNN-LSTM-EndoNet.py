import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load dataset
data_dir = '/kaggle/input/endoscopy'
image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

# Split dataset into train and validation (80:20)
train_size = int(0.8 * len(image_dataset))
val_size = len(image_dataset) - train_size
train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

# Apply validation transforms to validation dataset
val_dataset.dataset.transform = data_transforms['val']

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Get class names and number of classes
class_names = image_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Define the hybrid ViT + CNN + LSTM model
class ViT_CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=1):
        super(ViT_CNN_LSTM_Hybrid, self).__init__()
        
        # Load pretrained ViT
        self.vit = models.vit_b_16(pretrained=True)
        vit_feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # Remove classification head
        
        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # Reduce to 7x7
            nn.Flatten()
        )
        cnn_feature_dim = 256 * 7 * 7
        
        # Combine ViT and CNN features
        self.feature_dim = vit_feature_dim + cnn_feature_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # ViT features
        vit_features = self.vit(x)  # Shape: [batch_size, vit_feature_dim]
        
        # CNN features
        cnn_features = self.cnn(x)  # Shape: [batch_size, cnn_feature_dim]
        
        # Combine features
        combined_features = torch.cat((vit_features, cnn_features), dim=1)  # Shape: [batch_size, feature_dim]
        
        # Reshape for LSTM (treat as single-step sequence)
        combined_features = combined_features.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(combined_features)  # lstm_out: [batch_size, 1, hidden_size]
        lstm_out = lstm_out[:, -1, :]  # Take last time step: [batch_size, hidden_size]
        
        # Classifier
        output = self.classifier(lstm_out)  # Shape: [batch_size, num_classes]
        return output

# Initialize model
model = ViT_CNN_LSTM_Hybrid(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for ViT

# Training loop
num_epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
    val_acc = 100 * correct / total if total > 0 else 0.0
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Print and save final accuracies
if len(train_accuracies) > 0 and len(val_accuracies) > 0:
    print("\n" + "="*50)
    print(f"FINAL RESULTS:")
    print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print("="*50)
    
    with open('/kaggle/working/accuracies.txt', 'w') as f:
        f.write(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%\n")
else:
    print("Error: No accuracies recorded. Check dataset or training loop.")

# Plot loss and accuracy curves
plt.figure(figsize=(12, 4))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title(f'Accuracy Curves\nFinal Train: {train_accuracies[-1]:.2f}%, Val: {val_accuracies[-1]:.2f}%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
if num_classes == 2:
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
else:
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()