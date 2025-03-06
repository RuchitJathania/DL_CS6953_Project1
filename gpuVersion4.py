import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load CIFAR-10 dataset from pickle files
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


cifar10_dir = 'DL_CS6953_Project1/kaggle/input/DLProj1/cifar-10-python/cifar-10-batches-py'

# Load training batches
train_data = []
train_labels = []

for i in range(1, 6):  # CIFAR-10 has 5 training batches
    batch_dict = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
    train_data.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

# Convert to PyTorch tensors and normalize
train_images = torch.tensor(np.vstack(train_data), dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Load test data
test_batch = load_cifar_batch('DL_CS6953_Project1/kaggle/input/DLProj1/cifar_test_nolabel.pkl')
test_images = torch.tensor(test_batch[b'data'], dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0

# Define Normalization Transform
normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


# Custom Dataset Class
class CIFARDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


# Create Datasets
train_dataset = CIFARDataset(train_images, train_labels, transform=normalize_transform)
test_dataset = CIFARDataset(test_images, labels=None, transform=normalize_transform)

# Create DataLoaders (num_workers=0 to avoid Windows multiprocessing issues)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        return F.relu(out)


class LargerResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LargerResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # **⬇️ Reduce filters**
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)  # **2 residual block**
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # **2 residual block**
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # **2 residual block**

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)  # **⬇️ Reduce FC layer size**

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        layers += [BasicBlock(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Training Loop
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses, train_accuracies, test_accuracies = [], [], []
    scaler = torch.amp.GradScaler("cuda")  # Fix for deprecated GradScaler

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc if test_acc is not None else 0)

        print(f"Epoch [{epoch + 1}/{num_epochs}] → Loss: {train_loss}, Train Acc: {train_acc}%, Test Acc: {test_acc}%")


# Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            # Check if test dataset has labels
            if isinstance(batch, tuple) and len(batch) == 2:
                images, labels = batch
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                images = batch.to(device, non_blocking=True)
                labels = None  # No labels for test set

            outputs = model(images)

            # Only calculate accuracy if labels exist
            if labels is not None:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    return 100 * correct / total if total > 0 else None  # Return accuracy only if labels exist


# Ensure Windows multiprocessing compatibility
if __name__ == "__main__":
    model = LargerResNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=15)

    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
