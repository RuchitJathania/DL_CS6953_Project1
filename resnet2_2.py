import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torch.utils.data import Dataset, DataLoader

batch_size_in = 1400


def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats, inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


class CIFARDataset(Dataset):
    def __init__(self, image_list, label_list=None, transform=None):
        """
        image_list: List of NumPy arrays of shape (10000, 32, 32, 3)
        label_list: Single list of 50000 labels
        transform: Transformations to apply to each image
        """
        self.images = image_list if label_list is None else np.concatenate(image_list, axis=0)
        self.labels = np.array(label_list) if label_list is not None else None  # Convert labels to NumPy array
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert NumPy image to PIL (transform expects PIL image)
        image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)  # Convert to (C, H, W) format
        toPILImage = tt.ToPILImage()
        image = toPILImage(image)  # Convert to PIL image

        if self.transform:
            image = self.transform(image)  # Apply transforms
        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image


cifar10_dir = 'DL_CS6953_Project1/kaggle/input/DLProj1/cifar-10-python/cifar-10-batches-py'

train_data = []
train_labels = []
for i in range(1, 6):  # CIFAR-10 has 5 training batches
    batch_dict = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
    image_data = batch_dict[b'data']
    train_labels_batch = batch_dict[b'labels']
    train_images = image_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)  # N x W x H x C format
    train_data.append(train_images)
    train_labels.extend(train_labels_batch)

# Get Train Images Data:
train_dataset = CIFARDataset(train_data, train_labels, transform=train_tfms)
train_loader = DataLoader(train_dataset, batch_size=batch_size_in, shuffle=True, pin_memory=True)

# Get Test Images data
test_batch = load_cifar_batch('DL_CS6953_Project1/kaggle/input/DLProj1/cifar_test_nolabel.pkl')
test_images = test_batch[b'data']  # Already in N x W x H x C format

test_dataset = CIFARDataset(test_images, label_list=None, transform=valid_tfms)
test_loader = DataLoader(test_dataset, batch_size=batch_size_in, shuffle=False, pin_memory=True)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


train_dl = DeviceDataLoader(train_loader, device)
test_dl = DeviceDataLoader(test_loader, device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        # images, labels = batch
        images = batch
        out = self(images)  # Generate predictions
        # loss = F.cross_entropy(out, labels)  # Calculate loss
        loss = None
        # acc = accuracy(out, labels)  # Calculate accuracy
        acc = None
        # return {'val_loss': loss.detach(), 'val_acc': acc}
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {}, val_loss: {}, val_acc: {}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet_v1(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 224, pool=True)
        self.conv4 = conv_block(224, 416, pool=True)
        self.res2 = nn.Sequential(conv_block(416, 416), conv_block(416, 416))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(416, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


model = to_device(ResNet_v1(3, 10), device)
print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")


# model

@torch.no_grad()
def evaluate(model, val_loader, has_labels=True):
    model.eval()
    outputs = []
    for batch in val_loader:
        if has_labels:
            outputs.append(model.validation_step(batch))  # Regular validation
        else:
            images = batch.to(device)
            preds = model(images)  # Only predict, no labels
            outputs.append(preds)

    if has_labels:
        return model.validation_epoch_end(outputs)
    else:
        return {'val_loss': None, 'val_acc': None}
        # return outputs  # Return raw predictions for test set


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader, has_labels=False)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


history = [evaluate(model, test_dl, has_labels=False)]
print(history)

epochs = 20
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4

opt_func = torch.optim.Adam
history += fit_one_cycle(epochs, max_lr, model, train_dl, test_dl,
                         grad_clip=grad_clip,
                         weight_decay=weight_decay,
                         opt_func=opt_func)

model.eval()

# Store predictions
predictions = []

# Disable gradient computation for inference
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)  # Move images to same device as model
        outputs = model(images)  # Get model predictions
        _, predicted = torch.max(outputs, 1)  # Get class with the highest probability
        predictions.extend(predicted.cpu().numpy())  # Convert tensor to list

# Create a DataFrame for submission
submission_df = pd.DataFrame({'ID': list(range(0, len(predictions))), 'Labels': predictions})

# Save to CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")