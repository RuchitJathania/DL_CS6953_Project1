import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as tt
# from albumentations import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# import albumentations as A
from lookahead_optim import Lookahead
import time
from plotTensorBoard2 import plotAndSave

plot_save_dir = r'checkpoints/resnet3_v1_3'
summary_dir = 'summaries/' + "resnet3_v1_3"
writer = SummaryWriter('summaries/' + "resnet3_v1_3")

# Set device (Use GPU or MPS if available)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

cifar10_dir = 'kaggle/input/DLProj1/cifar-10-python/cifar-10-batches-py'
test_data_dir = 'kaggle/input/DLProj1/cifar_test_nolabel.pkl'

config = {
    'conv_kernel_sizes': [3, 3, 3],
    'num_blocks': [4, 4, 3],
    'num_channels': 64,
    'shortcut_kernel_sizes': [1, 1, 1],
    'avg_pool_kernel_size': 8,
    'drop': 0,  # proportion for dropout
    'squeeze_and_excitation': 1,  # True=1, False=0
    'max_epochs': 200,
    'optim': "sgd",
    'lr_sched': "CosineAnnealingLR",
    'momentum': 0.9,
    'lr': 0.1,
    'weight_decay': 0.0005,
    'batch_size': 128,
    'num_workers': 16 if device.type == "cuda" else 0,
    'persistent': True if device.type == "cuda" else False,
    'resume_ckpt': 0,  # 0 if not resuming, else path to checkpoint
    'data_augmentation': 1,  # True=1, False=0
    'data_normalize': 1,  # True=1, False=0
    'grad_clip': 0.1,
    'lookahead': 1,
    'save_dir': r'checkpoints/resnet3_v1_3'
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

normalize = tt.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
size = (32, 32)

transform_train = tt.Compose([
    # tt.RandomResizedCrop(size),
    tt.Resize(size),
    tt.RandomCrop(32, padding=4),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(10),
    tt.RandomAffine(0, shear=5, scale=(0.9, 1.1)),
    tt.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # had hue=0.5 before
    tt.ToImage(), tt.ToDtype(torch.float32, scale=True),  # tt.ToTensor(),
    normalize
])

transform_test = tt.Compose([tt.Resize(size), tt.ToImage(), tt.ToDtype(torch.float32, scale=True), normalize])


def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


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
            # print("IMAGE IS TYPE: ", type(image))
            image = self.transform(image)  # Apply transforms
            # image = self.transform(image=image)["image"]  # Apply transforms
        if self.labels is not None:
            label = self.labels[idx] if len(self.labels) > 1 else self.labels
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image


def getDataLoaders(train_data_dir, test_pkl_dir, batch_size_in, train_tfms, valid_tfms):
    train_data = []
    train_labels = []
    for i in range(1, 6):  # CIFAR-10 has 5 training batches
        batch_dict = load_cifar_batch(os.path.join(train_data_dir, f'data_batch_{i}'))
        image_data = batch_dict[b'data']
        train_labels_batch = batch_dict[b'labels']
        train_images = image_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)  # N x W x H x C format
        train_data.append(train_images)
        train_labels.extend(train_labels_batch)

    pin_Mem = device.type == "cuda"

    # Get Train Images Data:
    full_train_dataset = CIFARDataset(train_data, train_labels, transform=train_tfms)
    print("Total Training Samples: ", len(full_train_dataset))

    # Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size_in, shuffle=True,
                              num_workers=config['num_workers'],
                              persistent_workers=config['persistent'], pin_memory=pin_Mem)
    print("Train dataloader batch size:", train_loader.batch_size)
    valid_data = []
    valid_labels = []
    valid_dict = load_cifar_batch(os.path.join(train_data_dir, f'val_batch'))
    valid_image_data = valid_dict[b'data']
    valid_labels_batch = valid_dict[b'labels']
    valid_images = valid_image_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)  # N x W x H x C format
    valid_data.append(valid_images)
    valid_labels.extend(valid_labels_batch)
    valid_dataset = CIFARDataset(valid_data, valid_labels, transform=valid_tfms)
    valid_loader = DataLoader(valid_dataset, batch_size=int(batch_size_in / 4), shuffle=False, pin_memory=pin_Mem)

    # Get Test Images data
    test_batch = load_cifar_batch(test_pkl_dir)
    test_images = test_batch[b'data']  # Already in N x W x H x C format
    test_dataset = CIFARDataset(test_images, label_list=None, transform=valid_tfms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_in, shuffle=False, pin_memory=pin_Mem)

    return train_loader, valid_loader, test_loader


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=(device.type == "cuda"))


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, shortcut_kernel_size=1, drop=0.4):
        super(BasicBlock, self).__init__()
        self.drop = drop
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, stride=stride,
                               padding=int(conv_kernel_size / 2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1,
                               padding=int(conv_kernel_size / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=shortcut_kernel_size, stride=stride,
                          padding=int(shortcut_kernel_size / 2), bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.drop:
            out = self.dropout(out)
        return out


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1, stride=stride, groups=groups, bias=bias)


class SEBlock(nn.Module):
    # Squeeze and Excitation block
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=mid_channels, bias=True)
        self.activ = nn.ReLU(inplace=True)

        self.conv2 = conv1x1(in_channels=mid_channels, out_channels=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, conv_kernel_sizes=None, shortcut_kernel_sizes=None, num_classes=10,
                 num_channels=32, avg_pool_kernel_size=4, drop=None, squeeze_and_excitation=None):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.avg_pool_kernel_size = int(32 / (2 ** (len(num_blocks) - 1)))

        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation

        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=self.num_channels)

        self.residual_layers = []
        for n in range(len(num_blocks)):
            stride = 1 if n == 0 else 2
            conv_kernel_size = conv_kernel_sizes[n] if conv_kernel_sizes else 3
            shortcut_kernel_size = shortcut_kernel_sizes[n] if shortcut_kernel_sizes else 1
            self.residual_layers.append(self._make_layer(
                block,
                self.num_channels * (2 ** n),
                num_blocks[n],
                stride=stride,
                conv_kernel_size=conv_kernel_size,
                shortcut_kernel_size=shortcut_kernel_size
            ))
        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.linear = nn.Linear(self.num_channels * (2 ** n) * block.expansion, num_classes)

        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def _make_layer(self, block, planes, num_blocks, stride, conv_kernel_size, shortcut_kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_kernel_size, shortcut_kernel_size, drop=self.drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation: out = self.seblock(out)
        for layer in self.residual_layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        if self.drop: out = self.dropout(out)
        out = self.linear(out)
        return out


def project1_model(config=None):
    # Best Model
    net = ResNet(
        block=BasicBlock,
        num_blocks=[4, 4, 3],  # N: number of Residual Layers | Bi:Residual blocks in Residual Layer i
        conv_kernel_sizes=[3, 3, 3],  # Fi: Conv. kernel size in Residual Layer i
        shortcut_kernel_sizes=[1, 1, 1],  # Ki: Skip connection kernel size in Residual Layer i
        num_channels=64,  # Ci: # channels in Residual Layer i
        avg_pool_kernel_size=8,  # P: Average pool kernel size
        drop=0,  # use dropout with drop proportion
        squeeze_and_excitation=1  # Enable/disable Squeeze-and-Excitation Block
    )
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    return net, total_params


# Training
def train(epoch, config, train_loader, model, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_losses = []
    train_acc = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print('Epoch: {}'.format(epoch), end="  | ")
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if config["grad_clip"]: nn.utils.clip_grad_value_(model.parameters(), clip_value=config["grad_clip"])
        optimizer.step()

        train_loss += loss.item()
        train_losses.append(train_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_acc.append(100. * correct / total)
        print('Batch_idx: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (
            batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    mean_train_loss = np.mean(train_losses)
    mean_train_acc = np.mean(train_acc)
    print('Epoch: {} | Train Loss: {} | Train Acc: {}'.format(epoch, mean_train_loss, mean_train_acc))
    writer.add_scalar('Loss/train_loss', mean_train_loss, epoch)
    writer.add_scalar('Accuracy/train_accuracy', mean_train_acc, epoch)


# Testing
def test(epoch, config, savename, valid_loader, model, criterion):
    global best_acc
    model.eval()
    test_loss = 0
    test_losses = []
    test_acc = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_losses.append(test_loss)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_acc.append(100. * correct / total)
            # print('Batch_idx: %d | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (
            # batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        mean_test_loss = np.mean(test_losses)
        mean_test_acc = np.mean(test_acc)
        print('Epoch: {} | Test Loss: {} | Test Acc: {}'.format(epoch, mean_test_loss, mean_test_acc))
        writer.add_scalar('Loss/test_loss', mean_test_loss, epoch)
        writer.add_scalar('Accuracy/test_accuracy', mean_test_acc, epoch)

        # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'config': config
        }
        torch.save(state, os.path.join('./checkpoints/', savename, 'resnet3_v1_3.th'))
        best_acc = acc


best_acc = 0


def main():
    train_loader, valid_loader, test_loader = getDataLoaders(cifar10_dir, test_data_dir, config['batch_size'],
                                                             train_tfms=transform_train, valid_tfms=transform_test)

    # train_loader = DeviceDataLoader(train_loader, device)
    # valid_loader = DeviceDataLoader(valid_loader, device)
    # test_loader = DeviceDataLoader(test_loader, device)
    print(len(train_loader))
    print(train_loader.batch_size)
    # Code below to test if dataloader is working properly:
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    images, labels = next(iter(train_loader))
    images, labels = images[:10].cpu(), labels[:10].cpu()  # Take first 10 images if batch size at least 10

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # 2 rows, 5 columns

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
        # img = img * 0.2023 + 0.4914  # Simple normalization adjustment (approximation)
        img = img * np.array(CIFAR10_MEAN) + np.array(CIFAR10_STD)
        ax.imshow(img.clamp(0, 1))  # Clamp values between 0 and 1
        ax.set_title(f"Label: {cifar10_classes[labels[i]]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model, total_params = project1_model(config=config)
    model = model.to(device)
    config['total_params'] = total_params
    print(model)
    print('Total Parameters: ', total_params)

    if total_params > 5_000_000:
        print("===============================")
        print("Total parameters exceeding 5M")
        print("===============================")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config[
        "weight_decay"])

    # optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    optimizer = Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    main_start_time = time.time()
    for epoch in range(start_epoch, config["max_epochs"]):
        train_start_time = time.time()
        train(epoch, config, train_loader, model, optimizer, criterion)
        epoch_training_time = time.time() - train_start_time
        print("Epoch Train Time: ", epoch_training_time)
        test(epoch, config, savename="resnet3_v1_3", valid_loader=valid_loader, model=model, criterion=criterion)
        scheduler.step()
    training_time = time.time() - main_start_time
    print("Total Time: ", training_time)
    writer.close()

    # Get the csv file of predictions on Test Dataset:
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
    submission_df.to_csv('submissionTest7.csv', index=False)

    print("Submission file saved as 'submissionTest7.csv'")

    # Plot and Save Test Loss and Accuracy over epochs:
    plotAndSave(plot_save_dir, summary_dir)

    # Plot 10 random validation images and inference predictions:
    checkpoint_path = os.path.join(config['save_dir'], 'resnet3_v1_3.th')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Pick 10 random images from the validation set
    valid_images, valid_labels = next(iter(valid_loader))
    indices = random.sample(range(len(valid_images)), 10)  # Select 10 random indices
    images = valid_images[indices].to(device)
    labels = valid_labels[indices].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Get predicted class index

    # Create a plot for visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        img = img * torch.tensor(CIFAR10_MEAN).to(img.dtype) + torch.tensor(CIFAR10_STD).to(
            img.dtype)  # Denormalization
        img = img.clamp(0, 1)  # Ensure values are in valid range for display

        true_label = cifar10_classes[labels[i].item()]
        pred_label = cifar10_classes[preds[i].item()]

        color = "green" if true_label == pred_label else "red"
        ax.imshow(img)
        ax.set_title(pred_label, color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(plot_save_dir, "Inferences_plot.png")
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
