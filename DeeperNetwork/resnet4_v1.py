import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms.v2 as tt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('summaries/' + "model4_v1")

config = {
    "arch": "resnet110",  # Default model architecture
    "workers": 1,  # Number of data loading workers
    "epochs": 1,  # Total number of epochs
    "start_epoch": 0,  # Manual epoch number (useful for restarts)
    "batch_size": 128,  # Mini-batch size
    "lr": 0.1,  # Initial learning rate
    "momentum": 0.9,  # Momentum
    "weight_decay": 1e-4,  # Weight decay
    "print_freq": 50,  # Print frequency
    "resume": "",  # Path to latest checkpoint (empty means no resume)
    "evaluate": True,  # Whether to evaluate the model
    "pretrained": False,  # Whether to use a pre-trained model
    "half": False,  # Whether to use half-precision (float16)
    "save_dir": "save_temp",  # Directory to save trained models
    "save_every": 10,  # Save checkpoints every specified number of epochs
}

# Set device (Use GPU or MPS if available)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

cifar10_dir = 'kaggle/input/DLProj1/cifar-10-python/cifar-10-batches-py'
test_data_dir = 'kaggle/input/DLProj1/cifar_test_nolabel.pkl'

CIFAR10_MEAN = (0.485, 0.456, 0.406)
CIFAR10_STD = (0.229, 0.224, 0.225)

normalize = tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
size = (32, 32)

transform_train = tt.Compose([
    # tt.RandomResizedCrop(size),
    tt.Resize(size),
    tt.RandomCrop(32, padding=4),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(10),
    tt.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # had hue=0.5 before
    tt.ToTensor(),
    normalize
])

transform_test = tt.Compose([tt.Resize(size), tt.ToTensor(), normalize])


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
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size_in, shuffle=True, pin_memory=pin_Mem)
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


__all__ = ['ResNet', 'resnet110']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start_time = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.to(device)
        input_var = input.to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
    epoch_time = time.time() - start_time
    print('Epoch: {} | Train Loss: {} | Train Acc: {} | Time: {}'.format(epoch, losses.val, top1.val, epoch_time))
    writer.add_scalar('Loss/train_loss', losses.val, epoch)
    writer.add_scalar('Accuracy/train_accuracy', top1.val, epoch)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start_time = time.time()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target_var = target.to(device)
            input_var = input.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        epoch_time = time.time() - start_time
        print('Epoch: {} | Test Loss: {} | Test Acc: {} | Time: {}'.format(epoch, losses.val, top1.val, epoch_time))
        writer.add_scalar('Loss/test_loss', losses.val, epoch)
        writer.add_scalar('Accuracy/test_accuracy', top1.val, epoch)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

    train_loader, valid_loader, test_loader = getDataLoaders(cifar10_dir, test_data_dir, config['batch_size'],
                                                             train_tfms=transform_train, valid_tfms=transform_test)

    # Code below to test if dataloader is working properly:
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    images, labels = next(iter(train_loader))
    images, labels = images[:10].cpu(), labels[:10].cpu()  # Take first 10 images if batch size at least 10

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # 2 rows, 5 columns

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
        img = img * np.array(CIFAR10_MEAN) + np.array(CIFAR10_STD)  # Simple normalization adjustment
        ax.imshow(img.clamp(0, 1))  # Clamp values between 0 and 1
        ax.set_title(f"Label: {cifar10_classes[labels[i]]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    model = ResNet(BasicBlock, [18, 18, 18])
    total_params = 0
    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print('Total Parameters: ', total_params)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=config['start_epoch'] - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['lr'] * 0.1

    best_prec1 = 0

    for epoch in range(0, config['epochs']):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(valid_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
    writer.close()
    print("Best Test Accuracy: ", best_prec1)

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
    submission_df.to_csv('submissionTest5_resnet110.csv', index=False)

    print("Submission file saved as 'submissionTest5_resnet110.csv'")
