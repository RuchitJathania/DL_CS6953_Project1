# ResNet Implementation for CIFAR-10 Classification

This repository contains a ResNet-based implementation for CIFAR-10 image classification with a parameter constraint of 5 million. The best-performing model (`resnet3_v1_4.py`) is included along with essential utility files.

## Requirements

To run this project, you need a system with:
- Python 3.10
- PyTorch with CUDA (for NVIDIA GPUs) or MPS (for Mac M1/M2 chips)

### Install Dependencies
Make sure you have a compatible version of PyTorch installed. You can set up the environment as follows:

#### For NVIDIA GPU (CUDA) Users:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

You can replace the final 'cu118' with your cuda version, 'cu124' for CUDA 12.4 or 'cu126' for CUDA 12.6

#### For Mac M1/M2 Users (MPS Support):
```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly

for conda environment

or

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Make sure to have MacOS 12.3 or later and Xcode command-line tools

#### For regular Pytorch install which uses CPU only:
```bash
pip3 install torch torchvision torchaudio' 

#### Get the Dataset:

From either kaggle competition download, or "https://www.cs.toronto.edu/~kriz/cifar.html", and make sure to use `test_batch` from original dataset as `valid_batch` for code. 

## Running the Code

### 1. Adjust Directory Paths
Before running, ensure that the directory paths in `resnet3_v1_4.py` match your system's structure.

Edit these lines in `resnet3_v1_4.py` to reflect your file locations:
plot_save_dir = os.path.join("checkpoints", "resnet3_v1_4") 
summary_dir = os.path.join("summaries", "resnet3_v1_4") 
cifar10_dir = 'kaggle/input/DLProj1/cifar-10-python/cifar-10-batches-py' 
test_data_dir = 'kaggle/input/DLProj1/cifar_test_nolabel.pkl'

If your dataset is stored elsewhere, update these paths accordingly.

### 2. Run the Best Model
To train the best-performing model, run:

`python BestPerformingModel/resnet3_v1_4.py`

This will:
- Load the CIFAR-10 dataset.
- Train the model using the ResNet architecture.
- Save model checkpoints and logs.

### 3. Visualizing Training Progress
To visualize accuracy and loss:

`plotTensorboard2.py` is used within the main model code, or you can modify the script and run it separately from model training.

### Project Structure:

├── BestPerformingModel/  
│   ├── resnet3_v1_4.py  # Main model script  
│   ├── lookahead_optim.py  # Lookahead optimizer  
│   ├── plotTensorboard2.py  # Utility for TensorBoard visualization  
├── checkpoints/  # Model checkpoints will be stored here  
├── summaries/  # TensorBoard logs  
├── datasets/  # CIFAR-10 dataset (if stored locally)  
├── README.md  # This file  

### Adjusting Parameters:
 In main training file, you can adjust the parameters like batch size, number of epochs, number of workers for dataloader, etc. in `config` dictionary. Adjust accordingly to your resources and preferences.
 
