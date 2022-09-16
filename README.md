# MetaMask: Revisiting Dimensional Confounder for Self-Supervised Learning
MetaMask contains the official implementation of the NIPS 2022 paper:
MetaMask: Revisiting Dimensional Confounder for Self-Supervised Learning. The code is based on the [Barlowtwins implementation](https://github.com/IgorSusmelj/barlowtwins).

We provide the result and the pre-trained model for SimCLR+MetaMask method based on the Cifar10 dataset. 

## Installation

**Requirements**

* Linux with Python3.7
* [PyTorch](https://pytorch.org/get-started/locally/) == 1.10.1
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* lightly == 1.0.8
* pytorch-lightning==1.5.10
* setuptools == 52.0.0
* CUDA 11.1

**Build MetaMask**
* Create a virtual environment.
```angular2html
conda create -n metamask python=3.7
conda activate metamask
```
* Install PyTorch. We use pytorch1.10.1 in our experiments. To install pytorch-1.10.1:
```angular2html
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
* Install other requirements.
```angular2html
pip install lightly==1.0.8
pip install setuptools==52.0.0
pip install pytorch-lightning==1.5.10
```

## Code Structure
- **data**: Dataset files (Files for Cifar10 will be downloaded automatically).
- **models**
  - **mask_generator.py**: Code for generating masks.
  - **simsiam.py**: Code for feature extraction.
- **loss.py**: Code for BarlowTwinsLoss.
- **meta-mask.py**: Code for the training and testing pipeline.
- **utils.py**: Code for KNN prediction.

## Models
The classification accuracy (top 1) of SimCLR+MetaMask is 86.01. We provide the [model](https://drive.google.com/file/d/1Xe-3hzmR5V4M_uACk3sWFtDMTozaJ96q/view?usp=sharing) for download.


## Getting Started

### Training in Command Line

To train a model, run
```angular2html
python metamask.py
```
The parameters that can be modified are detailed in metamask.py.
### Evaluation in Command Line
To evaluate the trained models, run
```angular2html
python metamask.py --ckpt-path "path to the checkpoint file" --eval-only
```
