# CIFAR-10-MODELS

## Folders
Below is the introduction of folders in this repository.

### ResNet
A residual network implementation especially for the CIFAR-10 dataset. Number of blocks in each layer can be modified.

### LeNet
The LeNet-5 implementation variant for the CIFAR-10 dataset. It can achieve a moderate result (70% accuracy on CIFAR-10 test set).

### FD_ACC
Scripts in this folder can calculate the ResNet or LeNet architecture FID (Frechet inception distance) and accuracy of given a dataset.

### rotation_invariance
Scripts are used to calculate the rotation prediction accuracy for a given dataset.

### preprocess
Scripts in this folder can be used to preproces or sample a customized CIFAR-10 dataset.

### model
This folder contains the parameters of trained ResNet and LeNet models.

## Code Execution
Since each folder in this repository works as a Python package, module files can be executed from the PROJECT_DIR as below
```
python -m PACKAGE_NAME.MODULE_NAME
```
