# CIFAR-10-MODELS

Benchmark methods in this repository are from the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9655472&tag=1).

## Methods
- GI (grayscale invariance) from EI (effective invariance)
- RI (rotation invariance) from EI
- FD (Frechet distance)
- PS (prediction score) from score_based_methods
- ES (entropy score) from score_based_methods
- RP (rotation prediction)

## Folders
Below is the introduction of folders in this repository.

### autoeval_regression


### FD_ACC
Scripts in this folder can calculate the ResNet or LeNet architecture FID (Frechet inception distance) and accuracy of given a dataset.

### EI
Scripts in this folder can be used to calculate the effective invariacne of a given dataset (grayscale & rotation invariance).

### RP
Scripts are used to calculate the rotation prediction accuracy for a given dataset.

### preprocess
Scripts in this folder can be used to preprocess or sample a customized CIFAR-10 dataset.

### model
This folder contains the weight parameters of rotation prediction fully connected layers.

### score_based_methods
It contains files to calculate the prediction score or entropy score of a dataset with a given threshold.

## Dependencies
All Python dependencies for this project are specified in `requirements.txt` and the Python interpreter version is 3.8.10. The experiments were conducted on Ubuntu 18.04 with  and 4 Geforce RTX 2080Ti GPU.

## Code Execution
Since each folder in this repository works as a Python package, module files can be executed from the PROJECT_DIR as below
```bash
python -m PACKAGE_NAME.MODULE_NAME
```
