# med-vision-ai

This repository contains an implementation of carotid artery segmentation from ultrasound images using a U-Net architecture in PyTorch.

Main notebook:
- UNet-Carotid-Artery-Segmentation.ipynb

The notebook demonstrates the complete pipeline for medical image segmentation including data loading, preprocessing, training, validation, evaluation, and visualization.

---

## Project Description

Carotid artery segmentation from ultrasound images is an important task in medical image analysis.  
In this project, a U-Net model is trained to perform binary segmentation of the carotid artery using expert-annotated masks.

The goal is to accurately segment the carotid artery region from grayscale ultrasound images.

---

## Model Architecture

- Model: U-Net
- Input channels: 1 (grayscale images)
- Output channels: 1 (binary mask)
- Loss function: BCEWithLogitsLoss
- Optimizer: Adam
- Image size: 512 x 512

---

## Training Details

The model was trained for 10 epochs.

Training and validation loss per epoch:

Epoch [1/10] Train Loss: 1.1335 | Val Loss: 1.0435  
Epoch [2/10] Train Loss: 1.0348 | Val Loss: 0.9073  
Epoch [3/10] Train Loss: 0.9374 | Val Loss: 0.7054  
Epoch [4/10] Train Loss: 0.5743 | Val Loss: 0.7682  
Epoch [5/10] Train Loss: 0.3754 | Val Loss: 0.3121  
Epoch [6/10] Train Loss: 0.2442 | Val Loss: 0.2474  
Epoch [7/10] Train Loss: 0.1952 | Val Loss: 0.1657  
Epoch [8/10] Train Loss: 0.1659 | Val Loss: 0.1452  
Epoch [9/10] Train Loss: 0.1440 | Val Loss: 0.1872  
Epoch [10/10] Train Loss: 0.1399 | Val Loss: 0.1320  

---

## Loss Curve

<img width="691" height="393" alt="Training and Validation Loss Curve" src="https://github.com/user-attachments/assets/7e2113bf-9091-43f7-bb72-8d50f4fddb01" />

---

## Evaluation

The model was evaluated using the Dice Similarity Coefficient.

Mean Dice Score:
0.8878039050669897

---

## Segmentation Results

Comparison of an ultrasound image, its ground truth mask, and the predicted mask:

<img width="950" height="315" alt="Ground Truth vs Predicted Mask" src="https://github.com/user-attachments/assets/9773fbed-b738-485f-9036-23dacaab83b1" />

---

## Dataset

The dataset used in this project is available on Kaggle:

https://www.kaggle.com/datasets/orvile/carotid-ultrasound-images

The dataset contains ultrasound images of the common carotid artery along with expert-annotated segmentation masks.

---

## How to Run

1. Clone the repository
2. Download the dataset from Kaggle
3. Update dataset paths in the notebook
4. Open UNet-Carotid-Artery-Segmentation.ipynb
5. Run all cells to train and evaluate the model

---

## Future Work

- Train for more epochs with early stopping
- Add data augmentation
- Use Dice loss or combined BCE + Dice loss
- Experiment with Attention U-Net
- Improve evaluation with more metrics
