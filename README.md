# Brain-Tumor-Segmentation

https://www.kaggle.com/code/faheemkhaskheli9/breast-tumor-segmentation

In this code i have trained segmentation model to segment tumor in breast MRI, also tumor can be either benign or malignant.

This code first merse all masks and labels, to create single mask representing tumor and its type.

Segmentation model is used to segment the tumor in Breast MRI.

# Dataset

https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

# Preprocessing

I have used Square padding, and resize both images and mask, after which images were normalized.

# Architecture

I have used UNET Architecture, with resnet as backbone.
