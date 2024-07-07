
# Background Blurring using UNet Segmentation
This repository contains a project where a UNet-based segmentation model is employed to blur the background of images using the LFW (Labeled Faces in the Wild) dataset.


## Overview
In this project, we explore the application of a UNet architecture for image segmentation to blur the background of images. The model is trained using the LFW dataset, which contains labeled images of faces in various backgrounds and lighting conditions. The goal is to accurately segment the background from the main subject (face) and apply a blur effect to enhance the subject's prominence.


## Dataset
The LFW (Labeled Faces in the Wild) dataset is used in this project. It consists of already ground-truth segmentations.

## Methodology
- Data Preprocessing
Images are preprocessed to match the shapes needed for the model.

- Model Architecture
UNet: A convolutional neural network architecture well-suited for image segmentation tasks.
Encoder: Downsampling path to extract features.
Decoder: Upsampling path to generate segmentation masks.

![image](https://github.com/denisangel2k2/unet-blurring/assets/57831211/bc501e60-9261-4f56-9043-c480421d0436)

- Training
The model is trained on the LFW dataset using cross-entropy loss.
Adam optimizer with a ReduceLROnPlateau learning rate scheduler is employed to optimize the model parameters.
Also wandb was used for hyperparameter tuning.
The model was trained on a Laptop Nvidia GeForce RTX 3050 GPU.

- Inference
After training, the model can be used to predict segmentation masks for new images.
Post-processing involves applying a blur effect to the segmented background regions.

## Performance

- Mean Pixel Accuracy (mpa)
The ratio of correctly predicted pixels to the total number of pixels.

- Intersection over Union (IoU)
Measures the overlap between the predicted mask and the ground truth mask.

- Weighted Intersection over Union (Weighted IoU)
Calculates the IoU metric weighted by the frequency of each class in the masks.

## Requirements

- Python 3.x
- PyTorch

## Screenshots
Ground-truth vs Predicted
![image](https://github.com/denisangel2k2/unet-blurring/assets/57831211/2f43bcfb-a609-460f-a1bd-76d749e8a952)

![image](https://github.com/denisangel2k2/unet-blurring/assets/57831211/9c8fdfaa-f6cb-4573-a3b5-0341e034010f)
![image](https://github.com/denisangel2k2/unet-blurring/assets/57831211/3a1dce17-d105-49ac-a711-5e782d861167)

