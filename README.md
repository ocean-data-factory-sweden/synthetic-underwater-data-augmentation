# A methodology to detect deepwater corals using Generative Adversarial Networks

## Data
All data and models are published at the Swedish National Data Service under the DOI: https://doi.org/10.5878/hp35-4809.

## Data Augmentation


### Frame Tracking Data Augmentation

?????


### Synthetic Data Augmentation
Follow the steps below to reproduce the synthetic data augmentation experiment using StyleGAN2 and DiffAugment.
#### Step 1: Set up the environment

Clone the PyTorch implementation of StyleGAN2 with DiffAugment from the GitHub repository [1]:

```git clone https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2-pytorch```

#### Step 2: Train StyleGAN2
Train the StyleGAN2 model with the following hyperparameters:

- Image size: $512\times512$
- Optimizer: Adam with momentum parameters $\beta_1=0$, $\beta_2=0.99$
- Learning rate $0.002$ except for the mapping network which which used $100$ times lower learning rate
- Equalized learning rate approach: Enabled [2]
- Objective function: Improved loss from the original GAN paper, $R_1$ regularization, and regularization parameter $\gamma = 10$
- Activation function: Leaky ReLU with slope set to $\alpha=0.2$
- Batch size: $8$
- Training length: $500k$ image iterations (approximately $1222$ epochs)

#### Step 3: Monitor training and select the final model
During training, generate images every $40k$ iteration. Observe the quality of the generated images and select the weights for the final model when the quality stops improving. Use the exponential moving average of the generator weights with decay $0.999$.

#### Step 4: Apply DiffAugment
Use the PyTorch implementation of DiffAugment provided by the paper [1]. Apply the following augmentation techniques:

- Color: Adjust brightness, saturation, and contrast
- Translation: Resize the image and pad the remaining pixels with zeros
- Cutout: Cut out a random square of the image and pad it with zeros

Use all three transformations as recommended by the authors when training with limited data.






## Object Detection

#### Step 1: Prepare the datasets

- Collect 3599 ($80$%) of the initial and frame-tracking generated images (total of 4499) for the YOLO+FrameTrack model.
- Collect 2407 images ($90$%) of the initial and synthetically generated images (total of 2675) for the YOLO+Synthetic model.

#### Step 2: Set up YOLOv4 environment
Clone the YOLOv4 repository [3] and set up the environment as described in the official documentation.

#### Step 3: Prepare pre-trained weights
Download the pre-trained weights for the convolutional layers of the model trained on the MS COCO dataset.

#### Step 4: Configure the models
Use the default configurations for the models' training and set the width and height of the network to $512 \times 512$ pixels. 
Resize every image to this size during both training and detection.

#### Step 5: Apply data augmentation techniques
Employ the following data augmentation techniques during training:

- Random adjustments to saturation, hue, and exposure
- Mosaic (combines 4 training images into one image)
- Mixup (generates a new image by combining two random images)
- Blur (randomly blurs the background $50$% of the time)

#### Step 6: Train the models
Train the networks with the following settings:

- Batch size: $64$
- Total batch iterations: $6000$
- Mini-batch size: $2$

#### Step 7: Monitor training and select the final models
After the burn-in period, calculate the mAP@0.5 for every $4^{th}$ epoch on the validation set. Use this metric, along with the loss, to determine when to stop training.

#### References
[1] [Differentiable Augmentation for Data-Efficient GAN Training-Github](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2-pytorch)

[2] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

[3] [YOLOv4-Darknet](https://github.com/AlexeyAB/darknet)



