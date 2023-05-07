# A methodology to detect deepwater corals using Generative Adversarial Networks

## Data
All data and models are published at the Swedish National Data Service under the DOI: ???

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

```bash /opt/local/bin/run_py_job.sh -e stylegan -p gpu-shannon -c 8 -s train.py -- --outdir=out_dir --data=resized_images --gpus=1 --workers 2```

```bash /opt/local/bin/run_py_job.sh -e stylegan -p gpu-shannon -c 8 generate.py --output=out_dir --num-rows=10 --num-cols=12```

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

- 3599 ($80$%) of the initial and frame-tracking generated images (total of 4499) for the YOLO+FrameTrack model.
- 2407 images ($90$%) of the initial and synthetically generated images (total of 2675) for the YOLO+Synthetic model.

#### Step 2: Set up YOLOv4 environment
Clone the YOLOv4 repository [3] and set up the environment as described in the official documentation.
```
git clone https://github.com/AlexeyAB/darknet
```
```
# change makefile to have GPU and OPENCV enabled (edit makefile to enable GPU and opencv)
  cd darknet
  sed -i 's/OPENCV=0/OPENCV=1/' Makefile
  sed -i 's/GPU=0/GPU=1/' Makefile
  sed -i 's/CUDNN=0/CUDNN=1/' Makefile
  sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```
```
# make darknet (builds darknet to use the darknet executable file to run or train object detectors)
make
```

#### Step 3: Prepare pre-trained weights
Download the pre-trained weights for the convolutional layers of the model trained on the MS COCO dataset.

```wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137```

#### Step 4: Configure Files for Training
Use the default configurations for the models' training and set the width and height of the network to $512 \times 512$ pixels. 
Resize every image to this size during both training and detection (yolo-obj.cfg).

- Edit the max_batches = classes*2000 but not less than number of training images or 6000 in case we have 3 classes
- steps = 80% of max_batches, 90% of max_batches
- network size width = 416, height = 416 or any value multiple of 32
- Change number of classes (search yolo)
- Change filters to = (classes + 5) * 3 in each convolutional before each yolo layer

```# move the custom .cfg to cfg folder
cp /yolo-obj.cfg ./cfg
```
```
# move the obj.names and obj.data files to data folder
cp /obj.names ./data
cp /obj.data  ./data
```
```
# move the train.txt and valid.txt and test.txt files data folder
cp train.txt ./data
cp valid.txt  ./data
cp test.txt  ./data
```

#### Step 5: Apply data augmentation techniques
Employ the following data augmentation techniques during training (in cfg file):

- Random adjustments to saturation, hue, and exposure
- Mosaic (combines 4 training images into one image)
- Mixup (generates a new image by combining two random images)
- Blur (randomly blurs the background $50$% of the time)

#### Step 6: Train the models
Train the networks with the following settings:

- Batch size: $64$
- Total batch iterations: $6000$
- Mini-batch size: $2$

```
 #copy over both datasets into the root directory of the Colab VM
cp /obj.zip ../
cp /test.zip ../

unzip ../test.zip -d data/
```

```
# train your custom detector
!./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137 -dont_show -map
```


#### Step 7: Monitor training and select the final models
After the burn-in period, calculate the mAP@0.5 for every $4^{th}$ epoch on the validation set. Use this metric, along with the loss, to determine when to stop training.

```
#Checking the Mean Average Precision (mAP)
./darknet detector map data/obj.data cfg/yolo-obj.cfg /backup/yolo-obj_last.weights -thresh 0.75
```

#### Step 8: Test the model

```
# test the detector 
./darknet detector test data/obj.data cfg/yolov4-obj.cfg /backup/yolov4-obj_last.weights /images/example.jpg
```

#### References
[1] [Differentiable Augmentation for Data-Efficient GAN Training-Github](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2-pytorch)

[2] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

[3] [YOLOv4-Darknet](https://github.com/AlexeyAB/darknet)



