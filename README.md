> reference: https://github.com/achen353/Face-Mask-Detector 
# Idea
This repo aim to compare the performance between different scale dataset with pre-trained/non-pretrained model and different fine-tuned strategy.   

# Data
## Source
We use below two datasets as training and validation data.
- MaskedFace-Net [(MFN)](https://github.com/cabani/MaskedFace-Net)
- Flickr-Faces-HQ Dataset[(FFHQ)](https://github.com/NVlabs/ffhq-dataset)

We use the images from google search to build a testing set that has the different distribution from training and validation set.

## Structure
The dataset path should contain below three directories
- train
- valid
- test

Each above directory should contain below directories and each directory contains the images
- face_with_mask_correct
- face_with_mask_incorrect
- face_no_mask

## Different Scale
### Large
We assume the large scale dataset has 50,000 ~ 60,000 training images for each class.
### Small
We assume the small scale dataset has 200 training images for each class.

# Model
- CV model: MobileNetV3_large
- A Classifier follow MobileNetV3_large 

# Usage
```shell
# load the pre-train weight of imagenet
python src/train.py --loaded --dataset_path=<dataset_path>nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN
# fixed the weight of the CV model
python src/train.py --fixed --dataset_path=<dataset_path>
# load the pre-train weight of imagenet and fixed the CV model
python src/train.py --loaded --fixed --dataset_path=<dataset_path>
python src/train.py --dataset_path=<dataset_path>
```
