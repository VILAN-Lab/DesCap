# DesCap
This repository contains the main code for the paper "Region-Focused Network for Dense Captioning"


## Scene graph
We extract the scene graph by https://github.com/jz462/Large-Scale-VRD.pytorch

## Different backbones in experiments
For faster-rcnn, we take the 10-100 bounding boxes as the image features.
For DETR, we follow the setting as the paper mentioned.

## Running
```
python train.py      # for faster-rcnn backbone
python train_detr.py # for detr backbone
```




