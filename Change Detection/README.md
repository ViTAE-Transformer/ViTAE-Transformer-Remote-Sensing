<h1 align="center"> An Empirical Study of Remote Sensing Pretraining </h1> 

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#usage">Usage</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#statement">Statement</a> |
</p >

## Current applications

> **Scene Recognition: Please see [Remote Sensing Pretraining for Scene Recognition](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Scene%20Recognition)**;

> **Sementic Segmentation: Please see [Remote Sensing Pretraining for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Semantic%20Segmentation)**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **Change Detection: Please see [Usage](#usage) for a quick start**;

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

## Updates

***011/04/2022***

The baiduyun links of change detection models are provided.

***07/04/2022***

The paper is post on arxiv!

***06/04/2022***

The pretrained models for ResNet-50, Swin-T and ViTAEv2-S are released. The code for pretraining and downstream tasks are also provided for reference.

## Introduction

This repository contains codes, models and test results for the paper "An Empirical Study of Remote Sensing Pretraining". 

The aerial images are usually obtained by a camera in a birdview perspective lying on the planes or satellites, perceiving a large scope of land uses and land covers, whose scene is usually difficult to be interpreted since the interference of the scene-irrelevant regions and the complicated spatial distribution of land objects. Although deep learning has largely reshaped remote sensing research for aerial image understanding and made a great success. However, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining. To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and [ViTAE](https://arxiv.org/pdf/2202.10108.pdf), which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including scene recognition, semantic segmentation, object detection, and ***#change detection#*** using the CNN and vision transformers backbones. 


<figure>
<div align="center">
<img src=../Figs/cd.png width="100%">
</div>
<figcaption align = "center"><b>Fig. - Visual change detection results. The first and second row separately show the change detection results of a sample image from the CDD and LEVIR datasets. Here, (a) and (k), (b) and (l) are the first and second temporals of the same regions. (c) and (m) are ground truth change annotations. (d) and (n) are the results of the IMP-ResNet-50 based BIT, while (e) and (o), (f) and (p), (f) and (o), (g) and (q), (h) and (r), (i) and (s), (g) and (t) are the results from the SeCo-ResNet-50, RSP-ResNet-50, IMP-Swin-T, RSP-Swin-T, IMP-ViTAEv2-S, and RSP-ViTAEv2-S backbones, respectively. </b></figcaption>
</figure>


## Results and Models
### CDD

| Method | Backbone |Input size  | F1  | Model |
| ------ | -------- |---------- | ------- | --- |
| BIT| RSP-ResNet-50-E300|256 × 256 | 96.00 |  [google](https://drive.google.com/file/d/1SMNY93e5zKLFtzSyCeM7I5Wb7_qC61uP/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1mg0etrKMprRKKF69373bew?pwd=29u4) |
| BIT| RSP-Swin-T-E300 |256 × 256 | 95.21 |  [google](https://drive.google.com/file/d/1GhVXtT8fhi7yfJjJFbQPt95Prw3dAZ6L/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1gSgU5ZH6Fs0-RF21navLXA?pwd=1y6s) |
| BIT| RSP-ViTAEv2-S-E100 |256 × 256  | 96.81 | [google](https://drive.google.com/file/d/1ZGmx1lgzATJwy6Wk_HRFZRrFRgFWF-S6/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1qwr1TKmiQ5LuQ5ZC3oGX3w?pwd=q3vd) |

### LEVIR

| Method | Backbone |Input size | F1 | Model |
| ------ | -------- |---------- | ------- | --- |
| BIT| RSP-ResNet-50-E300 |256 × 256| 90.10 | [google](https://drive.google.com/file/d/1TX1VCCIcH0lsj6pObXj3u5PJTXB3_18o/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1urzS_zuW1FxQzACihaCoZQ?pwd=p2vr) |
| BIT| RSP-Swin-T-E300 |256 × 256| 90.10 | [google](https://drive.google.com/file/d/1MfaYBJdeCWg2qgqyjwOYr-KJpjCVHwHe/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1h7gptokQLFA4o17CdOHUdw?pwd=ba5x) |
| BIT| RSP-ViTAEv2-S-E100 |256 × 256 | 90.93 |  [google](https://drive.google.com/file/d/1z5ge7vN6d8tXKn82W6uNNNl5DAMbLb5v/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1ew3ZItT7HtuQP273yGJphg?pwd=utr9) |

## Usage

### Installation

Please refer to [Readme.md](https://github.com/RSCD-Lab/Siam-NestedUNet/blob/master/README.md).

### Data Preparation

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [LEVIR](https://justchenhao.github.io/LEVIR/)

For LEVIR dataset, clip the images to 256 × 256 patches and change the structure to 

```
│─train
│   ├─A
│   ├─B
│   └─OUT
│─val
│   ├─A
│   ├─B
│   └─OUT
└─test
    ├─A
    ├─B
    └─OUT
```
A: t1 image

B: t2 image

OUT: Binary label map


### Training

Training the BIT with RSP-ResNet-50 backbone on CDD dataset: 

```
python train.py \
--backbone 'resnet' --dataset 'cdd' --mode 'rsp_300'
```

### Inference

Evaluation using RSP-Swin-T on LEVIR dataset

```
python eval.py \
--backbone 'swin' --dataset 'levir' --mode 'rsp_300' \
--path [model path]
```

Predicting the change detection map using RSP-ViTAEv2-S on LEVIR dataset

```
python visualization.py \
--backbone 'vitae' --dataset 'levir' --mode 'rsp_100' \
--path [model path]
```

## Other Links

> **Scene Recognition: Please see [Remote Sensing Pretraining for Scene Recognition](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Scene%20Recognition)**;

> **Sementic Segmentation: Please see [Remote Sensing Pretraining for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Semantic%20Segmentation)**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;


## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .

## References

The codes are mainly borrowed from [SNUNet-CD](https://github.com/RSCD-Lab/Siam-NestedUNet) and [BIT-CD](https://github.com/justchenhao/BIT_CD)


