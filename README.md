<h1 align="center"> An Empirical Study of Remote Sensing Pretraining </h1> 

<p align="center">
<a href="http://arxiv.org/abs/2204.02825"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<h5 align="center"><em>Di Wang, Jing Zhang, Bo Du, Gui-Song Xia and Dacheng Tao</em></h5>

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

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

## Updates

***011/04/2022***

The baiduyun links of pretrained models are provided.

***07/04/2022***

The paper is post on arxiv!

***06/04/2022***

The pretrained models for ResNet-50, Swin-T and ViTAEv2-S are released. The code for pretraining and downstream tasks are also provided for reference.

## Introduction

This repository contains codes, models and test results for the paper "[An Empirical Study of Remote Sensing Pretraining](http://arxiv.org/abs/2204.02825)". 

The aerial images are usually obtained by a camera in a birdview perspective lying on the planes or satellites, perceiving a large scope of land uses and land covers, whose scene is usually difficult to be interpreted since the interference of the scene-irrelevant regions and the complicated spatial distribution of land objects. Although deep learning has largely reshaped remote sensing research for aerial image understanding and made a great success. However, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining. To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and [ViTAE](https://arxiv.org./abs/2202.10108), which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including scene recognition, semantic segmentation, object detection, and change detection using the CNN and vision transformers backbones. 


<figure>
<div align="center">
<img src=Figs/aerialscene.png width="70%">
</div>
<figcaption align = "center"><b>Fig. - (a) and (b) are the natural image and aerial image belonging to the "park" category. (c) and (d) are two aerial images from the "school" category. Despite the distinct view difference of (a) and (b), (b) contains the playground that is unusual in the park scenes but usually exists in the school scenes like (d). On the other hand, (c) and (d) show different colors as well as significantly different spatial distributions of land objects like playground and swimming pool. </b></figcaption>
</figure>

## Results and Models

### MillionAID
|Backbone | Input size | Acc@1 | Acc@5 | Param(M) | Pretrained model|
|-------- | ---------- | ----- | ----- | -------- | ----------|
RSP-ResNet-50-E300 | 224 × 224 | 98.99 | 99.82| 23.6 | [google](https://drive.google.com/file/d/1K3P4_fDfcBRGqpKoSdSa6OXS4xC1xLC9/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1q7VI1wj2Vp0N5jvaXZOuCA?pwd=5npb)| [//]: & [Log](https://drive.google.com/file/d/1y1GtAs2vbLHwrIK_dVvXyGXrArhiVsIi/view?usp=sharing)|
RSP-Swin-T-E300 | 224 × 224 | 98.59 | 99.88 | 27.6| [google](https://drive.google.com/file/d/1G5wjbjIHepmT6VVOuW03bWmyvrhcfe1F/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1Hs94fI7mF12CX-Sf69gumA?pwd=7579) | [//]: & [Log](https://drive.google.com/file/d/1Ld7qxWgwOrjba08F3DxrPjfJi_wk9biR/view?usp=sharing)|
RSP-ViTAEv2-S-E100 | 224 × 224 | 98.97 |  99.88 |18.8 | [google](https://drive.google.com/file/d/1cDB69frN-NxCyoy8lghjx6NiH1JriYUc/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1riMV07wtMpmrJzEtr_T-Ag?pwd=hm8p) | [//]: & [Log](https://drive.google.com/file/d/1KEgMqSlu0-q9ZodKAH92qTvCvmcjWhKn/view?usp=sharing)|

## Usage

Please refer to [Readme.md](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/blob/main/Scene%20Recognition/README.md) for installation, dataset preparation, training and inference.

## Citation

If this repo is useful for your research, please consider citation

```
@article{wang2022rsp,
  title={An Empirical Study of Remote Sensing Pretraining},
  author={Wang, Di and Zhang, Jing and Du, Bo and Xia, Gui-Song and Tao, Dacheng},
  journal={arXiv preprint arXiv:2204.02825},
  year={2022}
}
```

## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .
