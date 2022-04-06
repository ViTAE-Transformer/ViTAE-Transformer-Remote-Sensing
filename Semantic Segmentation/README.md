<h1 align="left"> An Empirical Study of Remote Sensing Pretraining </h1> 

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#usage">Usage</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#statement">Statement</a> |
</p >

## Current applications

> **Scene Recognition: Please see [Remote Sensing Pretraining for Scene Recognition](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Pretraining%20%26%20SceneRecoginition)**;

> **Sementic Segmentation: Please see [Usage](#usage) for a quick start**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

## Updates

***06/04/2022***
- The pretrained models for ResNet-50, Swin-T and ViTAEv2-S are released. The code for pretraining and scene recognition task are also provided for reference.

## Introduction

This repository contains the pretraining and scene  codes, models and test results for the paper "An Empirical Study of Remote Sensing Pretraining". 

The aerial images are usually obtained by a camera in a birdview perspective lying on the planes or satellites, perceiving a large scope of land uses and land covers, whose scene is usually difficult to be interpreted since the interference of the scene-irrelevant regions and the complicated spatial distribution of land objects. Although deep learning has largely reshaped remote sensing research for aerial image understanding and made a great success. However, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining. To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and [ViTAE](https://arxiv.org/pdf/2202.10108.pdf), which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including scene recognition, ***#semantic segmentation#***, object detection, and change detection using the CNN and vision transformers backbones. 

<figure>
<div align="center">
<img src=../Figs/seg.png width="100%">
</div>
<figcaption align = "center"><b>Fig. - Segmentation maps of the UperNet with different backbones on the Potsdam dataset. (a) Ground Truth. (b) IMP-ResNet-50. (c) SeCo-ResNet-50. (d) RSP-ResNet-50. (e) IMP-Swin-T. (f) RSP-Swin-T. (g) IMP-ViTAEv2-S. (h) RSP-ViTAEv2-S. </b></figcaption>
</figure>

## Results and Models
### ISPRS Potsdam

| Method | Backbone | Crop size | Lr schd | mF1 | Config | Log | Model |
| ------ | -------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| RSP-ResNet-50-E300 | 512 × 512 | 80k | 89.94 | [google drive](https://drive.google.com/file/d/1H9QL9CGWlf0ogvf5PLTYIftYfg1SlPeQ/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1UuNoUTZK90RaCtFUO8JX2hOWhy5rUA93/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1doT7lRkkP-d4FEAZEnMGZnwZRDftQM3m/view?usp=sharing) |
| UperNet| RSP-Swin-T-E300 | 512 × 512 | 80k | 90.03 | [google drive](https://drive.google.com/file/d/1hOCvbj82Qx36cN9WGl-qn6dOX6HBh6FT/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1QjWMScAA0MGY1pcENYFGO-0JmAz4AhaL/view?usp=sharing) | [googld drive](https://drive.google.com/file/d/1HruZN0CvY9COjffqbqdMi-20aMfrmVSH/view?usp=sharing) |
| UperNet| RSP-ViTAEv2-S-E100 | 512 × 512 | 80k | 90.64 | [google drive](https://drive.google.com/file/d/11lfzupcUnXFyHqVX1iarh8DpS7yEcpaC/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1OkwppWBdWqI4pSxSWU9SzVMON7qGKeFL/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1LnSsDfSS0TrIP34GdU06Vv-CFPO73WoJ/view?usp=sharing) |

### iSAID

| Method | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | -------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| RSP-ResNet-50-E300 | 896 × 896 | 80k | 61.26 | [google drive](https://drive.google.com/file/d/1MqQliF_UpTbavfOcFQBPMDtOXHOIBpHb/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1ZMvjKeGfwWZzP3_VzHHSkhdbXP9dKyDT/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1RQ-xGJFLhV50jHV1TARNRp6Te0ceL7Ex/view?usp=sharing) |
| UperNet| RSP-Swin-T-E300 | 896 × 896 | 80k | 64.10 | [google drive](https://drive.google.com/file/d/1ivPG1-yrXHaoot6ojWjFbfpW_G0D6vCW/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1btz19JL_kfijBCBvBpvNCLmNuyEn19oZ/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1TjcLQ3GTsmoEG_6kwyAohElMKdURmvQq/view?usp=sharing) |
| UperNet| RSP-ViTAEv2-E100 | 896 × 896 | 80k | 64.25 | [google drive](https://drive.google.com/file/d/1FNEOm5K8AESWqDGJy5rpwNxh8eGx-Yzh/view?usp=sharing) | [google drive](https://drive.google.com/file/d/17YRZl_LrMb5wsH7uzh8jqKnhxyDXQcgh/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1B57j-VgW0qNm1PYX7vKjjsferZexsrmJ/view?usp=sharing) |

## Usage

### Install

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation

### Training & Evaluation

Training and evaluation the UperNet with RSP-ResNet-50 backbone on Potsdam dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/upernet/upernet_our_r50_512x512_80k_potsdam_epoch300.py \
--launcher 'pytorch'
```
### Prediction
 
Predicting the segmentaion map of RSP-Swin-T on iSAID dataset

```
python tools/test.py configs/swin/upernet_swin_tiny_patch4_window7_896x896_80k_isaid.py \
[model path] --show-dir [img save path] \
--eval mIoU 'mFscore'
```

*Note: when training the ViTAEv2, please add `--cfg-options 'find_unused_parameters'=True`*

## Other Links

> **Scene Recognition: Please see [Remote Sensing Pretraining for Scene Recognition](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Pretraining%20%26%20SceneRecoginition)**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .

## References

The codes are mainly borrowed from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 
