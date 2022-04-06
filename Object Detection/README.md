<h1 align="left"> An Empirical Study of Remote Sensing Pretraining </h1> 

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

> **Object Detection: Please see [Usage](#usage)** for a quick start;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

## Updates

***06/04/2022***
- The pretrained models for ResNet-50, Swin-T and ViTAEv2-S are released. The code for pretraining and scene recognition task are also provided for reference.

## Introduction

This repository contains the pretraining and scene  codes, models and test results for the paper "An Empirical Study of Remote Sensing Pretraining". 

The aerial images are usually obtained by a camera in a birdview perspective lying on the planes or satellites, perceiving a large scope of land uses and land covers, whose scene is usually difficult to be interpreted since the interference of the scene-irrelevant regions and the complicated spatial distribution of land objects. Although deep learning has largely reshaped remote sensing research for aerial image understanding and made a great success. However, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining. To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and [ViTAE](https://arxiv.org/pdf/2202.10108.pdf), which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including scene recognition, semantic segmentation, ***#object detection#***, and change detection using the CNN and vision transformers backbones.

<figure>
<div align="center">
<img src=../Figs/det.png width="70%">
</div>
<figcaption align = "center"><b>Fig. -Visual detection results of the ORCN model with the ViTAEv2-S backbones on the DOTA testing set. LV: large vehicle. SV: small vehicle. BR: Bridge. IMP: IMP-ViTAEv2-S. RSP: RSP-ViTAEv2-S. </b></figcaption>
</figure>

## Results and Models
### DOTA

| Method | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ | -------- | --------- | ------- | --- | ------ | --- |
| ORCN| RSP-ResNet-50-E300-FPN | 1x | 76.50 | [google drive](https://drive.google.com/file/d/1gICsQ-k0j-W3v7CeV0iMvuhNLkd793Rp/view?usp=sharing) | [google drive](https://drive.google.com/file/d/15krhmAlAlveVq3lA6rk_ytkyLZGMN-kj/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1D6oAvi7D2ntjKKJD5eb0SSddQEL07UUa/view?usp=sharing) |
| ORCN| RSP-Swin-T-E300-FPN | 1x | 75.68 | [google drive](https://drive.google.com/file/d/1HQBjJN9oiilVdEnyQJc7a3RcQWVvPWKw/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1wlC6i6Ez0Eg9195GqBbzMsA5POseVuZW/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1EdsVrWeCD2k33oSQpRfv43aaZUTY4hUv/view?usp=sharing) |
| ORCN| RSP-ViTAEv2-S-E100-FPN |  1x | 77.72 | [google drive](https://drive.google.com/file/d/1QennCzqLZZZSu9QRk7VVSewPzq8MGZyh/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1hN5NBrTQTC0o0elCvKZ5mKtqMub9YZc5/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1aXhHibFWQF3nFgXHSDYotDYaXFn-8H_F/view?usp=sharing) |

### iSAID

| Method | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ | -------- | --------- | ------- | --- | ------ | --- |
| ORCN| RSP-ResNet-50-E300-FPN | 3x | 90.10 | [google drive](https://drive.google.com/file/d/1ECgLzx3tYo_j8PRK4YxfuFVzpn8zsUeF/view?usp=sharing) | [google drive](https://drive.google.com/file/d/106-9j9-shrlyVvRfSnDscyHnGDYhZgGK/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1d7OF8ziHD_0oEtDV5A6j40Xj4htKvxim/view?usp=sharing) |
| ORCN| RSP-Swin-T-E300-FPN | 3x | 89.91 | [google drive](https://drive.google.com/file/d/1h_Ue3N1doO7pOXtz3KcPkb6m1LHnLLEE/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1ajCFD2kd6Heop4UDRy7IYy_jIY-Tl6RK/view?usp=sharing) | [google drive](https://drive.google.com/file/d/12yCoe9EYc4Xm9318yNETzOJapEmD2gvz/view?usp=sharing) |
| ORCN| RSP-ViTAEv2-S-E100-FPN |  3x | 90.24 | [google drive](https://drive.google.com/file/d/1PWwYr2sY1iyIgV1PhLeDsYFjuaNRHoLX/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1UTk-LUI6NAppFwnh0bEhSpAJEa6zm0xX/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1y1pUw-QwpU6VA6kwdE_h8W_UHcrpIoy7/view?usp=sharing) |

## Usage

### Installation

Please refer to [install.md](https://github.com/jbwang1997/OBBDetection/blob/master/docs/install.md) for installation and dataset preparation

### Training 

Training the ORCN with RSP-ResNet-50 backbone on DOTA dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=50002 tools/train.py \
configs/obb/oriented_rcnn/faster_rcnn_orpn_rsp_r50_fpn_1x_dota10.py \
--launcher 'pytorch' --options 'find_unused_parameters'=True
```
### Inference

Predicting the saving detection map using RSP-Swin-T on DOTA dataset

```
python tools/test.py configs/obb/oriented_rcnn/faster_rcnn_orpn_our_rsp_swin_fpn_1x_dota10.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_swin_fpn_1x_dota10/latest.pth \
--format-only --show-dir [saved map path] \
--options save_dir=[merged result path] nproc=1
```

Evaluation the saving detection map using RSP-ViTAEv2-S on HRSC2016 dataset

```
python tools/test.py configs/obb/oriented_rcnn/faster_rcnn_orpn_our_rsp_vitae_fpn_3x_hrsc.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_vitae_fpn_3x_hrsc/latest.pth \
--out [result file] --eval 'mAP' \
--show-dir [saved map path]
```

*Note: when predicting the ViTAEv2, please create the saved map path in advance, while the saved map path and the path that can reach the result file should be constructed before the HRSC2016 testing set evaluation.*

## Other Links

> **Scene Recognition: Please see [Remote Sensing Pretraining for Scene Recognition](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Scene%20Recognition)**;

> **Sementic Segmentation: Please see [Remote Sensing Pretraining for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Semantic%20Segmentation)**;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .

## References

The codes are mainly borrowed from [OBBDetection](https://github.com/jbwang1997/OBBDetection) 

