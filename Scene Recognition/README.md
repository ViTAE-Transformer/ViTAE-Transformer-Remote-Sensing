<h1 align="center"> An Empirical Study of Remote Sensing Pretraining </h1> 

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#usage">Usage</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#statement">Statement</a> |
</p >

## Current applications

> **Scene Recognition: Please see [Usage](#usage) for a quick start**;

> **Sementic Segmentation: Please see [Remote Sensing Pretraining for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Semantic%20Segmentation)**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

## Updates

***011/04/2022***

The baiduyun links of scene recognition models are provided.

***07/04/2022***

The paper is post on arxiv!

***06/04/2022***

The pretrained models for ResNet-50, Swin-T and ViTAEv2-S are released. The code for pretraining and scene recognition task are also provided for reference.

## Introduction

This repository contains codes, models and test results for the paper "An Empirical Study of Remote Sensing Pretraining". 

The aerial images are usually obtained by a camera in a birdview perspective lying on the planes or satellites, perceiving a large scope of land uses and land covers, whose scene is usually difficult to be interpreted since the interference of the scene-irrelevant regions and the complicated spatial distribution of land objects. Although deep learning has largely reshaped remote sensing research for aerial image understanding and made a great success. However, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining (RSP). To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and [ViTAE](https://arxiv.org/pdf/2202.10108.pdf), which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including ***#scene recognition#***, semantic segmentation, object detection, and change detection using the CNN and vision transformers backbones. 

## Results and Models

### UCM (8:2)

|Backbone | Input size | Acc@1 (μ±σ) | Model|
|-------- | ---------- | ----- | ----------|
RSP-ResNet-50-E300 | 224 × 224 | 99.48 ± 0.10  | [google](https://drive.google.com/file/d/1g2-LkineSNMYBFt8ijv_Ev0ONBfEfQYS/view?usp=sharing) & [baidu](https://pan.baidu.com/s/17WM-gEa219hpMP-W4MpGhQ?pwd=bwep) |
RSP-Swin-T-E300 | 224 × 224 | 99.52 ± 0.00 | [google](https://drive.google.com/file/d/1XDjl7ppYNTQnNIj94jiIwpoQnO90IT1V/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1JBPcDEpdoEGSz0vsKKjNuQ?pwd=sp6j) |
RSP-ViTAEv2-S-E100 | 224 × 224 | 99.90 ± 0.13 | [google](https://drive.google.com/file/d/1-tsc6qFkpZZtiLHZ7mvEcV6UjgYQbHtv/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1OIY9URCkGjzxmmvNFRwlTA?pwd=sjvw) |

### AID (2:8)

|Backbone | Input size | Acc@1 (μ±σ) | Model|
|-------- | ---------- | ----- | ----------|
RSP-ResNet-50-E300 | 224 × 224 | 96.81 ± 0.03  | [google](https://drive.google.com/file/d/1Ibp_AuwvJHHGCGNeEK2Qbk9_-Zt2CRys/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1FQhMAFkSYL999afPt7OfLQ?pwd=nt3b) |
RSP-Swin-T-E300 | 224 × 224 | 96.89 ± 0.08 | [google](https://drive.google.com/file/d/1f66ToOZUAUG4F-R2W3E9-ITf3tR40Qg0/view?usp=sharing) & [baidu](https://pan.baidu.com/s/16HJ_UFgZIMvb9C8aHSD6bA?pwd=p79q) |
RSP-ViTAEv2-S-E100 | 224 × 224 | 96.91 ± 0.06 | [google](https://drive.google.com/file/d/1x27PF3Qewty8u--ssn8g1llpOhvVCjdf/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1Jnlv6o2HjFkg6xfLhm09AA?pwd=hfvr) |

### AID (5:5)

|Backbone | Input size | Acc@1 (μ±σ) | Model|
|-------- | ---------- | ----- | ----------|
RSP-ResNet-50-300 | 224 × 224 | 97.89 ± 0.08  | [google](https://drive.google.com/file/d/1AG1Yo7_KvBluy7HGeajdS_dczpqa6no2/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1R0uiJCukAJ7qwNjme2D6AQ?pwd=45yq) |
RSP-Swin-T-E300 | 224 × 224 | 98.30 ± 0.04 | [google](https://drive.google.com/file/d/1cCNIcIPDsvc2oM6vSdvlZRwL0JcCHL-j/view?usp=sharing) & [baidu](https://pan.baidu.com/s/12Z0tzP8ie75O9jqY1vEFYg?pwd=efds) |
RSP-ViTAEv2-S-E100 | 224 × 224 | 98.22 ± 0.09 | [google](https://drive.google.com/file/d/1MtOlWukreMCtx5c4W4bS_YaRrZtIzFLD/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1ODFQpst-ytNNEzVZR3ewBQ?pwd=uc56) |

### NWPU-RESISC (1:9)

|Backbone | Input size | Acc@1 (μ±σ) | Model|
|-------- | ---------- | ----- | ----------|
RSP-ResNet-50-E300 | 224 × 224 | 93.93 ± 0.10  | [google](https://drive.google.com/file/d/1IIChmwXBzNqjbOIULeTJ-zBITSoNq0u-/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1EEw3CPAsdW2bKjJllpshVQ?pwd=np2x) |
RSP-Swin-T-E300 | 224 × 224 | 93.02 ± 0.12 | [google](https://drive.google.com/file/d/1dDhljOeGKdv3exjlrKKLVIVWVHZAXbV_/view?usp=sharing) & [baidu](https://pan.baidu.com/s/11aBRnPkcSIloia-AYRLXAA?pwd=8q5a) |
RSP-ViTAEv2-S-E100 | 224 × 224 | 94.41 ± 0.11 | [google](https://drive.google.com/file/d/1uGj8w7Io2_RLtktGapN1twN54itHaeQa/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1vPcU7G4ngxBtgr54UNB9aA?pwd=af5w)  |

### NWPU-RESISC (2:8)

|Backbone | Input size | Acc@1 (μ±σ) | Model|
|-------- | ---------- | ----- | ----------|
RSP-ResNet-50-E300 | 224 × 224 | 95.02 ± 0.06  | [google](https://drive.google.com/file/d/1ITmL4Hz3KXaSVxyFUHWuTylTvs4CLmuZ/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1YjN5rXZ_-Fz7-G7UtZl4NA?pwd=4f6i) |
RSP-Swin-T-E300 | 224 × 224 | 94.51 ± 0.05 | [google](https://drive.google.com/file/d/1h6Oy7j651uNClMz-PkqlPMCgZut2nQGF/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1e5Zlpf1hAyS-ChWxybVy_A?pwd=rx9w) |
RSP-ViTAEv2-S-E100 | 224 × 224 | 95.60 ± 0.06 | [google](https://drive.google.com/file/d/1ASfFu997r6NKVg1W88_36H-HMQvTyhhD/view?usp=sharing) & [baidu](https://pan.baidu.com/s/1Phs9d1-byXn2J6yo6J84ng?pwd=mikc) |

## Usage

### Installation

1. Create a conda virtual environment and activate it

```
conda create -n rsp python=3.8 -y
conda activate rsp
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
```
- Install apex (optional)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- Install other requirements:

```
pip install pyyaml yacs pillow
```

2. Clone this repo

```
git clone https://github.com/DotWang/Remote-Sensing-Pretraining.git
```

### Data Preparation

We use the MillionAID dataset for pretraining, and fine tune the pretrained model on UCM/AID/NWPU-RESISC45 datasets. For each dataset, we firstly merge all images together, and then split them to training and validation sets, where their information are separately recoded in  `train_label.txt` and `valid_label.txt`. Note we only consider the third-level categories (totally 51 classes) for MillionAID dataset. The form in `train_label.txt` is exemplified as

```
P0960374.jpg dry_field 0
P0973343.jpg dry_field 0
P0235595.jpg dry_field 0
P0740591.jpg dry_field 0
P0099281.jpg dry_field 0
P0285964.jpg dry_field 0
...
```
Here, 0 is the training id of category for corresponded image.

### Training

* For pretraining, take ResNet-50 as an example, training on MillionAID dataset with 4 GPU and 512 batch size

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 6666 main.py \
--dataset 'millionAID' --model 'resnet' --exp_num 1 \
--batch-size 128 --epochs 300 --img_size 224 --split 100 \
--lr 5e-4  --weight_decay 0.05 --gpu_num 4 \
--output [model save path]
```

* When repeatedly finetuning the pretrained ViTAE model on AID dataset with the setting of (2:8) in 5 times
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 7777 main.py \
--dataset 'aid' --model 'vitae_win' --ratio 28 --exp_num 5 \
--batch-size 64 --epochs 200 --img_size 224 --split 1 \
--lr 5e-4  --weight_decay 0.05 --gpu_num 1 \
--output [model save path] \
--pretrained [pretraind vitae path]
```

### Inference

* Evaluate the existing model
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 8888 main.py \
--dataset 'nwpuresisc' --model 'vitae_win' --ratio 28 --exp_num 5 \
--batch-size 64 --epochs 200 --img_size 224 --split 100 \
--lr 5e-4  --weight_decay 0.05 --gpu_num 1 \
--output [log save path] \
--resume [model load path] \
--eval
```

*Note: When pretraining the Swin model, please uncomment `_update_config_from_file(config, args.cfg)` in `config.py`, and add*
```
--cfg configs/swin_tiny_patch4_window7_224.yaml
```

## Other Links

> **Sementic Segmentation: Please see [Remote Sensing Pretraining for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Semantic%20Segmentation)**;

> **Object Detection: Please see [Remote Sensing Pretraining for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Object%20Detection)**;

> **Change Detection: Please see [Remote Sensing Pretraining for Change Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)**;

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .

## References

The codes of Pretraining & Recognition part mainly from [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
