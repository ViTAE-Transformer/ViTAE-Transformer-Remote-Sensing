> ### :alarm_clock: The repo of the paper "An Empirical Study of Remote Sensing Pretraining" has been moved to [RSP](https://github.com/ViTAE-Transformer/RSP)
<h1 align="center">Remote Sensing</h1>


This repo contains a comprehensive list of our research works related to **Remote Sensing**. For any related questions, please contact <strong><a href="https://dotwang.github.io">Di Wang</a></strong> at [wd74108520@gmail.com](mailto:wd74108520@gmail.com) or [d_wang@whu.edu.cn](mailto:d_wang@whu.edu.cn).


## Overview

[1. An Empirical Study of Remote Sensing Pretraining [TGRS-2022]](#rsp)

<a href="https://arxiv.org/abs/2204.02825"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://ieeexplore.ieee.org/document/9782149"><img src="https://img.shields.io/badge/TGRS-Paper-red"></a> <a href="https://github.com/ViTAE-Transformer/RSP"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/RSP.svg?logo=github&label=Stars"></a>

[2. Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model [TGRS-2022]](#rvsa)

<a href="https://arxiv.org/abs/2208.03987"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://ieeexplore.ieee.org/document/9956816"><img src="https://img.shields.io/badge/TGRS-Paper-red"></a> <a href="https://github.com/https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/Remote-Sensing-RVSA.svg?logo=github&label=Stars"></a>

[3. SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [NeurIPS-2023]](#samrs)

<a href="https://arxiv.org/abs/2305.02034"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" > <a href="https://github.com/ViTAE-Transformer/SAMRS"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/SAMRS.svg?logo=github&label=Stars"></a>

## Projects

### <span id="rsp">📘 An Empirical Study of Remote Sensing Pretraining [TGRS-2022]</span>

<em>Di Wang<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Bo Du, Gui-Song Xia and Dacheng Tao</em>

[Paper](https://arxiv.org/abs/2204.02825) |  [Github Code](https://github.com/ViTAE-Transformer/RSP) | [BibTex](./assets/TGRS_2022_RSP/rsp.bib)

We train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and ViTAE, which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and remote sensing pretraining (RSP) on a series of downstream tasks including scene recognition, semantic segmentation, object detection, and change detection using the CNN and vision transformers backbones.

<img src="./assets/TGRS_2022_RSP/rsp.png" width="100%">

***

### <span id="rvsa">📘 Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model [TGRS-2022]</span>

<em>Di Wang<sup>&#8727;</sup>, Qiming Zhang<sup>&#8727;</sup>, Yufei Xu<sup>&#8727;</sup>, Jing Zhang, Bo Du, Dacheng Tao and Liangpei Zhang</em>.

[Paper](https://arxiv.org/abs/2208.03987) |  [Github Code](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA) | [BibTex](./assets/TGRS_2022_RVSA/rvsa.bib)

We resort to plain vision transformers with about 100M and make the first attempt to propose large vision models customized for RS tasks and propose a new rotated varied-size window attention (RVSA) to substitute the original full attention to handle the large image size and objects of various orientations in RS images. The RVSA could significantly reduce the computational cost and memory footprint while learn better object representation by extracting rich context from the generated diverse windows.

<img src="./assets/TGRS_2022_RVSA/rvsa.png" width="100%">

***

### <span id="samrs">📘 SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model [NeurIPS-2023]</span>

<em>Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao and Liangpei Zhang</em>.

[Paper](https://arxiv.org/abs/2305.02034) |  [Github Code](https://github.com/ViTAE-Transformer/SAMRS) | [BibTex](./assets/arXiv_2023_SAMRS/samrs.bib)

In this study, we leverage <a href="https://arxiv.org/abs/2304.02643"> SAM </a> and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS surpasses existing high-resolution RS segmentation datasets in size by several orders of magnitude, and provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects. We hope it could facilitate research in RS segmentation, particularly in large model pre-training.

<img src="./assets/arXiv_2023_SAMRS/samrs.png" width="100%">
