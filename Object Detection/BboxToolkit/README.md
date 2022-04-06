# BboxToolkit

BboxToolkit is a light codebase collecting some practical functions for the special-shape detection, such as oriented detection.
The whole project is written by python, which can run in different platform without compliation. 
We use this project to support the oriented detection benchmark [OBBDetection](http://github.com/jbwang1997/OBBDetection).

**news**: We are now developing the BboxToolkit v2.0 intended to support the new OBBDetection based on MMdetection v2.10.

## Main Features

- **Various type of Bboxes**

    We define three different type of bounding boxes in BboxToolkit. They are horizontal bounding boxes (HBB), oriented bounding boxes (OBB), and 4 point polygon (POLY). Each type of boxes can convert to others easily.

- **Convinence for usage**

    The functions in BboxToolkit will decide the box type according to the input shape. There is no need to concern about the input box type, when use BboxToolkit.

## License

This project is released under the [Apache 2.0 license](LICENSE)

## Installation

BboxToolkit requires following dependencies:

+ Python > 3
+ Numpy
+ Opencv-python
+ Shapely
+ Terminaltables
+ Pillow

BboxToolkit will automatically install dependencies when you install, so this section is mostly for your reference.

```shell
git clone https://github.com/jbwang1997/BboxToolkit
cd BboxToolkit
pip install -v -e . # or "python setup.py develop"
```

## Usage

Please reference [USAGE.md](USAGE.md) for detail.

## Ackownledgement

BboxToolkit refers to [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit), [MMCV](https://github.com/open-mmlab/mmcv), and [MMDetection](https://github.com/open-mmlab/mmdetection). [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) is the toolbox for [DOTA](https://arxiv.org/abs/1711.10398) Dataset [MMCV](https://github.com/open-mmlab/mmcv) is a foundational python library for computer vision. [MMDetection](https://github.com/open-mmlab/mmdetection) is an open source object detection toolbox based on PyTorch.
