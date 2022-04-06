## Definition
There are three types of bboxes defining in BboxToolkit.

![bbox definination](definition.png)

**HBB** is denoted by the left-top point and right-bottom points.
**The last dimension of HBB should be 4**.

**OBB** is denoted by center point(x, y), width(w), height(h) and theta.
The width is the length of the longer side. The height is the length of the shorter side. Theta is the angle between the longer side and the x-axis.
**The last dimension of OBB should be 5**.

**POLY** is denoted by four point coordinates.
The order of these points doesn't matter, but the adjacent points should be a side of POLY.
**The last dimension of POLY should be 8**

## Tools

We implemented some tools using BboxToolkit in `/BboxToolkit/tools/`.

### img_split

[img_split.py](tools/img_split.py) can split large image into small patches by sliding windows. This tool is usually used on large aerial images like images in DOTA.

**important arguments**

- `--base_json`: Loads arguments from a json file. We have some default json files in `split_configs`.

- `--load_type`: Decides the loading function using in img_split. The function need to be implemented in `BboxToolkit.datasets` as `load_{load_type}`. For example, [load_dota](BboxToolkit/datasets/DOTAio.py) , [load_dior](BboxToolkit/datasets/DIORio.py), etc.

- `--img_dirs`, `--ann_dirs`: The image and annotation files' path. These arguments can have more than one inputs but should be aligned.

- `--sizes`, `--gaps`: Decide the pathes' size and overlap of splitting.

- `--save_dir`: The path to save the splitted images and annotations

**example**
```shell
python img_split.py --base_json split_configs/dota1_0/ss_train.json

# ``or``

python image_split.py --load_type dota --img_dirs {image path} --ann_dirs {annotation path} --sizes 1024 --gaps 200 --save_dir {saving path}
```

**note**: The json file starting with ss means 'single scale', and ms means 'multiple scales'.


The structure of splitted dataset:

```
save_dir
├── images
│   ├──0001_0001.png
│   ├──0001_0002.png
│   ...
│   └──xxxx_xxxx.png
│
└── annfiles
    ├── split_config.json
    ├── patch_annfile.pkl
    └── ori_annfile.pkl
```

Where `split_config.json` saves the arugments of splitting and can be reloaded in `img_splitted.py`.
The `patch_annfile.pkl` saves the annotations of patches.
The `ori_annfile.pkl` saves the annotations of large images.

### visualize

[visualize.py](tools/visualize.py) can draw different types of boxes on images.

**important arguments**

- `--base_json`: Loads arguments from a json file. We have some default json files in `vis_configs`.

- `--load_type`: Decide the loading function. Same as the `--load_type` arugments in img_split.py

- `--img_dir`, `--ann_dir`: The image and annotation files' path. 

- `--show_off`: Shut down the online visualization. Set as True when you need to save the visualized images.

- `--save_dir`: The path to save the visualized images.

- `--score_thr`: The score threshold to filter the boxes with low confidence.

- `--colors`: The colors of bboxes of different classes. It should be a string or a filepath. please refer [Visualization](##Visualzation) for details.

**example**

```shell
# load json config 
# first you need to change the arguments in json for your occasion
python visualize.py --base_json vis_configs/dota1_0/config.json

# or you can directly specify arguments in the command.
python visualize.py --load_type dota_submission --img_dir {image path} --ann_dir {annotation path} --score_thr 0.3
```

**note**: 

- If your want different colors for different classes, you can modify the `colors` in json or command. `colors` recives a filepath or str of colors splitting by `|`.

- In json, we only show the `dota_submission` case, which can visualize the detection results on dota. For other cases, you can change load_type to visualize datasets or splitted `.pkl`.

### cal_mAP

[cal_mAP.py](tools/cal_mAP.py) is used to calculate the mAP. 

**important arguments**

- `--img_dir`:  the path of images. The ground truths and results share the same images.

- `--gt_type`, `--gt_ann_dir`: The loading type and annotation path of ground truths.

- `--res_type`, `--res_ann_dir`: The loading type and annotation path of results.

- `--save_dir`: The path to save the visualized images.

- `--iou_thr`: The IoU threshold to decide whether a box is positive.

**example**

```shell
python cal_mAP.py --img_dir {image path} --gt_type dota --gt_ann_dir {dota annotation path} --res_type dota_submission --res_ann_dir {dota submission} --iou_thr 0.5
```

## Transform

All transformation functions can be found at [here](BboxToolkit/transforms.py).

It's very convenient to convert bbox type in BboxToolkit. If you know the exact type of bboxes and want convert it to another type (i.e., `hbb`, `obb`, `poly`),
you can use the transformation functions which are named in a regular form `{START_BTYPE}2{END_BTYPE}` (e.g. `obb2hbb`, `poly2obb`, ...).

If you are not sure of the bbox type, we also provide `bbox2type` which will automatically decide the bbox type be the ndarray shape and select correct transformation functions.

example
```shell
# bboxes type: np.ndarray shape: n, 8
# want to transform bboxes to obb

import BboxToolkit as bt

obbs = bt.poly2obb(bboxes)
# or
obbs = bt.bbox2type(bboxes, 'obb')
```

## Operations

BboxToolkit provides some common opertaions in box. All functions can judge the type of bboxes by ndarry shape and do the right mannar.

### [Geometry](BboxToolkit/geometry.py)

`bbox_overlaps` can calculate IoUs or IoFs between two bboxes.

`bbox_nms` do the NMS for all type of bboxes.

`bbox_area_nms` use areas of bboxes to replace the score in NMS.

### [Move](BboxToolkit/move.py)

`translate` can translate bboxes.

`flip` can flip bboxes.

`warp` can warp bboxes. Used in affine transformation or project transformation.

## Visualization

to be continue
