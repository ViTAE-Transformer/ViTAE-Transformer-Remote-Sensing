import os
import time
import numpy as np
import os.path as osp

from PIL import Image
from functools import partial
from multiprocessing import Pool
from .misc import img_exts


def load_msra_td500(img_dir, ann_dir=None, classes=None, nproc=10):
    if classes is not None:
        print('load_msra_td500 loads all objects as `text`, arguments classes is no use')
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert ann_dir is None or osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'

    imgpaths = [f for f in os.listdir(img_dir) if f[-4:] in img_exts]
    _load_func = partial(_load_msra_td500_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir)

    print('Starting loading MSRA_TD500 dataset information.')
    start_time = time.time()
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, imgpaths)
        pool.close()
    else:
        contents = list(map(_load_func, imgpaths))
    end_time = time.time()
    print(f'Finishing loading MSRA_TD500, get {len(contents)} images, ',
          f'using {end_time-start_time:.3f}s.')
    return contents, ['text']


def _load_msra_td500_single(imgfile, img_dir, ann_dir):
    img_id, _ = osp.splitext(imgfile)
    gtfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.gt')
    content = _load_msra_td500_gt(gtfile)

    imgfile = osp.join(img_dir, imgfile)
    width, height = Image.open(imgfile).size
    content.update(dict(width=width, height=height, filename=imgfile, id=img_id))
    return content


def _load_msra_td500_gt(gtfile):
    bboxes, diffs = [], []
    if gtfile is None:
        pass
    elif not osp.isfile(gtfile):
        print(f'Cannot find {gtfile}, treated as empty gtfile')
    else:
        with open(gtfile, 'r') as f:
            for line in f:
                items = line.strip().split(' ')

                diffs.append(int(items[1]))

                l_x, t_y = float(items[2]), float(items[3])
                w, h = float(items[4]), float(items[5])
                theta = -float(items[6])
                x = l_x + w / 2
                y = t_y + h / 2
                bboxes.append([x, y, w, h, theta])

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes \
            else np.zeros((0, 5), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs \
            else np.zeros((0, ), dtype=np.int64)
    labels = np.zeros((bboxes.shape[0], ), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(ann=ann)
