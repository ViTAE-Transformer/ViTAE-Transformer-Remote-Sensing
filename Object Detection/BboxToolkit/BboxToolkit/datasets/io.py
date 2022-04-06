import os
import os.path as osp
import pickle
import time
import numpy as np

from multiprocessing import Pool

from ..utils import get_bbox_dim
from .misc import read_img_info, change_cls_order, get_classes


def load_imgs(img_dir, ann_dir=None, classes=None, nproc=10,
              def_bbox_type='poly'):
    assert def_bbox_type in ['hbb', 'obb', 'poly', None]
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    if ann_dir is not None:
        print('ann_dir is no use in load_imgs function')

    print('Starting loading images information')
    start_time = time.time()
    imgpaths = [osp.join(img_dir, imgfile)
                for imgfile in os.listdir(img_dir)]
    if nproc > 1:
        pool = Pool(nproc)
        infos = pool.map(read_img_info, imgpaths)
        pool.close()
    else:
        infos = list(map(read_img_info, imgpaths))

    if def_bbox_type is not None:
        for info in infos:
            if info is None:
                continue
            bbox_dim = get_bbox_dim(def_bbox_type)
            bboxes = np.zeros((0, bbox_dim), dtype=np.float32)
            labels = np.zeros((0, ), dtype=np.int64)
            info['ann'] = dict(bboxes=bboxes, labels=labels)
    classes = () if classes is None else classes
    end_time = time.time()
    print(f'Finishing loading images, get {len(infos)} iamges,',
          f'using {end_time-start_time:.3f}s.')
    return infos, classes


def load_pkl(ann_dir, img_dir=None, classes=None, nproc=10):
    assert osp.isfile(ann_dir), f'The {ann_dir} is not an existing pkl file!'
    assert img_dir is None or osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'

    print('Starting loading pkl information')
    start_time = time.time()
    data = pickle.load(open(ann_dir, 'rb'))
    old_classes, contents = data['cls'], data['content']

    if img_dir is not None:
        imgpaths = [osp.join(img_dir, content['filename'])
                    for content in contents]
        if nproc > 1:
            pool = Pool(nproc)
            infos = pool.map(read_img_info, imgpaths)
            pool.close()
        else:
            infos = list(map(read_img_info, imgpaths))

        for info, content in zip(infos, contents):
            content.update(info)

    if classes is None:
        classes = old_classes
    else:
        classes = get_classes(classes)
        change_cls_order(contents, old_classes, classes)
    end_time = time.time()
    print(f'Finishing loading pkl, get {len(contents)} iamges,',
          f'using {end_time-start_time:.3f}s.')
    return contents, classes


def save_pkl(save_dir, contents, classes):
    assert save_dir.endswith('.pkl')
    filepath = osp.split(save_dir)[0]
    if not osp.exists(filepath):
        os.makedirs(filepath)

    data = dict(cls=classes, content=contents)
    pickle.dump(data, open(save_dir, 'wb'))
