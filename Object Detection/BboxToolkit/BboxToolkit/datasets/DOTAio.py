import re
import os
import time
import zipfile

import os.path as osp
import numpy as np

from PIL import Image
from functools import reduce, partial
from multiprocessing import Pool
from collections import defaultdict

from .io import load_imgs
from .misc import get_classes, img_exts
from ..utils import get_bbox_type
from ..transforms import bbox2type


def load_dota(img_dir, ann_dir=None, classes=None, nproc=10):
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert ann_dir is None or osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'
    classes = get_classes('DOTA' if classes is None else classes)
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    print('Starting loading DOTA dataset information.')
    start_time = time.time()
    _load_func = partial(_load_dota_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir,
                         cls2lbl=cls2lbl)
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    contents = [c for c in contents if c is not None]
    end_time = time.time()
    print(f'Finishing loading DOTA, get {len(contents)} iamges,',
          f'using {end_time-start_time:.3f}s.')

    return contents, classes


def _load_dota_single(imgfile, img_dir, ann_dir, cls2lbl):
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size
    txtfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.txt')
    content = _load_dota_txt(txtfile, cls2lbl)

    content.update(dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def _load_dota_txt(txtfile, cls2lbl):
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    if items[8] not in cls2lbl:
                        continue
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(cls2lbl[items[8]])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
            np.zeros((0, 8), dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) if labels else \
            np.zeros((0, ), dtype=np.int64)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
            np.zeros((0, ), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)


def load_dota_submission(ann_dir, img_dir=None, classes=None, nproc=10):
    classes = get_classes('DOTA' if classes is None else classes)
    assert osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'
    assert img_dir is None or osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'

    file_pattern = r'Task[1|2]_(.*)\.txt'
    cls2file_mapper = dict()
    for f in os.listdir(ann_dir):
        match_objs = re.match(file_pattern, f)
        if match_objs is None:
            fname, _ = osp.splitext(f)
            cls2file_mapper[fname] = f
        else:
            cls2file_mapper[match_objs.group(1)] = f

    print('Starting loading DOTA submission information')
    start_time = time.time()
    infos_per_cls = []
    for cls in classes:
        if cls not in cls2file_mapper:
            infos_per_cls.append(dict())
        else:
            subfile = osp.join(ann_dir, cls2file_mapper[cls])
            infos_per_cls.append(_load_dota_submission_txt(subfile))

    if img_dir is not None:
        contents, _ = load_imgs(img_dir, nproc=nproc, def_bbox_type='poly')
    else:
        all_id = reduce(lambda x, y: x|y, [d.keys() for d in infos_per_cls])
        contents = [{'id':i} for i in all_id]

    for content in contents:
        bboxes, scores, labels = [], [], []
        for i, infos_dict in enumerate(infos_per_cls):
            infos = infos_dict.get(content['id'], dict())
            bboxes.append(infos.get('bboxes', np.zeros((0, 8), dtype=np.float32)))
            scores.append(infos.get('scores', np.zeros((0, ), dtype=np.float32)))
            labels.append(np.zeros((bboxes[-1].shape[0], ), dtype=np.int64) + i)

        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)
        content['ann'] = dict(bboxes=bboxes, labels=labels, scores=scores)
    end_time = time.time()
    print(f'Finishing loading DOTA submission, get {len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')
    return contents, classes


def _load_dota_submission_txt(subfile):
    if not osp.isfile(subfile):
        print(f"Can't find {subfile}, treated as empty subfile")
        return dict()

    collector = defaultdict(list)
    with open(subfile, 'r') as f:
        for line in f:
            img_id, score, *bboxes = line.split(' ')
            bboxes_info = bboxes + [score]
            bboxes_info = [float(i) for i in bboxes_info]
            collector[img_id].append(bboxes_info)

    anns_dict = dict()
    for img_id, info_list in collector.items():
        infos = np.array(info_list, dtype=np.float32)
        bboxes, scores = infos[:, :-1], infos[:, -1]
        bboxes = bbox2type(bboxes, 'poly')
        anns_dict[img_id] = dict(bboxes=bboxes, scores=scores)
    return anns_dict


def save_dota_submission(save_dir, id_list, dets_list, task='Task1', classes=None, with_zipfile=True):
    assert task in ['Task1', 'Task2']
    classes = get_classes('DOTA' if classes is None else classes)

    if osp.exists(save_dir):
        raise ValueError(f'The save_dir should be a non-exist path, but {save_dir} is existing')
    os.makedirs(save_dir)

    files = [osp.join(save_dir ,task+'_'+cls+'.txt') for cls in classes]
    file_objs = [open(f, 'w') for f in files]
    for img_id, dets_per_cls in zip(id_list, dets_list):
        for f, dets in zip(file_objs, dets_per_cls):
            bboxes, scores = dets[:, :-1], dets[:, -1]

            if task == 'Task1':
                if get_bbox_type(bboxes) == 'poly' and bboxes.shape[-1] != 8:
                    bboxes = bbox2type(bboxes, 'obb')
                bboxes = bbox2type(bboxes, 'poly')
            else:
                bboxes = bbox2type(bboxes, 'hbb')

            for bbox, score in zip(bboxes, scores):
                txt_element = [img_id, str(score)] + ['%.2f'%(p) for p in bbox]
                f.writelines(' '.join(txt_element)+'\n')

    for f in file_objs:
        f.close()

    if with_zipfile:
        target_name = osp.split(save_dir)[-1]
        with zipfile.ZipFile(osp.join(save_dir, target_name+'.zip'), 'w',
                             zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])
