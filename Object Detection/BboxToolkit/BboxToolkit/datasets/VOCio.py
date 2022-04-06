import os
import time
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image
from multiprocessing import Pool
from functools import partial

from .misc import get_classes, img_exts


def load_voc(img_dir, ann_dir=None, classes=None, nproc=10):
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert ann_dir is None or osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'
    classes = get_classes('VOC' if classes is None else classes)
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    contents = []
    print(f'Starting loading VOC dataset information.')
    start_time = time.time()
    _load_func = partial(_load_voc_single,
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
    print(f'Finishing loading VOC dataset, get {len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')

    return contents, classes


def _load_voc_single(imgfile, img_dir, ann_dir, cls2lbl):
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    xmlfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.xml')
    content = _load_voc_xml(xmlfile, cls2lbl)

    if not ('width' in content and 'height' in content):
        imgpath = osp.join(img_dir, imgfile)
        size = Image.open(imgpath).size
        content.update(dict(width=size[0], height=size[1]))
    content.update(dict(filename=imgfile, id=img_id))
    return content


def _load_voc_xml(xmlfile, cls2lbl):
    content, bboxes, labels, diffs, truncs = \
            dict(), list(), list(), list(), list()

    if xmlfile is None:
        pass
    elif not osp.isfile(xmlfile):
        print(f"Can't find {xmlfile}, treated as empty xmlfile")
    else:
        tree = ET.parse(xmlfile)
        root = tree.getroot()

        size = root.find('size')
        if size is not None:
            content['width'] = int(size.find('width').text)
            content['height'] = int(size.find('height').text)

        for obj in root.findall('object'):
            cls = obj.find('name').text.lower()
            if cls not in cls2lbl:
                continue
            labels.append(cls2lbl[cls])

            diff = obj.find('difficult')
            diffs.append(0 if diff is None else int(diff.text))

            trunc = obj.find('truncated')
            truncs.append(0 if trunc is None else int(trunc.text))

            bnd_box = obj.find('bndbox')
            bboxes.append([
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text)
            ])

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes \
            else np.zeros((0, 4), dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) if labels \
            else np.zeros((0, ), dtype=np.int64)
    diffs = np.array(diffs, dtype=np.int64) if diffs \
            else np.zeros((0, ), dtype=np.int64)
    truncs = np.array(truncs, dtype=np.int64) if truncs \
            else np.zeros((0, ), dtype=np.int64)
    anns = dict(bboxes=bboxes, labels=labels, diffs=diffs,
                truncs=truncs)
    content['ann'] = anns
    return content
