import BboxToolkit as bt

import os
import cv2
import time
import json
import logging
import argparse
import datetime
import itertools
import numpy as np
import os.path as osp

from math import ceil
from functools import partial, reduce
from multiprocessing import Pool, Manager

cv2.setNumThreads(1)


def add_parser(parser):
    #argument for processing
    parser.add_argument('--base_json', type=str, default=None,
                        help='json config file for split images')
    parser.add_argument('--nproc', type=int, default=20,
                        help='the procession number')

    #argument for loading data
    parser.add_argument('--load_type', type=str, default=None,
                        help='loading function type')
    parser.add_argument('--img_dirs', nargs='+', type=str, default=None,
                        help='images dirs, must give a value')
    parser.add_argument('--ann_dirs', nargs='+', type=str, default=None,
                        help='annotations dirs, optional')
    parser.add_argument('--classes', type=str, default=None,
                        help='the classes for loading data')
    parser.add_argument('--prior_annfile', type=str, default=None,
                        help='prior annotations merge to data')
    parser.add_argument('--merge_type', type=str, default='addition',
                        help='prior annotations merging method')

    #argument for splitting image
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024],
                        help='the sizes of sliding windows')
    parser.add_argument('--gaps', nargs='+', type=int, default=[512],
                        help='the steps of sliding widnows')
    parser.add_argument('--rates', nargs='+', type=float, default=[1.],
                        help='same as DOTA devkit rate, but only change windows size')
    parser.add_argument('--img_rate_thr', type=float, default=0.6,
                        help='the minimal rate of image in window and window')
    parser.add_argument('--iof_thr', type=float, default=0.7,
                        help='the minimal iof between a object and a window')
    parser.add_argument('--no_padding', action='store_true',
                        help='not padding patches to regular size')
    parser.add_argument('--padding_value', nargs='+',type=int, default=[0],
                        help='padding value, 1 or channel number')

    #argument for saving
    parser.add_argument('--filter_empty', action='store_true',
                        help='filter out empty patches, speed up trining splitting')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='to save pkl and splitted images')
    parser.add_argument('--save_ext', type=str, default='.png',
                        help='the extension of saving images')


def abspath(path):
    if isinstance(path, (list, tuple)):
        return type(path)([abspath(p) for p in path])
    if path is None:
        return path
    if isinstance(path, str):
        return osp.abspath(path)
    raise TypeError('Invalid path type.')


def parse_args():
    parser = argparse.ArgumentParser(description='Splitting images')
    add_parser(parser)
    args = parser.parse_args()

    if args.base_json is not None:
        with open(args.base_json, 'r') as f:
            prior_config = json.load(f)

        for action in parser._actions:
            if action.dest not in prior_config or \
               not hasattr(action, 'default'):
                continue
            action.default = prior_config[action.dest]
            args = parser.parse_args()

    # assert arguments
    assert args.load_type is not None, "argument load_type can't be None"
    assert args.img_dirs is not None, "argument img_dirs can't be None"
    args.img_dirs = abspath(args.img_dirs)
    assert args.ann_dirs is None or len(args.ann_dirs) == len(args.img_dirs)
    args.ann_dirs = abspath(args.ann_dirs)
    if args.classes is not None and osp.isfile(args.classes):
        args.classes = abspath(args.classes)
    assert args.prior_annfile is None or args.prior_annfile.endswith('.pkl')
    args.prior_annfile = abspath(args.prior_annfile)
    assert args.merge_type in ['addition', 'replace']
    assert len(args.sizes) == len(args.gaps)
    assert len(args.sizes) == 1 or len(args.rates) == 1
    assert args.save_ext in bt.img_exts
    assert args.iof_thr >= 0 and args.iof_thr < 1
    assert args.iof_thr >= 0 and args.iof_thr <= 1
    assert not osp.exists(args.save_dir), \
            f'{osp.join(args.save_dir)} already exists'
    args.save_dir = abspath(args.save_dir)
    return args


def get_sliding_window(info, sizes, gaps, img_rate_thr):
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for size, gap in zip(sizes, gaps):
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width-size)/step+1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1]+size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height-size)/step+1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1]+size > height:
            y_start[-1] = height - size

        start = np.array(list(itertools.product(x_start, y_start)),
                         dtype=np.int64)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
            (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
            (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]


def get_window_obj(info, windows, iof_thr):
    bboxes = info['ann']['bboxes']
    iofs = bt.bbox_overlaps(bboxes, windows, mode='iof')

    window_anns = []
    for i in range(windows.shape[0]):
        win_iofs = iofs[:, i]
        pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

        win_ann = dict()
        for k, v in info['ann'].items():
            try:
                win_ann[k] = v[pos_inds]
            except TypeError:
                win_ann[k] = [v[i] for i in pos_inds]
        win_ann['trunc'] = win_iofs[pos_inds] < 1
        window_anns.append(win_ann)
    return window_anns


def crop_and_save_img(info, windows, window_anns, img_dir, no_padding,
                      padding_value, filter_empty, save_dir, img_ext):
    img = cv2.imread(osp.join(img_dir, info['filename']))
    patch_infos = []
    for i in range(windows.shape[0]):
        ann = window_anns[i]
        if filter_empty and (ann['bboxes'].size == 0):
            continue

        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()
        ann['bboxes'] = bt.translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        patch_info['id'] = info['id'] + f'_{i:04d}'
        patch_info['ori_id'] = info['id']

        patch = img[y_start:y_stop, x_start:x_stop]
        if not no_padding:
            height = y_stop - y_start
            width = x_stop - x_start
            if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty(
                    (height, width, patch.shape[-1]), dtype=np.uint8)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[0]
        patch_info['width'] = patch.shape[1]

        cv2.imwrite(osp.join(save_dir, patch_info['id']+img_ext), patch)
        patch_info['filename'] = patch_info['id'] + img_ext
        patch_infos.append(patch_info)

    return patch_infos


def single_split(arguments, sizes, gaps, img_rate_thr, iof_thr, no_padding,
                 padding_value, filter_empty, save_dir, img_ext, lock,
                 prog, total, logger):
    info, img_dir = arguments
    windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
    window_anns = get_window_obj(info, windows, iof_thr)
    patch_infos = crop_and_save_img(info, windows, window_anns, img_dir, no_padding,
                                    padding_value, filter_empty, save_dir, img_ext)
    assert patch_infos or (filter_empty and info['ann']['bboxes'].size == 0)

    lock.acquire()
    prog.value += 1
    msg = f'({prog.value/total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {info['filename']}"
    msg += ' - ' + f"width: {info['width']:<5d}"
    msg += ' - ' + f"height: {info['height']:<5d}"
    msg += ' - ' + f"Objects: {len(info['ann']['bboxes']):<5d}"
    msg += ' - ' + f"Windows: {windows.shape[0]:<5d}"
    msg += ' - ' + f"Patches: {len(patch_infos)}"
    logger.info(msg)
    lock.release()

    return patch_infos


def setup_logger(log_path):
    logger = logging.getLogger('img split')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = osp.join(log_path, now + '.log')
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, 'w')
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def main():
    args = parse_args()

    if args.ann_dirs is None:
        args.ann_dirs = [None for _ in range(len(args.img_dirs))]
    padding_value = args.padding_value[0] \
            if len(args.padding_value) == 1 else args.padding_value
    sizes, gaps = [], []
    for rate in args.rates:
        sizes += [int(size / rate) for size in args.sizes]
        gaps += [int(gap / rate) for gap in args.gaps]
    save_imgs = osp.join(args.save_dir, 'images')
    save_files = osp.join(args.save_dir, 'annfiles')
    os.makedirs(save_imgs)
    os.makedirs(save_files)
    logger = setup_logger(save_files)

    print('Loading original data!!!')
    infos, img_dirs = [], []
    load_func = getattr(bt.datasets, 'load_'+args.load_type)
    for img_dir, ann_dir in zip(args.img_dirs, args.ann_dirs):
        _infos, classes = load_func(
            img_dir=img_dir,
            ann_dir=ann_dir,
            classes=args.classes,
            nproc=args.nproc)
        _img_dirs = [img_dir for _ in range(len(_infos))]
        infos.extend(_infos)
        img_dirs.extend(_img_dirs)
    if args.prior_annfile is not None:
        prior_infos, _ = bt.load_pkl(args.prior_annfile, classes=classes)
        bt.merge_prior_contents(infos, prior_infos, merge_type=args.merge_type)

    print('Start splitting images!!!')
    start = time.time()
    manager = Manager()
    worker = partial(single_split,
                     sizes=sizes,
                     gaps=gaps,
                     img_rate_thr=args.img_rate_thr,
                     iof_thr=args.iof_thr,
                     no_padding=args.no_padding,
                     padding_value=padding_value,
                     filter_empty=args.filter_empty,
                     save_dir=save_imgs,
                     img_ext=args.save_ext,
                     lock=manager.Lock(),
                     prog=manager.Value('i', 0),
                     total=len(infos),
                     logger=logger)

    if args.nproc > 1:
        pool = Pool(args.nproc)
        patch_infos = pool.map(worker, zip(infos, img_dirs))
        pool.close()
    else:
        patch_infos = list(map(worker, zip(infos, img_dirs)))

    patch_infos = reduce(lambda x, y: x+y, patch_infos)
    stop = time.time()
    logger.info(f'Finish splitting images in {int(stop - start)} second!!!')
    logger.info(f'Total images number: {len(patch_infos)}')

    print('Save information of splitted dataset!!!')
    arg_dict = vars(args)
    arg_dict.pop('base_json', None)
    with open(osp.join(save_files, 'split_config.json'), 'w') as f:
        json.dump(arg_dict, f, indent=4)
        json_str = json.dumps(arg_dict, indent=4)
        logger.info(json_str)
    bt.save_pkl(osp.join(save_files, 'ori_annfile.pkl'), infos, classes)
    bt.save_pkl(osp.join(save_files, 'patch_annfile.pkl'), patch_infos, classes)


if __name__ == '__main__':
    main()
