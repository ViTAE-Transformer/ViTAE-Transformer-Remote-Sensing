import BboxToolkit as bt
import os
import json
import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from multiprocessing import Pool, Manager
from functools import partial


def add_parser(parser):
    #argument for processing
    parser.add_argument('--base_json', type=str, default=None,
                        help='json config file for split images')

    # arguments for loading data
    parser.add_argument('--load_type', type=str, help='dataset and save form')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotations')
    parser.add_argument('--classes', type=str, default=None,
                        help='the classes to load, a filepath or classes joined by `|`')
    parser.add_argument('--prior_annfile', type=str, default=None,
                        help='prior annotations merge to data')
    parser.add_argument('--merge_type', type=str, default='addition',
                        help='prior annotations merging method')
    parser.add_argument('--load_nproc', type=int, default=10,
                        help='the procession number for loading data')

    # arguments for selecting content
    parser.add_argument('--skip_empty', action='store_true',
                        help='whether show images without objects')
    parser.add_argument('--random_vis', action='store_true',
                        help='whether to shuffle the order of images')
    parser.add_argument('--ids', type=str, default=None,
                        help='choice id to visualize')
    parser.add_argument('--show_off', action='store_true',
                        help='stop showing images')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='whether to save images and where to save images')
    parser.add_argument('--vis_nproc', type=int, default=10,
                        help='the procession number for visualizing')

    # arguments for visualisation
    parser.add_argument('--shown_btype', default=None,
                        help='the bbox type shown in images')
    parser.add_argument('--shown_names', type=str, default=None,
                        help='class names shown in picture')
    parser.add_argument('--score_thr', type=float, default=0.2,
                        help='the score threshold for bboxes')
    parser.add_argument('--colors', type=str, default='green',
                        help='the thickness for bboxes')
    parser.add_argument('--thickness', type=float, default=2.,
                        help='the thickness for bboxes')
    parser.add_argument('--text_off', action='store_true',
                        help='without text visualization')
    parser.add_argument('--font_size', type=float, default=10,
                        help='the thickness for font')
    parser.add_argument('--wait_time', type=int, default=0,
                        help='wait time for showing images')


def abspath(path):
    if isinstance(path, (list, tuple)):
        return type(path)([abspath(p) for p in path])
    if path is None:
        return path
    if isinstance(path, str):
        return osp.abspath(path)
    raise TypeError('Invalid path type.')


def parse_args():
    parser = argparse.ArgumentParser(description='visualization')
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
    assert args.img_dir is not None, "argument img_dir can't be None"
    args.img_dir = abspath(args.img_dir)
    args.ann_dir = abspath(args.ann_dir)
    if args.classes is not None and osp.isfile(args.classes):
        args.classes = abspath(args.classes)
    assert args.prior_annfile is None or args.prior_annfile.endswith('.pkl')
    args.prior_annfile = abspath(args.prior_annfile)
    args.ids = abspath(args.ids)
    assert args.merge_type in ['addition', 'replace']
    assert args.save_dir or (not args.show_off)
    args.save_dir = abspath(args.save_dir)
    assert args.shown_btype in [None, 'hbb', 'obb', 'poly']
    if args.shown_names is not None and osp.isfile(args.shown_names):
        args.shown_names = abspath(args.shown_names)
    if args.colors is not None and osp.isfile(args.colors):
        args.colors = abspath(args.colors)
    return args


def single_vis(task, btype, class_names, colors, thickness, text_off, font_size, show_off,
               wait_time, lock, prog, total):
    imgpath, out_file, bboxes, labels, scores = task
    bboxes = bt.bbox2type(bboxes, btype) if btype else bboxes
    bt.imshow_bboxes(imgpath, bboxes, labels, scores,
                     class_names=class_names,
                     colors=colors,
                     thickness=thickness,
                     with_text=(not text_off),
                     font_size=font_size,
                     show=(not show_off),
                     wait_time=wait_time,
                     out_file=out_file)

    lock.acquire()
    prog.value += 1
    msg = f'({prog.value/total:3.1%} {prog.value}:{total})'
    msg += ' - '  + f"Filename: {osp.split(imgpath)[-1]}"
    print(msg)
    lock.release()


def main():
    args = parse_args()

    print(f'{args.load_type} loading!')
    load_func = getattr(bt.datasets, 'load_'+args.load_type)
    contents, classes = load_func(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        classes=args.classes,
        nproc=args.load_nproc)
    if args.prior_annfile is not None:
        prior, _ = bt.load_pkl(args.prior_annfile, classes=classes)
        bt.merge_prior_contents(contents, prior, merge_type=args.merge_type)

    shown_names = classes if args.shown_names is None \
            else bt.get_classes(args.shown_names)
    assert len(shown_names) == len(classes)

    if isinstance(args.ids, (list, type(None))):
        ids = args.ids
    elif isinstance(args.ids, str):
        if osp.isfile(args.ids):
            with open(args.ids, 'r') as f:
                ids = [l.strip() for l in f]
        else:
            ids = args.ids.split('|')
    else:
        raise TypeError('Wrong base_json input in `ids`')

    tasks, max_label = [], 0
    for content in contents:
        if ids is not None and content['id'] not in ids:
            pass

        imgpath = osp.join(args.img_dir, content['filename'])
        out_file = osp.join(args.save_dir, content['filename']) \
                if args.save_dir else None
        if 'ann' in content:
            ann = content['ann']
            bboxes = ann['bboxes']
            labels = ann['labels']
            scores = ann.get('scores', None)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float)
            labels = np.zeros((0, ), dtype=np.int)
            scores = None

        if (scores is not None) and (args.score_thr > 0):
            bboxes = bboxes[scores > args.score_thr]
            labels = labels[scores > args.score_thr]
            scores = scores[scores > args.score_thr]

        if args.skip_empty and bboxes.size == 0:
            continue

        if labels.size > 0:
            max_label = max(max_label, labels.max())
        tasks.append((imgpath, out_file, bboxes, labels, scores))

    if args.colors == 'random':
        args.colors = bt.random_colors(max_label + 1)

    if args.random_vis:
        shuffle(tasks)

    if args.save_dir and (not osp.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    if args.show_off:
        plt.switch_backend('Agg')

    manager = Manager()
    _vis_func = partial(single_vis,
                        btype=args.shown_btype,
                        class_names=shown_names,
                        colors=args.colors,
                        thickness=args.thickness,
                        text_off=args.text_off,
                        font_size=args.font_size,
                        show_off=args.show_off,
                        wait_time=args.wait_time,
                        lock=manager.Lock(),
                        prog=manager.Value('i', 0),
                        total=len(tasks))
    if args.show_off and args.vis_nproc > 1:
        pool = Pool(args.vis_nproc)
        pool.map(_vis_func, tasks)
        pool.close()
    else:
        list(map(_vis_func, tasks))

    if args.save_dir:
        arg_dict = vars(args)
        arg_dict.pop('base_json', None)
        with open(osp.join(args.save_dir, 'vis_config.json'), 'w') as f:
            json.dump(arg_dict, f, indent=4)


if __name__ == '__main__':
    main()
