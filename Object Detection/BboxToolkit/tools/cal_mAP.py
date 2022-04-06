import BboxToolkit as bt
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='calculate mAP')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--gt_type', type=str,
                        help='ground truth dataset type')
    parser.add_argument('--gt_ann_dir', type=str,
                        help='path to ground truth annotations')
    parser.add_argument('--res_type', type=str,
                        help='results dataset type')
    parser.add_argument('--res_ann_dir', type=str,
                        help='path to results annotations')
    parser.add_argument('--classes', nargs='+', type=str, default=None,
                        help='the classes, and order for loading data')
    parser.add_argument('--nproc', type=int, default=10,
                        help='the procession number')

    parser.add_argument('--ign_diff', type=int, default=1,
                        help='ignore diffcult object in dataset')
    parser.add_argument('--iou_thr', type=float, default=0.5,
                        help='iou threshold for calculating mAP')
    parser.add_argument('--voc_metric', type=str, default='07',
                        help='use voc07 metric or voc12 metric')
    args = parser.parse_args()

    assert args.img_dir is not None, "argument img_dir can't be None"
    assert args.gt_type is not None, "argument gt_type can't be None"
    assert args.gt_ann_dir is not None, "argument gt_ann_dir can't be None"
    assert args.res_type is not None, "argument res_type can't be None"
    assert args.res_ann_dir is not None, "argument res_ann_dir can't be None"
    assert args.voc_metric in ['07', '12']
    return args


def main():
    args = parse_args()

    print('Loading ground truth and results information')
    gt_load_func = getattr(bt.datasets, 'load_'+args.gt_type)
    res_load_func = getattr(bt.datasets, 'load_'+args.res_type)
    gt_infos, gt_cls = gt_load_func(
        img_dir=args.img_dir,
        ann_dir=args.gt_ann_dir,
        classes=args.classes,
        nproc=args.nproc)
    res_infos, res_cls = res_load_func(
        img_dir=args.img_dir,
        ann_dir=args.res_ann_dir,
        classes=args.classes,
        nproc=args.nproc)
    bt.change_cls_order(res_infos, res_cls, gt_cls)

    print('Parsing ground truth and results information')
    id_mapper = {info['id']: i for i, info in enumerate(res_infos)}
    gts, res = [], []
    for gt_info in gt_infos:
        img_id = gt_info['id']
        res_info = res_infos[id_mapper[img_id]]
        assert 'scores' in res_info['ann'], \
                "f{args.res_type} don't have scores information"

        res_bboxes = res_info['ann']['bboxes']
        res_labels = res_info['ann']['labels']
        res_scores = res_info['ann']['scores']
        res_dets = np.concatenate(
            [res_bboxes, res_scores[..., None]], axis=1)
        res_dets = [res_dets[res_labels == i] for i in range(len(gt_cls))]
        res.append(res_dets)

        gt_bboxes = gt_info['ann']['bboxes']
        gt_labels = gt_info['ann']['labels']
        diffs = gt_info['ann'].get(
            'diffs', np.zeros(gt_bboxes.shape[0], dtype=np.int))
        gt_ann = {}
        if args.ign_diff > 0:
            gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
            gt_ann['labels_ignore'] = gt_labels[diffs == 1]
            gt_bboxes = gt_bboxes[diffs == 0]
            gt_labels = gt_labels[diffs == 0]
        gt_ann['bboxes'] = gt_bboxes
        gt_ann['labels'] = gt_labels
        gts.append(gt_ann)

    print('Starting calculating mAP')
    bt.eval_map(res, gts,
                iou_thr=args.iou_thr,
                use_07_metric=args.voc_metric=='07',
                nproc=args.nproc,
                dataset=gt_cls)


if __name__ == '__main__':
    main()
