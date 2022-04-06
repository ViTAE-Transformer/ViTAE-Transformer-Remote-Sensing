import numpy as np
import shapely.geometry as shgeo

from .transforms import bbox2type
from .utils import get_bbox_type


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof']
    assert get_bbox_type(bboxes1) != 'notype'
    assert get_bbox_type(bboxes2) != 'notype'
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return np.zeros((rows, 1), dtype=np.float32) \
                if is_aligned else np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = bbox2type(bboxes1, 'hbb')
    hbboxes2 = bbox2type(bboxes2, 'hbb')
    if not is_aligned:
        hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    if get_bbox_type(bboxes1) == 'hbb' and get_bbox_type(bboxes2) == 'hbb':
        overlaps = h_overlaps
        areas1 = (hbboxes1[..., 2] - hbboxes1[..., 0]) * (
            hbboxes1[..., 3] - hbboxes1[..., 1])

        if mode == 'iou':
            areas2 = (hbboxes2[..., 2] - hbboxes2[..., 0]) * (
                hbboxes2[..., 3] - hbboxes2[..., 1])
            unions = areas1 + areas2 - overlaps
        else:
            unions = areas1


    else:
        polys1 = bbox2type(bboxes1, 'poly')
        polys2 = bbox2type(bboxes2, 'poly')
        sg_polys1 = [shgeo.Polygon(p) for p in polys1.reshape(rows, -1, 2)]
        sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]

        overlaps = np.zeros(h_overlaps.shape)
        for p in zip(*np.nonzero(h_overlaps)):
            overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area

        if mode == 'iou':
            unions = np.zeros(h_overlaps.shape, dtype=np.float32)
            for p in zip(*np.nonzero(h_overlaps)):
                unions[p] = sg_polys1[p[0]].union(sg_polys2[p[-1]]).area
        else:
            unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
            if not is_aligned:
                unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def bbox_areas(bboxes):
    bbox_type = get_bbox_type(bboxes)
    assert bbox_type != 'notype'

    if bbox_type == 'hbb':
        areas = (bboxes[..., 2] - bboxes[..., 0]) * (
            bboxes[..., 3] - bboxes[..., 1])

    if bbox_type == 'obb':
        areas = bboxes[..., 2] * bboxes[..., 3]

    if bbox_type == 'poly':
        areas = np.zeros(bboxes.shape[:-1], dtype=np.float32)
        bboxes = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        for i in range(4):
            areas += 0.5 * (bboxes[..., i, 0] * bboxes[..., (i+1)%4, 1] -
                            bboxes[..., (i+1)%4, 0] * bboxes[..., i, 1])
        areas = np.abs(areas)
    return areas


def bbox_nms(bboxes, scores, iou_thr=0.5, score_thr=0.01):
    assert get_bbox_type(bboxes) != 'notype'
    order = scores.argsort()[::-1]
    order = order[scores[order] > score_thr]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        keep_bbox = bboxes[[i]]
        other_bboxes = bboxes[order[1:]]
        ious = bbox_overlaps(keep_bbox, other_bboxes)

        idx = np.where(ious <= iou_thr)[1]
        order = order[idx + 1]

    return np.array(keep, dtype=np.int64)


def bbox_area_nms(bboxes, iou_thr=0.5):
    assert get_bbox_type(bboxes) != 'notype'
    areas = bbox_areas(bboxes)
    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        keep_bbox = bboxes[[i]]
        other_bboxes = bboxes[order[1:]]
        ious = bbox_overlaps(keep_bbox, other_bboxes)

        idx = np.where(ious <= iou_thr)[1]
        order = order[idx + 1]

    return np.array(keep, dtype=np.int64)
