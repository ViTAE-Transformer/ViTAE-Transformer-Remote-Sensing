import numpy as np

from .utils import get_bbox_type, regular_obb
from .transforms import bbox2type


def translate(bboxes, x, y):
    assert get_bbox_type(bboxes) != 'notype'

    if get_bbox_type(bboxes) == 'obb':
        translated = bboxes.copy()
        translated[..., :2] = translated[..., :2] + \
                np.array([x, y], dtype=np.float32)
    else:
        dim = bboxes.shape[-1]
        translated = bboxes + \
                np.array([x, y]*int(dim/2), dtype=np.float32)
    return translated


def flip(bboxes, W, H, direction='horizontal'):
    assert get_bbox_type(bboxes) != 'notype'
    assert direction in ['horizontal', 'vertical']

    flipped = bboxes.copy()
    if get_bbox_type(bboxes) == 'poly':
        if direction == 'horizontal':
            flipped[..., 0::2] = W - bboxes[..., 0::2]
        else:
            flipped[..., 1::2] = H - bboxes[..., 1::2]

    if get_bbox_type(bboxes) == 'obb':
        if direction == 'horizontal':
            flipped[..., 0] = W - bboxes[..., 0]
        else:
            flipped[..., 1] = H - bboxes[..., 1]
        flipped[..., 4] = -flipped[..., 4]
        flipped = regular_obb(flipped)

    if get_bbox_type(bboxes) == 'hbb':
        if direction == 'horizontal':
            flipped[..., 0::4] = W - bboxes[..., 2::4]
            flipped[..., 2::4] = W - bboxes[..., 0::4]
        else:
            flipped[..., 1::4] = H - bboxes[..., 3::4]
            flipped[..., 3::4] = H - bboxes[..., 1::4]
    return flipped


def warp(bboxes, M, keep_type=False):
    ori_type = get_bbox_type(bboxes)
    assert ori_type != 'notype'
    assert M.ndim == 2

    polys = bbox2type(bboxes, 'poly')
    shape = polys.shape
    group_pts = polys.reshape(*shape[:-1], shape[-1]//2, 2)
    group_pts = np.insert(group_pts, 2, 1, axis=-1)
    warped_pts = np.matmul(group_pts, M.T)

    if M.shape[0] == 3:
        warped_pts = (warped_pts / warped_pts[..., -1:])[..., :-1]
    warped_pts = warped_pts.reshape(*shape)
    if keep_type:
        warped_pts = bbox2type(warped_pts, ori_type)
    return warped_pts
