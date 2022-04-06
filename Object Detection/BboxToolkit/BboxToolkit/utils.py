import numpy as np
from . import pi


def get_bbox_type(bboxes, with_score=False):
    dim = bboxes.shape[-1]
    if with_score:
        dim -= 1

    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim == 8:
        return 'poly'
    return 'notype'


def get_bbox_dim(bbox_type, with_score=False):
    if bbox_type == 'hbb':
        dim = 4
    elif bbox_type == 'obb':
        dim = 5
    elif bbox_type == 'poly':
        dim = 8
    else:
        raise ValueError(f"don't know {bbox_type} bbox dim")

    if with_score:
        dim += 1
    return dim


def choice_by_type(hbb_op, obb_op, poly_op, bboxes_or_type,
                   with_score=False):
    if isinstance(bboxes_or_type, np.ndarray):
        bbox_type = get_bbox_type(bboxes_or_type, with_score)
    elif isinstance(bboxes_or_type, str):
        bbox_type = bboxes_or_type
    else:
        raise TypeError(f'need np.ndarray or str,',
                        f'but get {type(bboxes_or_type)}')

    if bbox_type == 'hbb':
        return hbb_op
    elif bbox_type == 'obb':
        return obb_op
    elif bbox_type == 'poly':
        return poly_op
    else:
        raise ValueError('notype bboxes is not suppert')


def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def regular_obb(obboxes):
    x, y, w, h, theta = [obboxes[..., i] for i in range(5)]
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return np.stack([x, y, w_regular, h_regular, theta_regular], axis=-1)
