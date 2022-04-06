import numpy as np

from .. import pi
from ..utils import regular_obb

from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection


def draw_hbb(ax,
             bboxes,
             texts,
             color,
             thickness=1.,
             font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    patches, edge_colors = [], []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        if texts is not None:
            ax.text(xmin,
                    ymin,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    fontsize=font_size,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)


def draw_obb(ax,
             bboxes,
             texts,
             color,
             thickness=1.,
             font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    bboxes = regular_obb(bboxes)
    ctr, w, h, t = np.split(bboxes, (2, 3, 4), axis=1)
    Cos, Sin = np.cos(t), np.sin(t)
    vec1 = np.concatenate(
        [-w/2 * Cos, w/2 * Sin], axis=1)
    vec2 = np.concatenate(
        [-h/2 * Sin, -h/2 * Cos], axis=1)
    anchors = ctr + vec1 + vec2
    angles = -t * 180 / pi
    new_obbs = np.concatenate([anchors, w, h, angles], axis=1)

    patches, edge_colors = [], []
    for i, bbox in enumerate(new_obbs):
        x, y, w, h, angle = bbox
        if texts is not None:
            ax.text(x,
                    y,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    rotation=angle,
                    rotation_mode='anchor',
                    fontsize=font_size,
                    transform_rotates_text=True,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Rectangle((x, y), w, h, angle))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)


def draw_poly(ax,
              bboxes,
              texts,
              color,
              thickness=1.,
              font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    pts = bboxes.reshape(-1, 4, 2)
    top_pts_idx = np.argsort(pts[..., 1], axis=1)[:, :2]
    top_pts_idx = top_pts_idx[..., None].repeat(2, axis=2)
    top_pts = np.take_along_axis(pts, top_pts_idx, axis=1)

    x_sort_idx = np.argsort(top_pts[..., 0], axis=1)
    left_idx, right_idx = x_sort_idx[:, :1], x_sort_idx[:, 1:]
    left_idx = left_idx[..., None].repeat(2, axis=2)
    left_pts = np.take_along_axis(top_pts, left_idx, axis=1).squeeze(1)
    right_idx = right_idx[..., None].repeat(2, axis=2)
    right_pts = np.take_along_axis(top_pts, right_idx, axis=1).squeeze(1)

    x2 = right_pts[:, 1] - left_pts[:, 1]
    x1 = right_pts[:, 0] - left_pts[:, 0]
    angles = np.arctan2(x2, x1) / pi * 180

    patches, edge_colors = [], []
    for i, (pt, anchor, angle) in enumerate(zip(
        pts, left_pts, angles)):
        x, y = anchor
        if texts is not None:
            ax.text(x,
                    y,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    rotation=angle,
                    rotation_mode='anchor',
                    fontsize=font_size,
                    transform_rotates_text=True,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Polygon(pt))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)
