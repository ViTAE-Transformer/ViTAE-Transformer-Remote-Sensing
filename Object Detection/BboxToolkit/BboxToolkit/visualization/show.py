import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from .colors import colors_val, random_colors
from .draw import draw_hbb, draw_obb, draw_poly
from ..utils import choice_by_type

EPS = 1e-2

MAPPING = {
    6: (0, 0, 63),
    9: (0, 191, 127),
    1: (0, 63, 0),
    7: (0, 63, 127),
    8: (0, 63, 191),
    3: (0, 63, 255),
    2: (0, 127, 63),
    5: (0, 127, 127),
    4: (0, 0, 127),
    14: (0, 0, 191),
    13: (0, 0, 255),
    11: (0, 63, 63),
    10: (0, 127, 191),
    0: (0, 127, 255),
    12: (0, 100, 155),
    }
    
for i in range(15):
    MAPPING[i] = tuple(np.array(MAPPING[i])/255.0)
    
PALETTE = MAPPING


def plt_init(win_name, width, height):
    if win_name is None or win_name == '':
        win_name = str(time.time())
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    return ax, fig


def get_img_from_fig(fig, width, height):
    stream, _ = fig.canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype=np.uint8)
    img_rgba = buffer.reshape(height, width, 4)
    img, _ = np.split(img_rgba, [3], axis=2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  scores=None,
                  class_names=None,
                  colors='green',
                  thickness=1,
                  with_text=True,
                  font_size=10,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img)
    else:
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#    print('img',img.shape)
#    print('scores',scores.shape)
#    print('labels',labels.shape)
#    print('bboxes',bboxes.shape)

    if isinstance(bboxes, list):
        assert labels is None and scores is None
        with_score = True
    else:
        if scores is None:
            with_score = False
        else:
            bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)
            with_score = True

        if labels is None or labels.size == 0:
            bboxes = [bboxes]
        else:
            bboxes = [bboxes[labels == i] for i in range(labels.max()+1)]

    colors = colors_val(colors)
    if len(colors) == 1:
        colors = colors * len(bboxes)
    assert len(colors) >= len(bboxes)

    draw_func = choice_by_type(
        draw_hbb, draw_obb, draw_poly, bboxes[0], with_score)

    height, width = img.shape[:2]
    ax, fig = plt_init(win_name, width, height)
    plt.imshow(img)

    for i, cls_bboxes in enumerate(bboxes):
        if with_score:
            cls_bboxes, cls_scores = cls_bboxes[:, :-1], cls_bboxes[:, -1]

        if not with_text:
            texts = None
        else:
            texts = []
            for j in range(len(cls_bboxes)):
                text = f'cls: {i}' if class_names is None else class_names[i]
                if with_score:
                    text += f'|{cls_scores[j]:.02f}'
                texts.append(text)

        draw_func(ax, cls_bboxes, texts, PALETTE[i], thickness, font_size)

    drawed_img = get_img_from_fig(fig, width, height)
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, drawed_img)

    plt.close(fig)
    return drawed_img
