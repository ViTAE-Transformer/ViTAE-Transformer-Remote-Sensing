'''Contain the functions related to color. The defination of colors is same
to matplotlib.

Reference:
    https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py
    https://matplotlib.org/stable/api/colors_api.html
'''
import numpy as np
import os.path as osp
import matplotlib.colors as mpl_colors

from collections.abc import Iterable


def list_named_colors(show=None, color_format='rgb255'):
    ''' List all named colors in matplotlib.

    Args:
        show (None | str): the way to list the colors. if show is None,
            this function only return color_dict. Else show is 'print',
            this function print names and colors on screen. Or, this
            function treats show as a file and put names and colors in
            this file.
        color_format (str): the format of color, only can be 'rgb',
            'rgb255', 'hex'.

    Returns:
        if out is None, reture
    '''
    # Build format functions for different color_format.
    assert color_format in ['rgb', 'rgb255', 'hex']
    if color_format == 'rgb':
        def _format_func(color):
            color = mpl_colors.to_rgb(color)
            color_str = f'({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})'
            return color, color_str
    elif color_format == 'rgb255':
        def _format_func(color):
            color = mpl_colors.to_rgb(color)
            color = tuple([int(round(255*c)) for c in color])
            return color, str(color)
    else:
        def _format_func(color):
            color = mpl_colors.to_hex(color)
            return color, color

    # Use _format_func to format colors.
    ori_color_dict = mpl_colors.get_named_colors_mapping()
    color_dict, color_doc = {}, ''
    for name, color in ori_color_dict.items():
        color, color_str = _format_func(color)
        color_dict[name] = color
        color_doc += name + '$' + ' '*max(25-len(name), 1) + color_str + '\n'

    # Different ways to show names and colors.
    if show is not None:
        assert isinstance(show, str)
        if show == 'print':
            print(color_doc)
        else:
            assert not osp.exists(show)
            with open(show, 'w') as f:
                f.writelines(color_doc)
    return color_dict


def single_color_val(color):
    '''Convert single color to rgba format defined in matplotlib.
    A single color can be Iterable, int, float and str. All int
    will be divided by 255 to follow the color defination in
    matplotlib.
    '''
    # Convert Iterable, int, float to list.
    if isinstance(color, str):
        color = color.split('$')[0].strip(' ')
    elif isinstance(color, Iterable):
        color = [c/255 if isinstance(c, int) else c for c in color]
    elif isinstance(color, int):
        color = (color/255, color/255, color/255)
    elif isinstance(color, float):
        color = (color, color, color)

    # Assert wheather color is valid.
    assert mpl_colors.is_color_like(color) , \
            f'{color} is not a legal color in matplotlib.colors'
    return mpl_colors.to_rgb(color)


def colors_val(colors):
    '''Convert colors to rgba format. Colors should be Iterable or str.
    If colors is str, functions will first try to treat colors as a file
    and read lines from it. If the file is not existing, the function
    will split the str by '|'.
    '''
    if isinstance(colors, str):
        if osp.isfile(colors):
            with open(colors, 'r') as f:
                colors = [line.strip() for line in f]
        else:
            colors = colors.split('|')
    return [single_color_val(c) for c in colors]


def random_colors(num, cmap=None):
    '''Random generate colors.

    Args:
        num (int): number of colors to generate.
        cmap (matplotlib cmap): refer to matplotlib cmap.

    Returns:
        several colors.
    '''
    if cmap is None:
        return colors_val(np.random.rand(num, 3))
    else:
        return colors_val(cmap(np.random.rand(num)))
