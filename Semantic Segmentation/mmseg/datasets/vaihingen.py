from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class VaihingenDataset(CustomDataset):
    """Vaihingen dataset.

    In segmentation map annotation for Vaihingen, 5 stands for background, which
    is not included in 5 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'Impervious surfaces', 'Buildings', 'Low vegetation', 'Trees', 'Cars', 'Clutter')

    PALETTE = [[255, 255, 255], 
            [0, 0, 255], 
            [0, 255, 255], 
            [0, 255, 0],
            [255, 255, 0], 
            [255, 0, 0]]

    def __init__(self, **kwargs):
        super(VaihingenDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=5,
            **kwargs)
