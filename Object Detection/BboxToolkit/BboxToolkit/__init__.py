pi = 3.141592
from .datasets import *
import BboxToolkit.datasets

from .visualization import *
import BboxToolkit.visualization

from .evaluation import *
import BboxToolkit.evaluation

from .utils import (get_bbox_type, get_bbox_dim, choice_by_type,
                    regular_theta, regular_obb)
from .transforms import (poly2hbb, poly2obb, rectpoly2obb, obb2poly, obb2hbb,
                         hbb2poly, hbb2obb, bbox2type)
from .geometry import bbox_overlaps, bbox_areas, bbox_nms
from .move import translate, flip, warp
