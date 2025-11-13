# -*- coding: utf-8 -*-
"""
hybrid.masks
============

Public entrypoint for mask utilities:
- Primitive shape generators (hard/soft disks, ellipses, rectangles, Tukey windows)
- Cropping helpers (sides, bbox, center, square)
- High-level FOV builders (soft circular mask + application to arrays)
"""

from .shapes import (
    tukey2d,
    cosine_disk,
    hard_disk,
    soft_disk,
    hard_ellipse,
    soft_ellipse,
    hard_rect,
    soft_rect,
)

from .crop import (
    crop_sides,
    crop_to_bbox,
    crop_center,
    crop_square,
)

from .build import (
    build_soft_disk_mask,
    apply_mask_to_array,
    build_and_apply_fov,
)

__all__ = [
    # shapes
    "tukey2d",
    "cosine_disk",
    "hard_disk",
    "soft_disk",
    "hard_ellipse",
    "soft_ellipse",
    "hard_rect",
    "soft_rect",
    # crop
    "crop_sides",
    "crop_to_bbox",
    "crop_center",
    "crop_square",
    # build
    "build_soft_disk_mask",
    "apply_mask_to_array",
    "build_and_apply_fov",
]

__version__ = "0.1.0"
