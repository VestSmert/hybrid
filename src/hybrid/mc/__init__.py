"""
hybrid.mc
==============

Motion correction algorithms.

Modules
-------
pwrigid : Piecewise-rigid (tile-based) motion correction that estimates
           subpixel shifts per tile and interpolates a dense flow field.

Notes
-----
- For shift estimation we typically use a DoG-filtered copy of the data.
- Shifts are applied to the raw/flat-fielded frames.
"""

from .pwrigid import pwrigid_movie, PWMotionResult

import importlib as _importlib
pwrigid = _importlib.import_module(".pwrigid", __name__)

__all__ = [
    "pwrigid_movie",
    "PWMotionResult",
    "pwrigid",
]