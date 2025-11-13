# -*- coding: utf-8 -*-
"""
hybrid.flatfield
================

Flat-field estimation and application utilities.

Public API
----------
- estimate_flatfield : build a flat-field model (masked-aware) from a frame/stack
- apply_flatfield    : apply a given (normalized) flat map to data
- apply_from_model   : convenience wrapper that consumes the estimator's dict
- prepare_gain       : turn a normalized flat map into a multiplicative gain

Typical usage
-------------
>>> from hybrid.flatfield import estimate_flatfield, apply_from_model
>>> model = estimate_flatfield(stack, mask=soft_mask, blur_px=60)
>>> stack_ff = apply_from_model(stack, model, mask=soft_mask, renormalize="robust")
"""

from __future__ import annotations

from .estimate import estimate_flatfield
from .apply import apply_flatfield, apply_from_model

__all__ = [
    "estimate_flatfield",
    "apply_flatfield",
    "apply_from_model"
]
