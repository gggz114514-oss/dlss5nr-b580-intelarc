"""Selected serial NR256 fast stack after layout/crop complete-call validation.

This is the exact factory tested by the paired NR256, 1080p residual and 390-frame
suite, exposed without another wrapper or change in construction. Callers still
own toolchain selection, graph lifetime and serialization. Exact branch unchanged.
"""
from layout_crop_stack_v1 import Stack
