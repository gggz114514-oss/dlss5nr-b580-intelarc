"""Runner4 comparison API. Parent stays stdlib; fixed CPU child owns NumPy.

Every original output field and byte/finite/source sequence validation is
provided by the frozen bounded comparator. Failures propagate to the existing
NUMERIC_FAILURE stop path. No fallback comparison or array mutation occurs.
"""
from metrics_streaming_reference import header
from metrics_subprocess import compare_pair, compare_frames
