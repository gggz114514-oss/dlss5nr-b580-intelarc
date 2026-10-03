"""NR reset-frame correctness executor and components under validation."""
from .front import reset_front_features, project_front_features
from .pre_mlp import PreMLP
from .tensor_math import sm89_f16_dot, cubic_activation
from .pre_block import PreBlock
from .c32_block import C32SwinBlock
from .multihead_block import MultiHeadSwinBlock
from .split_block import SplitSwinBlock
from .vit_block import VitBlock
from .decoder import DecoderInputUpsample,DecoderUpsampleSwin
from .post import ResetPostBlock
from .noise import NativeNoiseTable
from .executor import ResetNR,ResetNR256
from .sigmoid import NativeSigmoidTable
from .reciprocal import NativeReciprocalTable,NativeDimensionReciprocalTable
from .temporal import ZeroMotionNR,IntegerMotionNR,MotionNR,MotionNR256

__all__ = ['reset_front_features', 'project_front_features', 'PreMLP', 'PreBlock', 'C32SwinBlock', 'MultiHeadSwinBlock', 'SplitSwinBlock', 'VitBlock', 'DecoderInputUpsample', 'DecoderUpsampleSwin', 'ResetPostBlock', 'NativeNoiseTable', 'ResetNR', 'ResetNR256', 'sm89_f16_dot', 'cubic_activation']
__all__ += ['NativeSigmoidTable','ZeroMotionNR','IntegerMotionNR','MotionNR','MotionNR256']
__all__ += ['NativeReciprocalTable','NativeDimensionReciprocalTable']

from .execution import use_arithmetic_backend,current_arithmetic_backend
__all__ += ['use_arithmetic_backend','current_arithmetic_backend']
