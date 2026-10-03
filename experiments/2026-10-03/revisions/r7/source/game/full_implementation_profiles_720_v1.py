"""Fixed-current720 implementation arms, CPU-only and explicit about variants.

These profiles are code candidates, not cache availability or speed claims.
The accepted six-option baseline is retained in every arm. GPU preparation,
same-arm graph checks and separate visual acceptance precede game exposure.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import json
from types import MappingProxyType

BASELINE_CONSTRUCTOR = MappingProxyType({
    'controlled': True, 'graph_capture_policy': 'all', 'combo_modes': (480,540,720),
    'c512_qkv_library_720': True, 'native_k8_720': True,
    'decoder_gather_unround_720': True, 'c32_hidden_native_720': True,
    'c512_probability_unround_720': True, 'c128_pairwise_720': True,
    'c128_dual_qkv_720': False, 'c64_attention_project_720': True,
    'c128_attention_project_720': True,
})
BASELINE_NUMERIC = MappingProxyType({
    'branch_accum_families': ('c128',),
    'history_value': 'fp32_fractional', 'front_noise': 'native_both',
})

@dataclass(frozen=True)
class Arm:
    label: str
    packages: tuple[str,...]
    constructor: tuple = ()
    numeric: tuple = ()
    conditional: tuple[str,...] = ()

    def mode_options(self):
        return {**copy.deepcopy(dict(BASELINE_CONSTRUCTOR)),
                **copy.deepcopy(dict(self.constructor)),
                'numeric_cleanup_720': {**copy.deepcopy(dict(BASELINE_NUMERIC)),
                                       **copy.deepcopy(dict(self.numeric))}}

    @property
    def identity(self):
        text=json.dumps(self.mode_options(),sort_keys=True,separators=(',',':'))
        return 'full-implementation-720-v1:'+hashlib.sha256(text.encode()).hexdigest()

_NATIVE_VIT = (
    ('vit_norm_fma',True), ('vit_exp_fma',True),
    ('vit_exp_zero_constant',True), ('vit_denominator','ordered_fused'),
)
_FULL_K_VIT = (('vit_qkv_full_k',True),('vit_projection_full_k',True))
_DECODER_MERGE = (('decoder_merge_unround_720',True),('decoder_merge_native_fma_720',True))
_BRANCHES = (('branch_accum_families',('c64','c128','c256')),)
_POST = (('post_sigmoid','native'),('post_store','rne'))
_HISTORY = (('history_value','fp32_all_paths'),('history_reciprocal','native'),('history_coord','direct_pixel'))

ARMS = MappingProxyType({
    'accepted_baseline': Arm('现役已验收基准：C512＋K8',()),
    'p01_vit_native': Arm('ViT 原生乘加与分母',('P01',),numeric=_NATIVE_VIT),
    'p01_vit_fp32_denominator': Arm('ViT 原生乘加与 FP32 分母归约',('P01',),numeric=(('vit_norm_fma',True),('vit_exp_fma',True),('vit_exp_zero_constant',True),('vit_denominator','fp32_reduction'))),
    'p02_decoder_native': Arm('Decoder 四级去舍入及原生乘加',('P02',),constructor=_DECODER_MERGE+(('post_native_fma_720','fp32_fma'),)),
    'p03_decoder_full_k': Arm('Decoder 输入完整累加',('P03',),numeric=(('decoder_input_full_k',True),)),
    'p03_vit_full_k': Arm('ViT 输入/输出完整累加',('P03',),numeric=_FULL_K_VIT),
    'p04_all_branches': Arm('C64/C128/C256 原生分支累加',('P04',),numeric=_BRANCHES),
    'p05_history_native': Arm('历史原生采样、倒数和像素坐标',('P05',),numeric=_HISTORY,conditional=('near motion','fractional motion')),
    'p05_dimension_reciprocal': Arm('历史尺寸原生倒数',('P05',),numeric=(('history_dimension_rcp','native'),),conditional=('alternative coordinate implementation',)),
    'p06_post_native': Arm('输出原生曲线与就近舍入',('P06',),numeric=_POST,conditional=('style0','style1','style2')),
    'numerical_all': Arm('数值遗留清理组合',('P01','P02','P03','P04','P05','P06'),
                         constructor=_DECODER_MERGE+(('post_native_fma_720','fp32_fma'),),
                         numeric=_NATIVE_VIT+_FULL_K_VIT+_BRANCHES+_POST+_HISTORY+(('decoder_input_full_k',True),),
                         conditional=('near motion','fractional motion','style0','style1','style2')),
})

def arm(name):
    if type(name) is not str or name not in ARMS:
        raise ValueError('Unknown fixed720 implementation arm')
    return ARMS[name]

def validate_profile(name, options_type):
    value=arm(name)
    options=value.mode_options()
    numeric=options_type.parse(options['numeric_cleanup_720'])
    if numeric.history_coord=='direct_pixel' and numeric.history_dimension_rcp=='native':
        raise ValueError('Pixel coordinates and normalized dimension reciprocals are alternative paths')
    if not {'c512_qkv_library_720','native_k8_720'} <= options.keys():
        raise ValueError('Missing accepted matrix baseline')
    return {'arm':name,'label':value.label,'packages':list(value.packages),
            'identity':value.identity,'constructor':options,
            'effective_numeric':numeric.to_dict(),'numeric_identity':numeric.identity,
            'conditional':list(value.conditional),'fixed_geometry':[720,1280],
            'cache_ready':False,'GPU_checked':False,'measured_gain_ms':None}
