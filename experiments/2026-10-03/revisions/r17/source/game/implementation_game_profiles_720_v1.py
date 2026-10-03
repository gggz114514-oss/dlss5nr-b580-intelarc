"""CPU-only checkbox descriptors for the comprehensive current720 payload.

Being listed is not permission to compile in-game. The host's installed
availability list is populated only from a frozen source/cache test receipt.
"""
from __future__ import annotations
from types import MappingProxyType
import json
from pathlib import Path
from full_implementation_profiles_720_v1 import ARMS
from implementation_hooks_720_v1 import parse_hooks

# Runtime configuration already supplies controlled/combo/capture policy.
# Only the accepted mathematics options belong in a checkbox override.
CONFIG_FIELDS = frozenset({'controlled','combo_modes','graph_capture_policy'})

def _numerical_descriptors():
    result={}
    for name,value in ARMS.items():
        if name=='accepted_baseline':
            continue
        options=value.mode_options()
        result['audit_'+name]={
            'label':value.label,'packages':value.packages,
            'legacy':tuple((k,v) for k,v in options.items()
                           if k not in CONFIG_FIELDS and k!='numeric_cleanup_720'),
            'numeric':tuple(options['numeric_cleanup_720'].items()),
            'hooks':(), 'pending_validation':True,
        }
    return result

# Main integration adds frozen worker descriptors here after reviewing their
# producer/consumer and ownership interfaces, never by importing worker code.
def _worker_payload():
    path=Path(__file__).with_name('IMPLEMENTATION_WORKER_PROFILES_720.json')
    if not path.exists():
        return {}, ()
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!='current720-cold-worker-checkbox-v1' or data.get('GPU_qualified') is not False:
        raise RuntimeError('Unknown checkbox source payload')
    result={}
    for name,row in data['profiles'].items():
        result[name]={**row,'legacy':tuple(tuple(v) for v in row['legacy']),
                      'numeric':tuple(tuple(v) for v in row['numeric']),
                      'packages':tuple(row['packages']),'hooks':parse_hooks(row['hooks'])}
    return result,parse_hooks(data.get('common_admission_hooks') or ())

_workers, COMMON_ADMISSION = _worker_payload()
WORKER_DESCRIPTORS = MappingProxyType(_workers)
DESCRIPTORS = MappingProxyType({**_numerical_descriptors(), **WORKER_DESCRIPTORS})

def combine_hooks(values):
    """Combine disjoint flags into one installation per verified module.

Boolean flags enable independent sites. Lists/tuples enumerate sites.
Different scalar implementations of the same site are explicitly exclusive.
"""
    collected={}
    for hooks in values:
        for spec in hooks:
            if spec.module not in collected:
                collected[spec.module]={'module':spec.module,'call':'installed',
                                        'stage':spec.stage,'kwargs':spec.kwargs}
                continue
            current=collected[spec.module]
            if current['stage']!=spec.stage:
                raise ValueError('同一模块不能在两个安装阶段执行')
            for key,value in spec.kwargs.items():
                old=current['kwargs'].get(key,value)
                if type(old) is bool and type(value) is bool:
                    value=old or value
                elif isinstance(old,(list,tuple)) and isinstance(value,(list,tuple)):
                    # Order is stable; no repeated consumer/producer sites.
                    value=list(old)+[item for item in value if item not in old]
                elif old!=value:
                    neutral={
                        'audit_swin_720_v1':{'qk_norm':'ordered_half','denominator':'ordered_half'},
                        'audit_front_decoder_post_720_v1':{'post_entry_mode':'baseline'},
                        'audit_vit_c512_720_v1':{'c512_qkv_backend':'library','denominator':'reference',
                                                 'profile':'numeric'},
                    }.get(spec.module,{})
                    if key in neutral and old==neutral[key]:
                        pass
                    elif key in neutral and value==neutral[key]:
                        value=old
                    elif spec.module=='audit_front_decoder_post_720_v1' and key=='post_entry_mode' and {old,value}<={'native','native_mlp'}:
                        value='native_mlp'
                    else:
                        raise ValueError(f'同一模块的 {key} 选择了不同实现')
                current['kwargs'][key]=value
    return parse_hooks(tuple(collected[k] for k in sorted(collected)))

def normalize_mode_options(result, hooks):
    """One executor per position, with explicit producer/consumer composition."""
    result=dict(result)
    hooks=combine_hooks((tuple(hooks),COMMON_ADMISSION)) if COMMON_ADMISSION and result.get('numeric_cleanup_720') else tuple(hooks)
    rows={spec.module:spec.to_dict() for spec in hooks}
    numeric=dict(result.get('numeric_cleanup_720',{}))
    front=rows.get('audit_front_decoder_post_720_v1')
    vit=rows.get('audit_vit_c512_720_v1')
    if front:
        kwargs=front['kwargs']
        if kwargs.get('merge_native'):
            result['decoder_merge_unround_720']=False
            result['decoder_merge_native_fma_720']=False
        if kwargs.get('decoder_full_k'):
            numeric['decoder_input_full_k']=False
        if kwargs.get('post_tail'):
            if numeric.get('post_store')=='native_rtz':
                raise ValueError('原生截断输出与完整就近舍入尾段是两种实现')
            if numeric.get('post_sigmoid')=='native': kwargs['post_sigmoid']='native'
            numeric['post_sigmoid']='table'
            numeric['post_store']='reference'
        if kwargs.get('post_entry_mode') in ('native','native_mlp'):
            if result.get('post_native_fma_720')=='fp16_fma':
                raise ValueError('输出入口的 FP16 和 FP32 乘加是两种实现')
            result['post_native_fma_720']=None
        result['post_rgb_tail_720']=False
    if vit and vit['kwargs'].get('profile')=='complete':
        kwargs=vit['kwargs']
        for name in ('vit_qkv_full_k','vit_projection_full_k','vit_exp_zero_constant','vit_norm_fma','vit_exp_fma'):
            kwargs[name]=bool(kwargs.get(name,False) or numeric.get(name,False))
            numeric[name]=False
        denominator=numeric.get('vit_denominator','reference')
        if denominator!='reference':
            if kwargs.get('denominator','reference') not in ('reference',denominator):
                raise ValueError('全局注意力分母选择了两种实现')
            kwargs['denominator']=denominator
        numeric['vit_denominator']='reference'
        if front and kwargs.get('encoder_pool_final'):
            kwargs['use_front_pool_helper']=True
            front['kwargs']['pool_c512_transition']=False
            front['kwargs']['down_channels']=sorted(set(front['kwargs'].get('down_channels',()))|{512})
    if numeric: result['numeric_cleanup_720']=numeric
    if rows: result['implementation_hooks_720']=[rows[name] for name in sorted(rows)]
    return result
