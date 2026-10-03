"""Conservative complete-use proof for moving FP8 rounding to a producer store.

All value consumers must reach the existing same FP8 conversion through only
reviewed half views/copies (or constant zero padding). Raw arithmetic consumers
and escaping results reject fusion. Allocation-only shape reads are ignored.
"""
from collections import defaultdict

QUANTIZE='nr_backend.triton_fp8._kernel'
MATRIX='fast_matrices_v3._matmul'
VIEWS={'aten.reshape.default','aten.slice.Tensor','aten.select.int','aten.transpose.int',
       'aten.unsqueeze.default','aten.alias.default','aten.contiguous.default','aten.to.dtype','aten.to.device'}
COPIES={'aten.reshape.default','aten.contiguous.default','aten.clone.default',
        'aten.to.dtype','aten.to.device','aten.repeat_interleave.self_int'}
SHAPE_ONLY={'aten.empty_like.default'}

def key(value):return value['storage'],value['revision']

def full_half(value):
    if value['dtype']!='torch.float16' or value['offset'] or value['logical_bytes']!=value['storage_bytes']:return False
    stride=1
    for n,s in reversed(list(zip(value['shape'],value['stride']))):
        if n>1 and s!=stride:return False
        stride*=n
    return True

def plan(record):
    events=record['events'];consumers=defaultdict(list);writes=defaultdict(list);root=key(record['result'])
    for event in events:
        refs=event['reads'].items() if event['kind']=='triton' else enumerate(event['reads'])
        for role,value in refs:consumers[key(value)].append((event,role,value))
        outputs=event['writes'].values() if event['kind']=='triton' else event['writes']
        for value in outputs:writes[value['storage']].append((event['id'],value['revision']))

    def visit(value_key,active):
        if value_key==root:return set(),[dict(reason='escaping_body_result',value=list(value_key))]
        if value_key in active:return set(),[dict(reason='cycle',value=list(value_key))]
        if any(revision>value_key[1] for _,revision in writes[value_key[0]]):
            return set(),[dict(reason='later_storage_write',value=list(value_key))]
        active=active|{value_key};terminals=set();reasons=[]
        for event,role,value in consumers[value_key]:
            name=event['name']
            if event['kind']=='triton':
                if name==QUANTIZE and role=='X' and value['dtype']=='torch.float16':terminals.add(event['id'])
                else:reasons.append(dict(event=event['id'],name=name,role=role,reason='raw_kernel_consumer'))
                continue
            if name in SHAPE_ONLY:continue
            outputs=event.get('outputs',[])
            if event['writes'] and any(key(v)[0]==value_key[0] for v in event['writes']):
                reasons.append(dict(event=event['id'],name=name,reason='mutation'));continue
            if (name in VIEWS and outputs and all(key(v)==value_key and v['dtype']=='torch.float16' for v in outputs)):
                continue # Later aliases still read the same storage revision.
            preserving=name in COPIES
            if name=='aten.pad.default':
                args=event['arguments'];kwargs=event['keyword_arguments']
                mode=args[2] if len(args)>2 else kwargs.get('mode','constant')
                val=args[3] if len(args)>3 else kwargs.get('value')
                preserving=mode=='constant' and val in (None,0)
            if (preserving and len(event['reads'])==1 and value['dtype']=='torch.float16'
                    and outputs and all(v['dtype']=='torch.float16' for v in outputs)):
                for target in {key(v) for v in outputs}:
                    if target!=value_key:
                        found,failed=visit(target,active);terminals.update(found);reasons.extend(failed)
                continue
            reasons.append(dict(event=event['id'],name=name,reason='raw_tensor_consumer'))
        return terminals,reasons

    selected=[];rejected=[]
    for event in events:
        if event['kind']!='triton' or event['name']!=MATRIX:continue
        output=event['writes']['OUT'];params=event['constants']
        terminals,reasons=visit(key(output),set())
        if not full_half(output):reasons.append(dict(reason='not_full_half_output'))
        if params['INT8'] or params['BATCHED'] or params['BK']!=32:reasons.append(dict(reason='unsupported_matrix_contract'))
        if not terminals:reasons.append(dict(reason='no_quantized_consumer'))
        row=dict(kernel_ordinal=event['kernel_ordinal'],event=event['id'],constants=params,
                 launch_options=event['launch_options'],grid=event['grid'],
                 operands={n:{k:v[k] for k in ('shape','stride','dtype','offset','logical_bytes')} for n,v in event['reads'].items()},
                 output={k:output[k] for k in ('shape','stride','dtype','offset','logical_bytes')},
                 quantizers=sorted(terminals))
        if reasons:row['reasons']=reasons;rejected.append(row)
        else:selected.append(row)
    return dict(selected=selected,rejected=rejected,matrix_calls=len(selected)+len(rejected),
                removable_quantizers=len({q for row in selected for q in row['quantizers']}))
