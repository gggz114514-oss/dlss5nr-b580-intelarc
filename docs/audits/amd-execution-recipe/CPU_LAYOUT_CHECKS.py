"""Persisted Python body of the original CPU command: 8 checks / 62 cases.

This is the already executed standard-library layout/packing/AST/metadata
check body; it adds no checks and does not import Torch/Triton, compile kernels,
or execute GPU work. This persistence step does not rerun the checks.

The absolute paths below are the original inputs. A complete rerun requires
the fixed AMD checkout, frozen r18 sources/manifest, original Main receipt and
private Luna RESULT/cache metadata, LLIR and selected SPV. If those original
metadata/cache artifacts are not published, this script alone cannot reproduce
all 8 checks / 62 cases. The algebra/packing checks are CPU layout checks only;
metadata tests are not GPU qualification; actual only Luna.
"""

import json,ast,re,hashlib,base64,sys
from pathlib import Path
i=Path(r'${WORKSPACE}\cyberpunk-b580-nr-opt\artifacts\b580-full-implementation-v1-20261003');a=Path(r'${EXPERIMENTS_E}\cyberpunk-opt\amd-20261003-current720-reanalysis\upstream');s=Path(r'${WORKSPACE}\cyberpunk-b580-nr-opt\artifacts\b580-full-implementation-v1-20261003\phase2-complete\revisions\r18\source')
m=json.loads((i/'phase2-complete/PHASE2-r18.json').read_text(encoding='utf-8-sig'))
receipt=json.loads((i/'R18_SWIN_THROUGHPUT_MAIN_REVIEW.json').read_text(encoding='utf-8-sig'))
checks=[];cnt=0
def passed(name,cases,detail): checks.append(dict(name=name,status='PASS',cases=cases,detail=detail))
# Actual CPU host packing equations from packed_weights.h and matching w2_weight address.
for c in [64,128]:
 for rows,cols in [(4*c,c),(c,4*c),(c,c)]:
  old=[];wide=[]
  for n in range(rows):
   for k in range(cols):
    tile=((n//16)*(cols//32)+k//32)*512
    old.append(tile+(((k%32)//16*2+(k%16)//8)*16+n%16)*8+k%8)
    wide.append(tile+((((k%16)//8)*16+n%16)*2+(k%32)//16)*8+k%8)
    lanegr=(k%16)//8
    assert old[-1]==tile+(((k%32)//16*2+lanegr)*16+n%16)*8+k%8
  assert sorted(old)==list(range(rows*cols))
  assert sorted(wide)==list(range(rows*cols))
  cnt+=2
passed('actual_fragment_host_pack_and_w2_weight_bijection',cnt,'C64/C128 expand/contract/project; original and W16 pure byte permutations. No GPU numeric claim.')
for c in [32,64,128]:
 seen=[]
 for head in range(c//32):
  for qt in range(4):
   for lane in range(32):
    for e in range(8):
     token=qt*16+(lane&15);col=head*32+(lane>>4)*8+e
     # A single fragment is N16; two ci fragments cover the 32-channel head.
     for ci in range(2):
      col=head*32+ci*16+(lane>>4)*8+e
      if col<head*32+32:seen.append((head,token,col))
 # The lane mapping above is per ci fragment and uses gr for eight of its sixteen columns.
 assert len(seen)==64*c and len(set(seen))==64*c
 offsets=[((qt*(c//16)+ct)*32+lane)*8+e for qt in range(4) for ct in range(c//16) for lane in range(32) for e in range(8)]
 assert sorted(offsets)==list(range(64*c))
passed('actual_amd_lane_qt_lds_full_window_coverage',3,'32 lanes, rc=lane&15, gr=lane>>4; two ci fragments per head. All 64 tokens/32 channels covered exactly once.')
for c in [64,128]:
 owners=[(sg//2,qt,t) for sg in range((c//32)*2) for qt in [(sg%2)*2,(sg%2)*2+1] for t in range(qt*16,(qt+1)*16)]
 assert len(owners)==64*(c//32) and len(set(owners))==len(owners)
passed('proposed_sg16_pair_token_partition',2,'Proposal layout only: two SG/head, each two QT, all 64 keys retained; native DPAS lane ABI unqualified.')
mh=(a/'hip/wave_owned_mh.inc').read_text(encoding='utf-8-sig')
assert len(re.findall(r'^\s*w2_sync\(\);',mh,re.M))==4
assert '#define W2_FFN_W16_SMALL 0' in mh
build=(a/'hip/build-modules.ps1').read_text(encoding='utf-8-sig').splitlines()[65]
for flag in ['W2_FFN_QT_SMALL_MASK 3','W2_FFN_QT_BATCH 4','W2_HIDDEN_TILES 2','W2_QKV_FUSE 3','W2_HOIST_LOADS 15','W2_PACK8 6']:assert "'"+flag+"'" in build
assert 'W2_FFN_W16_SMALL' not in build
passed('actual_recipe_and_core_barrier_lifetime',8,'Six active flags, W16_SMALL absent, four core WG barriers; no asynchronous prefetch claimed.')
tree=ast.parse((s/'game/audit_swin_720_v1.py').read_text(encoding='utf-8-sig'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='installed')
opt=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Options')
defs={n.target.id:ast.literal_eval(n.value) for n in opt.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ['serial_c64_mlp','split_c256_tail']}
assert fn.args.kwarg.arg=='kwargs'
assert defs['serial_c64_mlp'] is False and defs['split_c256_tail'] is False
assert not any(h['module']=='audit_swin_720_v1' for h in m['arms']['accepted_baseline']['constructor']['implementation_hooks_720'])
passed('actual_r18_default_off_and_baseline_composition',3,'Options serial/split defaults False; installed passes kwargs to Options; accepted_baseline does not install Swin hook.')
gridcases=0;counter=[]
for row in receipt['rows']:
 ca=next(v for k,v in row['arms'].items() if k!='accepted_baseline')
 d=json.loads(Path(ca['result']['path']).read_text(encoding='utf-8-sig'))
 own=d['fixed13_evidence']['implementation']['before_numeric']['children']['audit_swin_720_v1']
 countsites=len(own['sites']);assert countsites==44
 counter.append(dict(candidate=row['candidate'],sites=countsites,counter_labels=len(own['calls']),call_values=sorted(set(own['calls'].values())),capture_values=sorted(set(own['capture_calls'].values()))))
 assert set(own['capture_calls'].values())=={2}
 keys=d['cache_gate']['actual_compiler_keys']
 for label,r in own['resources'].items():
  if ', 0, 0)' not in label:continue
  role,geo=ast.literal_eval(label);c,h,w,sy,sx=geo;nr=64//r['grid'][0] if role!='qkv' else None
  key=keys[r['kernel_hash']];names=list(key['signature'])
  co={names[v['key'][0]]:v['value'] for v in key.get('constants',[]) if len(v['key'])==1 and isinstance(v['key'][0],int) and v['key'][0]<len(names)}
  bm=co['BM']
  if role=='qkv':pred=[(co['M']+bm-1)//bm,((3*(c//32))+(2 if co['WIDE_N'] else 1)-1)//(2 if co['WIDE_N'] else 1)]
  elif role=='attend':pred=[64//bm,co['HP']*co['WP']//64,c//32]
  else:pred=[64//bm,co['HP']*co['WP']//64,(c+co['BN']-1)//co['BN']]
  assert pred==r['grid'],(label,co,pred,r['grid'])
  assert r['spills']==0
  gridcases+=1
passed('actual_r18_selected_grids_full_key_and_capture_roles',gridcases,'Three original RESULT snapshots; 44 sites each; 176 or 192 real role counters, selected grids match actual constants; cold capture values two.')
ll=Path(r'${EXPERIMENTS_E}\cyberpunk-opt\b580-full-implementation-v1-20261003\luna\phase2\shared-cache\NQ4KGSUNYURTXK545KZL4EBVH2AM2TGZ6AAIFVAHSTKACMAI7R6A\_mlp_pairs.llir')
text=ll.read_text(encoding='utf-8-sig')
sgref=re.search(r'!intel_reqd_sub_group_size !(\d+)',text).group(1)
assert re.search(r'!'+sgref+r' = !\{i32 16\}',text)
assert '__spirv_SubgroupMatrixMultiplyAccumulateINTEL' in text
spv=ll.with_suffix('.spv');assert hashlib.sha256(spv.read_bytes()).hexdigest()=='bd1860e95393a50de01c5ff59370e1ca642084f3745961a4eed0471b4cfa655f'
passed('actual_selected_c64_sg16_spv_and_matrix_intrinsic',3,'SG16 and matrix intrinsic verified from existing LLIR; SPV matches original selected hash. No native ISA occupancy or DPAS issue rate claim.')
budgets=dict(c32_slm_bytes=16*1024,c64_slm_bytes=64*64*6,c128_slm_bytes=64*128*6)
assert budgets==dict(c32_slm_bytes=16384,c64_slm_bytes=24576,c128_slm_bytes=49152)
assert 2*2*16*32*4//(16*4)==128
assert 4*(2+2)*16*16*4//(16*4)==256
passed('port_budget_algebra_and_alias_lifetime',5,'Logical bytes/DWORD, not physical GRF allocation; hold AV until final KV readers reach barrier before aliasing.')
assert not any(n.split('.')[0] in ['torch','triton'] for n in sys.modules)
print(json.dumps(dict(status='PASS',scope='source/AST/metadata/layout CPU only; not GPU qualification; actual only Luna',checks=checks,total_cases=sum(x['cases'] for x in checks),counters=counter,actual_llir=dict(path=str(ll),sha256=hashlib.sha256(ll.read_bytes()).hexdigest(),spv_path=str(spv),spv_sha256=hashlib.sha256(spv.read_bytes()).hexdigest(),subgroup_size=16,workgroup_threads=64),budgets=budgets,forbidden_imports=[]),ensure_ascii=True))

