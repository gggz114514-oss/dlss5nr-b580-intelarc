"""Build a publication-only source snapshot and explicit, sanitized evidence excerpts.

No repository history, weights, runtime binaries, arrays, media, SSH material or
raw traces are exported. Original research files are never edited.
"""
import ast,hashlib,importlib.metadata,json,re,shutil,subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
EXACT=BASE/'nr-b580';FAST=BASE/'nr-b580-int8';R=EXACT/'reference'
OUT=BASE/'nr-b580-public'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
receipt_dir=D/'publication-v0.1.0-pre';receipt_dir.mkdir(exist_ok=True)
manifest_path=OUT/'evidence/source-manifest.json';assert not manifest_path.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

sources=[]
def copy(source,target,source_id):
    source=Path(source);target=OUT/target
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==source.read_bytes(),str(target)
    else:shutil.copyfile(source,target)
    assert target.read_bytes()==source.read_bytes()
    sources.append(dict(path=target.relative_to(OUT).as_posix(),source=source_id,
        sha256=sha(source),bytes=source.stat().st_size,byte_identical=True))

checkpoint=js(R/'backend-checkpoint.json')
exact_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXACT,text=True).strip()
fast_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FAST,text=True).strip()
backend=list((EXACT/'backend/nr_backend').glob('*.py'));assert len(backend)==35
for p in sorted(backend):
    rel=p.relative_to(EXACT).as_posix()
    assert p.read_bytes()==subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=EXACT)
    assert sha(p)==checkpoint['current_backend_source_files'][rel.replace('/','\\')]
    copy(p,rel,'exact/'+rel)

for name in ('pre_replay.cpp','c32_replay.cpp','c64_replay.cpp','c128_replay.cpp',
             'c256_replay.cpp','c512_replay.cpp','vit_replay.cpp','decoder_replay.cpp',
             'post_replay.cpp','half_math.cpp','noise_lut.cpp','mufu_queries.cpp','sigmoid.cpp'):
    rel='reference/native-replay/'+name;copy(EXACT/rel,rel,'exact/'+rel)
for rel in ('reference/native-trace/nr_nvapi_trace.cpp','reference/native_front_capture.py'):
    copy(EXACT/rel,rel,'exact/'+rel)

pending=['strided_batched_v2','batched_branched_mlp_v2','fused_split_ffwd_v2',
    'fused_c32_mlp_lut_v1','cubic_lut_constant_v1','fused_vit_projection_v3',
    'fused_swin_native_half_v1','native_half_head_layout_v1','fused_dynamic_front_v1',
    'graph_front_v6','graph_history_warp_v2','residual_scale_v1']
seen=set()
while pending:
    name=pending.pop()
    if name in seen:continue
    p=FAST/'experimental'/f'{name}.py';assert p.is_file(),name
    seen.add(name)
    for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
        mods=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
              [a.name for a in node.names] if isinstance(node,ast.Import) else [])
        for module in mods:
            first=module.split('.')[0]
            if (FAST/'experimental'/f'{first}.py').is_file():pending.append(first)
for name in sorted(seen):
    copy(FAST/'experimental'/f'{name}.py',f'snapshots/fast-kernels/{name}.py',f'fast/experimental/{name}.py')

private_path=re.compile(r'(?:[A-Za-z]:[/\\]|192\.168\.|/Users/|\\\\)')
drop={'sources','gate_artifacts','baseline_sources','frozen_artifacts','current_sources',
      'input_path','output_path','download_url','source_inputs','command','cwd','traceback',
      'source','files','source_files','source_receipts','held_files','remote','host','loginUser'}
def sanitize(value):
    if isinstance(value,dict):
        return {('<local-path>' if private_path.search(str(k)) else k):sanitize(v)
                for k,v in value.items() if k not in drop and not private_path.search(str(k))}
    if isinstance(value,list):return [sanitize(x) for x in value]
    if isinstance(value,str) and private_path.search(value):return '<local-path omitted>'
    return value

# The selected fields are excerpts, never a claim to publish the raw reports.
reports=[
 ('results/pre-mlp-evidence-verification.json','historical',('independent_replay','holdout_files','rounding')),
 ('results/pre-holdouts-v1/pre-outputs-validation.json','historical',('scope','cases')),
 ('results/c64-holdouts-v1/pre-c32-c64-chain-validation.json','historical',('scope','holdout_hashes_verified','cases')),
 ('results/c512-holdouts-v2/encoder-prefix-validation.json','historical',('scope','cases','comparisons','all_byte_equal')),
 ('results/vit-holdouts-v2/encoder-prefix-validation.json','historical',('scope','cases','comparisons','all_byte_equal')),
 ('results/post-holdouts-v1/encoder-prefix-validation.json','historical',('scope','cases','comparisons','all_byte_equal')),
 ('results/reset-executor-v1/validation.json','historical',('scope','gpu','cases','complete_migration')),
 ('results/reset-executor-v1/visual-validation.json','historical',('scope','files_verified','cases','complete_migration')),
 ('half-fma-validation.json','historical',('cases','oracle','devices','source_sha256')),
 ('results/reset-full-1920x1080-v3/validation.json','historical',('scope','cases','runs','byte_equal','mismatches','bytes','output_sha256','native_sha256','complete_migration')),
 ('results/motion-tex-v1/texture-validation-v3.json','historical',('scope','cases','comparisons','all_byte_equal','rule')),
 ('results/texture-full-basis-v1/coordinate-precision-probe-v1.json','historical',('scope','cases','tested','candidates','results','best')),
 ('video-fixture-diversity-correction-v1.json','historical-correction',('scope','reason','affected','correction','cases','summary')),
 ('real-flow-864x480-full-audit-v2.json','historical',('scope','all_byte_equal','frames','comparisons','bytes_compared','passed','complete_migration')),
 ('motion-range-stage-full-audit-v1.json','historical',('scope','all_byte_equal','cases','frames','comparisons','passed','complete_migration')),
 ('experimental/attention-weights-stage-v1/full-regression-saved-audit-v1.json','current-exact',
    ('audit_script_sha256','suite_sha256','staged_sources','full_frame_comparisons','all_byte_equal',
     'persisted_rgb_and_private_reread_equal','uses_int8_or_xmx','complete_migration')),
 ('experimental/attention-weights-stage-v1/main-864x480-saved-audit-v1.json','current-exact',
    ('scope','audit_script_sha256','frames','full_frame_comparisons','all_byte_equal','persisted_rgb_and_private_reread_equal','complete_migration')),
 ('experimental/native-half-attention-v1/validation.json','current-fast',('scope','passed','body','whole_body','complete_migration')),
 ('results/nr256-native-parity-v1/saved-audit-v1.json','current-fast',
    ('passed','report_sha256','auditor_sha256','complete_output_receipts','summary','body_diagnostic','b580_over_4060_complete_host_ratio','native_equivalence_claim_for_fast_mode','limitation')),
 ('results/native-half-long1080-v1/saved-audit-v1.json','current-fast',
    ('passed','report_sha256','auditor_sha256','frames_verified','unique_arrays_reread','no_additional_visual_review_required_for_identical_output','limitation')),
]
index=[]
for ordinal,(rel,status,fields) in enumerate(reports,1):
    path=R/rel
    if not path.exists():path=D/rel
    assert path.exists(),rel
    report=js(path)
    excerpt={key:sanitize(report[key]) for key in fields if key in report}
    target=f'evidence/excerpts/{ordinal:02d}.json'
    row=dict(report_id=rel,status=status,original_report_sha256=sha(path),original_report_bytes=path.stat().st_size,
             excerpt_path=target,excerpt_fields=list(excerpt),raw_arrays_public=False)
    write(OUT/target,dict(provenance=row,excerpt=excerpt))
    row['excerpt_sha256']=sha(OUT/target);index.append(row)

stage=js(D/'experimental/attention-weights-stage-v1/full-regression-saved-audit-v1.json')
assert stage['all_byte_equal'] and stage['persisted_rgb_and_private_reread_equal'] and stage['full_frame_comparisons']==55
assert {p.name:sha(p) for p in backend}==stage['staged_sources']
index_doc=dict(schema=1,reference='Fixed SF-v2 RTX4060 Laptop, driver616.86',
    limitation='Excerpts and hash commitments, not public raw tensors or independent reproduction. Historical and current scopes are separate.',
    reports=index,current_exact_stage_comparisons=55,complete_migration=False)
write(OUT/'evidence/index.json',index_doc)
write(manifest_path,dict(schema=1,exact_local_commit=exact_commit,fast_local_commit=fast_commit,
    backend_files=35,fast_module_files=len(seen),files=sources,
    checkpoint_original_sha256=sha(R/'backend-checkpoint.json')))
packages={}
for dist in importlib.metadata.distributions():
    name=dist.metadata.get('Name','')
    if any(key in name.lower() for key in ('torch','triton','numpy','sycl','dpcpp','mkl')):
        packages[name]=dist.version
write(OUT/'evidence/environment.json',dict(platform='Windows x64',target_gpu='Intel Arc B580',
    reference_gpu='NVIDIA RTX4060 Laptop',reference_driver='616.86',python_packages=packages,
    scope='Installed package metadata of the measured workstation; not a dependency installer or broad compatibility guarantee.'))
write(receipt_dir/'source-export-v1.json',dict(builder_sha256=sha(__file__),public_directory=str(OUT),
    exact_commit=exact_commit,fast_commit=fast_commit,source_files=len(sources),evidence_reports=len(index),
    source_manifest_sha256=sha(manifest_path),index_sha256=sha(OUT/'evidence/index.json'),passed=True))
print(json.dumps(dict(source_files=len(sources),exact_files=35,fast_modules=len(seen),reports=len(index),passed=True),indent=2))
