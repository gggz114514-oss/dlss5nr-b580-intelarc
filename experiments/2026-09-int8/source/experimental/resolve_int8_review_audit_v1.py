"""Main's bounded resolution of two Windows path false positives in Luna's audit.

Reuse the preserved completed audit; do not rerun inference, video decoding,
or the source/array audit. Authenticate its primary artifacts and resolve the
two reported path issues using actual filesystem identity. Preserve the
original audit issues and failed GPU report, with a separate resolution.
"""
import hashlib
import json
from pathlib import Path

OUT=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-residual1080-audit-resolution-v1')
RESULT=OUT/'main-resolution.json'
assert not RESULT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={
    OUT/'luna-existing-output.json':'fa1fb2b942fee40e1aaf0df9d3bd9229a289a1e7293068a3280460c4f1b88087',
    OUT/'provenance.json':'d7f3c371096cecbd19cbaccfcac98f09a039bce052516f7e92c6fdd6ca005693',
    OUT/'luna-existing-tool-code.js':'c75a99597135d374005ba2647880e0d1b47f19482b69c6e00d0e9ff0951f4bca',
}
for p,h in pins.items():assert sha(p)==h,p
a=js(OUT/'luna-existing-output.json')
assert sha(a['manifest'])==a['manifest_sha256']
m=js(a['manifest'])
primary={**a['primary'],**{'original_'+k:a['original'][k] for k in ('report','log','lease')}}
for record in primary.values():
    p=Path(record['path'])
    assert sha(p)==record['sha256'] and p.stat().st_size==record['bytes'],p
r=js(primary['report']['path']);q=js(primary['receipt']['path'])
original=js(primary['original_report']['path'])
assert original['passed'] is False and original['phase']=='failed'
assert js(primary['original_lease']['path'])['returncode']==1
assert r['passed'] is True and r['phase']=='completed'
assert js(primary['lease']['path'])['returncode']==0
assert a['issue_count']==len(a['issues'])==2
pairs=[
    ('receipt.source.path',q['source']['path'],r['original_video']['path']),
    ('receipt.output.path',q['output']['path'],m['video']),
]
assert a['issues']==[f'{label}:actual={left!r}:expected={right!r}' for label,left,right in pairs]
resolved=[]
for label,left,right in pairs:
    lp,rp=Path(left),Path(right)
    assert lp.resolve()==rp.resolve() and lp.samefile(rp),(label,left,right)
    resolved.append(dict(label=label,original_path=left,comparison_path=right,
                         resolved_path=str(lp.resolve()),same_file=True))
assert r['remux_receipt']==q and a['receipt_snapshot_equal']
assert r['all390_decoded_frames_byte_equal'] and r['all_packet_timestamps_equal']
assert len(r['decoded_frames'])==390
assert all(f['frame']==i and f['byte_equal'] and f['bytes']==6635520
           for i,f in enumerate(r['decoded_frames']))
assert all(r['probe'][k]=='bt709' for k in ('color_space','color_primaries','color_transfer'))
assert r['probe']['color_range']=='tv'
assert r['human_review']=='pending' and r['new_quality_approved'] is False
assert r['candidate_promoted'] is False
for key,count in (('sources',794),('exact_gate',269)):
    s=a['source_counts'][key]
    assert s['listed']==s['checked']==count
    assert s['missing']==s['sha_mismatches']==s['bad_sha']==0
report=dict(passed=True,scope=__doc__,resolved_by='main agent, after user-reported audit anomaly',
    evidence_sources={str(p):h for p,h in pins.items()},source_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
    primary_artifacts_reauthenticated=primary,original_luna_issues=a['issues'],resolved_path_issues=resolved,
    unresolved_issues=[],original_failed_report_preserved=True,
    inherited_source_counts=a['source_counts'],inherited_array_storage=a['array_storage'],
    inherited_images=a['images'],decoded_frames=r['decoded_frames'],probe=r['probe'],
    timing=a['timing'],timing_scope='Paired resident residual pipeline; excludes IO/upload, flow estimation, decode, encode and JIT',
    video=r['video'],no_model_rerun=True,no_decode_rerun=True,no_full_audit_rerun=True,
    human_review='pending',new_quality_approved=False,candidate_promoted=False,
    limitation='The recovery report has no spill resource fields; zero fields scanned is not a fresh zero-spill test. Model checks remain inherited execution evidence.',
    audit_failure='Two raw-string Windows path comparisons were false positives. Subsequent auditor script had SyntaxError and did not execute; main resolved issues without retrying that script.')
RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
print(json.dumps(dict(passed=True,resolved_path_issues=2,result=str(RESULT),sha256=sha(RESULT),video=r['video'],human_review='pending')))
