"""Authenticate390 candidate outputs against the approved NR256 receipt.

Reread complete saved low tensors and motion arrays; compare saved full-composite
hashes and state records. Full composite comparisons happened in the GPU run.
This audit does not independently execute the model or reconstruct full outputs.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=D/'results/k8-tiled-long1080-v1';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
path=out/'validation.json';r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
old=js(D/'results/batched-residual-long1080-review-v2/validation.json')
assert r['frames_completed']==len(r['frames'])==len(old['frames'])==390
assert set(r['k8_calls'])=={'pre','post'} and r['k8_calls']['pre']==r['k8_calls']['post']>0
assert r['all_outputs_match_approved_sequence'] and r['independent_uninterrupted_history']
assert r['reset_reproduces_first_frame'] and r['caller_ownership_guards_passed']
assert r['held_outputs_survive_replay'] and r['lut_bytes_unchanged'] and r['graph_replays']==393
assert len(r['graphs'])==2 and all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'])
assert not r['new_quality_change'] and not r['new_video_encoded'] and not r['new_output_arrays']
review=r['human_review_reference'];assert sha(review['path'])==review['sha256']
assert js(review['path'])['nr256_residual_visually_accepted']
verified={}
for i,(row,prior) in enumerate(zip(r['frames'],old['frames'])):
    assert row['frame']==i and row['reset']==(i==0) and row['next_seed']==i+1
    assert row['input_rgb8_sha256']==prior['input_rgb8_sha256']
    assert row['low_nr']==prior['low_nr'] and row['motion']==prior['motion']
    for key,shape,dtype in [('low_nr',(256,256,3),'<f2'),('motion',(1080,1920,2),'<f2')]:
        meta=row[key]
        if meta['path'] not in verified:
            a=arrays.load(meta);assert a.shape==shape and a.dtype.str==dtype and np.isfinite(a).all()
            verified[meta['path']]=Path(meta['path']).stat().st_size
    assert row['low_raw_sha256']==row['low_nr']['raw_sha256']==prior['runs']['batched']['low_raw_sha256']
    assert row['full_raw_sha256']==prior['runs']['batched']['full_raw_sha256']
    assert row['effective_dispatch']==prior['runs']['batched']['effective_dispatch']
    assert all(row[k] for k in ('low_bytes_equal','full_hash_equal','private_byte_equal_low','held_outputs_unchanged','inputs_unchanged'))
audit=dict(passed=True,report_sha256=sha(path),lease_sha256=sha(lease),auditor_sha256=sha(__file__),
    frames_verified=390,unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),
    no_additional_visual_review_required_for_identical_output=True,limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2),flush=True)
