"""Independently reread saved full frames and native captures after a tiled run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_auxiliary_captures import authenticated, js, sha

R=Path(__file__).resolve().parent; root=R.parent
p=argparse.ArgumentParser(); p.add_argument('--version', choices=['1','2'], required=True)
a=p.parse_args(); folder=R/f'results/tiled-i32-1080-v{a.version}'
destination=folder/'saved-audit.json'; assert not destination.exists()
report_path=folder/'validation.json'; report=js(report_path)
assert report['all_byte_equal'] and 'error' not in report
assert not report['promoted'] and not report['complete_migration']
for path,digest in report['source_files'].items(): assert sha(root/path)==digest, path
lease_path=R/f'full-nr-tiled-v{a.version}.log.lease.json'; lease=js(lease_path)
assert lease['returncode']==0 and lease['owner']==f'nr-b580-full-nr-tiled-v{a.version}'
assert Path(lease['cwd'])==root and lease['finished_unix']>lease['started_unix']
inputs=R/'inputs/flow-full-1920x1080-v3'; manifest=js(inputs/'manifest.json')
assert sha(inputs/'manifest.json')==report['input_manifest_sha256']
assert len(manifest['frames'])==13 and manifest['unique_adjacent_image_contents']==12
metadata=[R/f'4060-real-flow-1920x1080-{kind}-v3.json'for kind in('sequence','repeat')]
assert sha(metadata[0])==report['native_metadata_sha256']
native, repeat=[authenticated(js(path)) for path in metadata]
rows=[]
for run,(mode,count) in zip(report['runs'], [('baseline',3),('candidate',13)], strict=True):
    assert run['mode']==mode and len(run['frames'])==count
    for i,(frame,spec) in enumerate(zip(run['frames'],manifest['frames'][:count],strict=True)):
        assert sha(inputs/spec['file'])==spec['sha256'] and sha(inputs/spec['motion_file'])==spec['motion_sha256']
        assert frame['frame']==i and frame['reset']==spec['reset']==(i in(0,12))
        assert frame['next_seed']==(1 if spec['reset'] else i+1)
        for suffix in ('input.rgba32f.bin','motion.rg32f.bin','depth.r32f.bin','output.rgba32f.bin'):
            assert (native/f"{spec['file']}_{suffix}").read_bytes()==(repeat/f"{spec['file']}_{suffix}").read_bytes()
        expected=np.fromfile(native/f"{spec['file']}_output.rgba32f.bin",'<f4').reshape(1080,1920,4)[...,:3].copy()
        rgb_path=folder/f'{mode}-frame{i:02d}-rgb32f.npy'; private_path=folder/f'{mode}-frame{i:02d}-private-rgb16f.npy'
        rgb=np.load(rgb_path,allow_pickle=False); private=np.load(private_path,allow_pickle=False)
        assert rgb.shape==private.shape==expected.shape
        assert rgb.dtype==np.dtype('<f4') and private.dtype==np.dtype('<f2')
        assert np.isfinite(rgb).all() and rgb.tobytes()==expected.tobytes()
        assert private.tobytes()==expected.astype('<f2').tobytes()
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert frame['rgb_sha256']==frame['native_rgb_sha256']==digest
        assert frame['byte_equal'] and frame['private_history_byte_equal'] and frame['byte_mismatches']==0
        assert frame['bytes']==rgb.nbytes and frame['seconds']>0
        assert all(frame['arithmetic_dispatches'][k]>0 for k in ('dense','batched','half_fma'))
        rows.append(dict(mode=mode,frame=i,bytes=rgb.nbytes,rgb_raw_sha256=digest,
                         rgb_npy_sha256=sha(rgb_path),private_npy_sha256=sha(private_path)))
    if count==13: assert run['frames'][0]['rgb_sha256']==run['frames'][12]['rgb_sha256']
base=float(np.mean([f['seconds']for f in report['runs'][0]['frames'][1:]]))
matched=float(np.mean([f['seconds']for f in report['runs'][1]['frames'][1:3]]))
warm=float(np.mean([f['seconds']for f in report['runs'][1]['frames'][1:]]))
for key,value in dict(baseline_warm_mean_seconds=base,candidate_matching_frames_mean_seconds=matched,
                      candidate_all_warm_mean_seconds=warm,paired_warm_speedup=base/matched).items():
    assert report[key]==value
result=dict(all_saved_outputs_and_history_byte_equal=True, comparisons=len(rows),candidate_frames=13,
            bytes_compared=sum(r['bytes']for r in rows), report_sha256=sha(report_path),
            audit_script_sha256=sha(Path(__file__)), lease_sha256=sha(lease_path),
            log_sha256=sha(Path(lease['log'])), native_metadata={str(p.relative_to(root)):sha(p)for p in metadata},
            source_files=report['source_files'],frames=rows,baseline_warm_mean_seconds=base,
            candidate_matching_frames_mean_seconds=matched,candidate_all_warm_mean_seconds=warm,
            paired_warm_speedup=base/matched,timing_scope=report['timing_scope'],scope=report['scope'],complete_migration=False)
destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items()if k not in ('frames','source_files','native_metadata')}))
