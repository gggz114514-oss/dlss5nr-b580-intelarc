"""Authenticate depth uploads and measure their effect on the native NR graph."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def authenticated(meta):
    d=R/'results'/Path(meta['runDirectory']).name/'output'
    assert meta['completedAllInputs']and meta['exitCode']==0 and not meta['timedOut']
    assert meta['runtimeSha256'].lower()=='6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
    for item in meta['outputs']:assert sha(d/item['name'])==item['sha256'].lower()
    for item in meta['traceFiles']:assert sha(d/'nvapi-trace'/item['name'])==item['sha256'].lower()
    return d
def trace_info(d):
    events=[json.loads(s)for s in(d/'nvapi-trace/events.jsonl').read_text().splitlines()]
    launches=[e for e in events if e['event']=='launch']
    pres=[e for e in launches if e['name']=='cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8']
    posts=[e for e in launches if e['name'].startswith('cc_tinlayout_fused_post_block_swin_1h_32_fp8')]
    assert len(pres)==len(posts)==4
    return events,pres,posts
def pre_regions(d,events,pre):
    params=(d/f"nvapi-trace/launch-{pre['sequence']}-0.bin").read_bytes()
    snapshot=next(e for e in events if e['event']=='buffer_scheduled'and e['sequence']==pre['sequence']and e['label']=='after-pre')
    arena=(d/'nvapi-trace'/snapshot['file']).read_bytes();base=snapshot['gpu_va'];skip=struct.unpack_from('<Q',params,216)[0]-base;down=struct.unpack_from('<Q',params,248)[0]-base
    assert struct.unpack_from('<II',params,208)==(256,256)and struct.unpack_from('<II',params,240)==(320,320)
    assert 0<=skip and down-skip==320*320*32 and down+160*160*32<=len(arena)
    return arena[skip:down],arena[down:down+160*160*32],params
p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['zero','one','field','field-inverted','step','none']);a=p.parse_args()
build=js(R/'4060-depth-v1-build.json');source=js(R/'video2dlssnr-depth-v1.source.json');assert build['archiveSha256']==source['sha256']==sha(R/'video2dlssnr-depth-v1.zip')
baseline=authenticated(js(R/'4060-motion-fine-sequence-v2.json'));base_trace=authenticated(js(R/'4060-motion-fine-trace-v2.json'));be,bp,bpost=trace_info(base_trace)
report=dict(scope='Native reference only; default model controls, 256x256 fine motion, four reset/temporal frames; no claim for other presets/masks/UI',source_manifest_sha256=sha(R/'video2dlssnr-depth-v1.source.json'),cases=[])
y,x=np.indices((256,256),dtype='i4')
for name in a.cases:
    dirs=[]
    for kind in('sequence','trace'):
        m=js(R/f'4060-depth-{name}-fine-{kind}-v1.json');d=authenticated(m)
        assert m['executableSha256'].lower()==build['binarySha256'].lower()and m['depthProbeCase']==name and m['resetPattern']==[1,0,0,1]
        if kind=='trace':assert m['traceDllSha256'].lower()==build['traceDllSha256'].lower()
        assert m['depthBound']==(name!='none')and m['depthInverted']==(name=='field-inverted');dirs.append(d)
    native,traced=dirs;events,pres,posts=trace_info(traced);frames=[]
    for i in range(4):
        filename=f'frame{i:02d}.png';c=js(native/f'{filename}_capture.json');assert c['depth_probe_mode']==name and c['depth_bound']==(name!='none')and c['depth_inverted']==(name=='field-inverted')
        assert bool(c['reset'])==(i in(0,3))and c['style']==0 and c['intensity']==c['local_tone']==c['local_structure']==1 and c['auto_mask']==c['ui_correction']==0
        for suffix in('input.rgba32f.bin','motion.rg32f.bin','depth.r32f.bin','output.rgba32f.bin'):
            raw=(native/f'{filename}_{suffix}').read_bytes();assert raw==(traced/f'{filename}_{suffix}').read_bytes()
            if suffix in('input.rgba32f.bin','motion.rg32f.bin'):assert raw==(baseline/f'{filename}_{suffix}').read_bytes()
        phase=0 if c['reset']else i
        expected=np.ones((256,256),dtype='f4')if name=='one'else np.zeros((256,256),dtype='f4')
        if name in('field','field-inverted'):expected=((x*5+y*3+phase*17)%1025).astype('f4')/np.float32(1024)
        if name=='step':expected=(((x+phase*13)//32+y//32)%2).astype('f4')
        depth=np.fromfile(native/f'{filename}_depth.r32f.bin','<f4').reshape(256,256);assert depth.tobytes()==expected.tobytes()
        rgb=np.fromfile(native/f'{filename}_output.rgba32f.bin','<f4').reshape(256,256,4)[...,:3].copy();original=np.fromfile(baseline/f'{filename}_output.rgba32f.bin','<f4').reshape(256,256,4)[...,:3].copy();assert np.isfinite(rgb).all()
        skip,down,params=pre_regions(traced,events,pres[i]);bs,bd,bparams=pre_regions(base_trace,be,bp[i])
        frames.append(dict(frame=i,explicit_reset=c['reset'],depth_sha256=sha(native/f'{filename}_depth.r32f.bin'),depth_range=[float(depth.min()),float(depth.max())],depth_nonzero_values=int(np.count_nonzero(depth)),rgb_equals_zero_depth=rgb.tobytes()==original.tobytes(),rgb_byte_mismatches=int(np.count_nonzero(rgb.view('u1')!=original.view('u1'))),native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),pre_skip_equals_zero_depth=skip==bs,pre_down_equals_zero_depth=down==bd,pre_scalar_bytes40_216_equal=params[40:216]==bparams[40:216],pre_texture_bound=[bool(struct.unpack_from('<Q',params,k)[0])for k in range(0,40,8)],pre_seed=struct.unpack_from('<I',params,200)[0],post_kernel=posts[i]['name'],post_kernel_equals_baseline=posts[i]['name']==bpost[i]['name']))
    row=dict(case=name,depth_bound=name!='none',depth_inverted=name=='field-inverted',reference=str(native),trace=str(traced),all_hashes_verified=True,paired_input_and_output_byte_equal=True,uploaded_depth_matches_independent_formula=True,frames=frames);report['cases'].append(row)
    print(json.dumps(dict(case=name,rgb_equal=[f['rgb_equals_zero_depth']for f in frames],pre_equal=[f['pre_skip_equals_zero_depth']and f['pre_down_equals_zero_depth']for f in frames],texture_bindings=[f['pre_texture_bound']for f in frames])),flush=True)
report['all_tested_depth_cases_rgb_unchanged']=all(f['rgb_equals_zero_depth']for c in report['cases']for f in c['frames'])
target=R/('depth-sweep-v1-analysis.json'if len(a.cases)==6 else'depth-sweep-v1-partial-analysis.json');target.write_text(json.dumps(report,indent=2));print(target)
