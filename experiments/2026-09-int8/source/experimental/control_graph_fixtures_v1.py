"""Authenticated native display/private oracles for graph control preservation."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
from PIL import Image
from analyze_auxiliary_captures import authenticated, expected_aux, js, sha, trace
from nr_backend.controlled_temporal import NRControls

R = Path(__file__).resolve().parents[2] / 'nr-b580/reference'


def control(value):
    return NRControls(style=value['style'], intensity=value['intensity'],
                      local_tone=value['tone'], local_structure=value['structure'])


def oracle(path, dtype):
    value = np.fromfile(path, dtype).reshape(256, 256, 4)[..., :3].copy()
    half = value.astype('f2')
    assert np.isfinite(value).all() and half.astype(value.dtype).tobytes() == value.tobytes()
    meta = dict(format='native-rgba-select-rgb-to-f16', path=str(path), sha256=sha(path),
                rgba_dtype=np.dtype(dtype).str, shape=[256, 256, 3], dtype='<f2',
                raw_sha256=hashlib.sha256(half.tobytes()).hexdigest())
    return half, meta


def load():
    inputs = R / 'inputs/temporal-v1'
    manifest_path = inputs / 'manifest.json'
    manifest = js(manifest_path)
    assert sha(Path(manifest['source'])) == manifest['source_sha256']
    sources = [Path(__file__), manifest_path, R / 'analyze_auxiliary_captures.py']
    single_chain = R / 'results/live-single-private-chain-v1'
    assert js(single_chain / 'validation.json')['all_byte_equal']
    single_run = js(single_chain / 'run.json')
    binding_chain = R / 'results/live-binding-private-chain-v1'
    assert js(binding_chain / 'validation.json')['all_rgb_byte_equal']
    binding_run = js(binding_chain / 'run.json')
    aux_analysis_path = R / 'live-auxiliary-sweep-v1-analysis.json'
    aux_analysis = js(aux_analysis_path)
    assert aux_analysis['all_captures_authenticated']
    sources += [aux_analysis_path, single_chain / 'validation.json', single_chain / 'run.json',
                binding_chain / 'validation.json', binding_chain / 'run.json']
    cases = []
    for name in ('intensity', 'tone', 'structure', 'style', 'auto-on', 'mask-add', 'mask-remove'):
        single = name in ('intensity', 'tone', 'structure', 'style')
        measure = None if single else next(c for c in aux_analysis['cases'] if c['case'] == name)
        version = 1 if single else measure['capture_version']
        prefix = f'4060-live-single-controls-{name}-fine' if single else f'4060-liveaux-{name}-fine'
        meta_paths = [R / f'{prefix}-{kind}-v{version}.json' for kind in ('sequence', 'trace')]
        sources += meta_paths
        metas = [js(path) for path in meta_paths]
        native, repeat = [authenticated(meta) for meta in metas]
        _, _, pres, _ = trace(repeat)
        if single:
            requests = metas[0]['liveControlRequests']
            assert requests == metas[1]['liveControlRequests'] and len(requests) == 4
        case = dict(name=name, frames=[])
        for i, spec in enumerate(manifest['frames']):
            filename = spec['file']
            input_path = inputs / filename
            assert sha(input_path) == spec['sha256']
            sources.append(input_path)
            pixels = np.asarray(Image.open(input_path).convert('RGB'), dtype='f4') / 255
            suffixes = ('input.rgba32f.bin', 'motion.rg32f.bin', 'depth.r32f.bin', 'output.rgba32f.bin')
            for suffix in suffixes + (() if single else ('control-mask.rgba32f.bin',)):
                assert (native / f'{filename}_{suffix}').read_bytes() == (repeat / f'{filename}_{suffix}').read_bytes()
            captured = np.fromfile(native / f'{filename}_input.rgba32f.bin', '<f4').reshape(256, 256, 4)[..., :3]
            assert pixels.astype('f2').astype('f4').tobytes() == captured.tobytes()
            assert not np.fromfile(native / f'{filename}_depth.r32f.bin', '<f4').any()
            motion_path = native / f'{filename}_motion.rg32f.bin'
            motion = np.fromfile(motion_path, '<f4').reshape(256, 256, 2).copy()
            pre_path = repeat / f"nvapi-trace/launch-{pres[i]['sequence']}-0.bin"
            params = pre_path.read_bytes()
            seed = struct.unpack_from('<I', params, 200)[0]
            history = bool(struct.unpack_from('<Q', params, 8)[0])
            bound, mask = False, None
            if single:
                requested = control(requests[i]['requested'])
                chain_case = f'{name}-frame{i:02d}-post'
                record = next(c for c in single_run['cases'] if c['name'] == chain_case)
                private_path = single_chain / f'outputs/{chain_case}/post256.rgba16f.bin'
                item = next(x for x in record['outputs'] if x['name'] == private_path.name)
                assert record['exitCode'] == 0 and not record['timedOut'] and sha(private_path) == item['sha256'].lower()
            else:
                changed = i in (1, 2)
                requested = NRControls(auto_mask=changed if name == 'auto-on' else False)
                bound = changed if name == 'mask-add' else not changed if name == 'mask-remove' else False
                assert seed == measure['frames'][i]['native_seed'] and history == measure['frames'][i]['native_history_bound']
                assert bool(struct.unpack_from('<Q', params, 32)[0]) == bound
                mask_path = native / f'{filename}_control-mask.rgba32f.bin'
                mask = expected_aux('mask-field', i)['control-mask.rgba32f.bin']
                assert mask.tobytes() == mask_path.read_bytes()
                sources.append(mask_path)
                if name.startswith('mask-'):
                    chain_case = f'{name}-frame{i:02d}'
                    record = next(c for c in binding_run['cases'] if c['name'] == chain_case)
                    private_path = binding_chain / f'outputs/{chain_case}/post256.rgba16f.bin'
                    item = next(x for x in record['outputs'] if x['name'] == private_path.name)
                    assert record['exitCode'] == 0 and not record['timedOut'] and sha(private_path) == item['sha256'].lower()
                else:
                    private_path = native / f'{filename}_output.rgba32f.bin'
            reset = bool(spec['reset']) if single else i == 0
            assert bool(js(native / f'{filename}_capture.json')['reset']) == reset
            target_path = native / f'{filename}_output.rgba32f.bin'
            target, target_meta = oracle(target_path, '<f4')
            private, private_meta = oracle(private_path, '<f4' if name == 'auto-on' else '<f2')
            sources += [motion_path, pre_path, target_path, private_path]
            case['frames'].append(dict(pixels=pixels, motion=motion, mask=mask if bound else None,
                controls=requested, reset=reset, seed=seed, history_bound=history,
                target=target, target_meta=target_meta, private=private, private_meta=private_meta))
        assert len(case['frames']) == 4
        cases.append(case)
    return cases, {str(path): sha(path) for path in sources}
