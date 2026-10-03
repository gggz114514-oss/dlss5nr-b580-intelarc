"""CPU preparation of an isolated runtime view; model/resource bytes stay intact."""
from __future__ import annotations
import ast
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib
import release_contracts as c

SUPPORT = ('fast_cached_runtime_v1.py', 'portable_cache.py', 'runtime_environment.py')
REQUIRED_ASSETS = (
    'exact/model-assets/sf-v2/WEIGHTS_HT.bin',
    'exact/model-assets/noise-sm89-v2/manifest.json',
    'exact/model-assets/sigmoid-sm89-v1/manifest.json',
    'templates/profile-v1.json',
    'data/reference/experimental/cubic-fp8-lut-v3.npy',
    'data/reference/cubic-fp8-fused-cpu-v3.json',
    'data/reference/cubic-fp8-fused-xpu-v3.json',
)
WEIGHTS_SHA = '836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4'

def write_json(path, value):
    p = c.plain_path(path)
    payload = json.dumps(report_value(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)

def report_value(value):
    if is_dataclass(value):
        return report_value(asdict(value))
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            encoded = ('@tuple:'+json.dumps(key, separators=(',', ':')) if isinstance(key, tuple) else
                       '@int:'+str(key) if type(key) is int else key)
            c.require(type(encoded) is str and encoded not in out, 'invalid/colliding report key')
            out[encoded] = report_value(child)
        return out
    if isinstance(value, (tuple, list)):
        return [report_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (bool, str, int, float):
        return value
    # Actual device NamedTuple and enums are descriptive metadata, never
    # substituted for numeric counters, spilled bytes or execution proofs.
    if hasattr(value, '_asdict'):
        return report_value(value._asdict())
    raise TypeError('unsupported durable metadata type: ' + type(value).__name__)

def _profile(value, bundle_root, rows, destination=None):
    """Validate compressed arrays before changing ONLY their path fields."""
    refs = []
    def visit(v):
        if isinstance(v, list):
            return [visit(x) for x in v]
        if not isinstance(v, dict):
            return v
        out = {k: visit(x) for k, x in v.items()}
        if 'path' in v and 'sha256' in v:
            text = v['path']
            c.require(type(text) is str, 'profile array path must be text')
            name = text[8:] if text.startswith('${ROOT}/') else text
            name = c.relative(name)
            c.require(name in rows and rows[name]['sha256'] == v['sha256'], 'profile array not pinned in assets: ' + name)
            p = c.under(bundle_root, name)
            data = p.read_bytes()
            c.require(v.get('format') == 'npy+zlib' and len(data) == v['stored_bytes'] and
                      hashlib.sha256(data).hexdigest() == v['sha256'], 'compressed profile array bytes changed: ' + name)
            decoder = zlib.decompressobj()
            expected = v['raw_bytes']
            c.require(type(expected) is int and 0 < expected <= 16*1024*1024, 'invalid profile array size')
            unpacked = decoder.decompress(data, expected+65536+1)
            c.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
                      hashlib.sha256(unpacked).hexdigest() == v['npy_sha256'], 'profile NPY SHA/decompression mismatch: ' + name)
            actual = c.npy_bytes(unpacked, shape=v['shape'])
            c.require(actual['raw_sha256'] == v['raw_sha256'] and v['dtype'] == '<f4' and
                      expected == 4*__import__('math').prod(v['shape']), 'profile array raw contract mismatch: ' + name)
            refs.append(dict(relative_path=name, raw_sha256=actual['raw_sha256']))
            if destination is not None:
                out['path'] = str(c.under(destination, name))
        return out
    c.require(value.get('schema') == 1 and value.get('profile') == 'nr256-reviewed-v1' and
              len(value.get('vit_hidden_scales', [])) == 8 and len(value.get('c512', [])) == 16, 'incomplete real profile')
    c.require(all(row.get('variants', {}).get('floor16') for row in value['c512']), 'C512 floor16 calibration missing')
    rebound = visit(value)
    c.require(refs, 'profile arrays absent')
    return rebound, refs

def validate_asset_bundle(root):
    root = c.plain_path(root)
    manifest = c.read_json(root/'assets.json')
    c.require(manifest.get('schema') == 'nr-release-assets-v1', 'wrong assets.json schema')
    rows = {r['relative_path']: r for r in c.source_rows(manifest)}
    for name, row in rows.items():
        p = c.under(root, name)
        c.require(p.is_file() and c.sha(p) == row['sha256'], 'asset missing/SHA mismatch: ' + name)
        if 'bytes' in row:
            c.require(p.stat().st_size == row['bytes'], 'asset size mismatch: ' + name)
    for name in REQUIRED_ASSETS:
        c.require(name in rows, 'external asset required: ' + name)
    c.require(rows[REQUIRED_ASSETS[0]]['sha256'] == WEIGHTS_SHA, 'SF-v2 weights identity mismatch')
    # Complete native LUT domains, never a sampled/synthetic substitute.
    for prefix in ('exact/model-assets/noise-sm89-v2', 'exact/model-assets/sigmoid-sm89-v1'):
        value = c.read_json(root/prefix/'manifest.json')
        for name, expected in value['files'].items():
            rel = c.relative(prefix+'/'+name)
            c.require(rel in rows and rows[rel]['sha256'] == expected, 'native model LUT missing: ' + rel)
    template = manifest.get('profile_template', 'templates/profile-v1.json')
    c.require(template in rows, 'profile template must be pinned')
    _, refs = _profile(c.read_json(c.under(root, template)), root, rows)
    return dict(schema='nr-release-assets-check-v1', manifest=c.file_record(root/'assets.json'),
                file_count=len(rows), profile_template=template, profile_array_count=len(refs),
                profile_array_receipts=refs, driver_receipt=manifest.get('driver', 'not supplied'),
                GPU_executed=False)

def runtime_dependencies(runtime_root, asset_root):
    result, missing = {}, []
    for prefix in ('toolchain', 'host-helpers'):
        roots = [p/prefix for p in (asset_root, runtime_root) if (p/prefix).is_dir()]
        if not roots:
            missing.append(prefix)
            continue
        # Prefer the explicitly pinned bundle, otherwise inventory the actual
        # user-supplied runtime dependency tree before use.
        for name, path in c.all_files(roots[0]).items():
            if '__pycache__' not in path.parts and path.suffix != '.pyc':
                result[prefix+'/'+name] = path
    for name in ('toolchain/triton/__init__.py',
                 'toolchain/triton/backends/intel/lib/libsycl-spir64-unknown-unknown.bc',
                 'host-helpers/manifest.json', 'host-helpers/fast-cache-identity.json'):
        if name not in result:
            missing.append(name)
    return result, sorted(set(missing))

def source_selection(repo, source_manifest, experiment_source=None, experiment_manifest=None, arm=None):
    report, _ = c.validate_sources(repo, source_manifest)
    baseline = c.read_json(source_manifest)
    rows = c.source_rows(baseline)
    sources = {r['relative_path'][len('current/runtime/'):]: c.under(repo, r['relative_path'])
               for r in rows if r['relative_path'].startswith('current/runtime/')}
    constructor, controls, reference = dict(c.OPTIONS), dict(c.CONTROLS), None
    if experiment_source is not None:
        c.require(experiment_manifest is not None and arm, 'frozen experiment requires --manifest and --arm')
        exp_report, trees = c.validate_sources(experiment_source, experiment_manifest, exact=True)
        value = c.read_json(experiment_manifest)
        c.require(arm in value.get('arms', {}), 'unknown archived arm: ' + str(arm))
        spec = value['arms'][arm]
        constructor = spec['constructor']
        controls = c.validate_controls(spec.get('controls', {}))
        c.validate_constructor(constructor, trees['game/nr_game_fullsize.py'])
        for row in c.source_rows(value):
            sources[row['relative_path']] = c.under(experiment_source, row['relative_path'])
        reference = dict(arm=arm, revision=value.get('revision'), archived_manifest=c.file_record(experiment_manifest),
                         source=exp_report, original_source_root=value.get('source_root'),
                         result_claims_inherited=False, private_fixture_reproduced=False)
    else:
        c.require(experiment_manifest is None and arm is None, '--manifest/--arm need --source-root')
    c.validate_constructor(constructor, ast.parse(sources['game/nr_game_fullsize.py'].read_text('utf-8-sig')))
    actual_root = experiment_source if experiment_source is not None else repo/'current/runtime'
    return sources, dict(source=report, experiment_reference=reference, constructor=constructor, controls=controls,
                         numeric_identity=c.numeric_identity(actual_root,constructor))

def build_view(output, sources, runtime_root, asset_root, asset_report):
    view = c.under(output, 'runtime')
    c.require(not view.exists(), 'use a fresh output: runtime view already exists')
    dependencies, missing = runtime_dependencies(runtime_root, asset_root)
    c.require(not missing, 'missing real runtime dependencies: ' + ', '.join(missing))
    rows = c.source_rows(c.read_json(asset_root/'assets.json'))
    assets = {r['relative_path']: c.under(asset_root, r['relative_path']) for r in rows}
    copies = dict(dependencies)
    for name, path in sources.items():
        if name in copies:
            c.require(c.sha(copies[name]) == c.sha(path), 'source/dependency collision: ' + name)
        copies[name] = path
    for name, path in assets.items():
        if name in copies:
            c.require(c.sha(copies[name]) == c.sha(path), 'asset cannot replace published source: ' + name)
        copies[name] = path
    support_pins = c.read_json(c.HERE/'runtime_support/SOURCE_PINS.json')
    for name in SUPPORT:
        original = c.HERE/'runtime_support'/name
        c.require(c.sha(original) == support_pins['files'][name], 'runtime support drift: ' + name)
        if name in copies:
            c.require(c.sha(copies[name]) == c.sha(original), 'unexpected portable bootstrap version: ' + name)
        copies[name] = original
    view.mkdir(parents=True)
    inventory = {}
    for name, path in sorted(copies.items()):
        target = c.under(view, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        c.require(c.sha(path) == c.sha(target), 'runtime view copy changed: ' + name)
        inventory[name] = dict(source=c.file_record(path), copy=c.file_record(target))
    manifest = c.read_json(asset_root/'assets.json')
    profile_template = manifest.get('profile_template', 'templates/profile-v1.json')
    profile, refs = _profile(c.read_json(asset_root/profile_template), asset_root,
                            {r['relative_path']: r for r in rows}, destination=view)
    profile_path = view/'data/product-v1/profile-v1.json'
    c.require(not profile_path.exists(), 'bundle must supply a template, not an already absolute installed profile')
    write_json(profile_path, profile)
    # The old libdevice path is a portable cache identity token. A fresh public
    # compile must resolve a real local file. Rebind only this derived JSON;
    # both phases use the same actual portable fast_cache_options function.
    identity_path = view/'host-helpers/fast-cache-identity.json'
    identity = c.read_json(identity_path)
    device = view/'toolchain/triton/backends/intel/lib/libsycl-spir64-unknown-unknown.bc'
    c.require(c.sha(device) == identity['libdevice_sha256'], 'portable libdevice SHA mismatch')
    before_identity = c.file_record(identity_path)
    identity['libdevice_key'] = str(device)
    identity_path.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
    return view, dict(copied_files=inventory, profile=c.file_record(profile_path),
        profile_template=c.file_record(asset_root/profile_template), profile_arrays=refs,
        portable_libdevice_identity=dict(before=before_identity, after=c.file_record(identity_path),
                                        reason='new local compiler cache; no old absolute path token is opened'),
        files={name:c.sha(path) for name,path in c.all_files(view).items()},
        assets=asset_report, no_published_or_asset_file_changed=True)
