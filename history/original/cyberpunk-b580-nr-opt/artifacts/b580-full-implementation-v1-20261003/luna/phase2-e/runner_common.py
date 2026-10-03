"""CPU/stdlib PHASE2 contracts. Runtime and PHASE1 are strictly read-only."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(p for p in SCRIPT_DIR.parents if p.name == 'b580-full-implementation-v1-20261003')
PROJECT = ROOT.parent.parent
V4 = PROJECT / 'artifacts/vit-current-provider-gpu-runner-v4-20261002'
HERE = V4
RUNTIME = Path(r'G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime')
EVIDENCE_DATA = Path(r'D:\Codex-NR-Experiments\cyberpunk-opt\b580-full-implementation-v1-20261003\luna\phase2')
E_DATA = Path(r'E:\Codex-NR-Experiments\cyberpunk-opt\b580-full-implementation-v1-20261003\luna\phase2')
DATA = (E_DATA if SCRIPT_DIR == ROOT/'luna/phase2-e' else
        Path(r'D:\Codex-NR-Experiments\cyberpunk-opt\b580-full-implementation-v1-20261003\reviews\metrics-vectorized\runner4-candidate')
        if SCRIPT_DIR.is_relative_to(ROOT/'reviews/metrics-vectorized') else EVIDENCE_DATA)
SEED_ROOT = EVIDENCE_DATA.parent / 'phase1/attempt-03-shared-cache/shared-cache'
PHASE1_MANIFEST = ROOT / 'PHASE1.json'
PHASE1_SHA256 = '68c2dfdc4ee5f1f507e1012a8d64cc4ac9c092eac4a26397eca16be9d9e4df84'
V4_FREEZE_SHA256 = '98f4389d7942c219d43cd6073b2dbef288260694d08dbea01b7fd7f05556acea'
V4_PINS_SHA256 = 'c973ffdff807d31348a5ff07083def022f0e3ce8c9cb8e21c3813060c9c122ec'
SCHEMA = 'current720-full-implementation-phase2-v1'
TAGS, PREPARE_TAGS = ('PARENT_OFF',), ('PRECOMPILE_PARENT',)
FRAME_SHAPE = (720, 1280, 3)
CONTROLS = dict(style=0, intensity=1.0, local_tone=1.0, local_structure=1.0,
                auto_mask=False, skin_structure=None)
WORKLOAD = dict(fixed_frames=13, captures=2, replays=11, raw_events=50,
                warmup_cycles=2, measured_cycles=3, frames_per_cycle=13,
                height=720, width=1280, history_warp='fused', graph_replay=True,
                variant='unrounded', controls=CONTROLS)
HOOK_MODULES = frozenset({'audit_swin_720_v1', 'audit_vit_c512_720_v1',
                        'audit_front_decoder_post_720_v1', 'audit_history_host_720_v1'})
ACCEPTED_FLAGS = ('decoder_gather_unround_720', 'c32_hidden_native_720',
                  'c512_probability_unround_720', 'c128_pairwise_720',
                  'c64_attention_project_720', 'c128_attention_project_720')
CAPTURE_GATES = ('c32_hidden_native_ten', 'c512_decoder_all_eight',
                 'c512_probability_all_sixteen', 'c64_all_eight', 'c128_all_twelve',
                 'c128_pairwise_twelve', 'c64_attention_project_eight',
                 'c128_attention_project_twelve', 'c512_library_all_sixteen', 'native_k8_pre_post')
ARM = ''
MANIFEST_PATH = None
MANIFEST_SHA256 = None
_MANIFEST = None
sys.dont_write_bytecode = True

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def digest_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

def record(path):
    path = no_reparse(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)

def checked(row):
    require(isinstance(row, dict) and {'path', 'sha256'} <= set(row), 'Hashed file reference required')
    path = no_reparse(row['path']).resolve(strict=True)
    require(sha(path) == row['sha256'], 'File drift: ' + str(path))
    return path

def no_reparse(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()),
                'Path crosses a link/junction: ' + str(part))
        if part.exists():
            flags = getattr(part.lstat(), 'st_file_attributes', 0)
            require(not flags & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024),
                    'Path crosses a reparse point: ' + str(part))
    return path.resolve()

def output_path(path):
    path = no_reparse(path)
    require(path.is_relative_to(DATA.resolve()) and path != DATA.resolve(),
            'PHASE2 writes must stay beneath the selected task-owned phase2 root: ' + str(path))
    return path

def write(path, value, *, replace=False):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if replace else 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return record(path)

def relative_name(value):
    require(type(value) is str and value and '\\' not in value and ':' not in value,
            'Use canonical relative_path with forward slashes')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == value
            and all(p not in ('', '.') for p in value.split('/')), 'Unsafe relative source: ' + value)
    return value

def source_rows_digest(rows):
    return digest_obj(sorted([dict(relative_path=r['relative_path'], sha256=r['sha256'])
                             for r in rows], key=lambda r: r['relative_path']))

def validate_manifest(value, *, expected_source_root=None):
    require(isinstance(value, dict) and value.get('schema') == SCHEMA, 'Wrong PHASE2 manifest schema')
    require({'source_root', 'source_files', 'source_manifest_sha256', 'arms',
             'parent_manifest', 'cache_seed_root'} <= set(value), 'Incomplete PHASE2 manifest')
    source = no_reparse(value['source_root']).resolve(strict=True)
    revision = value.get('revision', 1)
    require(type(revision) is int and revision >= 1, 'Positive integer Main source revision required')
    expected = (ROOT / 'phase2-complete/source' if revision == 1 else
                ROOT / ('phase2-complete/revisions/r%d/source' % revision)) if expected_source_root is None else Path(expected_source_root)
    require(source == expected.resolve() and source.is_dir(), 'Use Main frozen phase2-complete/source only')
    rows = value['source_files']
    require(isinstance(rows, list) and rows, 'Empty source manifest')
    names = set()
    for row in rows:
        require(isinstance(row, dict) and {'relative_path', 'sha256'} <= set(row), 'Bad source row')
        rel = relative_name(row['relative_path'])
        require(rel.casefold() not in names, 'Duplicate/case-aliased source path: ' + rel)
        names.add(rel.casefold())
        require(type(row['sha256']) is str and re.fullmatch('[a-f0-9]{64}', row['sha256']), 'Invalid source SHA')
        path = no_reparse(source / rel)
        require(path.is_relative_to(source) and path.is_file() and sha(path) == row['sha256'],
                'Frozen PHASE2 source drift: ' + rel)
        if 'path' in row:
            require(no_reparse(row['path']) == path, 'Source row path differs from relative_path')
    # Accept canonical sorted relative rows and Main's canonical relative->SHA map.
    # Both cover exactly the same complete byte inventory; no hash is optional.
    proofs = {'sorted_relative_rows': source_rows_digest(rows),
              'relative_sha_map': digest_obj({r['relative_path']: r['sha256'] for r in rows}),
              'manifest_rows': digest_obj(rows)}
    require(value['source_manifest_sha256'] in proofs.values(), 'Source manifest digest mismatch')
    actual = set()
    for path in source.rglob('*'):
        no_reparse(path)
        if path.is_file():
            actual.add(path.relative_to(source).as_posix().casefold())
    require(actual == names, 'Frozen source inventory has missing or unpinned extra files')
    fullsize = source / 'game/nr_game_fullsize.py'
    require(fullsize.is_file(), 'Missing FullsizeGameModes source')
    parsed = ast.parse(fullsize.read_text(encoding='utf-8-sig'))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'FullsizeGameModes')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    known = {a.arg for a in init.args.args + init.args.kwonlyargs} - {'self', 'exact_root', 'profile_path', 'profile_sha256'}
    arms = value['arms']
    require(isinstance(arms, dict) and arms and any(a.get('baseline') is True for a in arms.values()), 'Missing baseline arm')
    for name, arm in arms.items():
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) and name not in ('.', '..', 'all'), 'Unsafe arm name')
        require({'constructor', 'numeric_identity', 'selectors', 'packages', 'baseline'} <= set(arm), 'Incomplete arm: ' + name)
        constructor = arm['constructor']
        require(isinstance(constructor, dict) and not set(constructor) - known, 'Unknown constructor fields: ' + name)
        require(constructor.get('controlled') is True and constructor.get('graph_capture_policy') == 'all'
                and constructor.get('c512_qkv_library_720') is True and constructor.get('native_k8_720') is True
                and constructor.get('c128_dual_qkv_720', False) is False, 'Accepted C512+K8 mode changed')
        require(all(constructor.get(f) is True for f in ACCEPTED_FLAGS), 'Accepted six flags changed: ' + name)
        require(constructor.get('combo_modes') == [480, 540, 720], 'Accepted combo geometry changed')
        require(type(arm['baseline']) is bool and isinstance(arm['selectors'], (list, dict))
                and isinstance(arm['packages'], list), 'Malformed arm selectors/packages/baseline')
        require(arm['numeric_identity'] is None or type(arm['numeric_identity']) is str, 'Bad numeric identity')
        hooks = constructor.get('implementation_hooks_720') or []
        require(isinstance(hooks, list) and len({h['module'] for h in hooks}) == len(hooks), 'Duplicate hooks')
        for hook in hooks:
            require(hook.get('module') in HOOK_MODULES and hook.get('call', 'installed') == 'installed'
                    and hook.get('stage', 'before_numeric') in ('before_numeric', 'after_numeric')
                    and isinstance(hook.get('kwargs', {}), dict), 'Unknown implementation hook')
    if 'priority_order' in value or 'remaining_order' in value:
        order = value.get('priority_order', []) + value.get('remaining_order', [])
        require(len(order) == len(set(order)) and set(order) == set(arms), 'Priority/remaining order does not cover the exact registry')
    return dict(source_root=str(source), source_pin_count=len(rows), digest_encodings=proofs)

def configure(manifest_path, arm, expected_sha256=None):
    global _MANIFEST, MANIFEST_PATH, MANIFEST_SHA256, ARM
    MANIFEST_PATH = no_reparse(manifest_path).resolve(strict=True)
    MANIFEST_SHA256 = sha(MANIFEST_PATH)
    if expected_sha256:
        require(MANIFEST_SHA256 == expected_sha256, 'Manifest changed since supervisor pin')
    _MANIFEST = read(MANIFEST_PATH)
    revision = _MANIFEST.get('revision', 1)
    require(MANIFEST_PATH in ((ROOT / 'PHASE2.json').resolve(),
            (ROOT / ('phase2-complete/PHASE2-r%d.json' % revision)).resolve()), 'Use Main latest or matching immutable archived PHASE2 manifest')
    archive = ROOT / ('phase2-complete/PHASE2-r%d.json' % revision)
    require(archive.is_file() and sha(archive) == MANIFEST_SHA256, 'Latest manifest differs from immutable Main archive')
    validate_manifest(_MANIFEST)
    ARM = arm
    require(arm in _MANIFEST['arms'], 'Unknown arm: ' + arm)
    parent = _MANIFEST['parent_manifest']
    if type(parent) is str:
        require(no_reparse(parent) == PHASE1_MANIFEST.resolve(), 'Unexpected parent manifest')
    else:
        require(checked(parent) == PHASE1_MANIFEST.resolve() and parent['sha256'] == PHASE1_SHA256,
                'Unexpected parent manifest identity')
    require(sha(PHASE1_MANIFEST) == PHASE1_SHA256, 'PHASE1 parent manifest changed')
    require(no_reparse(_MANIFEST['cache_seed_root']) == SEED_ROOT.resolve(), 'Unexpected cache_seed_root')
    if 'composition' in _MANIFEST:
        composition = checked(_MANIFEST['composition'])
        require(composition == (ROOT / 'COMPOSITION.json').resolve(), 'Foreign composition receipt')
    ready = verify_ready()
    if revision == ready['target_manifest_revision']:
        require(MANIFEST_SHA256 == ready['target_manifest']['sha256'], 'Main manifest changed inside the runner target revision')
    return _MANIFEST

def manifest():
    require(_MANIFEST is not None and sha(MANIFEST_PATH) == MANIFEST_SHA256, 'Missing/drifted pinned manifest')
    return _MANIFEST

def source_root(tag='PARENT_OFF'):
    require(tag in TAGS + PREPARE_TAGS, 'Unknown worker tag')
    return Path(manifest()['source_root']).resolve()

def mode_options(tag='PARENT_OFF', config=None):
    source_root(tag)
    return copy.deepcopy(manifest()['arms'][ARM]['constructor'])

def selected_controls():
    value = dict(CONTROLS, **manifest()['arms'][ARM].get('controls', {}))
    require(not set(value)-set(CONTROLS) and type(value['style']) is int and value['style'] in (0,1,2)
            and type(value['auto_mask']) is bool, 'Unsupported control fields/style')
    for name in ('intensity','local_tone','local_structure','skin_structure'):
        v = value[name]
        require(v is None and name == 'skin_structure' or type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 2,
                'Control must be finite and in actual NRControls domain [0,2]: ' + name)
    return value

def workload():
    result = copy.deepcopy(WORKLOAD)
    result['controls'] = selected_controls()
    return result

def import_file(name, path):
    path = Path(path).resolve(strict=True)
    module = sys.modules.get(name)
    if module is not None:
        require(Path(module.__file__).resolve() == path, 'Foreign module binding: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module

def effective_numeric(tag='PARENT_OFF', config=None):
    constructor = mode_options(tag)
    options = constructor.get('numeric_cleanup_720')
    expected = manifest()['arms'][ARM]['numeric_identity']
    if options is None:
        require(expected is None, 'Numeric identity without options')
        return dict(values=None, identity=None, suite_identity=None, declared_identity=None, children=[])
    module = import_file('_phase2_numeric_options', source_root(tag) / 'game/numeric_cleanup_options_720_v1.py')
    parsed = module.NumericCleanupOptions.parse(options)
    require(expected is None or parsed.identity == expected, 'Effective numerical identity differs from manifest')
    rel = 'game/implementation_hooks_720_v1.py'
    path = source_root(tag) / rel
    pin = next(row['sha256'] for row in manifest()['source_files'] if row['relative_path'] == rel)
    require(sha(no_reparse(path)) == pin, 'Frozen implementation hook identity source drift')
    hooks_module = import_file('_phase2_implementation_hooks', path)
    hooks = hooks_module.parse_hooks(constructor.get('implementation_hooks_720') or [])
    hooks_id = hooks_module.hooks_identity(hooks) if hooks else None
    suite_identity = parsed.identity + ':' + hooks_id if hooks_id is not None else parsed.identity
    # Child guards bind NumericCleanupOptions.identity. The composed suite binds
    # that base plus the parsed selected hook specs, independently of snapshots.
    return dict(values=parsed.to_dict(), identity=parsed.identity, suite_identity=suite_identity,
                suite_hooks_identity=hooks_id, suite_identity_source=record(path), declared_identity=expected,
                children=[label for label, _, _ in parsed.selected_scopes()] if parsed.active else [])


def expected_graph_counts(workload):
    cycles = workload['warmup_cycles'] + workload['measured_cycles']
    require(workload['captures'] + workload['replays'] == workload['fixed_frames'], 'Invalid fixed frame route workload')
    warm_frames = workload['frames_per_cycle'] * cycles
    return dict(entries=workload['captures'], capture_routes=workload['captures'],
                replay_routes=workload['replays']+warm_frames, warm_frames=warm_frames,
                # Frozen GraphFront.forward performs one counted graph replay
                # for each processed frame, including each newly built entry.
                graph_replay_counter=workload['fixed_frames']+warm_frames)

def verify_ready():
    path = SCRIPT_DIR / 'READY.json'
    value = read(path)
    require(value['schema'] == 'b580-phase2-runner-ready-v1' and value['GPU_executed'] is False,
            'Missing CPU-only runner READY')
    for rel, expected in value['files'].items():
        require(sha(SCRIPT_DIR / relative_name(rel)) == expected, 'Runner changed after READY: ' + rel)
    for row in value['read_only_dependencies']:
        checked(row)
    return value

def runtime_pins():
    require(sha(V4 / 'FREEZE.json') == V4_FREEZE_SHA256 and sha(V4 / 'SOURCE_PINS.json') == V4_PINS_SHA256,
            'Frozen v4 runtime/fixture pin changed')
    pins = read(V4 / 'SOURCE_PINS.json')
    freeze = read(V4 / 'FREEZE.json')
    for rel in ('runner_common.py', 'module_sources.py', 'FIXTURE_MANIFEST.json'):
        require(sha(V4 / rel) == freeze['files'][rel], 'Frozen v4 dependency changed')
    return pins

def allowed_source_map():
    result = {str((source_root() / r['relative_path']).resolve()): r['sha256']
              for r in manifest()['source_files']}
    old = runtime_pins()
    for row in old['current_runtime_sources'] + old['runtime_assets']:
        result[str(Path(row['path']).resolve())] = row['sha256']
    return result

def fixture(scan=False):
    runtime_pins()
    helper = import_file('_phase2_v4_common', V4 / 'runner_common.py')
    value, protocols = helper.fixture(scan=scan)
    require(value['controls'] == CONTROLS and len(value['frames']) == 13, 'Frozen fixture controls/frame count changed')
    return value, protocols

def verify_sources():
    manifest()
    source = validate_manifest(manifest())
    verify_ready()
    pins = runtime_pins()
    for row in pins['current_runtime_sources'] + pins['runtime_assets'] + pins['unchanged_evidence']:
        checked(row)
    return dict(source, manifest=record(MANIFEST_PATH), parent_manifest=record(PHASE1_MANIFEST),
                fixture_manifest=record(V4 / 'FIXTURE_MANIFEST.json'))

def cache_inventory(root):
    root = no_reparse(root)
    return {p.relative_to(root).as_posix(): sha(no_reparse(p))
            for p in sorted(root.rglob('*')) if p.is_file()}

def verify_prepared(prepared):
    prepared = output_path(prepared)
    value = read(prepared / 'READY.json')
    require(value['schema'] == 'b580-phase2-precompiled-ready-v1' and value['arm'] == ARM
            and value['manifest_sha256'] == MANIFEST_SHA256
            and value['constructor_sha256'] == digest_obj(mode_options())
            and value['workload'] == workload(), 'No matching PHASE2 prepared receipt')
    child = read(checked(value['child']))
    proc = read(checked(value['process']))
    require(child['completed'] is True and child['phase'] == 'precompile'
            and proc['returncode'] == 0 and proc['timed_out'] is False,
            'Precompile model/retirement did not complete')
    require(value['compiler_keys'] == child['cache_gate']['actual_compiler_keys'] and value['compiler_keys'],
            'Prepared actual compiler keys differ')
    cache = output_path(value['cache'])
    require(cache_inventory(cache) == value['cache_inventory'], 'Prepared cache drifted')
    return value

def stats(samples):
    require(samples and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
            'Expected finite positive device event timings')
    ordered = sorted(samples)
    n = len(samples)
    return dict(count=n, samples_ms=samples, mean_ms=sum(samples) / n,
                p50_ms=(ordered[(n-1)//2] + ordered[n//2]) / 2,
                p95_ms=ordered[math.ceil(.95*n)-1])
