"""Portable, stdlib-only contracts. Never imports the model or device libraries."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTROLS = dict(style=0, intensity=1.0, local_tone=1.0, local_structure=1.0,
                auto_mask=False, skin_structure=None)
FLAGS = ('decoder_gather_unround_720', 'c32_hidden_native_720',
         'c512_probability_unround_720', 'c128_pairwise_720',
         'c64_attention_project_720', 'c128_attention_project_720')
NUMERIC_BASE = dict(branch_accum_families=['c128'], history_value='fp32_fractional', front_noise='native_both')
NUMERIC_BASE_IDENTITY = 'numeric-cleanup-720-v1:85549e9c272011a4ed4974a4d341f3e28280723afe3dabf29c9dd0d9af043938'
OPTIONS = dict(controlled=True, graph_capture_policy='all', combo_modes=[480, 540, 720],
               c512_qkv_library_720=True, native_k8_720=True,
               c128_dual_qkv_720=False, numeric_cleanup_720=NUMERIC_BASE, **dict.fromkeys(FLAGS, True))
GATES = ('c32_hidden_native_ten', 'c512_decoder_all_eight',
         'c512_probability_all_sixteen', 'c64_all_eight', 'c128_all_twelve',
         'c128_pairwise_twelve', 'c64_attention_project_eight',
         'c128_attention_project_twelve', 'c512_library_all_sixteen', 'native_k8_pre_post')

class ContractError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise ContractError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ContractError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=invalid)

def relative(value):
    require(type(value) is str and value and '\\' not in value and ':' not in value,
            'canonical relative path required: ' + repr(value))
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and
            all(x not in ('', '.', '..') for x in value.split('/')), 'unsafe relative path: ' + value)
    return value

def plain_path(path):
    p = Path(path).absolute()
    for part in (p, *p.parents):
        require(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()),
                'path crosses a link: ' + str(part))
        if part.exists():
            require(not getattr(part.lstat(), 'st_file_attributes', 0) &
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024), 'path crosses a reparse point: ' + str(part))
    return p.resolve()

def under(root, name):
    root = plain_path(root)
    p = plain_path(root / relative(name))
    require(p.is_relative_to(root), 'path leaves root: ' + name)
    return p

def file_record(path):
    p = plain_path(path)
    require(p.is_file(), 'missing file: ' + str(p))
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def all_files(root):
    root = plain_path(root)
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = plain_path(Path(directory) / name)
            if p.is_file():
                result[p.relative_to(root).as_posix()] = p
    return result

def source_rows(manifest):
    rows = manifest.get('source_files', manifest.get('files'))
    require(isinstance(rows, list) and rows, 'empty source inventory')
    result, names = [], set()
    for row in rows:
        require(isinstance(row, dict), 'invalid source row')
        name = relative(row.get('relative_path', row.get('path')))
        require(name.casefold() not in names, 'duplicate/case-aliased source: ' + name)
        names.add(name.casefold())
        require(type(row.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'invalid SHA: ' + name)
        if 'bytes' in row:
            require(type(row['bytes']) is int and row['bytes'] >= 0, 'invalid byte count: ' + name)
        result.append(dict(row, relative_path=name))
    if 'source_manifest_sha256' in manifest:
        normalized = [dict(relative_path=r['relative_path'], sha256=r['sha256']) for r in result]
        proofs = (digest(sorted(normalized, key=lambda r: r['relative_path'])),
                  digest({r['relative_path']: r['sha256'] for r in result}), digest(rows))
        require(manifest['source_manifest_sha256'] in proofs, 'source inventory digest mismatch')
    return result

def validate_sources(root, manifest_path, *, exact=False):
    root, manifest_path = plain_path(root), plain_path(manifest_path)
    manifest = read_json(manifest_path)
    rows = source_rows(manifest)
    trees = {}
    for row in rows:
        p = under(root, row['relative_path'])
        require(p.is_file(), 'missing pinned module/file: ' + row['relative_path'])
        require(sha(p) == row['sha256'], 'source SHA mismatch: ' + row['relative_path'])
        if 'bytes' in row:
            require(p.stat().st_size == row['bytes'], 'source byte count mismatch: ' + row['relative_path'])
        if p.suffix == '.py':
            trees[row['relative_path']] = ast.parse(p.read_text('utf-8-sig'), filename=row['relative_path'])
    if exact:
        require(set(all_files(root)) == {r['relative_path'] for r in rows},
                'source inventory has extra or missing files')
    return dict(schema='nr-release-source-check-v1', manifest=file_record(manifest_path),
                root=str(root), files=len(rows), python_files=len(trees),
                source_inventory_sha256=digest({r['relative_path']: r['sha256'] for r in rows}),
                exact_inventory=exact, GPU_executed=False), trees

def validate_controls(value):
    require(isinstance(value, dict) and not set(value)-set(CONTROLS), 'unknown control field')
    result = dict(CONTROLS, **value)
    require(type(result['style']) is int and result['style'] in (0, 1, 2), 'invalid style')
    require(type(result['auto_mask']) is bool, 'auto_mask must be boolean')
    for name in ('intensity', 'local_tone', 'local_structure', 'skin_structure'):
        v = result[name]
        require(v is None and name == 'skin_structure' or
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 2, 'invalid control: ' + name)
    require(result['intensity'] >= 1, 'intensity < 1 needs the separate real native adapter validation context')
    return result

def validate_constructor(value, tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FullsizeGameModes')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    require({'__init__', 'select', 'process', 'close'} <= set(methods), 'fullsize API incomplete')
    init = methods['__init__']
    known = {a.arg for a in init.args.args + init.args.kwonlyargs} - {'self', 'exact_root', 'profile_path', 'profile_sha256'}
    require(isinstance(value, dict) and not set(value)-known, 'unknown FullsizeGameModes constructor fields')
    for key in ('controlled', 'c512_qkv_library_720', 'native_k8_720', *FLAGS):
        require(value.get(key) is True, 'accepted current720 flag changed: ' + key)
    require(value.get('c128_dual_qkv_720', False) is False and value.get('graph_capture_policy') == 'all' and
            value.get('combo_modes') == [480, 540, 720], 'accepted geometry/graph selection changed')
    numeric = value.get('numeric_cleanup_720')
    require(isinstance(numeric,dict) and set(NUMERIC_BASE) <= set(numeric) and
            'c128' in numeric['branch_accum_families'] and
            numeric['history_value'] in ('fp32_fractional','fp32_all_paths') and
            numeric['front_noise'] == 'native_both', 'accepted numeric baseline was omitted/changed')
    process = methods['process']
    require({'height', 'reset', 'history_warp', 'graph_replay', 'controls', 'variant'} <=
            {a.arg for a in process.args.kwonlyargs}, 'fullsize process signature changed')
    modes = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'MODES' for t in n.targets))
    row = next(v for k, v in zip(modes.keys, modes.values) if ast.literal_eval(k) == 720)
    require(isinstance(row, ast.Call) and isinstance(row.func, ast.Name) and row.func.id == 'Mode', '720 Mode contract missing')
    require(tuple(ast.literal_eval(a) for a in row.args) ==
            (720, (720, 1280), (720, 1280), (768, 1280), (0, 0)), '720 active/model/padding contract changed')
    return dict(input=[720, 1280, 3], motion=[720, 1280, 2], padding=[768, 1280],
                downsampling=False, history_warp='fused', graph_replay=True, variant='unrounded', constructor=value)

def stdlib_source_module(path, name):
    """Only the actual declaration modules with reviewed stdlib imports."""
    tree = ast.parse(Path(path).read_text('utf-8-sig'))
    allowed = {'__future__','collections','contextlib','dataclasses','hashlib','importlib',
               'json','pathlib','sys','threading','types','typing'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            require(all(a.name.split('.')[0] in allowed for a in node.names), 'nonstdlib declaration import')
        if isinstance(node,ast.ImportFrom):
            require(node.level == 0 and node.module.split('.')[0] in allowed, 'nonstdlib declaration import')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try: spec.loader.exec_module(module)
    finally:
        if previous is None: sys.modules.pop(name,None)
        else: sys.modules[name] = previous
    return module

def numeric_identity(source_root, constructor):
    module = stdlib_source_module(source_root/'game/numeric_cleanup_options_720_v1.py','_release_numeric_options_cpu')
    options = module.NumericCleanupOptions.parse(constructor['numeric_cleanup_720'])
    base = options.identity
    if constructor['numeric_cleanup_720'] == NUMERIC_BASE:
        require(base == NUMERIC_BASE_IDENTITY, 'actual numeric baseline identity drift')
    hooks = constructor.get('implementation_hooks_720') or []
    hooks_id = None
    if hooks:
        hooks_module = stdlib_source_module(source_root/'game/implementation_hooks_720_v1.py','_release_hooks_cpu')
        hooks_id = hooks_module.hooks_identity(hooks_module.parse_hooks(hooks))
    return dict(base_identity=base, suite_identity=base+':'+hooks_id if hooks_id else base,
                hooks_identity=hooks_id, options=options.to_dict())

def npy_bytes(data, *, shape=None, rgb=False):
    """Validate primitive float32 .NPY without NumPy, including payload/finite."""
    require(data[:6] == b'\x93NUMPY' and len(data) >= 10, 'not a NPY file')
    major, minor = data[6:8]
    require((major, minor) in ((1, 0), (2, 0), (3, 0)), 'unsupported NPY version')
    size_len = 2 if major == 1 else 4
    require(len(data) >= 8+size_len, 'truncated NPY header')
    length = int.from_bytes(data[8:8+size_len], 'little')
    require(0 < length <= 65536, 'oversized NPY header')
    end = 8+size_len+length
    header = ast.literal_eval(data[8+size_len:end].decode('utf-8' if major == 3 else 'latin1').strip())
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'invalid NPY header')
    require(header['descr'] == '<f4' and header['fortran_order'] is False, 'NPY requires little-endian float32 C order')
    dims = header['shape']
    require(type(dims) is tuple and dims and all(type(x) is int and x > 0 for x in dims), 'invalid NPY dimensions')
    require(shape is None or dims == tuple(shape), 'NPY shape mismatch')
    payload = memoryview(data)[end:]
    require(len(payload) == math.prod(dims)*4, 'NPY payload length mismatch')
    lo, hi = math.inf, -math.inf
    for (v,) in struct.iter_unpack('<f', payload):
        require(math.isfinite(v), 'nonfinite NPY payload')
        if rgb:
            require(0 <= v <= 1, 'RGB must be SDR [0,1]')
        lo, hi = min(lo, v), max(hi, v)
    return dict(shape=list(dims), dtype='<f4', finite=True, min=lo, max=hi,
                raw_sha256=hashlib.sha256(payload).hexdigest())

def validate_inputs(path):
    p = plain_path(path)
    value = read_json(p)
    require(value.get('schema') == 'nr-release-inputs-v1', 'wrong input manifest schema')
    require(value.get('height') == 720 and value.get('width') == 1280 and
            value.get('motion_contract') == 'current-to-previous pixel displacement', 'input geometry/motion contract mismatch')
    frames = value.get('frames')
    require(isinstance(frames, list) and len(frames) == 13, 'fixed13 input required')
    for ordinal, row in enumerate(frames):
        require(type(row.get('reset')) is bool and row['reset'] == (ordinal == 0), 'reset must restart only frame0')
        require(row.get('ordinal') == ordinal and type(row['ordinal']) is int, 'input order mismatch')
        for role, channels in (('rgb', 3), ('motion', 2)):
            ref = row[role]
            f = under(p.parent, ref['path'])
            require(sha(f) == ref['sha256'], 'input file SHA mismatch: ' + role)
            actual = npy_bytes(f.read_bytes(), shape=(720, 1280, channels), rgb=role == 'rgb')
            require(actual['raw_sha256'] == ref['raw_sha256'], 'input raw SHA mismatch: ' + role)
    return dict(manifest=file_record(p), frames=13, scenario=value.get('scenario', 'external NPY'))
