"""corr_strip_manifest_v1：编译清单**纯收集**（不编译、不派发内核）。

## 为什么要有它

Triton 的编译是**惰性**的：某个 ``@triton.jit`` 内核第一次被调用时才编。所以
「这一臂一共需要编译哪些内核」这个清单，**只能靠真跑一次链路才能发现**；
而真跑一次就意味着**串行编译** —— 在 corr_strip_v1 r3 上实测：06 个臂总共 9m08s，
其中 ``00-warm-off`` 一个臂独占 7.5 分钟（其余五臂各 ~18 秒）。那 7.5 分钟
就是「首次全量编译」，且**只用了一个进程**。

本模块把那 7.5 分钟拆成两段：

* **发现段（本模块）**：照常跑链路，但把 ``JITFunction._do_compile`` 换成
  「记录参数 + 回一个空壳内核」——**不编译、不派发**。链路照常跑完（数值无意义，
  但清单完整），耗时秒级。
* **编译段（``corr_strip_parcompile_v1``）**：拿清单在 16 个进程里并行编译，
  **不占 GPU**。

两段合起来仍走**同一套 cache key**，所以编译产物落在与串行时**完全相同的目录**里。

## 为什么回「空壳内核」而不是 ``None``

``JITFunction.run`` 里写着 ``kernel = self._do_compile(...)`` / ``if kernel is None: return None``。
返回 ``None`` 会让 ``run`` 提前返回、调用点拿不到东西，链路容易断；回一个
``launch_metadata() -> None`` / ``run() -> None`` 的空壳，则 ``run`` 的后续代码
（grid 规范化、属性检查、launch）**全部照常走**，只是真正什么都没派发。

## 清单里存了什么

足以在别的进程里**逐位重建** ``ASTSource`` 与 options：

    module / qualname      重建 JITFunction（cache_key 随源码自然复现）
    signature              {参数名: 类型串}      —— 已可直接 JSON 化
    constexprs             {路径: 标量}          —— 已可直接 JSON 化
    attrs                  {路径: [["tt.divisibility", 16]]} —— 纯数据
    options                backend options 的 __dict__（**必须过 _enc**）
    target                 (backend, arch-dict, warp_size) —— Intel 的 arch 是 dict
    cache_dir_name         预期目录名（由 get_cache_key 预先算出，供编译段自检）
    src_hash / backend_hash / options_hash / env_vars / cache_key_len
                           逐项留痕：对不上时直接定位是哪一项漂的

## ⚠️ 一个**静默**的坑：options / target 里的 tuple

`XPUOptions.hash()` 是对 `__dict__` 做 `f'{name}-{val}'` 直接拼串再 sha256，
而 `str()` 对 list 与 tuple **不同**（`[1, 1, 1]` vs `(1, 1, 1)`，**长度一样**）。
`options.__dict__` 里恰好有四个序列字段**本来就是 tuple**
（`cluster_dims`、`supported_fp8_dtypes`、`deprecated_fp8_dot_operand_dtypes`、
`allowed_dot_input_precisions`），`extern_libs` 是 tuple-of-tuple；
`target.arch` 也可能带 tuple。若这些字段**裸存**进 JSON，tuple 会被压成 list，
而 `parse_options()` **不会**把它们还原成 tuple —— 于是编译段重算出的 cache key
与清单**长度丝毫不差、内容全错**，产物落进另一个目录，正式臂全部重新编译。

实测：裸存时 329/329 条目录名全错（164 off + 165 nat）；按 `_enc/_dec` 还原后
329/329 全中。**所以这里一律走 `_enc`。**

``cache_dir_name`` 是自检用的：编译段编完后要确认产物**真的落在这个目录**，
否则说明重建的 ``ASTSource`` / options 与原件不等价，就**不能**当作命中。
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

RECORDS = []
_STATE = {'installed': False, 'orig': None, 'path': None, 'errors': [], 'unwrapped': 0}


def _b32(digest_hex):
    """sha256 摘要（hex） → Triton 落盘用的目录名。

    实测口径：``base32decode(目录名 + "=" * pad) == bytes.fromhex(_kernel.json["hash"])``，
    且目录名**不带** ``=`` 填充（32 字节 → 52 字符，Triton 落盘时把填充去掉了）。
    """
    import base64
    return base64.b32encode(bytes.fromhex(digest_hex)).decode('ascii').rstrip('=')


def _enc(value):
    """把任意值编码成 JSON-safe 结构（保留 tuple/list/dict 的区别）。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return {'@seq': 'tuple' if isinstance(value, tuple) else 'list',
                'v': [_enc(item) for item in value]}
    if isinstance(value, dict):
        return {'@map': [[_enc(k), _enc(v)] for k, v in value.items()]}
    return {'@repr': repr(value), '@type': type(value).__name__}


def _dec(node):
    """``_enc`` 的逆运算。"""
    if isinstance(node, dict):
        if '@seq' in node:
            items = [_dec(item) for item in node['v']]
            return tuple(items) if node['@seq'] == 'tuple' else items
        if '@map' in node:
            return {_dec(k): _dec(v) for k, v in node['@map']}
        if '@repr' in node:
            return node['@repr']          # 落到此处说明该值无法无损重建（编译段会报）
        return {k: _dec(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_dec(item) for item in node]
    return node


class _Noop:
    """万能空操作：能调用、能取属性、**布尔为假**、**数值为 0**。

    用来冒充内核对外的所有附属字段（``metadata.shared`` 之类），既让
    ``if kernel.asm:`` 走空分支，也不会在算术里炸掉。

    **单例**：产品代码里有 ``assert self.resources[key] == resource`` 这类
    「同 key 必同资源」的一致性检查；若每次都返回新实例，两次取值就不相等，
    断言会把链路打断（实测：卡在 ``decoder_gather_scope_v1.merge``）。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _Noop()

    def __getitem__(self, key):
        return _Noop()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __int__(self):
        return 0

    def __index__(self):
        return 0

    def __float__(self):
        return 0.0

    def __add__(self, other):
        return 0

    def __sub__(self, other):
        return 0

    def __mul__(self, other):
        return 0

    __radd__ = __rsub__ = __rmul__ = __add__

    def __hash__(self):
        return id(self)


class _Shell:
    """空壳内核：链路继续走，但什么也不派发。

    必须看起来像 ``CompiledKernel`` 到足以骗过两层包装：
    * ``JITFunction.run`` 读 ``launch_metadata`` / ``run`` / ``function`` / ``packed_metadata``；
    * Intel 后端 ``triton/backends/intel/track.py`` 在外层又包了一次 ``_do_compile``，
      退出时读 ``kernel.asm``，并在 ``TRACK_RUN`` 打开时读 ``kernel._init_handles``
      —— 缺了它，整个 harness 会以 ``AttributeError`` 中断（实测：清单只收到 34/165 条）。

    ``install`` 会先把 track 那层剥掉，这里的补全作为第二道保险。
    """

    function = None
    packed_metadata = None
    metadata = _Noop()          # 有人读 kernel.metadata.shared ⇒ 给个可容忍一切的假对象
    asm = None
    name = '_corr_strip_shell'
    n_regs = 0
    n_spills = 0
    n_max_threads = 1
    is_gluon = False

    def launch_metadata(self, *args, **kwargs):
        return None

    def run(self, *args, **kwargs):
        return None

    def _init_handles(self):
        return None

    def get_kernel(self):
        return None

    def __getattr__(self, name):        # 兜底：其余属性一律给空操作（且布尔为假）
        return _Noop()


_SHELL = _Shell()


def install(path):
    """装上收集钩子。**必须在第一帧之前、且在 ``install(mode)`` 生效之后调用**。"""
    from triton.runtime.jit import JITFunction

    if _STATE['installed']:
        return _STATE['orig']

    # Intel 后端 track.py 会在**第一次编译时**惰性地把 ``JITFunction._do_compile``
    # 再包一层，退出时去读 ``kernel.asm`` / ``kernel._init_handles`` / ``kernel.metadata``。
    # 我们的空壳内核经不起那套读取，所以：
    #   1) 先把已经存在的包装剥掉（functools.wraps 留了 __wrapped__）；
    #   2) 再把它的「只装一次」守卫按住，别让它回头又包上来。
    current = JITFunction._do_compile
    unwrapped = 0
    while hasattr(current, '__wrapped__'):
        current = current.__wrapped__
        unwrapped += 1
    try:
        from triton.backends.intel import track as _track
        guard = getattr(_track, 'decorate_jit', None)
        if isinstance(guard, list) and guard:
            guard[0] = False
    except BaseException:                                  # noqa: BLE001
        pass
    _STATE['unwrapped'] = unwrapped
    _STATE['orig'] = current
    JITFunction._do_compile = current       # 先恢复干净的原函数，再套我们的钩子
    _STATE['path'] = Path(path)

    def spy(self, key, signature, device, constexprs, options, attrs, warmup):
        record = {
            'module': getattr(self, '__module__', None),
            'module_file': getattr(sys.modules.get(getattr(self, '__module__', None)),
                                   '__file__', None),
            'qualname': getattr(self, '__qualname__', None),
            'name': getattr(self, '__name__', None),
            'kernel_cache_key': getattr(self, 'cache_key', None),
            'call_key': str(key),
            'signature': {str(k): _enc(v) for k, v in (signature or {}).items()},
            # 键可能是 tuple（Triton 用路径元组标识参数）⇒ 存成键值对列表，
            # 否则 _enc 出来的结构不能当 JSON 对象的键。
            'constexprs': [[_enc(k), _enc(v)] for k, v in (constexprs or {}).items()],
            'attrs': [[_enc(k), _enc(v)] for k, v in (attrs or {}).items()],
            # ⚠️ options / target **必须**过 _enc：它们里面有 tuple
            # （``cluster_dims=(1,1,1)``、``extern_libs=(('libdevice', path),)``、
            #  ``supported_fp8_dtypes`` / ``allowed_dot_input_precisions`` 等）。
            # 裸存进 JSON 会被压成 list，而 ``XPUOptions.hash()`` 是对 ``__dict__`` 的
            # ``f'{name}-{val}'`` 直接 sha256 —— ``[1, 1, 1]`` 与 ``(1, 1, 1)``
            # **长度相同、内容不同**，于是 worker 重算出的 cache key 悄悄偏掉。
            # 实测：不还原时 329/329 条目录名全错；还原后 329/329 全中。
            'options': _enc(dict(getattr(options, '__dict__', {}) or {})),
            'cache_dir_name': None,
            'error': None,
        }
        try:                                  # 预先算出**预期目录名**，供编译段自检
            from triton.compiler.compiler import (ASTSource, get_cache_key,
                                                  get_cache_invalidating_env_vars)
            kernel_cache, _, target, backend, _ = self.device_caches[device]
            record['target'] = _enc([target.backend, target.arch, target.warp_size])
            src = ASTSource(self, signature, constexprs, attrs)
            raw_key = get_cache_key(src, backend, options, get_cache_invalidating_env_vars())
            record['cache_key_len'] = len(raw_key)
            # 逐项留痕：万一将来 cache key 又对不上，直接从清单读出是哪一项漂的，
            # 不必再跑一次 GPU 现场去猜。
            record['src_hash'] = src.hash()
            record['backend_hash'] = backend.hash()
            record['options_hash'] = options.hash()
            record['env_vars'] = sorted(get_cache_invalidating_env_vars().items())
            # Triton 落盘时的目录名 = base32(sha256(cache_key) 的 32 字节摘要)。
            # 实测：base32decode(目录名) == 该目录 _kernel.json 的 "hash" 字段。
            digest = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()
            record['cache_dir_name'] = _b32(digest)
        except BaseException as exc:                       # noqa: BLE001
            record['error'] = repr(exc)
            _STATE['errors'].append(repr(exc))
            kernel_cache, _, _, _, _ = self.device_caches[device]
        try:
            kernel_cache[key] = _SHELL        # 同 key 后续不再进 _do_compile
        except BaseException as exc:                       # noqa: BLE001
            _STATE['errors'].append('cache-put: %r' % (exc,))
        RECORDS.append(record)
        return _SHELL

    JITFunction._do_compile = spy
    _STATE['installed'] = True
    return _STATE['orig']


def dump(path=None, mode=None):
    """把清单写盘，返回摘要 dict。"""
    import os
    nat_state = {}
    try:                                  # 编译段要沿用主进程的硬件 cvt 结论，不能自己重探
        import corr_strip_nat_v1 as nat
        nat_state = dict(hardware_cvt=getattr(nat, '_ROUND_FP8_HARDWARE_CVT', None),
                         allow_hw_probe=getattr(nat, '_ALLOW_HW_PROBE', None))
    except BaseException:                              # noqa: BLE001
        pass
    target = Path(path or _STATE['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    # ★ module_dirs：把「这些模块当时是从哪个目录 import 到的」一并记下。
    #   编译段必须能 import 到同样的模块；而采集遍的 sys.path 是 harness 自己配的
    #   （例如 c512_int8_ffn_rows_v1 / int8_ffn_segment_rows_v1 在
    #   ``reference/fullsize_rows_v1/`` 下），光靠 backend/experimental 两个根不够。
    #   与其在编译段硬编码路径，不如让清单自带。
    module_dirs = sorted({str(Path(r['module_file']).parent)
                          for r in RECORDS if r.get('module_file')})
    payload = dict(
        kind='corr-strip-compile-manifest',
        mode=mode or os.environ.get('NR_CORR_STRIP', '<unset>'),
        count=len(RECORDS),
        nat_state=nat_state,
        module_dirs=module_dirs,
        records=RECORDS,
        errors=_STATE['errors'],
        unwrapped=_STATE.get('unwrapped', 0),
    )
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    dirs = {r.get('cache_dir_name') for r in RECORDS if r.get('cache_dir_name')}
    return dict(path=str(target), records=len(RECORDS),
                unique_dirs=len(dirs), module_dirs=len(module_dirs),
                errors=len(_STATE['errors']))


def loads(path):
    """读回清单，并把编码结构还原成 Python 值。"""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for record in payload.get('records', []):
        out.append(dict(
            module=record['module'],
            qualname=record['qualname'],
            name=record['name'],
            kernel_cache_key=record.get('kernel_cache_key'),
            call_key=record.get('call_key'),
            signature={k: _dec(v) for k, v in (record.get('signature') or {}).items()},
            constexprs={_dec(k): _dec(v) for k, v in (record.get('constexprs') or [])},
            attrs={_dec(k): _dec(v) for k, v in (record.get('attrs') or [])},
            # options / target 走 _dec（与 spy 的 _enc 对称），把 tuple 找回来。
            # 旧清单是裸存的，_dec 对它们是无损直通（tuple 已在落盘时丢失，无法找回）。
            options=_dec(record.get('options') or {}),
            target=_dec(record.get('target')),
            src_hash=record.get('src_hash'),
            backend_hash=record.get('backend_hash'),
            options_hash=record.get('options_hash'),
            env_vars=record.get('env_vars'),
            cache_dir_name=record.get('cache_dir_name'),
        ))
    return out
