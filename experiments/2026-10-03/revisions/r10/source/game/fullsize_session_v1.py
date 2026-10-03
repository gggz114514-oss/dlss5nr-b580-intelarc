"""原尺寸（864x480，不降采样）NR 会话 —— 供实时链 `sr-fg` 使用。

产品 ``nr_runtime_v1.Session`` 在 864x480 输入上会套 ``Face480Scale``，把画布降采样到
**256x256**（142 有效行 + 顶部补 57），跑完再升回 864x480 —— 那就是产品在跑的 NR256。
本模块包一层，对外暴露与产品 Session **完全相同**的 ``process(rgb, motion, reset=)`` /
``close()`` 接口，内部走原尺寸路径：

  1. **不建、不套 scaler**：整帧 864x480 直接进 ``stack.model``；
  2. **装 ``rows_scopes.install(stack, 'both', None, None)``**：把两个定行（144 行 /
     64 行）INT8 FFN 调用换成全行实现；
  3. **``rows_scopes.dispatch_guard('both')`` 失败闭合**：跑出被替换的定行内核、或
     DEBUG 存储、或行数不对，当场抛错。

第 1、2 条合起来就是本仓库里"480 原尺寸快速版"的全部差异。

``c512`` / ``vforward`` / ``installed`` 三个闭包 **逐字移植**自
``reference/materials_v1/materials_entry_v1.py``（已审核的原尺寸入口），不新增算法、
不改计数契约（每帧 ``{c512:16, vit:8}``）。移植而不是 import，是因为那份是脚本
（逻辑写在 ``main()`` 里），且按本仓库惯例每轮各留一份自己的副本。

**产品源零改动**：不改 ``nr_runtime_v1.py``，不改 ``nr-b580/backend/nr_backend/*``
（那是精确线预编译包的 35 个受保护源，改一行整包失效）。
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import torch

EXACT = Path(__file__).resolve().parents[1] / 'exact'
FAST = Path(__file__).resolve().parents[1]
ROWS = FAST / 'game'

# 原尺寸只支持这一个输入尺寸；产品 Session 的 SOURCE_SIZES 里它是 (480, 864)。
FULLSIZE = (480, 864)
VARIANT = 'both'
EXPECTED_FFN = dict(c512=16, vit=8)

_SERIAL = threading.RLock()


@dataclass(frozen=True)
class FullsizeFrameResult:
    """与产品 ``FrameResult`` 同名同形的 ``.color``，另带 FFN 计数供对账。"""
    color: object
    low: object
    sequence: int
    reset_applied: bool
    ffn_calls: dict


class FullsizeSession:
    """原尺寸 NR 会话。接口对齐产品 Session，可直接交给 ``NativeCallback``。"""

    def __init__(self, exact_root, profile_path, profile_sha256, *, verify_paths=True):
        # rows_scopes_v1 / c512_int8_ffn_rows_v1 / int8_ffn_segment_rows_v1 /
        # rows_paths_v1 都在 fullsize_rows_v1 下；product/comfy 提供
        # runtime_environment。bootstrap() 只加了 EXACT/reference 与 int8 的
        # backend/experimental/product，这两个要自己补。
        for entry in (str(ROWS), str(FAST / 'experimental/fp8_unround_overlay/modules')):
            if entry not in sys.path:
                sys.path.insert(0, entry)

        import rows_paths_v1 as paths
        import rows_scopes_v1 as rows_scopes
        import nr_backend.split_block as split
        import nr_backend.vit_block as vit
        from nr_backend.execution import use_arithmetic_backend
        from nr_runtime_v1 import Session as ProductSession

        self._rows_scopes = rows_scopes
        self._split = split
        self._vit = vit
        self._use_arithmetic_backend = use_arithmetic_backend

        if verify_paths:
            paths.require(rows_scopes, ROWS / 'rows_scopes_v1.py')
            paths.require(rows_scopes.c512_kernels, ROWS / 'c512_int8_ffn_rows_v1.py')
            paths.require(rows_scopes.vit_kernels, ROWS / 'int8_ffn_segment_rows_v1.py')
            import nr_backend
            paths.require(nr_backend, FAST / 'experimental/fp8_unround_overlay/nr_backend/__init__.py')
            agreement = paths.duplicate_source_agreement()
            assert all(row['same'] for row in agreement.values()), \
                [name for name, row in agreement.items() if not row['same']]

        # 产品 Session 只用来持有 Stack 与 profile 校验；我们不调它的 process()，
        # 所以 Face480Scale 永远不会被建。
        self._product = ProductSession.create(exact_root, profile_path, profile_sha256)
        stack = self._stack = self._product._stack

        rows_scopes.install(stack, VARIANT, None, None)
        self._counters = rows_scopes.FfnCounters(stack).attach()
        self._installed_ffn = dict(
            variant=VARIANT,
            c512=type(stack.c512_int8.ffn).__name__,
            vit=type(stack.int8_vit.ffn).__name__,
            expected_calls_per_frame=dict(EXPECTED_FFN))
        self._closed = False
        self._failed = False

        # ------------------------------------------------------------------
        # 以下三个闭包逐字移植自 materials_entry_v1.py:172-230（见模块 docstring）
        # ------------------------------------------------------------------
        counts = dict(c512=0, vit=0)

        def c512(module, x):
            name = stack.c512_int8.modules[id(module)]
            # rows_scopes.install() 已把这个 callable 指向冻结的全行 INT8 内核。
            mlp = stack.c512_int8.ffn(name, x)
            h, w = x.shape[:2]
            sy, sx = module.window_shift
            padded = torch.nn.functional.pad(mlp, (0, 0, sx, (-w - sx) % 8, sy, (-h - sy) % 8))
            attended = split.q(module.attention(padded)[sy:sy + h, sx:sx + w])
            full = module.projection.forward_unquantized(attended, mlp)
            output = split.q(full)
            counts['c512'] += 1
            if module.final_weight is None:
                return (mlp, mlp, attended, output)
            top = (full[0::2, 0::2] + full[0::2, 1::2]).half()
            bottom = (full[1::2, 0::2] + full[1::2, 1::2]).half()
            pool = split.q(((top + bottom).half() * .25).half())
            pool = torch.nn.functional.pad(pool, (0, 0, 0, (-pool.shape[1]) % 4, 0, (-pool.shape[0]) % 4))
            final = split.q(split.dot(pool, module.final_weight, chunk_k=16))
            return (mlp, mlp, attended, output, pool, final)

        def vforward(module, x):
            index = stack.int8_vit.modules[id(module)]
            x = vit.q(x)
            mlp = vit.q(stack.int8_vit.ffn(index, x)[0])
            tokens = x.shape[0]
            z = (vit.dot(mlp[:, :512], module.qkv_weight[:512], chunk_k=16)
                 + vit.dot(mlp[:, 512:], module.qkv_weight[512:], chunk_k=16)).half().reshape(tokens, 32, 3, 32)
            query = vit.q((vit.normalize_c32(z[:, :, 0]) * 5.65625).half() * module.query_scale[None, :, None])
            key = vit.q(vit.normalize_c32(z[:, :, 1]))
            value = vit.q(z[:, :, 2])
            attended = vit.vit_attention(query.transpose(0, 1), key.transpose(0, 1),
                                         value.transpose(0, 1)).transpose(0, 1).reshape(tokens, 1024)
            counts['vit'] += 1
            return vit.q(vit.split_k_projection(attended, module.projection, (mlp * module.attn_skip).half()))

        @contextmanager
        def installed():
            excluded = (stack.graph, stack.rewrite, stack.call_guard, stack.compact_queries)
            stack.call_guard.validate()
            with ExitStack() as scopes:
                for component in stack.components:
                    if all(component is not item for item in excluded):
                        scopes.enter_context(component.installed())
                old = (split.SplitSwinBlock.forward, split.SplitSwinBlock.forward_boundaries,
                       vit.VitBlock.forward)
                split.SplitSwinBlock.forward = lambda m, x: c512(m, x)[-1]
                split.SplitSwinBlock.forward_boundaries = c512
                vit.VitBlock.forward = vforward
                try:
                    yield
                finally:
                    (split.SplitSwinBlock.forward, split.SplitSwinBlock.forward_boundaries,
                     vit.VitBlock.forward) = old
            stack.call_guard.validate()

        self._counts = counts
        self._installed = installed

    # ------------------------------------------------------------------ API

    def _ready(self):
        if self._closed:
            raise RuntimeError('NR session is closed')
        if self._failed:
            raise RuntimeError('NR execution failed; close and recreate the session')

    def process(self, rgb, motion, *, reset=False):
        """一帧。签名与产品 Session 一致；rgb/motion 是设备上 float32 HWC/HWC2。"""
        with _SERIAL:
            self._ready()
            if not isinstance(rgb, torch.Tensor) or not isinstance(motion, torch.Tensor):
                raise TypeError('Expected GPU tensors')
            size = tuple(rgb.shape[:2])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or size != FULLSIZE:
                raise ValueError(f'Fullsize NR expects HWC RGB {FULLSIZE[::-1]}, got {size}')
            if rgb.device.type != 'xpu' or rgb.dtype != torch.float32:
                raise ValueError('Color must be float32 SDR RGB on XPU')
            if motion.device != rgb.device or motion.dtype != torch.float32 or motion.shape != (*size, 2):
                raise ValueError('Motion must be same-device float32 HWC2 pixel displacement')

            before = dict(self._counts)
            self._counters.reset()
            try:
                with self._rows_scopes.dispatch_guard(VARIANT), \
                        self._installed(), torch.inference_mode(), \
                        self._use_arithmetic_backend('triton'):
                    low = self._stack.model(rgb, motion.float(), reset=bool(reset))
                    color = low.float()
                    torch.xpu.synchronize()
                # 计数契约：跑错调度不许静默通过。
                assert (self._counts['c512'] - before['c512'] == EXPECTED_FFN['c512']
                        and self._counts['vit'] - before['vit'] == EXPECTED_FFN['vit']), self._counts
                totals = self._counters.totals()
                assert totals == EXPECTED_FFN, (totals, EXPECTED_FFN)
            except BaseException:
                self._failed = True
                raise
            return FullsizeFrameResult(color, low, self._stack.model.next_seed,
                                       bool(reset), totals)

    def reset(self):
        with _SERIAL:
            self._ready()
            self._stack.model.reset()

    def warmup(self, frames=3):
        """把本形状要用的 Triton 内核全部编完，再让 worker 接管。

        为什么必须在 worker 之前：`nr_worker.cpp:2101` 的解码线程只等
        ``std::chrono::seconds(30)`` 就判 ``decoder_queue_timeout``。原尺寸路径
        的内核集与 NR256 **不同**（`product/precompile/README.md:37-39`），
        首帧要现场编译几十个内核 ⇒ 远超 30 s ⇒ 解码线程先超时，
        整个臂在 5 帧处中止（2026-09-20 17:21 实测，`block=decoder_queue_timeout`）。
        预热把编译搬到流水线之外，30 s 窗口里只剩执行。

        用零输入：内核按**形状**特化，与像素值无关。首帧带 ``reset=True``
        （复位路径），随后两帧走稳态路径，两类都要覆盖。预热后显式 ``reset()``，
        免得把预热的历史留给真实首帧。

        返回的 ``seconds`` 只用于记录装配成本，**不是**任何加速比的分母。
        """
        with _SERIAL:
            self._ready()
            h, w = FULLSIZE
            rgb = torch.zeros((h, w, 3), dtype=torch.float32, device='xpu')
            motion = torch.zeros((h, w, 2), dtype=torch.float32, device='xpu')
            t0 = time.perf_counter()
            for i in range(int(frames)):
                self.process(rgb, motion, reset=(i == 0))
            self._stack.model.reset()
            return dict(frames=int(frames), size=[h, w],
                        seconds=round(time.perf_counter() - t0, 3))

    def close(self):
        with _SERIAL:
            if not self._closed:
                self._product.close()
                self._stack = None
                self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------- 报告用

    def report(self):
        return dict(kind='fullsize-session', variant=VARIANT, input=list(FULLSIZE),
                    downsampling=False, scaler=None, installed_ffn=self._installed_ffn,
                    expected_ffn_per_frame=dict(EXPECTED_FFN),
                    note=('no Face480Scale; full-row INT8 FFN; forward ported verbatim '
                          'from reference/materials_v1/materials_entry_v1.py'))
