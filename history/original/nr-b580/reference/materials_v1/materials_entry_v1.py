"""块 1 在**新素材**上的执行入口 —— C512 前馈算术：INT8（现状）vs FP16（候选）。

与 ``reference/block1_v1/block1_entry_v1.py`` 逐字相同的算法，只有三处差别，
全部是为了让它能吃「不是那份 243 帧冻结素材」的输入：

  1. ``--exact-validation`` 指向本轮 ``exact_capture_v1.py`` 产出的 validation.json，
     不再写死 ``rows_paths_v1.EXACT_VALIDATION``；
  2. ``--baseline-validation`` 可选。给了就做「与冻结快速产物逐字节比较」，
     没给（新素材的常态）就只记录 ``rmse``（候选 vs 锚点），把 ``byte_equal`` 记成
     null —— 新素材上 int8 侧**就是**产品本身，没有更早的产物可复现，这是诚实口径；
  3. 帧数、输出尺寸都不再断言成常量，而是从锚点数组推导。

其余一律照抄：同一份 FP16 权重张量、同一条 ``rows_scopes.install`` 边界、
同样的计数契约（FP16 路径不进 ``c512_int8.ffn``，所以候选侧应为 ``{c512:0, vit:8}``）、
``DiskOnly`` 仍是参考侧默认，候选侧才允许 ``--allow-compile``。

阶段
  probe  少量帧冒烟，用于确认入口、计数与产物结构
  full   全量帧，产出可交给裁决工具链的候选帧
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROWS = Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/fullsize_rows_v1')

VARIANT = 'both'
ARITH = ('int8', 'fp16')


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, phase=args.phase, block='block1-c512-ffn-arith',
                  c512_arith=args.c512_arith, variant=VARIANT, frames=[],
                  performance_test=False, low_resolution_input=False,
                  exact_validation=str(args.exact_validation),
                  baseline_validation=str(args.baseline_validation) if args.baseline_validation else None,
                  arithmetic=('Frozen product FP16 weights/scales; C512 FFN in '
                              + ('frozen INT8 floor16 row-batched kernels'
                                 if args.c512_arith == 'int8' else
                                 'the model\'s own FP16 FFN path')
                              + '; ViT FFN frozen INT8 row-batched; generic full attention; eager diagnostic'),
                  candidate=('replaces only the C512 FFN arithmetic; ViT FFN, QKV dense, attention, '
                             'projection, final pooling, residual and Face480 scale untouched; '
                             'weights are the same FP16 tensors on both sides'),
                  byte_assertion=('recorded only; differences are expected and are not a failure'
                                  if args.baseline_validation is None else
                                  'recorded against the supplied frozen baseline; not a failure'))
    session = None
    fs_guard = None
    try:
        # The leaf-only rmdir guard must be installed before Triton is imported.
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        if str(ROWS) not in sys.path:
            sys.path.insert(0, str(ROWS))
        import rows_fscache_guard_v1 as fs_guard
        report['fs_guard'] = fs_guard.install()
        import rows_paths_v1 as paths
        sys.path.insert(0, str(paths.PRODUCT / 'comfy'))
        from runtime_environment import isolate
        isolate()
        sys.path.insert(0, str(paths.PRODUCT / 'precompile'))
        from fast_cached_runtime_v1 import bootstrap, DiskOnly
        bootstrap()
        for entry in (str(ROWS), str(HERE)):
            if entry not in sys.path[:8]:
                sys.path.insert(0, entry)
        if args.cache is not None:
            os.environ['TRITON_CACHE_DIR'] = str(Path(args.cache).resolve())
        import numpy as np
        import torch
        import nr_backend
        import nr_backend.split_block as split
        import nr_backend.vit_block as vit
        from nr_backend.execution import use_arithmetic_backend
        from nr_runtime_v1 import Session
        import rows_scopes_v1 as rows_scopes
        import c512_int8_ffn_rows_v1 as c512_rows
        import int8_ffn_segment_rows_v1 as vit_rows

        provider = paths.first_provider('nr_backend')
        agreement = paths.duplicate_source_agreement()
        assert all(row['same'] for row in agreement.values()), \
            [name for name, row in agreement.items() if not row['same']]
        report['imports'] = dict(
            entry=str(Path(__file__).resolve()),
            c512_rows=str(paths.require(c512_rows, ROWS / 'c512_int8_ffn_rows_v1.py')),
            vit_rows=str(paths.require(vit_rows, ROWS / 'int8_ffn_segment_rows_v1.py')),
            nr_backend=str(paths.require_directory(nr_backend, provider)),
            split_block=str(paths.require(split, provider / 'split_block.py')),
            vit_block=str(paths.require(vit, provider / 'vit_block.py')),
            runtime=str(paths.require(sys.modules['nr_runtime_v1'], paths.PRODUCT / 'nr_runtime_v1.py')),
            rows_entry=str(ROWS / 'rows_entry_v1.py'),
            backend_provider=str(provider),
            backend_duplicate_trees=[str(item) for item in paths.DUPLICATE_TREES],
            backend_duplicate_modules=len(agreement),
            backend_duplicates_identical=True)
        report['cache'] = os.environ.get('TRITON_CACHE_DIR')
        report['baseline_adapter_sha256'] = paths.BASELINE_ADAPTER_SHA256
        report['profile_sha256'] = paths.PROFILE_SHA256

        assert isinstance(report['frames'], list), type(report['frames']).__name__
        config = paths.read(paths.LOCAL_RUNTIME)
        capture = json.loads(Path(args.exact_validation).read_text(encoding='utf-8'))
        report['exact_validation_sha256'] = paths.sha(args.exact_validation)
        report['source'] = capture['source']
        report['source_sha256'] = capture['source_sha256']
        assert capture['passed'], 'exact capture did not pass'
        assert len(capture['frames']) > 0
        if args.frames is not None:
            assert len(capture['frames']) == args.frames, (len(capture['frames']), args.frames)
        report['frame_count'] = len(capture['frames'])
        report['input_contract'] = capture.get('input_contract')
        report['capture_entry_sha256'] = capture.get('capture_entry_sha256')

        # Only the original reviewed material has a frozen downstream identity to
        # re-assert; a new clip has none, and claiming one would be false.
        if Path(args.exact_validation).resolve() == Path(paths.EXACT_VALIDATION).resolve():
            report.update(paths.identity())
            report['frozen_reference'] = True
        else:
            report['frozen_reference'] = False
            report['frozen_reference_note'] = (
                'New material: no frozen fast-line artifact exists yet, so the INT8 side '
                'is the current product behaviour by construction and is not byte-compared '
                'against an older artifact.')

        for path, digest in capture['loaded_sources'].items():
            assert paths.sha(path) == digest, path
        report['loaded_source_count'] = len(capture['loaded_sources'])

        baseline_frames = None
        if args.baseline_validation is not None:
            baseline = json.loads(Path(args.baseline_validation).read_text(encoding='utf-8'))
            assert baseline['passed'], 'baseline validation did not pass'
            assert baseline['source_sha256'] == capture['source_sha256'], 'baseline is a different source'
            baseline_frames = baseline['frames']
            assert len(baseline_frames) == len(capture['frames']), \
                (len(baseline_frames), len(capture['frames']))
            report['baseline_validation_sha256'] = paths.sha(args.baseline_validation)
            report['baseline_frame_count'] = len(baseline_frames)

        session = Session.create(paths.NR, paths.PROFILE, config['profile_sha256'])
        stack = session._stack

        counts = dict(c512=0, vit=0)
        rows_scopes.install(stack, VARIANT, None, None)
        counters = rows_scopes.FfnCounters(stack).attach()
        report['installed_ffn'] = dict(variant=VARIANT, c512_arith=args.c512_arith,
                                       c512=type(stack.c512_int8.ffn).__name__,
                                       vit=type(stack.int8_vit.ffn).__name__,
                                       expected_calls_per_frame=dict(c512=16, vit=8))
        # The FP16 path never enters stack.c512_int8.ffn, so the counter at that
        # boundary is expected to read zero there. State both expectations so a
        # silent fall-back to the other arithmetic cannot pass as success.
        expected_counters = (dict(c512=16, vit=8) if args.c512_arith == 'int8'
                             else dict(c512=0, vit=8))
        report['expected_counters'] = expected_counters

        def c512(module, x):
            name = stack.c512_int8.modules[id(module)]
            if args.c512_arith == 'fp16':
                # Block 1 candidate: the model's own FP16 C512 feed-forward.
                mlp = module.ffwd_projection(module.ffwd(x), x)
            else:
                # rows_scopes.install() has already pointed this callable at the
                # frozen row-batched INT8 kernel, so this is the reference side.
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

        def one_frame(row):
            assert paths.sha(row['file']) == row['sha256']
            with np.load(row['file'], allow_pickle=False) as handle:
                rgb = torch.from_numpy(handle['rgb'].astype('f4')).to('xpu')
                motion = torch.from_numpy(handle['motion'].astype('f4')).to('xpu')
                exact = handle['exact'].astype('f4')
            before = dict(counts)
            counters.reset()
            torch.xpu.synchronize()
            start = time.perf_counter()
            with rows_scopes.dispatch_guard(VARIANT), \
                    installed(), torch.inference_mode(), use_arithmetic_backend('triton'):
                low = stack.model(rgb, motion.float(), reset=row['reset'])
                color = low.float()
                torch.xpu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            assert counts['c512'] - before['c512'] == 16 and counts['vit'] - before['vit'] == 8, counts
            totals = counters.totals()
            assert totals == expected_counters, (totals, expected_counters)
            return color.cpu().numpy(), exact, elapsed, totals

        selected = capture['frames'] if args.phase == 'full' else capture['frames'][:args.limit]
        report['selected_frames'] = len(selected)
        if args.limit is not None and args.phase != 'full':
            report['limit'] = args.limit

        # The reviewed INT8 path is fully covered by the frozen disk cache. The
        # model's FP16 feed-forward is a different kernel set and is not, so the
        # candidate side has to compile it once. Compilation is a warmup cost, not
        # part of the steady timing window, and DiskOnly stays the default so the
        # reference side can never silently compile anything.
        report['cache_mode'] = 'compile_allowed' if args.allow_compile else 'disk_only'

        @contextmanager
        def cache_scope():
            if args.allow_compile:
                yield None
            else:
                with DiskOnly() as disk:
                    yield disk

        arrays = args.out / 'frames'
        arrays.mkdir()
        with cache_scope() as cache:
            for row in selected:
                output, exact, elapsed, totals = one_frame(row)
                assert output.shape == exact.shape, (output.shape, exact.shape)
                assert np.isfinite(output).all()
                delta = (output.astype('f8') - exact) * 255
                result = dict(index=row['index'], input_sha256=row['sha256'],
                              ms=elapsed, ffn_calls=totals,
                              rmse=float(np.sqrt(np.mean(delta * delta))))
                if baseline_frames is not None:
                    frame = baseline_frames[row['index']]
                    assert paths.sha(frame['file']) == frame['sha256']
                    assert frame['input_sha256'] == row['sha256']
                    with np.load(frame['file'], allow_pickle=False) as handle:
                        expected_output = handle['fast']
                    assert output.shape == expected_output.shape, (output.shape, expected_output.shape)
                    assert output.dtype == expected_output.dtype, (output.dtype, expected_output.dtype)
                    result['byte_equal'] = output.tobytes() == expected_output.tobytes()
                    result['max_abs'] = float(np.max(np.abs(output - expected_output)))
                else:
                    result['byte_equal'] = None
                    result['max_abs'] = None
                path = arrays / f"{row['index']:04d}.npz"
                np.savez_compressed(path, fast=output)
                result.update(file=str(path), sha256=paths.sha(path))
                report['frames'].append(result)
                print(json.dumps(result), flush=True)
            report['cache_hits'] = None if cache is None else cache.hits
        timed = [row['ms'] for row in report['frames'][4:]]
        if timed:
            report['timing'] = dict(count=len(timed), median_ms=statistics.median(timed),
                                    mean_ms=statistics.mean(timed), min_ms=min(timed),
                                    max_ms=max(timed),
                                    scope='synchronized processing call; includes Python/scopes/guards; '
                                          'excludes upload, readback, IO, motion generation and codec')
        report['identical_frames'] = sum(1 for row in report['frames'] if row['byte_equal'])
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['adapter_file'] = str(Path(__file__).resolve())
        if fs_guard is not None:
            report['fs_guard'] = fs_guard.status()
        try:
            if session is not None:
                session.close()
        except BaseException:
            report['passed'] = False
            report['close_error'] = traceback.format_exc()
            raise
        finally:
            (args.out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['probe', 'full'], required=True)
    parser.add_argument('--c512-arith', dest='c512_arith', choices=list(ARITH), required=True)
    parser.add_argument('--exact-validation', dest='exact_validation', type=Path, required=True)
    parser.add_argument('--baseline-validation', dest='baseline_validation', type=Path, default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path,
                        help='Triton cache directory; defaults to the shared reviewed cache')
    parser.add_argument('--limit', type=int, default=2, help='probe phase: number of frames')
    parser.add_argument('--frames', type=int, default=None,
                        help='expected frame count; omit to accept whatever the capture holds')
    parser.add_argument('--allow-compile', dest='allow_compile', action='store_true', default=False,
                        help='let Triton compile the candidate-side kernels once instead of '
                             'requiring every kernel to hit the frozen disk cache')
    main(parser.parse_args())
