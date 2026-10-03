"""Bounded compile-first C512 attention/projection screen; no full model run.

Resources for every specialization are checked before candidate dispatch.
Then compare 4 shifts x 4 padding controls and a captured two-kernel replay.
No performance or full-frame/4060 parity claim follows from this local screen.
"""
from layout_crop_validation_env_v1 import *
import traceback
import numpy as np
import torch, triton
import c512_quad_queries_v2 as kernels
import window_block_attention_v3 as reference_attention
import c512_window_projection_v1 as reference_projection
from fork_fp8_jit_v2 import Fork

OUT = D/'results/c512-quad-queries-v2'
assert not OUT.exists()
OUT.mkdir()
report = dict(passed=False, phase='initializing', scope=__doc__, sources=sources,
    exact_gate=gate, queue_check=queue_check, resources=[], controls=[], graphs=[],
    candidate_dispatches=0, candidate_promoted=False, full_model_run=False,
    performance_claimed=False, nvidia_parity_claimed=False, saved_raw_tensors=False)
for name in ('c512_quad_queries_v2.py', 'screen_c512_quad_queries_v2.py',
             'audit_c512_quad_queries_v2.py', 'Run-C512QuadQueriesV2.cmd',
             'Run-C512QuadQueriesV2Audit.cmd', 'C512_QUAD_QUERY_DESIGN_V2.md',
             'window_block_attention_v3.py', 'c512_window_projection_v1.py',
             'fork_fp8_jit_v2.py', 'short_fp8_v2.py'):
    p = HERE/name
    sources[str(p)] = sha(p)
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda t: hashlib.sha256(raw(t)).hexdigest()


def save():
    p = OUT/'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.replace(OUT/'validation.json')


def phase(value):
    report['phase'] = value
    save()
    print(json.dumps(dict(phase=value)), flush=True)


def compile_checked(label, jit, args, grid, options):
    k = jit.warmup(*args, grid=grid, **options)
    k._init_handles()
    row = dict(label=label, hash=k.hash, spills=k.n_spills,
               registers=k.n_regs, shared_bytes=k.metadata.shared, grid=list(grid), options=options)
    report['resources'].append(row)
    save()
    assert isinstance(k.n_spills, int) and k.n_spills == 0, row
    return k


def packed_reference(value, inverse, shift):
    a = value.cpu().numpy()[..., inverse, :]
    hw = a.reshape(16, 2, 2, 8, 8, 32).transpose(0, 1, 3, 2, 4, 5).reshape(16, 16, 16, 32)
    sy, sx = shift
    crop = hw[:, sy:sy+12, sx:sx+12]
    return np.ascontiguousarray(crop.reshape(16, 3, 4, 3, 4, 32)
                              .transpose(0, 1, 3, 2, 4, 5).reshape(16, 9, 16, 32))


save()
fork = None
try:
    assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
    torch.set_num_threads(2)
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__,
                            device=torch.xpu.get_device_name())
    order = np.array([base+g%4+8*(g//4)+16*word for base in (0, 4, 32, 36)
                      for word in range(2) for g in range(8)], dtype='i8')
    inverse = np.argsort(order)
    shifts = [(0, 0), (0, 4), (4, 0), (4, 4)]
    report['geometry'] = []
    for shift in shifts:
        coords = []
        for g in kernels.tile_geometry(shift):
            tile, window, start = g['tile'], g['window'], g['query_start']
            for row in range(16):
                y, x = divmod(int(order[start+row]), 8)
                got = (window//2*8+y-shift[0], window%2*8+x-shift[1])
                want = (tile//3*4+row//4, tile%3*4+row%4)
                assert got == want
                coords.append(got)
        assert len(set(coords)) == 144 and set(coords) == {(y, x) for y in range(12) for x in range(12)}
        report['geometry'].append(dict(shift=list(shift), mapping=kernels.tile_geometry(shift),
                                        valid_queries=144, unique=True))
    rng = np.random.default_rng(481039)
    shape = (16, 2, 2, 64, 32)
    alphabet = np.array([-1., -.5, .25, .5, 1.], dtype='f2')
    host_qkv = [rng.choice(alphabet, shape) for _ in range(3)]
    q, k, v = [torch.from_numpy(a.copy()).to('xpu') for a in host_qkv]
    bias = torch.from_numpy(rng.choice(alphabet, (16, 64, 64))).to('xpu')
    weight = torch.from_numpy((rng.choice(alphabet, (512, 512))*.125).astype('f2')).to('xpu')
    residual = torch.from_numpy(rng.choice(alphabet, (12, 12, 512))).to('xpu')
    scale = torch.from_numpy(rng.choice(np.array([.5, 1., 2.], dtype='f2'), (512,))).to('xpu')
    inv = torch.from_numpy(inverse).to('xpu')
    attention_out = torch.empty((16, 9, 16, 32), device='xpu', dtype=torch.float16)
    full_out = torch.empty_like(residual)
    ref_attention_out = torch.empty_like(q)
    ref_full_out = torch.empty_like(residual)
    frozen = {n: digest(t) for n, t in [('bias', bias), ('weight', weight), ('residual', residual), ('scale', scale), ('inverse', inv)]}
    options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
    project_options = dict(num_warps=4, enable_fp_fusion=False)
    fork = Fork(ROOT)
    sources.update(fork.sources)
    with torch.inference_mode(), fork.installed():
        phase('compile_only_all_specializations')
        # Do not dispatch even a zero-input candidate before all resource gates pass.
        compile_checked('reference_attention', reference_attention._kernel,
            (q, k, v, bias, ref_attention_out, 4, 32), (2, 64), options)
        compiled = {}
        for shift in shifts:
            compiled[shift] = compile_checked('candidate_attention_'+str(shift), kernels._attend,
                (q, k, v, bias, attention_out, *shift), (9, 16), options)
            compile_checked('reference_projection_'+str(shift), reference_projection._project,
                (ref_attention_out, weight, residual, scale, inv, ref_full_out,
                 16, 16, 12, 12, *shift, 16, 32), (9, 16), project_options)
        compiled_project = compile_checked('candidate_projection', kernels._project,
            (attention_out, weight, residual, scale, full_out), (9, 16), project_options)
        report['all_resources_passed_before_candidate_dispatch'] = report['candidate_dispatches'] == 0
        phase('local_bytes_and_padding_controls')
        def launch(shift):
            assert kernels._attend[(9, 16)](q, k, v, bias, attention_out, *shift, **options) is compiled[shift]
            assert kernels._project[(9, 16)](attention_out, weight, residual, scale, full_out,
                                            **project_options) is compiled_project
            report['candidate_dispatches'] += 2
        for shift in shifts:
            padding = np.ones((2, 2, 64), dtype=bool)
            for y in range(12):
                for x in range(12):
                    yy, xx = y+shift[0], x+shift[1]
                    padding[yy//8, xx//8, inverse[(yy%8)*8+xx%8]] = False
            changed = [a.copy() for a in host_qkv]
            changed[1][:, padding, :], changed[2][:, padding, :] = 2, 4
            ignored = [a.copy() for a in host_qkv]
            ignored[0][:, padding, :] = 8
            cases = [('zero', [np.zeros_like(a) for a in host_qkv]), ('distinct', host_qkv),
                     ('padded_kv_changed', changed), ('ignored_q_changed', ignored)]
            outputs = {}
            for label, values in cases:
                for t, a in zip((q, k, v), values):
                    t.copy_(torch.from_numpy(a))
                ref, _, _ = reference_attention.forward(q, k, v, bias)
                ref_full, _, _ = reference_projection.forward(ref, weight, residual, scale, inv, shift=shift)
                launch(shift)
                want = packed_reference(ref, inverse, shift)
                assert raw(attention_out) == want.tobytes(), (shift, label, 'attention')
                assert raw(full_out) == raw(ref_full), (shift, label, 'unquantized projection')
                assert all(raw(t) == a.tobytes() for t, a in zip((q, k, v), values))
                outputs[label] = digest(attention_out)
                report['controls'].append(dict(shift=list(shift), case=label,
                    attention_byte_equal=True, projection_byte_equal=True, inputs_unchanged=True,
                    attention_sha256=outputs[label], projection_sha256=digest(full_out)))
                save()
            assert outputs['distinct'] == outputs['ignored_q_changed'] != outputs['padded_kv_changed']
            # A real two-kernel graph consumes changed live input without host layout copies.
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                launch(shift)
            for case, values in [('zero', cases[0][1]), ('distinct', host_qkv)]:
                for t, a in zip((q, k, v), values):
                    t.copy_(torch.from_numpy(a))
                ref, _, _ = reference_attention.forward(q, k, v, bias)
                ref_full, _, _ = reference_projection.forward(ref, weight, residual, scale, inv, shift=shift)
                graph.replay()
                assert raw(attention_out) == packed_reference(ref, inverse, shift).tobytes()
                assert raw(full_out) == raw(ref_full)
                report['graphs'].append(dict(shift=list(shift), case=case, live_input_byte_equal=True,
                    attention_sha256=digest(attention_out), projection_sha256=digest(full_out)))
            del graph
        assert frozen == {n: digest(t) for n, t in [('bias', bias), ('weight', weight), ('residual', residual), ('scale', scale), ('inverse', inv)]}
        report['constant_hashes'] = frozen
        report['constants_unchanged'] = True
        report['padded_kv_influence_preserved'] = report['ignored_queries_have_no_influence'] = True
    fork.verify_restored()
    report['fork_restored'] = True
    report['passed'], report['phase'] = True, 'completed'
except BaseException:
    report['error'] = traceback.format_exc()
    raise
finally:
    try:
        if fork is not None:
            fork.verify_restored()
        finalize_sources()
    except BaseException:
        report['passed'] = False
        report['finalization_error'] = traceback.format_exc()
        raise
    finally:
        save()
        print(json.dumps(dict(passed=report['passed'], phase=report['phase'], result=str(OUT/'validation.json'))), flush=True)
