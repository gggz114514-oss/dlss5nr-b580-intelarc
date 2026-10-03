"""Refresh stage diagnostics on the accepted range-repaired INT8 NR256 route.

One authenticated temporal car input; unchanged weights, ranges and kernels.
Intact body and 13 isolated stages are measured separately, seven rotating
rounds of ten graph replays. Stage medians are nonadditive diagnostic rankings.
No XPU profiler, graph timing events, new video, tensor dumps or promotion.
"""
from layout_crop_validation_env_v1 import *
from contextlib import contextmanager
import statistics, time, traceback

OUT = D / 'results/repaired-body-stages-v1'
assert not OUT.exists()
repair = receipt(D / 'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
projection = receipt(D / 'results/quantized-projection-store-v2/validation.json',
    '35293269e11a2d3ace1d40b017c688d07a9203a6ad52a872779ffa9809227089')
paired = receipt(D / 'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review_path = D / 'results/int8-ffn-range-repair-v1/user-review-v1.json'
assert sha(review_path) == '8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214'
assert js(review_path)['status'] == 'accepted_face_color_for_range_repair_v1'

import numpy as np
import torch, triton
from PIL import Image
from int8_ffn_calibrated_stack_v1 import Stack
from residual_scale_v1 import ResidualScale
from nr_backend.execution import use_arithmetic_backend
from full_body_dataflow_v1 import DetailedDataflow
from repaired_body_stage_graphs_v1 import Recorder, forward_staged, own_inputs, run_seeded, signature
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
assert triton.__version__.startswith('3.8.0')
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
manifest = js(manifest_path)
expected = {r['frame']: r for r in repair['paired']['runs']['repaired_int8'] if r['round'] == 0}
for p in (Path(__file__), HERE / 'repaired_body_stage_graphs_v1.py',
          HERE / 'audit_repaired_body_stages_v1.py', HERE / 'Run-RepairedBodyStagesV1.cmd',
          HERE / 'selected_body_stage_graphs_v1.py', review_path, manifest_path):
    sources[str(p)] = sha(p)

OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources,
    exact_gate=gate, queue_check=queue_check, candidate_promoted=False,
    new_quantization=False, recalibration=False, new_video=False, tensor_dumps=False,
    default_unchanged=True, warmup=[], stages=[], orders=[], full_body_samples_ms=[],
    accepted_reference=dict(path=str(review_path), sha256=sha(review_path)),
    input_scope='Car sample1 after actual sample0 reset, using accepted repaired private history',
    timing_scope='Isolated static graphs through host completion, with output copy; '
                 'excludes dynamic front, warp, history commit, scaling, flow, IO and JIT',
    diagnostic_medians_are_not_additive=True)
stack = None
graphs = []
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda t: hashlib.sha256(raw(t)).hexdigest()
values = lambda result: result if isinstance(result, tuple) else (result,)
COUNT_KEYS = ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')


def summary(analysis):
    full = analysis.rewrite_summary()
    return {k: full[k] for k in COUNT_KEYS}


def descriptor(t):
    return None if t is None else dict(shape=list(t.shape), stride=list(t.stride()),
        offset=t.storage_offset(), dtype=str(t.dtype), bytes=t.numel()*t.element_size(),
        raw_sha256=digest(t))


def save():
    tmp = OUT / 'progress.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(OUT / 'validation.json')


@contextmanager
def body_scopes():
    # Exactly the inner scopes of LayoutCropGraph + ShortFP8Graph. Avoid nesting
    # a second provenance analysis: the diagnostic supplies DetailedDataflow.
    with stack.rewrite.layout.installed(), stack.rewrite.vit_layout.installed(), \
            stack.rewrite.post_region.installed(), stack.rewrite.fork.installed(), \
            stack.rewrite.resource_gate():
        yield


save()
try:
    torch.set_num_threads(2)
    scales = [arrays.load(row['hidden_scale']) for row in repair['calibration']['layers']]
    stack = Stack(EXACT, hidden_scales=scales)
    sources.update(stack.rewrite.fork.sources)
    model, scaler = stack.model, ResidualScale(256)
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__,
                             device=torch.xpu.get_device_name())
    with torch.inference_mode():
        for i in (0, 1):
            spec = manifest['frames'][i]
            p, f = inputs / spec['file'], inputs / spec['motion_file']
            for path in (p, f):
                assert sha(path) == paired['sources'][str(path)]
                sources[str(path)] = sha(path)
            a = np.asarray(Image.open(p).convert('RGB')).astype('f4') / 255
            m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
            rgb, motion = torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')
            with stack.installed(), use_arithmetic_backend('triton'):
                canvas, flow = scaler.prepare(rgb, motion)
                low = model(canvas, flow, reset=i == 0)
                out = scaler.composite(rgb, canvas, low)
                torch.xpu.synchronize()
            assert digest(low) == expected[i]['low_raw_sha256']
            assert digest(out) == expected[i]['full_raw_sha256']
            assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            report['warmup'].append(dict(frame=i, byte_equal=True,
                low_raw_sha256=digest(low), full_raw_sha256=digest(out)))
        assert len(stack.graph.entries) == 2 and stack.graph.replays == 2
        assert stack.rewrite.scopes == 6
        report['capture_build_counts'] = [{k: b[k] for k in COUNT_KEYS} for b in stack.rewrite.builds]
        assert len(report['capture_build_counts']) == 6
        assert all(list(b.values()) == [643, 172, 395, 223] for b in report['capture_build_counts'])
        entry = stack.graph.last_entry
        static = {k: None if t is None else t.clone() for k, t in entry.inputs.items()}
        assert all(t is None or (t.stride() == entry.inputs[k].stride()
                   and t.storage_offset() == entry.inputs[k].storage_offset()) for k, t in static.items())
        original_inputs = {k: descriptor(t) for k, t in entry.inputs.items()}
        report['body_inputs'] = {k: descriptor(t) for k, t in static.items()}
        body_expected = raw(entry.output)
        assert hashlib.sha256(body_expected).hexdigest() == expected[1]['low_raw_sha256']
        report['expected_output_sha256'] = expected[1]['low_raw_sha256']
        previous, history_bytes, seed = model._previous, raw(model._previous), model.next_seed
        constant_hashes = [digest(t) for row in stack.int8_vit.packed for t in row]
        options = dict(sigmoid=model.sigmoid, blend_scale=model.blend_scale, return_float32=False)
        assert not set(options).intersection(static)
        report['phase'] = 'trace_validation'
        save()
        with stack.installed(), use_arithmetic_backend('triton'), body_scopes():
            full_trace = DetailedDataflow()
            with full_trace.installed():
                full = stack.rewrite.original(model, **static, **options)
            assert raw(full) == body_expected
            staged_trace = DetailedDataflow()
            recorder = Recorder(staged_trace)
            with staged_trace.installed():
                staged = forward_staged(stack.rewrite.layout, model, **static, stage=recorder, **options)
            assert raw(staged) == body_expected
            full_signature = signature(full_trace.events)
            assert signature(staged_trace.events) == full_signature
            assert summary(full_trace) == summary(staged_trace) == report['capture_build_counts'][0]
            assert len(recorder.stages) == 13
            report['full_trace_summary'] = summary(full_trace)
            report['full_staged_physical_sequence_equal'] = True
            report['full_staged_output_byte_equal'] = True
            trace_path = OUT / 'physical-sequence.json'
            trace_path.write_text(json.dumps(full_signature, indent=2) + '\n', encoding='utf-8')
            report['physical_sequence'] = dict(path=str(trace_path), sha256=sha(trace_path))
            report['stage_trace_summaries_cover_full_body'] = True
            assert sum(len(signature(i['events'])) for i in recorder.stages) == len(full_signature)
            assert sum(i['quantization_calls'] for i in recorder.stages) == 395
            assert sum(i['elided_fp8'] for i in recorder.stages) == 223

            stream, pool = torch.xpu.Stream(), torch.xpu.graph_pool_handle()

            def capture(fn):
                with stream:
                    for _ in range(2):
                        warm = values(fn())
                torch.xpu.synchronize()
                public = tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in warm)
                del warm
                graph = torch.xpu.XPUGraph()
                graphs.append(graph)
                with torch.xpu.graph(graph, stream=stream, pool=pool):
                    temporary = values(fn())
                    for target, source in zip(public, temporary):
                        target.copy_(source)
                del temporary
                graph.replay()
                torch.xpu.synchronize()
                return graph, public

            def intact():
                analysis = DetailedDataflow()
                with analysis.installed():
                    value = stack.rewrite.original(model, **static, **options)
                assert signature(analysis.events) == full_signature
                assert summary(analysis) == report['full_trace_summary']
                return value

            full_graph, full_public = capture(intact)
            assert raw(full_public[0]) == body_expected
            report['full_graph_output_byte_equal'] = True
            items = []
            report['phase'] = 'stage_capture'
            for item in recorder.stages:
                args, roots = own_inputs(item)
                expected_stage = [raw(t) for t in item['outputs']]
                root_hashes = [digest(r['tensor']) for r in roots]
                row = dict(name=item['name'], input_provenance=item['input_provenance'],
                    inputs=[descriptor(t) for t in args], outputs=[descriptor(t) for t in item['outputs']],
                    quantization_calls=item['quantization_calls'], elided_fp8=item['elided_fp8'],
                    triton_calls=len(signature(item['events'])), samples_ms=[])
                fn = lambda it=item, ar=args, rt=roots: run_seeded(it, ar, rt)[0]
                graph, public = capture(fn)
                assert [raw(t) for t in public] == expected_stage, item['name']
                row.update(physical_sequence_and_fp8_elisions_equal_full_stage=True,
                           captured_output_byte_equal=True)
                items.append(dict(row=row, graph=graph, public=public, args=args,
                    roots=roots, root_hashes=root_hashes, expected=expected_stage))
                report['stages'].append(row)
                print('Captured repaired stage ' + item['name'], flush=True)
                save()

            persistent = [*full_public, *[t for t in static.values() if t is not None]]
            for item in items:
                persistent += [*item['public'], *[r['tensor'] for r in item['roots']]]
            segments = torch.xpu.memory_snapshot(pool)
            assert all(not any(s['address'] <= t.data_ptr() < s['address'] + s['total_size']
                               for s in segments) for t in persistent)
            report['persistent_io_outside_shared_pool'] = True
            workloads = [dict(name='full_body', graph=full_graph, public=full_public,
                             expected=[body_expected], samples=report['full_body_samples_ms'])]
            workloads += [dict(name=i['row']['name'], graph=i['graph'], public=i['public'],
                expected=i['expected'], samples=i['row']['samples_ms']) for i in items]
            report['phase'] = 'timing'
            for repetition in range(7):
                offset = (repetition * 3) % len(workloads)
                order = workloads[offset:] + workloads[:offset]
                report['orders'].append([w['name'] for w in order])
                for workload in order:
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(10):
                        workload['graph'].replay()
                    torch.xpu.synchronize()
                    workload['samples'].append((time.perf_counter() - started) * 100)
                    assert [raw(t) for t in workload['public']] == workload['expected'], workload['name']
                print('Measured repaired stage round ' + str(repetition), flush=True)
                save()
            for item in items:
                item['row']['median_ms'] = statistics.median(item['row']['samples_ms'])
                assert [digest(r['tensor']) for r in item['roots']] == item['root_hashes']
            report['full_body_median_ms'] = statistics.median(report['full_body_samples_ms'])
            report['stage_inputs_unchanged'] = True
        assert {k: descriptor(t) for k, t in static.items()} == report['body_inputs']
        assert {k: descriptor(t) for k, t in entry.inputs.items()} == original_inputs
        assert raw(entry.output) == body_expected
        assert model._previous is previous and raw(previous) == history_bytes and model.next_seed == seed
        assert constant_hashes == [digest(t) for row in stack.int8_vit.packed for t in row]
        assert stack.rewrite.scopes == 6 and stack.graph.replays == 2
        with stack.installed():
            stack.graph._validate()
        stack.rewrite.verify_restored()
        report['history_seed_constants_static_inputs_and_graphs_unchanged'] = True
        report['graphs'] = stack.graph.metadata()
        report['resources'] = dict(short_fp8=stack.rewrite.resources,
            int8=stack.int8_vit.ffn_resources, c512=stack.rewrite.layout.resources,
            vit=stack.int8_vit.resources)
        assert all(group and all(v['spills'] == 0 for v in group.values()) for group in report['resources'].values())
    report.update(passed=True, phase='completed')
except BaseException:
    report.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for graph in graphs:
            graph.reset()
        if stack is not None:
            stack.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, phase='failed', finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, full_body_median_ms=report['full_body_median_ms'],
    stages_ms={s['name']: s['median_ms'] for s in report['stages']}), allow_nan=False), flush=True)
