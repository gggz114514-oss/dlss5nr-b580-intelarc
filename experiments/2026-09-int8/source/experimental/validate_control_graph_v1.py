"""Exact full-model v7 graphs retain live controls, masks, native display and history."""
import dataclasses
import gc
import json
import os
from pathlib import Path
import sys
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'results/control-graph-v7-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import sha
gate = authenticate_main()
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import numpy as np
import torch
from nr_backend.executor import ResetNR
from nr_backend.live_temporal import LiveControlledMotionNR
from nr_backend.controlled_temporal import NRControls
from nr_backend.execution import use_arithmetic_backend
from graph_front_v7 import GraphFront
from window_layout_v1 import WindowLayout
from fused_cached_matrices_v2 import FusedCachedMatrices
from swin_scheduling_v1 import SwinScheduling
from control_graph_fixtures_v1 import load
import compressed_arrays_v1 as arrays

cases, frozen = load()
names = ['Run-ControlGraphV1.cmd', 'control_graph_fixtures_v1.py', 'window_layout_v1.py', 'capture_body_v1.py',
    'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py', 'fused_activation_int8_v1.py',
    'static_weight_cache_v2.py', 'static_weight_cache_v1.py', 'fast_matrices_v3.py', 'swin_scheduling_v1.py',
    'compressed_arrays_v1.py', 'immutable_artifacts_v1.py', *[f'graph_front_v{i}.py' for i in range(1, 8)]]
for path in [Path(__file__), *[HERE / name for name in names], *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]:
    frozen[str(path)] = sha(path)
OUT.mkdir()
report = dict(scope=__doc__, passed=False, exact_gate=gate, sources=frozen, cases=[],
    complete_migration=False, precision='original exact arithmetic',
    contract='256x256 SDR, zero depth, same-sized synthetic fine motion and optional explicit RGBA mask; seven recorded control sequences, two rounds each, plus progress fallback',
    history='Own private unblended output; native display and private surfaces are comparison only',
    timing_scope='Correctness only; new graph capture/shape compilation and validation are not a speed benchmark')
adapter = None
base_method = ResetNR._forward_front

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def tensors(frame):
    return (torch.from_numpy(frame['pixels']).to('xpu'), torch.from_numpy(frame['motion']).to('xpu'),
            None if frame['mask'] is None else torch.from_numpy(frame['mask']).to('xpu'))

try:
    torch.set_num_threads(2)
    for case in cases:
        model = LiveControlledMotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
            EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1',
            controls=case['frames'][0]['controls']).to('xpu').eval()
        experiment = FusedCachedMatrices()
        experiment.select('baseline')
        scheduler, layout = SwinScheduling(enabled=True), WindowLayout()
        adapter = GraphFront(model, arithmetic=experiment)
        record = dict(name=case['name'], frames=[])
        report['cases'].append(record)
        held = []
        last_outputs = {}
        with torch.inference_mode(), experiment.installed(), scheduler.installed(), layout.installed(), adapter.installed():
            for repetition in range(2):
                for i, frame in enumerate(case['frames']):
                    rgb, motion, mask = tensors(frame)
                    before = adapter.replays
                    with use_arithmetic_backend('triton') as dispatch:
                        value = model(rgb, motion, controls=frame['controls'], control_mask=mask, reset=frame['reset'])
                        torch.xpu.synchronize()
                    actual, private = value.cpu().numpy(), model._previous.cpu().numpy()
                    display_equal = actual.astype('f4').tobytes() == frame['target'].astype('f4').tobytes()
                    private_equal = private.tobytes() == frame['private'].tobytes()
                    row = dict(round=repetition, frame=i, controls=dataclasses.asdict(frame['controls']),
                        reset=frame['reset'], native_history_bound=frame['history_bound'], native_seed=frame['seed'],
                        next_seed=model.next_seed, native_display_byte_equal=display_equal, native_private_byte_equal=private_equal,
                        output=frame['target_meta'] if display_equal else arrays.save(actual),
                        private=frame['private_meta'] if private_equal else arrays.save(private),
                        native_output=frame['target_meta'], native_private=frame['private_meta'],
                        runtime_dispatch=dict(dispatch), captured_dispatch=adapter.last_entry.dispatch,
                        graph_replays=adapter.replays, bound=mask is not None)
                    record['frames'].append(row)
                    save()
                    assert actual.dtype == np.dtype('f2') and display_equal and private_equal, (case['name'], repetition, i)
                    assert adapter.replays == before + 1 and dispatch['xpu_graph_replay'] == 1
                    assert model.next_seed == frame['seed'] + 1 and model.controls == frame['controls']
                    assert model._control_mask_bound == (mask is not None) and model._raw_private is None and model._control_mask is None
                    for key, old in last_outputs.items():
                        entry = adapter.entries[key]
                        if entry is not adapter.last_entry:
                            assert entry.output.cpu().numpy().tobytes() == old
                    for key, entry in adapter.entries.items():
                        if entry is adapter.last_entry:
                            last_outputs[key] = entry.output.cpu().numpy().tobytes()
                    for old_value, old in held:
                        assert old_value.cpu().numpy().tobytes() == old
                    if not held:
                        held.append((value, actual.tobytes()))
                    else:
                        value.zero_()
                        assert model._previous.cpu().numpy().tobytes() == private.tobytes()
                    assert rgb.cpu().numpy().tobytes() == frame['pixels'].tobytes()
                    assert motion.cpu().numpy().tobytes() == frame['motion'].tobytes()
                    if mask is not None:
                        assert mask.cpu().numpy().tobytes() == frame['mask'].tobytes()
                        mask.zero_()
                        assert model._previous.cpu().numpy().tobytes() == private.tobytes()
                    print(json.dumps(dict(case=case['name'], round=repetition, frame=i, display_equal=True, private_equal=True, seed=model.next_seed)), flush=True)
            frame = case['frames'][0]
            rgb, motion, mask = tensors(frame)
            before, marks = adapter.replays, []
            with use_arithmetic_backend('triton'):
                value = model(rgb, motion, controls=frame['controls'], control_mask=mask, reset=True, progress=marks.append)
            assert value.cpu().numpy().astype('f4').tobytes() == frame['target'].astype('f4').tobytes()
            assert model._previous.cpu().numpy().tobytes() == frame['private'].tobytes()
            assert adapter.replays == before
            assert marks == ['pre', 'encoder C32', 'encoder C64', 'encoder C128', 'encoder C256', 'encoder C512', 'ViT', 'decoder C512', 'decoder C256', 'decoder C128', 'decoder C64', 'decoder C32', 'RGB']
            record['progress_fallback'] = dict(byte_equal=True, marks=marks)
            seed, previous, controls, bound = model.next_seed, model._previous, model.controls, model._control_mask_bound
            old_private = previous.cpu().numpy().tobytes()
            bad = rgb.clone()
            bad[0, 0, 0] = float('nan')
            try:
                model(bad, motion, controls=NRControls(style=2, intensity=.25), reset=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid input accepted')
            assert model.next_seed == seed and model._previous is previous and model.controls is controls
            assert model._control_mask_bound == bound and model._raw_private is None and model._control_mask is None
            assert previous.cpu().numpy().tobytes() == old_private
            assert len({str(entry.graph.pool()) for entry in adapter.entries.values()}) == 1
            has_float32 = any(key[-1] for key in adapter.entries)
            assert has_float32 == case['name'].startswith('mask-')
            record.update(passed=True, invalid_input_preserves_state=True, held_outputs_survive=True,
                other_graph_outputs_survive=True, graphs=adapter.metadata(), has_float32_body_graph=has_float32)
        assert ResetNR._forward_front is base_method and '_forward_front' not in model.__dict__
        adapter.close()
        adapter.close()
        adapter = None
        held.clear()
        del model, experiment, scheduler, layout, rgb, motion, mask, value, previous, bad, entry
        gc.collect()
        torch.xpu.empty_cache()
        save()
    report.update(passed=True, complete_native_display_comparisons=56, complete_native_private_comparisons=56,
        progress_native_comparisons=7, class_entry_restored=True)
except Exception as error:
    report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    if adapter is not None:
        adapter.close()
    assert ResetNR._forward_front is base_method
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key: report[key] for key in ('passed', 'complete_native_display_comparisons', 'complete_native_private_comparisons', 'progress_native_comparisons')}), flush=True)
