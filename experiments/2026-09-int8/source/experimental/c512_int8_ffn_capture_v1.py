"""Eager-only trace of intact compact-layout C512 FFNs; no capture-time hooks."""
from contextlib import contextmanager
from selected_body_stage_graphs_v1 import Recorder, own_inputs, run_seeded, signature


@contextmanager
def record_ffns(layout, recorder):
    assert isinstance(recorder, Recorder) and 'split' not in layout.__dict__
    original = layout.split

    def split(module, x):
        assert 'forward' not in module.ffwd.__dict__
        assert 'forward' not in module.ffwd_projection.__dict__
        ffwd, project = module.ffwd.forward, module.ffwd_projection.forward
        # Move the adjacent projection into the recorder's callable. The outer
        # projection returns the already computed value: no tensor op changes.
        fn = lambda v: project(ffwd(v), v)
        replacement = lambda v: recorder(layout.modules[id(module)] + '.ffn', fn, v)
        identity = lambda value, residual: value
        module.ffwd.forward = replacement
        module.ffwd_projection.forward = identity
        try:
            return original(module, x)
        finally:
            valid = module.ffwd.forward is replacement and module.ffwd_projection.forward is identity
            del module.ffwd.forward
            del module.ffwd_projection.forward
            assert valid and module.ffwd.forward == ffwd and module.ffwd_projection.forward == project

    layout.split = split
    try:
        yield
    finally:
        valid = layout.split is split
        del layout.split
        assert valid and layout.split == original
