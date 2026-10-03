"""Reject replacement and version changes before graph replay or history commit.

Run only after functional and timing checks; the final no-op write intentionally
invalidates the immutable session. It does not change any LUT data bytes.
"""
import torch
from cubic_lut_constant_v1 import BUFFER


def verify(model, adapter, invoke):
    original = getattr(model, BUFFER)
    saved = original.cpu().numpy().tobytes()
    history = model._previous
    history_bytes = history.cpu().numpy().tobytes()
    seed, replays = model.next_seed, adapter.replays
    checks = []

    def rejected(kind):
        try:
            invoke()
        except RuntimeError as error:
            assert 'Model constants changed' in str(error), str(error)
            message = str(error)
        else:
            raise AssertionError('Changed graph LUT accepted')
        assert model.next_seed == seed and adapter.replays == replays
        assert model._previous is history
        assert history.cpu().numpy().tobytes() == history_bytes
        checks.append(dict(kind=kind, rejected=True, error=message,
                           history_and_seed_unchanged=True, no_graph_replay=True))

    with torch.inference_mode(False):
        replacement = original.clone()
    setattr(model, BUFFER, replacement)
    try:
        rejected('buffer_replacement')
    finally:
        setattr(model, BUFFER, original)
    with torch.no_grad():
        original.add_(0)
    rejected('in_place_version_change')
    assert original.cpu().numpy().tobytes() == saved
    return checks
