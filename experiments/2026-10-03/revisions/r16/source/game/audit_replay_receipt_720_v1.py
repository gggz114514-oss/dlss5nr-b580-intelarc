"""Reuse the existing lifecycle capture proof; warm frames keep scalar gates.

The source/pointer/stream/entry/publisher/provider gates remain in _Lifecycle.
Snapshots of compiler/capture counts are retained only at a genuine miss.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureToken:
    state: object
    frame: object
    epoch: int
    replays: int
    entries: int


def capture_token(child):
    from replay_lifecycle_audit_base_720_v1 import state_for
    state = state_for(getattr(child.modes, '_numeric_cleanup_owner', None))
    if state is None or not state.armed:
        return None
    state.require_frame()
    if not any(value is child for value in state.children.values()):
        raise RuntimeError('Capture token child is absent from the actual lifecycle owner')
    return CaptureToken(state, state.frame, state.capture_epoch, child.graph.replays, len(child.graph.entries))


def capture_delta(child, token):
    state = token.state
    state.require_frame()
    if state.frame is not token.frame or state.graph is not child.graph or state.pending is not None:
        raise RuntimeError('Capture token lost its actual frame/graph/publisher')
    added = state.capture_epoch - token.epoch
    if added not in (0, 1) or len(child.graph.entries) != token.entries + added:
        raise RuntimeError('Unexpected graph entry/capture generation change')
    if child.graph.replays != token.replays + 1 or state.consumed is None:
        raise RuntimeError('Frame must consume exactly one audited graph replay')
    _, seal, before_entry, before_graph = state.consumed
    if (not seal.current() or child.graph.last_entry is not seal.entry
            or seal.entry.replays != before_entry + 1
            or child.graph.replays != before_graph + 1):
        raise RuntimeError('Capture proof consumption/entry commit changed')
    receipt = state.last_capture_receipt if added else None
    if added and (receipt is None or receipt['epoch'] != state.capture_epoch or not receipt['passed']):
        raise RuntimeError('No actual cold capture/provider proof for the new generation')
    return added, child.graph.replays - token.replays, receipt
