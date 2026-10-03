"""Borrow only sealed graph input destinations, within one serialized frame."""
from threading import get_ident


def _key720(device, torch, *, temporal, raw):
    desc = lambda shape, dtype: (shape, dtype, device.type, device.index)
    return (desc((720, 1280, 3), torch.float32),
            desc((768, 1280, 16), torch.float16),
            desc((720, 1280, 3), torch.float32) if temporal else None,
            desc((720, 1280), torch.float32) if temporal else None, raw)


class GraphDestinationBroker720:
    """One front writer, actual EntrySeal and serial-frame replay authority.

    Lending does not consume the entry. The unchanged GraphFront lookup and
    lifecycle exit must still commit their one replay. A submitted lease stays
    strongly anchored until that proof is observed, including on failure.
    """
    def __init__(self, modes, front, producer):
        self.modes, self.session = modes, modes.session
        self.stack = self.session._stack
        self.model, self.graph = self.stack.model, self.stack.graph
        self.front, self.producer = front, producer
        self.torch, self.device = front.torch, front.device
        self.lease = self.staged = None
        self.acquisitions = self.submissions = self.retired_leases = 0
        self.cold_calls = 0
        self._suite = modes._numeric_cleanup_owner
        if (self._suite.session is not self.session or self._suite.children.get('front') is not front
                or front.model is not self.model or front.session is not self.session
                or front.front_producer is not producer or producer.model is not self.model
                or producer.session is not self.session or producer.graph is not self.graph
                or self.graph.closed or self.graph.entries):
            raise RuntimeError('Front broker requires its sole cold Numeric executor and controls scope')

    def _state(self):
        from replay_lifecycle_audit_base_720_v1 import state_for
        if (self.modes.session is not self.session or self.modes._numeric_cleanup_owner is not self._suite
                or self.session._stack is not self.stack or self.stack.model is not self.model
                or self.stack.graph is not self.graph or self.graph.closed
                or self._suite.children.get('front') is not self.front
                or self.front.front_producer is not self.producer
                or self.model.__dict__.get('_audit_fdp_front_producer') is not self.producer
                or self.session._failed):
            raise RuntimeError('Front broker lost its actual session/graph/single writer')
        state = state_for(self._suite)
        if state is None or not state.armed:
            return None
        state.require_frame()
        if state.graph is not self.graph or state.entries is not self.graph.entries:
            raise RuntimeError('Front broker graph is not the actual lifecycle graph')
        return state

    def _tensor(self, value, shape, dtype):
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device != self.device
                or not value.is_contiguous()):
            raise ValueError('Front broker tensor layout/device changed')

    def stage_history_components_720(self, numerator, reciprocal):
        if self.staged is not None:
            raise RuntimeError('Nested history components in one front executor')
        self._tensor(numerator, (720, 1280, 3), self.torch.float32)
        self._tensor(reciprocal, (720, 1280), self.torch.float32)
        state = self._state()
        self.staged = (numerator, reciprocal, None if state is None else state.frame)

    def end_history_components_720(self, numerator, reciprocal):
        if self.staged is None or self.staged[0] is not numerator or self.staged[1] is not reciprocal:
            self.session._failed = True
            raise RuntimeError('History component publication ownership changed')
        self.staged = None

    def _finish_previous(self, state):
        old = self.lease
        if old is None:
            return
        if old['frame'] is state.frame:
            raise RuntimeError('Two front writers requested the same graph frame')
        seal, entry = old['seal'], old['seal'].entry
        if (not old['submitted'] or not seal.current()
                or dict.get(state.entries, old['key']) is not entry
                or self.graph.last_entry is not entry
                or entry.replays != old['entry_replays'] + 1
                or self.graph.replays != old['graph_replays'] + 1):
            self.session._failed = True
            raise RuntimeError('Previous front lease has no actual single replay retirement proof')
        self.lease = None
        self.retired_leases += 1

    def acquire_front_720(self, rgb, previous, event):
        self._tensor(rgb, (720, 1280, 3), self.torch.float32)
        temporal = previous is not None
        if temporal:
            self._tensor(previous, (720, 1280, 3), self.torch.float32)
        state = self._state()
        if state is None:
            self.cold_calls += 1
            return None
        self._finish_previous(state)
        if state.consumed is not None:
            raise RuntimeError('Front destination requested after actual graph consumption')
        if state.pending is not None:
            self.cold_calls += 1
            return None
        if (not self.front._in_frame or self.front._front_event is not event
                or event.get('frontend') != ('normalized_history' if temporal else 'reset')
                or event.get('seed') != (self.model._next_seed if temporal else 0)):
            raise RuntimeError('Front lease lost the actual observed reset/history/seed frame')
        if event.get('history_input') == 'numerator_reciprocal':
            if (not temporal or self.staged is None or self.staged[0] is not previous
                    or self.staged[2] is not state.frame
                    or event.get('normalized_history_tensor_materialized') is not False):
                raise RuntimeError('Component front has no exact current numerator/reciprocal publication')
        elif self.staged is not None:
            raise RuntimeError('Staged components were relabeled as a normalized image')
        controls = self.model._controls
        key = _key720(self.device, self.torch, temporal=temporal,
                      raw=controls.style == 0 and controls.intensity < 1)
        seal = state.sealed.get(key)
        if seal is None:
            self.cold_calls += 1
            return None
        entry = seal.entry
        if dict.get(state.entries, key) is not entry or not seal.current():
            raise RuntimeError('Front static destination is no longer the sealed entry')
        out = entry.inputs['front']
        self._tensor(out, (768, 1280, 16), self.torch.float16)
        inputs = (rgb, previous) + (() if self.staged is None else self.staged[:2])
        if any(t is not None and out.data_ptr() == t.data_ptr() for t in inputs):
            raise RuntimeError('Front static destination aliases live RGB/history components')
        # Allocation belongs to the sealed graph; the front kernel writes it
        # on the same serial in-order queue before GraphFront.forward sees it.
        self.lease = dict(frame=state.frame, thread=get_ident(), key=key, seal=seal,
                          output=out, event=event, submitted=False,
                          entry_replays=entry.replays, graph_replays=self.graph.replays)
        self.acquisitions += 1
        return out

    def commit_front_720(self, output, event):
        state = self._state()
        lease = self.lease
        if lease is None:
            return  # complete cold/eager front; no graph-static publication
        if (state is None or lease['frame'] is not state.frame or lease['thread'] != get_ident()
                or lease['output'] is not output or lease['event'] is not event or lease['submitted']
                or not lease['seal'].current() or state.consumed is not None
                or not event.get('front_kernel_hash') or not event.get('front_binary_sha256')
                or event.get('direct_static_destination') is not True):
            self.session._failed = True
            raise RuntimeError('Front publication has no exact acquired destination/actual launch receipt')
        lease['submitted'] = True
        self.submissions += 1

    def release_retired(self):
        if self.graph.closed is not True or self.graph.entries or self.session._failed:
            raise RuntimeError('Front broker release needs a successful closed and empty graph')
        self.lease = self.staged = None

    def snapshot(self):
        return dict(schema='sealed-front-destination-broker720-v1', acquisitions=self.acquisitions,
                    actual_submissions=self.submissions, retired_replay_leases=self.retired_leases,
                    cold_or_eager_calls=self.cold_calls, retained_leases=int(self.lease is not None),
                    front_publication_copy_eliminated=self.submissions > 0,
                    qualification='CPU_ready_GPU_unverified', gpu_completion_claimed=False,
                    actual_entry_consumption='unchanged GraphFront lookup and lifecycle finish_frame')


def history_destinations(child):
    if not child.implementation.reuse_buffers:
        return None
    from replay_lifecycle_audit_base_720_v1 import state_for
    state = state_for(getattr(child.modes, '_numeric_cleanup_owner', None))
    if state is None or not state.armed or state.pending is not None:
        return None
    state.require_frame()
    if not child.in_frame or child.graph is not state.graph or child.graph.closed or state.consumed is not None:
        raise RuntimeError('History producer lost the actual graph frame owner')
    controls = child.model._controls
    raw = controls.style == 0 and controls.intensity < 1
    d = child.device
    desc = lambda shape, dtype: (shape, dtype, d.type, d.index)
    key = (desc((720, 1280, 3), child.torch.float32),
           desc((768, 1280, 16), child.torch.float16),
           desc((720, 1280, 3), child.torch.float32),
           desc((720, 1280), child.torch.float32), raw)
    seal = state.sealed.get(key)
    if seal is None:
        return None
    # This is a peek, not entry consumption. Actual GraphFront.forward still
    # performs its genuine lookup, live admission and replay commit gates.
    if dict.get(state.entries, key) is not seal.entry or not seal.current():
        raise RuntimeError('History graph destination is no longer sealed')
    result = (seal.entry.inputs['previous'], seal.entry.inputs['history_reciprocal'])
    for tensor, shape in zip(result, ((720, 1280, 3), (720, 1280))):
        if (tuple(tensor.shape) != shape or tensor.dtype != child.torch.float32
                or tensor.device != d or not tensor.is_contiguous()):
            raise RuntimeError('History graph destination layout changed')
    return result
