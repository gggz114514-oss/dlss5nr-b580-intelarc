"""Owned fixed-720 implementation admission, installed BEFORE the numeric suite.

This module is stdlib-only at import. It never selects/loads a device itself.
The existing history child remains the sampler, forward-registry and resource
owner; these immutable options are read when that child is constructed.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from threading import get_ident
import sys


@dataclass(frozen=True)
class HistoryImplementation720:
    pixel_rgb: bool = True
    drop_normalized: bool = True
    fused_near: bool = True
    reuse_buffers: bool = True
    receipt_capacity: int = 64
    direct_compiled_launch: bool = False
    input_summary: bool = False
    release_geometry_indices: bool = True
    release_retired_refs: bool = True
    component_front_input: bool = True

    def __post_init__(self):
        for name in ('pixel_rgb', 'drop_normalized', 'fused_near',
                     'reuse_buffers', 'direct_compiled_launch', 'input_summary',
                     'release_geometry_indices', 'release_retired_refs', 'component_front_input'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(name + ' must be bool')
        if type(self.receipt_capacity) is not int or self.receipt_capacity < 1:
            raise ValueError('receipt_capacity must be a positive integer')
        if self.fused_near and not self.pixel_rgb:
            raise ValueError('fused_near requires the pixel RGB sampler')


class HistoryAdmissionCounter:
    """Hook interface; there is exactly one actual numeric history child."""
    def __init__(self, modes, session, options):
        self.modes, self.session, self.options = modes, session, options
        self.child = None
        self.install_thread = get_ident()
        self.active = True
        self.preflight_calls = 0
        self.final_receipt = None
        self.front_consumer = None
        self.graph_broker = None
        self.geometry_receipt = None
        self.source_admission = None
        self.identity = ('history-host-graph-implementation-v1',) + tuple(asdict(options).items())

    def _actual_child(self):
        child = self.session.__dict__.get('_history_numeric_suite_720')
        suite = getattr(self.modes, '_numeric_cleanup_owner', None)
        if (self.modes.session is not self.session or child is None or suite is None
                or suite.session is not self.session or suite.children.get('history') is not child
                or child.modes is not self.modes or child.session is not self.session
                or child.implementation is not self.options or not child.active or child.retired
                or (self.child is not None and self.child is not child)):
            self.session._failed = True
            raise RuntimeError('Admission lost its actual single history child/suite/session')
        self.child = child
        return child

    def validate(self):
        if (not self.active or getattr(self.modes, '_audit_history_host_options_720', None) is not self.options):
            self.session._failed = True
            raise RuntimeError('History admission slot/active owner changed')
        return self._actual_child().validate()

    def preflight(self):
        self.validate()
        child = self._actual_child()
        if set(child._compiled) != child._specializations():
            raise RuntimeError('The main numeric child must preflight before deferred admission preflight')
        # The main suite already screened all specializations. Delegate its
        # cached preflight only while fresh; never build another executor.
        child.preflight()
        if self.options.component_front_input or self.options.reuse_buffers:
            suite = self.modes._numeric_cleanup_owner
            front = suite.children.get('front')
            for owner in getattr(self.modes, 'implementation_calls_720', {}).values():
                scope = owner.children.get('audit_front_decoder_post_720_v1')
                if scope is not None and callable(getattr(scope, 'apply_history_components', None)):
                    raise RuntimeError('Controls scope cannot be a second native front executor')
            if self.options.component_front_input and (front is None or
                    not callable(getattr(front, 'apply_history_components', None))):
                raise RuntimeError('Components require the actual Numeric front apply_history_components API')
            producer = None if front is None else getattr(front, 'front_producer', None)
            if producer is not None and self.graph_broker is None:
                from audit_graph_prepare_720_v1 import GraphDestinationBroker720
                self.graph_broker = GraphDestinationBroker720(self.modes, front, producer)
                producer.bind_front_destination(self.graph_broker)
            if self.options.component_front_input and self.front_consumer is None:
                from audit_history_front_input_720_v1 import bind_front_component_consumer
                self.front_consumer = bind_front_component_consumer(self.modes, front, self.graph_broker)
        self.preflight_calls += 1
        return self.snapshot()

    def validate_frame_context(self):
        # The sole sampler moves in NumericCleanup's selected-child phase.
        # HookCounter skips metadata with no thread field, then validates us
        # after its own participants have moved in the same transaction.
        # The actual numeric child's public validate() already checks its
        # current CPU owner, fixed scope and installed chain together.
        return self.validate()

    def _transfer_serial_thread(self, owner, *, serial_guard, previous_thread):
        """Optional HookCounter check AFTER selected numeric transfer.

        No independent sampler or second queue/stream is introduced. The main
        transaction moves/rolls back the actual child once; this counter has no
        independent sampler/thread field to migrate.
        """
        adapter = sys.modules.get('cyberpunk_nr_adapter')
        module = sys.modules.get('implementation_hooks_720_v1')
        if (adapter is None or module is None or type(owner) is not module.HookCounter
                or owner.modes is not self.modes or owner.session is not self.session
                or owner.children.get('audit_history_host_720_v1') is not self
                or serial_guard is not getattr(adapter, '_process_serial_lock', None)
                or not serial_guard._is_owned()):
            self.session._failed = True
            raise RuntimeError('History admission transfer requires its actual serial hook transaction')
        child = self._actual_child()
        child._require_serial_idle()
        if type(previous_thread) is not int or child.thread != get_ident():
            raise RuntimeError('History admission requires the already migrated real sampler')
        return get_ident()

    @property
    def calls(self):
        return {} if self.child is None else self.child.calls

    def snapshot(self):
        if self.final_receipt is not None:
            return self.final_receipt
        admission_only = not any(value for name, value in asdict(self.options).items() if name != 'receipt_capacity')
        original_math = (self.child is not None and admission_only and
                         (self.child.options.history_value, self.child.options.history_reciprocal,
                          self.child.options.history_dimension_rcp, self.child.options.history_coord) ==
                         ('fp32_fractional', 'table', 'table', 'reference'))
        return dict(schema='history-host-graph-implementation-v1',
                    options=asdict(self.options), identity=self.identity,
                    admission_only=admission_only, original_history_math=original_math,
                    control_label='admission only, original math' if original_math else 'selected history implementation',
                    install_thread=self.install_thread, stage='before_numeric',
                    active=self.active, preflight_calls=self.preflight_calls,
                    sampler_thread=None if self.child is None else self.child.thread,
                    metadata_owns_sampler_thread=False, geometry=self.geometry_receipt,
                    source_admission=None if self.source_admission is None else self.source_admission.receipts,
                    component_front_bound=self.front_consumer is not None,
                    component_front_calls=0 if self.front_consumer is None else self.front_consumer.calls,
                    front_destination_broker=None if self.graph_broker is None else self.graph_broker.snapshot(),
                    single_history_executor=True, gpu_checked=False,
                    actual_child=None if self.child is None else self.child.snapshot())


@contextmanager
def installed(modes, **kwargs):
    """Cold admission only; yield metadata for the integrated scope owner.

    Unwind AFTER numeric children/graph retirement. No shared registry patches
    or disabled guards: the numeric history child authenticates all new JITs.
    """
    options = HistoryImplementation720(**kwargs)
    session = modes.session
    if (session is None or modes.height != 720 or tuple(modes.source) != (720, 1280)
            or modes.variant != 'unrounded' or not modes.controlled
            or session._stack.graph.closed or session._stack.graph.entries):
        raise RuntimeError('History implementation needs a fresh owned fast 1280x720 session')
    slot = '_audit_history_host_options_720'
    if hasattr(modes, slot) or '_history_numeric_suite_720' in session.__dict__:
        raise RuntimeError('Install history implementation before the numeric child exactly once')
    setattr(modes, slot, options)
    model = session._stack.model
    if '_audit_history_host_modes_720' in model.__dict__:
        raise RuntimeError('History model admission already installed')
    model.__dict__['_audit_history_host_modes_720'] = modes
    counter = HistoryAdmissionCounter(modes, session, options)
    from audit_history_source_admission_720_v1 import SourceAdmission720
    counter.source_admission = SourceAdmission720()
    try:
        counter.source_admission.install()
        if options.release_geometry_indices:
            from audit_history_resources_720_v1 import release_unused_geometry_indices
            counter.geometry_receipt = release_unused_geometry_indices(modes.geometry)
        if options.release_retired_refs:
            from audit_history_resources_720_v1 import install_session_retirement
            install_session_retirement(session)
        yield counter
    finally:
        if getattr(modes, slot, None) is not options:
            session._failed = True
            raise RuntimeError('History implementation admission owner changed')
        if model.__dict__.get('_audit_history_host_modes_720') is not modes:
            session._failed = True
            raise RuntimeError('History model admission owner changed')
        model.__dict__.pop('_audit_history_host_modes_720', None)
        component = model.__dict__.get('_audit_history_front_components_720')
        if component is counter.front_consumer:
            model.__dict__.pop('_audit_history_front_components_720', None)
        graph = session._stack.graph if session._stack is not None else None
        from audit_history_resources_720_v1 import graph_retired, release_retired_history
        counter.final_receipt = counter.snapshot()
        counter.final_receipt['active'] = False
        if graph_retired(graph):
            if options.release_retired_refs:
                release_retired_history(session, counter.child)
            if counter.graph_broker is not None:
                counter.graph_broker.release_retired()
            counter.child = counter.front_consumer = None
            counter.graph_broker = None
        else:
            session._audit_history_admission_retired_720 = counter
        counter.source_admission.restore()
        counter.active = False
        delattr(modes, slot)
