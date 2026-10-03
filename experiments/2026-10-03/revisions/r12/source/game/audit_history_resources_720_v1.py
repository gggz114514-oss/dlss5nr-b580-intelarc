"""Fixed-720 geometry admission and verified Python retirement (stdlib-only)."""
from types import MethodType


def release_unused_geometry_indices(geometry):
    m = geometry.mode
    if (tuple(geometry.source) != (720, 1280) or tuple(m.model) != (720, 1280)
            or tuple(m.active) != (720, 1280) or tuple(m.internal) != (768, 1280)
            or tuple(m.inset) != (0, 0) or geometry.tables):
        raise ValueError('Only zero-origin fixed 720p geometry may release unused indices')
    geometry.rows = geometry.columns = None
    return dict(rows_used=False, columns_used=False, logical_resident_bytes_removed=16000,
                initialization_allocation_avoided=False)


def make_frame_geometry720(geometry_type, height, *, source, device, mode_for):
    """Optional main factory hook: avoid even the discarded index allocation.

    Use only for this frozen 720 source; other existing callers keep their
    original constructor unchanged.
    """
    if height != 720 or tuple(source) != (720, 1280):
        return geometry_type(height, source=source, device=device)
    from residual_scale_v1 import _axis
    geometry = geometry_type.__new__(geometry_type)
    geometry.mode, geometry.source = mode_for(720), tuple(source)
    geometry._axis_kernel, geometry.tables = _axis, {}
    geometry.rows = geometry.columns = None
    release_unused_geometry_indices(geometry)
    return geometry


def graph_retired(graph):
    return graph is not None and graph.closed is True and not graph.entries


def release_retired_history(session, child):
    """Command destruction must precede all reference release; no storage edits."""
    if child is None:
        return False
    if child.in_frame or child.active or not child.retired or not graph_retired(child.graph):
        return False
    if getattr(child, '_retired_receipt', None) is None:
        child._retired_receipt = child.snapshot()
    # These objects own model/stack, binary, tables or temporary tensors. Public
    # borrowed tensors have their own storage owners; no resize/set_/zero occurs.
    for name in ('modes', 'session', 'stack', 'model', 'provider', 'graph',
                 'pre_module', 'post_module', 'pre_weight', 'post_weight',
                 'reciprocal', 'dimensions', 'values', 'dimension_values',
                 'native_dimensions', 'native_dimension_owner', 'weights', 'targets',
                 'prepared_dimensions', 'prepared_dimension_owner',
                 'c512_blocks', 'constants', '_prepare_buffers', 'last_diagnostics',
                 'last_domain_buffers', 'input_summary', 'wrapper', 'previous_installed',
                 'joint_scope', 'dense', 'delegated', 'post_head', 'graph_original',
                 'idle_model_slots', 'geometry', '_compiled',
                 '_binary_objects', '_screened_launches', '_lifecycle_rest_720'):
        if name in child.__dict__:
            child.__dict__[name] = None
    if session.__dict__.get('_history_numeric_retired_720') is child:
        session.__dict__.pop('_history_numeric_retired_720')
    return True


def _closed_scope():
    raise RuntimeError('The released NR session has no installed execution scope')


def release_closed_session(session, graph):
    """Call only after the full product.close chain returned successfully."""
    product = session.__dict__.get('_product')
    if (not getattr(session, '_closed', False) or not graph_retired(graph)
            or (product is not None and (not getattr(product, '_closed', False)
                                       or getattr(product, '_stack', None) is not None))):
        return False
    child = session.__dict__.get('_history_numeric_retired_720')
    release_retired_history(session, child)
    counters = session.__dict__.get('_counters')
    if counters is not None and hasattr(counters, 'stack'):
        counters.stack = None
    session._counters = None
    session._installed = _closed_scope
    session._stack = None
    session.__dict__.pop('_audit_history_admission_retired_720', None)
    return True


class SessionRetirement720:
    def __init__(self, session):
        self.session, self.graph = session, session._stack.graph
        self.original = session.close
        self.closed = False

    def close(self):
        if self.closed:
            return
        session = self.session
        # If product/graph close fails, nothing below executes: anchors stay.
        self.original()
        if not release_closed_session(session, self.graph):
            raise RuntimeError('NR close returned without verified product/graph retirement')
        session.close = self.original
        self.closed = True
        self.graph = self.session = self.original = None


def install_session_retirement(session):
    previous = session.__dict__.get('_audit_python_retirement_720')
    if previous is not None:
        return previous
    owner = SessionRetirement720(session)
    session.close = owner.close
    session._audit_python_retirement_720 = owner
    return owner
