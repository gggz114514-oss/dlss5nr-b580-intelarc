"""Scoped host timings with an AST-identical, annotated MotionNR forward.

No GPU profiler, event, added fence, changed guard, or arithmetic substitution.
Durations include host submission/waits, not exclusive device execution. Only
temporary per-instance bindings change inside an already installed session.
"""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
import time
from types import MethodType
import nr_backend.temporal as temporal


def annotate_forward():
    tree = ast.parse(Path(temporal.__file__).read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionNR')
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    node = copy.deepcopy(original)
    wrapped = []

    def wrap(label, statements):
        wrapped.append(label)
        call = ast.Call(func=ast.Attribute(value=ast.Name(id='_host_spans', ctx=ast.Load()),
            attr='span', ctx=ast.Load()), args=[ast.Constant(value=label)], keywords=[])
        return ast.With(items=[ast.withitem(context_expr=call, optional_vars=None)], body=statements)

    def target(statement, name):
        return isinstance(statement, ast.Assign) and any(
            isinstance(v, ast.Name) and v.id == name for v in statement.targets)

    def self_target(statement, name):
        return isinstance(statement, ast.Assign) and any(isinstance(v, ast.Attribute)
            and isinstance(v.value, ast.Name) and v.value.id == 'self' and v.attr == name
            for v in statement.targets)

    output = []
    i = 0
    while i < len(node.body):
        statement = node.body[i]
        text = ast.unparse(statement)
        if isinstance(statement, ast.Expr) and text == 'self._validate_rgb(rgb)':
            output.append(wrap('rgb_validation', [statement]))
        elif isinstance(statement, ast.If) and 'torch.isfinite(motion)' in ast.unparse(statement.test):
            output.append(wrap('motion_value_checks', [statement]))
        elif isinstance(statement, ast.If) and ast.unparse(statement.test) == 'previous is None':
            # Preserve original branching and both warp variants. The front span
            # includes the history numerator*reciprocal multiplication.
            assert len(statement.body) == 1 and target(statement.body[0], 'front')
            assert len(statement.orelse) == 2 and isinstance(statement.orelse[0], ast.If)
            assert target(statement.orelse[1], 'front')
            statement.body = [wrap('front', statement.body)]
            statement.orelse = [wrap('history_warp', statement.orelse[:1]),
                               wrap('front', statement.orelse[1:])]
            output.append(statement)
        elif target(statement, 'result'):
            assert isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == 'self._forward_front'
            output.append(wrap('graph_call', [statement]))
        elif self_target(statement, '_previous'):
            assert self_target(node.body[i+1], '_next_seed')
            output.append(wrap('history_commit', node.body[i:i+2]))
            i += 1
        else:
            output.append(statement)
        i += 1
    node.body = output
    assert sorted(wrapped) == sorted(['rgb_validation', 'motion_value_checks',
        'front', 'history_warp', 'front', 'graph_call', 'history_commit'])

    class Strip(ast.NodeTransformer):
        def visit_With(self, value):
            assert len(value.items) == 1
            call = value.items[0].context_expr
            assert isinstance(call, ast.Call) and ast.unparse(call.func) == '_host_spans.span'
            return [self.visit(v) for v in value.body]

    stripped = Strip().visit(copy.deepcopy(node))
    assert ast.dump(stripped) == ast.dump(original), 'Instrumented forward changed original AST'
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    return compile(module, str(Path(__file__).resolve()) + ':annotated_forward', 'exec'), wrapped


FORWARD_CODE, ANNOTATIONS = annotate_forward()


class HostSpans:
    def __init__(self, stack):
        self.stack = stack
        self.rows, self.active, self.patches = [], [], []
        self.restored = False

    @contextmanager
    def span(self, name):
        row = dict(id=len(self.rows), name=name,
                   parent=self.active[-1] if self.active else None, start_ns=time.perf_counter_ns())
        self.rows.append(row)
        self.active.append(row['id'])
        try:
            yield
        finally:
            row['end_ns'] = time.perf_counter_ns()
            assert self.active.pop() == row['id']

    def patch(self, obj, name, replacement):
        existed = name in obj.__dict__
        previous = obj.__dict__.get(name)
        original = getattr(obj, name)
        setattr(obj, name, replacement)
        self.patches.append((obj, name, existed, previous, original, replacement))

    def measure(self, obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            with self.span(label):
                return original(*args, **kwargs)
        self.patch(obj, name, measured)

    @contextmanager
    def installed(self):
        s = self.stack
        assert not self.rows and not self.patches
        assert len(s.graph.entries) == 2 and '_forward_front' in s.model.__dict__
        assert s.call_guard.original.__func__ is temporal.MotionNR.forward
        # Snapshot AFTER FusedFront/warp scopes were installed. This retains their
        # actual bound helpers, without replacing any global module/class binding.
        namespace = dict(temporal.__dict__)
        namespace['_host_spans'] = self
        exec(FORWARD_CODE, namespace)
        self.patch(s.call_guard, 'original', MethodType(namespace['forward'], s.model))
        self.measure(s.call_guard, 'validate', 'call_guard')
        self.measure(s.int8_vit, 'validate_constants', 'vit_constants')
        if hasattr(s, 'c512_int8'):
            self.measure(s.c512_int8, 'validate_constants', 'c512_constants')
        self.measure(s.graph, '_validate', 'graph_validate')
        self.measure(s.graph, '_constants', 'graph_constants')
        self.measure(s.graph, '_complete', 'graph_completion_wait')
        try:
            yield self
        finally:
            assert not self.active
            for obj, name, existed, previous, original, replacement in reversed(self.patches):
                assert getattr(obj, name) is replacement, ('Host trace interference', name)
                if existed:
                    setattr(obj, name, previous)
                else:
                    delattr(obj, name)
                assert getattr(obj, name) == original
            self.patches.clear()
            self.restored = True

    def result(self):
        assert self.restored and not self.active and self.rows[0]['name'] == 'pipeline'
        root = self.rows[0]
        assert root['parent'] is None
        result = []
        for row in self.rows:
            children = [v for v in self.rows if v['parent'] == row['id']]
            elapsed = row['end_ns'] - row['start_ns']
            exclusive = elapsed - sum(v['end_ns'] - v['start_ns'] for v in children)
            assert 0 <= exclusive <= elapsed
            if row['parent'] is not None:
                parent = self.rows[row['parent']]
                assert parent['start_ns'] <= row['start_ns'] < row['end_ns'] <= parent['end_ns']
            result.append(dict(id=row['id'], name=row['name'], parent=row['parent'],
                start_ns=row['start_ns']-root['start_ns'], end_ns=row['end_ns']-root['start_ns'],
                inclusive_ns=elapsed, exclusive_ns=exclusive))
        assert sum(v['exclusive_ns'] for v in result) == root['end_ns']-root['start_ns']
        return dict(spans=result, host_ms=(root['end_ns']-root['start_ns'])/1e6,
            ast_identity_verified=True, bindings_restored=True, extra_gpu_fences=0,
            scope='Nested host duration including existing waits; not device kernel duration.')
