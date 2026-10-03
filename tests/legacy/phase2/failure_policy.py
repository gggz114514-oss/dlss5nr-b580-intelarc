"""CPU failure classification. Every exception stops the serial run."""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import re
import traceback

CATEGORIES = frozenset({
    'SOURCE_FAILURE', 'COMPILE_FAILURE', 'ADMISSION_FAILURE', 'ROLE_FAILURE',
    'CACHE_FAILURE', 'NUMERIC_FAILURE', 'CHILD_FAILURE', 'RETIREMENT_FAILURE',
    'PROCESS_FAILURE', 'RUNNER_FAILURE', 'INTERRUPTED',
})
SKIP_REASONS = frozenset({'LOW_STRENGTH_REQUIRES_REAL_ADAPTER_NATIVE_HARNESS'})


class Failure(RuntimeError):
    def __init__(self, category, operation, message, *, evidence=None):
        if category not in CATEGORIES:
            raise ValueError('Unknown fatal failure category: ' + str(category))
        super().__init__(message)
        self.category = category
        self.operation = operation
        self.evidence = evidence or {}


def source_frames(stack, source_root):
    """Only actual traceback filenames inside the frozen source count."""
    if source_root is None:
        return []
    root = Path(source_root).resolve()
    rows = []
    for filename, line in re.findall(r'File "([^"]+)", line (\d+)', stack):
        path = Path(filename).resolve()
        if path.is_relative_to(root):
            rows.append(dict(path=str(path), line=int(line)))
    return rows


def describe(exc, *, category='RUNNER_FAILURE', operation='serial run', source_root=None):
    stack = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if isinstance(exc, Failure):
        category, operation, detail = exc.category, exc.operation, exc.evidence
    else:
        detail = {}
        if not isinstance(exc, Exception):
            category = 'INTERRUPTED'
    original = exc
    visited = set()
    while original is not None and id(original) not in visited:
        visited.add(id(original))
        if isinstance(original, (AttributeError, ImportError, SyntaxError, NameError)) and source_frames(stack, source_root):
            category = 'SOURCE_FAILURE'
            break
        if type(original).__module__ == 'triton.compiler.errors':
            category = 'COMPILE_FAILURE'
            break
        original = original.__cause__ or original.__context__
    return dict(category=category, operation=operation, fatal=True, source_failure=category == 'SOURCE_FAILURE',
                exception_type=type(exc).__name__, message=str(exc), stack=stack, evidence=detail,
                frozen_source_frames=source_frames(stack, source_root), action='STOP_RUN_NO_FURTHER_LAUNCH')


@contextmanager
def guard(category, operation):
    try:
        yield
    except Failure:
        raise
    except Exception as exc:
        raise Failure(category, operation, str(exc)) from exc


def classified(category, operation):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with guard(category, operation):
                return fn(*args, **kwargs)
        return wrapped
    return decorator


def child_failure(result, *, operation, evidence, timed_out=False, source_root=None):
    category = 'CHILD_FAILURE'
    if not timed_out and isinstance(result, dict):
        events = result.get('failure_events', [])
        if isinstance(events, list) and events and events[0].get('category') in CATEGORIES:
            category = events[0]['category']
        else:
            # Old receipts remain immutable. Classify their original traceback,
            # never synthesize dispatch/capture/resource observations.
            stack = result.get('error') or ''
            if (source_frames(stack, source_root) and
                    re.search(r'^(?:AttributeError|ImportError|ModuleNotFoundError|NameError|SyntaxError):', stack, re.M)):
                category = 'SOURCE_FAILURE'
            elif result.get('cache_inventory_error'):
                category = 'CACHE_FAILURE'
            elif result.get('retirement_error'):
                category = 'RETIREMENT_FAILURE'
    return Failure(category, operation, 'Child failed; original result/stack/log preserved', evidence=evidence)


def pending_skip(reason, controls):
    # Explicit pre-launch condition only. No exception can be converted to skip.
    if reason not in SKIP_REASONS or controls['intensity'] >= 1.0:
        raise Failure('RUNNER_FAILURE', 'pending condition', 'Invalid successful skip')
    return dict(status='UNEXECUTED_REQUIRES_REAL_ADAPTER_NATIVE_HARNESS', reason_code=reason,
                selected_controls=controls, successful_skip=True, source_failure=False,
                GPU_launched=False, qualification_passed=False,
                reason='P17 raw cold token needs actual adapter validation context')
