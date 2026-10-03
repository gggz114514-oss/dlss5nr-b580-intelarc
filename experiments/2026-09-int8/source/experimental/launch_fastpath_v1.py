"""Product-side launch fast path: replace Triton's per-call preparation with a lookup.

Why this belongs in the product
-------------------------------
``triton.runtime.jit.JITFunction.run`` rebuilds a launch key on every call: it
binds arguments through ``exec``-generated code (measured 4.0 of 6.6 us), then
recomputes the specialization and looks the kernel up in the device cache. A
frame issues roughly 950 launches, so that preparation is a large share of the
host side of the frame, and the answer only depends on things that repeat.

Triton is a third-party package, so the change cannot live in its source. It is
installed here, from product code, at import time of the public runtime.

Why the key is safe
-------------------
The memo key is a strict refinement of Triton's own specialization -- per
non-constexpr tensor ``(dtype, data_ptr() & 15)``, every other argument its full
value -- so it can only miss, never collide. Shapes carried as ``tl.constexpr``
are part of the key, because constexpr arguments contribute their full value.
A miss simply re-enters the original ``run``.

Four safety nets keep this honest
---------------------------------
1. the key refines the real specialization (above);
2. a sampled self-check recomputes the live specialization every ``SAMPLE`` hits
   and retires the entry on mismatch -- counted in ``report()['stats']``, never
   silent;
3. a jit whose hit rate stays low is bypassed, so a call site with varying shapes
   cannot pay more than it saves;
4. anything uncovered (defaults, varargs, pre-run hooks, unhashable arguments,
   constexpr passed by keyword) falls back to the original ``run`` unchanged.

Scope
-----
Cuts ``run`` / ``getitem`` / ``launch`` / ``md`` / ``book``. The ``keyof`` and
``stream`` cuts stay out on purpose: ``keyof`` measured as zero benefit on this
workload, and ``stream`` patches ``torch.xpu.synchronize``, which the frame
boundary already owns.

``NR_LAUNCH_FASTPATH=off`` installs nothing. Installation is reversible and never
fatal: if the pinned Triton layout differs, the product keeps running on the
original path and ``report()['installed']`` says so.
"""

from __future__ import annotations

import os
import sys

SAMPLE = 64          # sampled self-checks per jit (one every SAMPLE hits)
BYPASS_AFTER = 256   # evaluate the hit rate only after this many calls
BYPASS_RATIO = 0.5   # below this hit rate the jit is bypassed

_EMPTY = ()
_INSTANCE = None


class _Fallback(Exception):
    """Key construction met an uncovered case -- fall back, never guess."""


class _Plan:
    """Per-JITFunction key plan, built once from its parameter list."""

    __slots__ = ('jit', 'names', 'name_set', 'tx', 'cx', 'key_of', 'calls', 'hits',
                 'verify', 'verify_fail', 'bypass')

    def __init__(self, jit, torch):
        params = list(jit.params)
        self.jit = jit
        self.names = tuple(jit.arg_names)
        self.name_set = frozenset(self.names)
        self.tx = tuple(i for i, p in enumerate(params) if not p.is_constexpr)
        self.cx = tuple(i for i, p in enumerate(params) if p.is_constexpr)
        self.calls = 0
        self.hits = 0
        self.verify = 0
        self.verify_fail = 0
        self.bypass = False

        tx, cx = self.tx, self.cx
        tensor_type = torch.Tensor

        def key_of(args):
            """Compress the arguments into a hashable key; raise _Fallback if unsure."""
            parts = []
            for i in tx:
                a = args[i]
                if isinstance(a, tensor_type):
                    parts.append((a.dtype, a.data_ptr() & 15))
                else:
                    hash(a)
                    parts.append(a)
            for i in cx:
                a = args[i]
                hash(a)
                parts.append(a)
            return tuple(parts)

        self.key_of = key_of


class LaunchFastPath:
    """Memoize launch preparation. Only class attributes are replaced."""

    CUTS = ('run', 'getitem', 'launch', 'md', 'book')
    DEFAULT = 'run,getitem,launch,md,book'

    def __init__(self, cuts=('run', 'getitem', 'launch', 'md', 'book'),
                 sample=SAMPLE, verbose=True):
        self.cuts = tuple(cuts)
        for name in self.cuts:
            if name not in self.CUTS:
                raise ValueError('unknown cut %r; known %r' % (name, list(self.CUTS)))
        self.sample = sample
        self.verbose = verbose
        self.plans = {}
        self.memo = {}
        self.gcache = {}
        self.stats = dict(hits=0, misses=0, fallback=0, verify=0, verify_fail=0,
                          bypass=0, evicted=0, launch=0, warmup=0,
                          gmemo_hits=0, gmemo_misses=0)
        self._original_run = None
        self._original_getitem = None
        self._jit_class = None
        self._kiface_class = None
        self._driver = None
        self._knobs = None
        self._ad = None
        self._last_jit = None
        self._last_plan = None
        self._installed = False
        self._torch = None

    # ---------------- install / uninstall ----------------

    def install(self):
        if self._installed:
            return self
        import torch
        from triton.runtime.jit import JITFunction, KernelInterface
        from triton.runtime import driver as triton_driver
        import triton.knobs as knobs

        self._torch = torch
        self._jit_class = JITFunction
        self._kiface_class = KernelInterface
        self._driver = triton_driver
        self._knobs = knobs
        # `driver.active` is a property that re-tests `self._active is None` on
        # every access and the launch段 touches it twice per launch; it only
        # changes in `set_active`, which this product never calls.
        self._ad = triton_driver.active

        if 'run' in self.cuts:
            self._original_run = JITFunction.run
            # A plain function, not a bound method: a bound method is not a
            # descriptor, so `jit.run(...)` would pass the jit twice.
            cuts = self

            def run_impl(jit, *args, grid, warmup, **kwargs):
                return cuts._run(jit, *args, grid=grid, warmup=warmup, **kwargs)

            self._run_impl = run_impl
            JITFunction.run = run_impl
        if 'getitem' in self.cuts:
            self._original_getitem = KernelInterface.__getitem__
            cuts = self

            def getitem_impl(jit, grid):
                return cuts._getitem(jit, grid)

            self._getitem_impl = getitem_impl
            KernelInterface.__getitem__ = getitem_impl
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return self
        if self._original_run is not None:
            self._jit_class.run = self._original_run
        if self._original_getitem is not None:
            self._kiface_class.__getitem__ = self._original_getitem
        self._installed = False
        return self

    # ---------------- plans ----------------

    def plan(self, jit):
        entry = self.plans.get(id(jit))
        if entry is not None and entry[0] is jit:
            return entry[1]
        plan = self._make_plan(jit)
        self.plans[id(jit)] = (jit, plan)
        return plan

    def _make_plan(self, jit):
        import inspect
        try:
            params = list(jit.params)
        except Exception:  # noqa: BLE001
            return None
        if not params:
            return None
        if any(p.has_default for p in params):
            return None
        if any(p._param.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            return None
        if jit.pre_run_hooks:
            return None
        if self._knobs.runtime.add_stages_inspection_hook is not None:
            return None
        return _Plan(jit, self._torch)

    # ---------------- cut 1: run ----------------

    def _run(self, jit, *args, grid, warmup, **kwargs):
        stats = self.stats
        if 'book' in self.cuts:
            # Monomorphic inline cache: the hot loop calls the same jit back to
            # back, so skip id() + dict lookup + tuple unpacking each time.
            if jit is self._last_jit:
                plan = self._last_plan
            else:
                plan = self.plan(jit)
                self._last_jit = jit
                self._last_plan = plan
        else:
            plan = self.plan(jit)
        if plan is None or plan.bypass:
            stats['fallback'] += 1
            return self._original_run(jit, *args, grid=grid, warmup=warmup, **kwargs)

        plan.calls += 1
        # Keyword arguments (including a constexpr passed by name) would make
        # zip(names, args) wrong -- fall back.
        if not plan.name_set.isdisjoint(kwargs):
            stats['fallback'] += 1
            return self._original_run(jit, *args, grid=grid, warmup=warmup, **kwargs)

        try:
            key = (id(jit), grid, tuple(kwargs.values()), plan.key_of(args))
            hash(key)
        except _Fallback:
            stats['fallback'] += 1
            return self._original_run(jit, *args, grid=grid, warmup=warmup, **kwargs)
        except TypeError:
            stats['fallback'] += 1
            return self._original_run(jit, *args, grid=grid, warmup=warmup, **kwargs)

        hit = self.memo.get(key)
        if hit is not None and hit[0] is jit:
            kernel = hit[1]
            plan.hits += 1
            stats['hits'] += 1
            if plan.hits % self.sample == 0:
                self._verify(plan, args, kwargs, key, kernel)
            if warmup:
                stats['warmup'] += 1
            else:
                stats['launch'] += 1
                self._launch(jit, kernel, args, grid, plan, hit)
            self._maybe_bypass(plan)
            return kernel

        stats['misses'] += 1
        kernel = self._original_run(jit, *args, grid=grid, warmup=warmup, **kwargs)
        if kernel is not None:
            self.memo[key] = self._entry(jit, kernel)
        self._maybe_bypass(plan)
        return kernel

    def _entry(self, jit, kernel):
        if 'launch' not in self.cuts:
            return (jit, kernel)
        # `kernel.run` is a property (compiler.py) that re-tests `self._run is
        # None` on every access; `.function` and `.packed_metadata` are plain
        # attribute lookups. All three are invariant after `_init_handles()`.
        try:
            return (jit, kernel, kernel.run, kernel.function, kernel.packed_metadata)
        except Exception:  # noqa: BLE001
            return (jit, kernel)

    def _maybe_bypass(self, plan):
        """A jit that keeps missing is not worth the key cost -- bypass it."""
        if plan.bypass or plan.calls < BYPASS_AFTER:
            return
        if plan.hits < plan.calls * BYPASS_RATIO:
            plan.bypass = True
            self.stats['bypass'] += 1
            stale = [k for k in self.memo if k[0] == id(plan.jit)]
            for k in stale:
                self.memo.pop(k, None)
            self.stats['evicted'] += len(stale)
            if self.verbose:
                print('!! launch-fastpath bypass %s.%s: hits=%d calls=%d'
                      % (plan.jit.fn.__module__, plan.jit.fn.__name__,
                         plan.hits, plan.calls), flush=True)

    def _verify(self, plan, args, kwargs, key, kernel):
        """Sampled self-check: recompute the live specialization and compare.

        A mismatch retires the entry and is counted -- self-healing, not silent.

        `run` fills in `debug` / `instrumentation_mode` before calling the binder
        (jit.py), and this replacement receives kwargs before that happens, so
        both must be supplied here or the real cache lookup would always miss and
        the check would degrade into a no-op.
        """
        stats = self.stats
        try:
            driver = self._driver
            knobs = self._knobs
            probe = dict(kwargs)
            probe['debug'] = probe.get('debug', plan.jit.debug) or knobs.runtime.debug
            probe['instrumentation_mode'] = knobs.compilation.instrumentation_mode
            device = driver.active.get_current_device()
            kernel_cache, kernel_key_cache, _, _, binder = plan.jit.device_caches[device]
            _, specialization, options = binder(*args, **probe)
            import triton.runtime.jit as jit_module
            live = kernel_cache.get(
                jit_module.compute_cache_key(kernel_key_cache, specialization, options), None)
        except Exception:  # noqa: BLE001
            return
        stats['verify'] += 1
        plan.verify += 1
        if live is not None and live is not kernel:
            stats['verify_fail'] += 1
            plan.verify_fail += 1
            self.memo.pop(key, None)
            if self.verbose:
                print('!! launch-fastpath self-check failed for %s.%s (key retired)'
                      % (plan.jit.fn.__module__, plan.jit.fn.__name__), flush=True)

    # ---------------- launch段 ----------------

    def _launch(self, jit, kernel, args, grid, plan, hit):
        """Mirrors the tail of `jit.py`'s run(), minus the preparation already done."""
        if 'launch' in self.cuts and not callable(grid):
            # `bound` only exists to produce `values` and to feed `grid(bound)`.
            values = args
        else:
            bound = dict(zip(plan.names, args))
            if callable(grid):
                grid = grid(bound)
            values = bound.values()

        size = len(grid)
        g0 = grid[0]
        g1 = grid[1] if size > 1 else 1
        g2 = grid[2] if size > 2 else 1

        driver = self._driver
        knobs = self._knobs
        active = self._ad if 'launch' in self.cuts else driver.active
        stream = active.get_current_stream(active.get_current_device())
        enter = knobs.runtime.launch_enter_hook
        exit_ = knobs.runtime.launch_exit_hook

        if 'md' in self.cuts:
            # `compiler.py` tests `if knobs.runtime.launch_enter_hook is None`,
            # which never holds in production (the default is a `HookChain`
            # instance), so every launch builds a `LazyDict` nobody consumes.
            md = (kernel.launch_metadata(grid, stream, *values)
                  if (enter.calls or exit_.calls) else None)
        else:
            md = kernel.launch_metadata(grid, stream, *values)

        if len(hit) == 5:
            runner, function, packed = hit[2], hit[3], hit[4]
        else:
            runner, function, packed = kernel.run, kernel.function, kernel.packed_metadata

        runner(g0, g1, g2, stream, function, packed, md, enter, exit_, *values)

    # ---------------- cut 2: __getitem__ ----------------

    def _getitem(self, jit, grid):
        """`jit[grid]` builds a fresh closure each call (jit.py); reuse it per grid."""
        try:
            key = (id(jit), grid)
            hash(key)
        except TypeError:
            self.stats['fallback'] += 1
            return self._original_getitem(jit, grid)
        entry = self.gcache.get(key)
        if entry is not None and entry[0] is jit:
            self.stats['gmemo_hits'] += 1
            return entry[1]
        self.stats['gmemo_misses'] += 1
        fn = self._original_getitem(jit, grid)
        self.gcache[key] = (jit, fn)
        return fn

    # ---------------- reporting ----------------

    def report(self):
        rows = []
        for jit, plan in self.plans.values():
            if plan is None:
                continue
            rows.append(dict(module=plan.jit.fn.__module__, name=plan.jit.fn.__name__,
                             calls=plan.calls, hits=plan.hits,
                             hit_rate=(plan.hits / plan.calls if plan.calls else 0.0),
                             bypass=plan.bypass, verify=plan.verify,
                             verify_fail=plan.verify_fail,
                             params=len(plan.names), constexpr=len(plan.cx)))
        rows.sort(key=lambda r: r['calls'], reverse=True)
        return dict(installed=self._installed, cuts=list(self.cuts), sample=self.sample,
                    stats=dict(self.stats), memo_entries=len(self.memo),
                    gcache_entries=len(self.gcache), per_jit=rows)


def install_product_fastpath(verbose=True):
    """Install the launch fast path according to ``NR_LAUNCH_FASTPATH``.

    ``off`` / empty installs nothing. Unknown entries install nothing either --
    a typo must not silently leave the fast path on. Installation failure is
    reported and swallowed: the product keeps running on the original path.
    """
    global _INSTANCE
    raw = os.environ.get('NR_LAUNCH_FASTPATH', LaunchFastPath.DEFAULT).strip()
    if raw.lower() in ('', 'off', 'none'):
        print('[launch-fastpath] off (NR_LAUNCH_FASTPATH=%r)' % raw, file=sys.stderr, flush=True)
        return None
    cuts = tuple(item for item in raw.split(',') if item)
    unknown = [item for item in cuts if item not in LaunchFastPath.CUTS]
    if unknown:
        print('[launch-fastpath] unknown entries %r; installing nothing' % (unknown,),
              file=sys.stderr, flush=True)
        return None
    if not cuts:
        return None
    try:
        _INSTANCE = LaunchFastPath(cuts=cuts, verbose=verbose).install()
    except Exception as exc:  # noqa: BLE001
        print('[launch-fastpath] not installed (%s: %s); original path in use'
              % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        return None
    print('[launch-fastpath] on: %s' % ','.join(cuts), file=sys.stderr, flush=True)
    return _INSTANCE


def report():
    """Frame report for the launch fast path (``installed`` False when off)."""
    if _INSTANCE is None:
        return dict(installed=False, cuts=[], stats={}, memo_entries=0,
                    gcache_entries=0, per_jit=[])
    return _INSTANCE.report()
