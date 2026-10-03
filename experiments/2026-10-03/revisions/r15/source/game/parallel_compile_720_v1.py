"""Cold actual-Tensor warmup via the pinned runtime AsyncCompileMode.

No device imports until an enabled Luna precompile batch. Readonly sessions
return before creating any executor/monitor or importing the async runtime.
Only the caller submits warmup/finalizes futures; no worker loads GPU handles.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
from threading import Event, RLock, Thread, get_ident
import time
import traceback

_SESSION = ContextVar("nr_actual720_cold_compile_session", default=None)
GIB = 1024 ** 3


@dataclass(frozen=True)
class CompileSpec:
    label: str
    jit: object
    grid: tuple
    args: tuple
    options: dict


def enabled():
    session = _SESSION.get()
    return bool(session is not None and session.precompile and session.workers > 1)


def warmup_specs(label, specs):
    session = _SESSION.get()
    if session is None or not session.precompile or session.workers <= 1:
        return None
    return session.warmup(label, specs)


def record_admitted(report, kernels, *, complete=False):
    """Cold selected-object accounting; never loads handles or launches a kernel."""
    if report is None:
        return
    hashes = set(report['serial_admitted_kernel_hashes'])
    hashes.update(kernel.hash for kernel in kernels)
    report['serial_admitted_kernel_hashes'] = sorted(hashes)
    if complete:
        report['serial_admission_complete'] = True
        report['compiled_first_choices_not_selected'] = [
            row for row in report['specs'] if row['kernel_hash'] not in hashes]


def windows_memory():
    """Process private/RSS bytes and system physical availability, CPU APIs only."""
    import ctypes
    if os.name != "nt":
        raise RuntimeError("The pinned compile memory probe requires Windows")
    from ctypes import wintypes as w
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(ProcessMemory), w.DWORD]
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    process = ProcessMemory(); process.cb = ctypes.sizeof(process)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)) or not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(process), process.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(total_physical_bytes=status.total_phys, available_physical_bytes=status.avail_phys,
                process_private_bytes=process.private, process_rss_bytes=process.rss,
                process_peak_rss_bytes=process.peak_rss)


class MemoryWatch:
    def __init__(self, probe, budget, reserve):
        self.probe, self.budget, self.reserve = probe, budget, reserve
        self.lock = RLock(); self.stop = Event(); self.error = None; self.thread = None
        self.first = self.probe(); self.samples = 0
        self.peak_private = self.first["process_private_bytes"]
        self.peak_rss = self.first["process_rss_bytes"]
        self.min_available = self.first["available_physical_bytes"]
        self.observe(self.first)

    def observe(self, row=None):
        with self.lock:
            try:
                row = self.probe() if row is None else row
                self.samples += 1
                self.peak_private = max(self.peak_private, row["process_private_bytes"])
                self.peak_rss = max(self.peak_rss, row["process_rss_bytes"])
                self.min_available = min(self.min_available, row["available_physical_bytes"])
                if self.min_available < self.reserve:
                    raise RuntimeError("Compile memory reserve violated; queued compiles will stop")
                if self.peak_private - self.first["process_private_bytes"] > self.budget:
                    raise RuntimeError("Compile process-private growth budget violated; queued compiles will stop")
            except BaseException as error:
                if self.error is None: self.error = error

    def check(self):
        self.observe()
        with self.lock:
            if self.error is not None: raise self.error

    def start(self):
        self.check()
        def sample():
            while not self.stop.wait(.25): self.observe()
        self.thread = Thread(target=sample, name="NR_COMPILE_MEMORY", daemon=False)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None: self.thread.join()
        self.observe()

    def snapshot(self):
        with self.lock:
            return dict(initial=self.first, peak_process_private_bytes=self.peak_private,
                peak_process_rss_bytes=self.peak_rss,
                peak_private_growth_bytes=self.peak_private-self.first["process_private_bytes"],
                min_available_physical_bytes=self.min_available, samples=self.samples,
                growth_budget_bytes=self.budget, system_reserve_bytes=self.reserve,
                violation=None if self.error is None else str(self.error),
                coverage="process private/RSS + system physical availability; descendant private peaks unknown",
                hard_peak_guarantee=False)


class CheckedExecutor:
    def __init__(self, workers, watch, report, lock):
        self.watch, self.report, self.lock = watch, report, lock
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="NR_CPU_COMPILE")
        self.active = 0

    def submit(self, compile_fn):
        submitted = time.perf_counter()
        def compile_only():
            self.watch.check()
            begin = time.perf_counter(); thread = get_ident()
            with self.lock:
                self.active += 1
                self.report["peak_active_compile_tasks"] = max(self.active, self.report["peak_active_compile_tasks"])
            try:
                kernel = compile_fn()
                self.watch.check()
                return kernel
            finally:
                with self.lock:
                    self.active -= 1
                    self.report["compile_tasks"].append(dict(thread=thread, queue_seconds=begin-submitted,
                        worker_wall_seconds=time.perf_counter()-begin))
        return self.pool.submit(compile_only)

    def shutdown(self):
        # Do not kill native compilers; drain all CPU work before GPU admission.
        self.pool.shutdown(wait=True, cancel_futures=False)


class ColdCompileSession:
    def __init__(self, *, precompile, workers=8, memory_budget_gib=32, reserve_gib=8,
                 progress=None, memory_probe=windows_memory):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("CPU compile workers must be an integer in 1..16")
        if memory_budget_gib <= 0 or reserve_gib < 8:
            raise ValueError("Positive compile budget and at least 8 GiB system reserve required")
        self.precompile, self.workers = precompile, workers
        self.budget, self.reserve = int(memory_budget_gib*GIB), int(reserve_gib*GIB)
        self.progress = progress or (lambda *args, **kwargs: None)
        self.memory_probe = memory_probe; self.owner = get_ident(); self.reports = []
        self.token = None; self.in_batch = False; self.lock = RLock()
        self.pools_started = 0

    def __enter__(self):
        if _SESSION.get() is not None: raise RuntimeError("Nested compile session forbidden")
        self.token = _SESSION.set(self)
        return self

    def __exit__(self, *args):
        if self.in_batch: raise RuntimeError("Compile session exited with CPU work still live")
        _SESSION.reset(self.token); self.token = None

    def _load_async_api(self):
        # Executed only by Luna during actual precompile. CPU regressions instead
        # execute the pinned stdlib-only async module AST, never import Triton.
        from triton.runtime._async_compile import AsyncCompileMode, FutureKernel
        return AsyncCompileMode, FutureKernel

    def warmup(self, label, specs):
        if not self.precompile: raise RuntimeError("Readonly phase cannot start parallel compilation")
        if get_ident() != self.owner or self.in_batch: raise RuntimeError("Warmup requires the single cold caller thread")
        specs = tuple(specs)
        if not specs: return None
        report = dict(label=label, requested_workers=self.workers, caller_thread=self.owner,
            requested_specs=len(specs), submitted_unique_runtime_keys=0, status="SUBMITTING",
            peak_active_compile_tasks=0, compile_tasks=[], specs=[], failures=[],
            serial_admitted_kernel_hashes=[], serial_admission_complete=False,
            compiled_first_choices_not_selected=None,
            init_handles_called_by_batch=False, launch_called_by_batch=False,
            candidate_strategy="required first choice only; conditional fallback stays serial")
        self.reports.append(report); self.in_batch = True
        executor = watch = mode = None; futures = []; started = time.perf_counter()
        try:
            watch = MemoryWatch(self.memory_probe, self.budget, self.reserve); watch.start()
            AsyncCompileMode, FutureKernel = self._load_async_api()
            executor = CheckedExecutor(self.workers, watch, report, self.lock)
            self.pools_started += 1
            mode = AsyncCompileMode(executor, ignore_errors=False)
            with mode:
                for spec in specs:
                    watch.check()
                    report["specs"].append(dict(label=spec.label, grid=list(spec.grid), options=dict(spec.options),
                        source=spec.jit.fn.__module__+"."+spec.jit.fn.__qualname__, status="SUBMITTING"))
                    # Real Tensor arguments are bound by the actual JIT on this
                    # owner thread; no cloned ASTSource/target or fake Tensor.
                    futures.append(spec.jit.warmup(*spec.args, grid=spec.grid, **spec.options))
                    report["specs"][-1]["status"] = "SUBMITTED_OR_JIT_CACHE_HIT"
            report["submitted_unique_runtime_keys"] = len(mode.raw_futures)
            for row, future in zip(report["specs"], futures):
                kernel = future.result() if isinstance(future, FutureKernel) else future
                if kernel is None: raise RuntimeError("Actual manual warmup returned no kernel")
                row.update(dict(status="CPU_COMPILED_PENDING_SERIAL_ADMISSION",
                    kernel_hash=kernel.hash, binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
                    handles_already_initialized=getattr(kernel, "module", None) is not None))
            watch.check(); report["status"] = "CPU_COMPILE_COMPLETE_PENDING_SERIAL_GPU_ADMISSION"
            self.progress("parallel_cpu_compile_done", label=label, unique_runtime_keys=len(mode.raw_futures))
            return report
        except BaseException:
            report["status"] = "FAILED_NO_GPU_ADMISSION"
            report["failures"].append(traceback.format_exc())
            # Real AsyncCompileMode.__exit__ stops on its first error. Drain and
            # finalize other completed jobs on this same caller before unwind.
            if mode is not None:
                for future in mode.future_kernels.values():
                    try: future.result()
                    except BaseException: report["failures"].append(traceback.format_exc())
            raise
        finally:
            if executor is not None: executor.shutdown()
            if watch is not None:
                watch.close(); report["memory"] = watch.snapshot()
            if mode is not None: report["submitted_unique_runtime_keys"] = len(mode.raw_futures)
            report["batch_wall_seconds"] = time.perf_counter()-started
            report["all_CPU_jobs_drained"] = True
            self.in_batch = False
            # A final sample can first detect pressure while draining. Preserve
            # an earlier compile exception; otherwise fail before any admission.
            if watch is not None and watch.error is not None and report["status"] != "FAILED_NO_GPU_ADMISSION":
                report["status"] = "FAILED_NO_GPU_ADMISSION"
                report["failures"].append(repr(watch.error))
                raise watch.error

    def snapshot(self):
        return dict(enabled=self.precompile and self.workers > 1,
                    requested_workers=self.workers, effective_workers=self.workers if self.precompile else 0,
                    batches=self.reports, pools_started=self.pools_started,
                    actual_peak_before_Luna="unknown")
