"""Serialize this task's GPU commands; bounded process tree and log, stdlib only."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('E:\\ComfyUI-aki-v3-IntelArc_20260722\\xess-tools\\work\\r4-route-completion-20260907')


class BasicLimit(ctypes.Structure):
    _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                ("Flags", w.DWORD), ("MinWS", ctypes.c_size_t),
                ("MaxWS", ctypes.c_size_t), ("Active", w.DWORD),
                ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD),
                ("Scheduling", w.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [("Basic", BasicLimit), ("IO", IoCounters),
                ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]


def job_api():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.CreateJobObjectW.restype = w.HANDLE
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.SetInformationJobObject.restype = w.BOOL
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.AssignProcessToJobObject.restype = w.BOOL
    k.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
    k.TerminateJobObject.restype = w.BOOL
    k.CloseHandle.argtypes = [w.HANDLE]
    k.CloseHandle.restype = w.BOOL
    job = k.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    lim = ExtendedLimit()
    lim.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k.SetInformationJobObject(job, 9, ctypes.byref(lim), ctypes.sizeof(lim)):
        error = ctypes.get_last_error()
        k.CloseHandle(job)
        raise ctypes.WinError(error)
    return k, job


def run(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or not 0 < args.timeout_seconds <= 1800:
        raise ValueError("command required; timeout must be >0 and <=1800 seconds")
    if not 0 <= args.wait_seconds <= 1800:
        raise ValueError("wait-seconds must be 0..1800")
    logpath = Path(args.log).resolve()
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "gpu.lock", "a+b", buffering=0) as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b"0")
        deadline = time.monotonic() + args.wait_seconds
        while True:
            if (ROOT / "STOP").exists():
                print("STOP requested; no command started", flush=True)
                return 130
            try:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    print("GPU lease busy; continue CPU work and retry later", flush=True)
                    return 75
                time.sleep(1)
        owner = ROOT / "gpu-owner.json"
        process = None
        api = job = None
        record = {"owner": args.owner, "lease_pid": os.getpid(), "command": command,
                  "cwd": args.cwd, "log": str(logpath), "started_unix": time.time()}
        try:
            owner.write_text(json.dumps(record, indent=2), encoding="utf-8")
            api, job = job_api()
            with open(logpath, "w", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=args.cwd, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                if not api.AssignProcessToJobObject(job, w.HANDLE(int(process._handle))):
                    error = ctypes.get_last_error()
                    subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=log, stderr=log, timeout=15)
                    raise ctypes.WinError(error)
                deadline = time.monotonic() + args.timeout_seconds
                while process.poll() is None:
                    stopping = (ROOT / "STOP").exists()
                    if stopping or time.monotonic() >= deadline:
                        api.TerminateJobObject(job, 130 if stopping else 124)
                        process.wait(timeout=15)
                        record["reason"] = "stop" if stopping else "timeout"
                        return 130 if stopping else 124
                    time.sleep(0.25)
                record["returncode"] = process.returncode
                return process.returncode
        finally:
            if api and job:
                api.CloseHandle(job)
            if process is not None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            record["finished_unix"] = time.time()
            logpath.with_suffix(logpath.suffix + ".lease.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
            owner.unlink(missing_ok=True)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
