"""Luna's optional CPU-only GET sampler; no game control, GPU imports or logs.

Use the actual control page URL. Poll /api/state at 1 Hz for one bounded run;
save first/end and revision/availability/error/loss changes, never frame dumps.
This file does not change the frozen v2 patch/helper and is not run by its author.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

PERF = Path("D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001")


def endpoint(url):
    value = urlsplit(url.strip())
    if (value.scheme != "http" or value.hostname != "127.0.0.1" or value.port is None or
            value.username or value.password or value.query or value.fragment or
            value.path not in ("", "/", "/api/state")):
        raise ValueError("use the actual http://127.0.0.1:PORT/ control-page URL")
    return f"http://127.0.0.1:{value.port}/api/state"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise URLError("loopback state redirects are not followed")


def extract(state):
    health = state.get("health") or {}
    temporal = health.get("temporal_diagnostics") or {}
    diagnostic = temporal.get("periodic_flash_v2")
    if not isinstance(diagnostic, dict):
        diagnostic = {"status": "unavailable", "reason": "periodic_flash_v2_missing_or_disabled"}
    return {"health_failed": health.get("failed"), "health_reason": health.get("reason"),
            "periodic_flash_v2": diagnostic}


class ChangeWriter:
    def __init__(self, stream):
        self.stream, self.key, self.last, self.samples, self.records = stream, None, None, 0, 0

    def write(self, kind, observation):
        self.stream.write(json.dumps({"kind": kind, "capture_utc": datetime.now(timezone.utc).isoformat(),
                                      "capture_monotonic_ns": time.monotonic_ns(), **observation},
                                     ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.records += 1

    def accept(self, observation):
        self.samples += 1
        diagnostic = observation["periodic_flash_v2"]
        header, counters = diagnostic.get("native_header") or {}, diagnostic.get("counters") or {}
        # Normal frame count/frame-ring overwrite changes are intentionally absent.
        key = (observation.get("health_failed"), observation.get("health_reason"),
               diagnostic.get("status"), diagnostic.get("reason"), diagnostic.get("revision"),
               header.get("dropped"), header.get("frames_dropped"), header.get("overwritten"),
               diagnostic.get("python_diagnostic_errors"),
               *(counters.get(name) for name in ("original_fallback", "nr_skipped", "failure", "game_reset")))
        if self.last is None or key != self.key:
            self.write("start" if self.last is None else "change", observation)
        self.key, self.last = key, observation

    def finish(self, stopped):
        self.write("end", {"stopped": stopped, "samples": self.samples, "records_before_end": self.records,
                           "last_observation": self.last, "negative_evidence_complete": False})


def output_path(value):
    path = Path(value) if value else PERF / ("periodic-flash-health-v2-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".jsonl")
    if path.drive.lower() != "d:" or not path.resolve().is_relative_to(PERF.resolve()):
        raise ValueError("capture output must stay inside the named D PERF directory")
    for part in (path, *path.parents):
        if part.exists() and (part.is_symlink() or part.is_junction()):
            raise ValueError("capture output cannot traverse a junction/symlink")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--output")
    args = parser.parse_args()
    url, output = endpoint(args.url), output_path(args.output)
    if not 1 <= args.seconds <= 600:
        parser.error("one capture must be between 1 and 600 seconds")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped, observed_ok = "duration", False
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        writer = ChangeWriter(stream)
        end = time.monotonic() + args.seconds
        try:
            while time.monotonic() < end:
                started = time.monotonic()
                try:
                    with opener.open(Request(url, method="GET"), timeout=1.5) as response:
                        body = response.read(2 * 1024 * 1024 + 1)
                    if len(body) > 2 * 1024 * 1024:
                        raise ValueError("state response exceeds 2 MiB")
                    observation = extract(json.loads(body))
                except Exception as error:
                    observation = {"periodic_flash_v2": {"status": "read_error", "reason":
                                    type(error).__name__ + ": " + str(error)[:240]}}
                observed_ok |= observation["periodic_flash_v2"].get("status") == "ok"
                writer.accept(observation)
                time.sleep(max(0.0, min(1.0 - (time.monotonic() - started), end - time.monotonic())))
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            writer.finish(stopped)
    print(json.dumps({"capture": str(output), "samples": writer.samples, "records": writer.records,
                      "observed_native_v2_ok": observed_ok, "negative_evidence_complete": False}, indent=2))
    return 0 if observed_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
