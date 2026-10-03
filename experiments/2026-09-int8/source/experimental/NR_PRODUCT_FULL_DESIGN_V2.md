# Consolidated NR product API acceptance — 2026-09-11

User chooses product first, further body optimization later. This suite freezes
the accepted floor16 arithmetic plus byte-identical native queries and decoder
gathers behind one public `Session`. It performs acceptance, not another tile or
quantization search. Exact/public repositories and historical factories stay.

The new runner is derived from the frozen `validate_c512_quad_full_v1.py` suite.
For stable downstream audit compatibility, report labels `floor16` and `compact`
are retained: floor16 is the approved pre-query/gather stack, and compact now
means the new consolidated product Session. The report explicitly records the
profile and decoder-gather resources. No previous runner or audit is modified.

The product candidate is constructed through `Session.create`, from a small
standalone calibration profile and 56 content-addressed compressed arrays on D:.
Every file is hash checked; profile weights reconstruct the original 1011 exact
model-buffer hashes. Model weights/noise/sigmoid/reciprocal assets remain in the
existing local model-assets tree. This does not establish distributability or
pretend that the whole portable installation is finished.

All candidate public video calls execute `Session.process`, including its lock,
shape and dtype validation, lazy reusable scaler, private history and completion.
Original/source output and 256 NR output must match the accepted floor16 bytes.
The frozen tests retain 16 padding/shift controls, 128 propagated C512 boundary
comparisons, 10 eager/live checks, 4 x13 paired 1080p calls per route, continuous
243-frame face history, caller modification isolation, held frames, reset and
constant/pool checks. No new quality-changing arithmetic or video is introduced.

Additional API checks reject unsupported source size, wrong motion channels,
non-tensor input and missing reset on source-size change. `Session.reset` clears
private history. Direct 256x256 API input must match the original stack, close is
idempotent, and calls after close are rejected. Source-size adapters for 1080p
and 864x480 must have the same complete filter table bytes as the frozen scalers.

Before the final native256 smoke, each route has 298 public replays and two graph
entries. Product has 12 body constructions; five decoder gathers each execute
12 times during construction, never as per-frame Python operations. Product
builds: 639 Triton /136 standalone FP8 /311 logical q /175 elided. Original:
643 /140 /315 /175. Captured arithmetic telemetry drops five generic half_fma
dispatches and four logical fp8 dispatches; the same boundaries now live inside
the gather kernel. All other telemetry and constants must match the old suite.

Timing retains the old paired body and whole resident-pipeline scopes, but the
candidate pipeline includes the new Session API. Means and medians are recorded
separately. Promotion requires complete correctness; this is product integration
with accepted quality, not a claim that every included historical variant was
individually faster end to end. Existing runtime guards remain enabled.

Main freezes and actually launches. Existing Luna/max only monitors and invokes
the fixed stored-result audit. Any exception, nonzero exit, stall, timeout or
audit failure is reported immediately without self-repair or rerun. Main will
read completed results after user notification; no normal main task polling.

## V2 input contract correction
V1 stopped on the first face frame because archived motion is FP16 while the public API requires FP32. V2 expands finite half values to float before BOTH routes, verifies an exact half roundtrip on all 243 records, and does the same for the final reset. Model, API, calibration, arithmetic, frozen expected output hashes and old evidence are unchanged.
