# Projection and FP8 store fusion

## v2 completed: numerical pass, no demonstrated speedup

After the user reported completion, main authenticated Luna's handoff, all three
primary artifacts, five critical sources and 269 exact-gate files. Luna's frozen
audit verified 830 unique files. Lease exit code was 0, elapsed 93.652 seconds.

All 243 face frames and three paired rounds of 13 car samples matched the
accepted range-repaired INT8 reference byte for byte, including low NR outputs
and reconstructed full RGB. Six eager/graph checks, private histories, held
outputs, reset, immutable constants and the fresh donor checks passed. This is
equivalence to the approved approximate route, not a new NVIDIA-exact claim.

| Route | Body median, ms | Resident temporal mean, ms | Physical Triton calls |
| --- | ---: | ---: | ---: |
| Repaired INT8 reference | 8.634570 | 11.796221 | 643 |
| Branched projection store | 8.752830 | 11.704512 | 607 |
| Branched plus C512 store | 8.713410 | 11.810312 | 592 |

Seven body samples each average ten static replays. The resident measurement
uses three rotated rounds, 33 temporal samples per route. Its scope includes
1080p preparation, NR256, history, residual reconstruction and completion;
decode, flow, upload, encode and JIT are excluded. Body and pipeline measures
are separate diagnostics and must not be subtracted as a peripheral-cost test.

Temporal mean changes were -0.7774% and +0.1195%. Round changes did not agree in
sign, and body medians were slightly worse. These data do not establish a
repeatable gain. Keep both variants as experiments; no default promotion.
Removing 51 calls alone did not reduce the measured latency meaningfully.

The C256 candidate's 16x64 compile reported 448 spills and was rejected before
dispatch; 16x32 reported zero and was used. Every executed new projection
specialization reported zero spill. A tile tradeoff may offset fusion savings,
but this test did not isolate its contribution. Register count metadata was 0
and is not evidence of zero register use.

Result SHA256: 35293269e11a2d3ace1d40b017c688d07a9203a6ad52a872779ffa9809227089.
Log SHA256: 5ea4001758f2c900c5db1b45fc792b3562ed2cccd4dd3c610cb019eb41e5bea1.
Lease SHA256: 137e0df61a938ea7259ec95b5e31c15ba452d962adc3a482dfe24bb8babf8b3e.
Handoff SHA256: 2873d5af29102c2e81d2b9797720a3b796028083884685b3582f711b6e133fb0.

Next diagnostic uses the unchanged repaired route. The last broad stage timing
predates selected layouts, post crop and INT8, so it cannot locate the current
best target. Refresh complete-body and isolated-stage graphs with identical
physical operation sequences, preserving FP8 provenance and tensor strides.
Stage medians are diagnostic rankings, not additive costs or promised savings.

## v1 setup failure and v2 correction

v1 exited with code 1 while constructing the second session, before warmup,
kernel execution or timing. The first session already had 40 registered INT8
buffers; the next session was still constructing its base model when it tried
to share from that template. The unchanged topology guard correctly rejected
the unequal buffer-name sets. This was a runner construction error, not evidence
about the fusion kernels or a GPU fault.

Frozen failure report SHA256:
dfc7135581ab18de960c4c2b2d762ca08fbebabb36c872b7b3b6762bc2259e73.
v1 sources, report, log and lease are preserved. The new v2 runner uses a fresh
base-only donor that never performs inference. All three INT8 sessions share
the donor's verified base buffers, then register their own 40 packed buffers.
Explicit checks verify base-buffer object identity, disjoint packed storage,
and unchanged donor topology/history/graphs. The topology guard is not relaxed.

`benchmark_quantized_projection_v2.py` and its v2 launcher/audit retain all
original byte and performance tests. Projection kernels and stack factory are
unchanged. New data goes to
`D:/Codex-NR-Experiments/nr-b580/reference/results/quantized-projection-store-v2/`.
Static parsing and constructor-wiring preflight passed. v2 was launched once
through the bounded GPU lease (owner `nr-quantized-projection-store-v2`, 1200 s,
exec session 43296) and assigned to the existing Luna/max reviewer, submission
`01a08aa1-66b0-7720-9333-5e61ef460a53`. Startup and final audit are prewritten.
Results remain pending; main does not monitor the running experiment and waits
for the user to report ordinary completion. Luna reports actual errors directly.

## Optimization and validation design

This experiment continues from the user-accepted static-range INT8 repair. Its
weights, quantization ranges and arithmetic remain the reference. It targets
separate half projection writes followed immediately by FP8 conversion, across
the C64/C128/C256 branched MLPs and non-pooling C512 attention projections.

The branched kernel retains the original sequence of branch dots, each half
addition boundary, scaled residual and final half-to-FP8 rounding. The existing
tile is attempted first; only zero-spill variants can execute, with bounded
smaller output tiles as fallbacks. A scope entered only by BranchedMLP.forward
leaves direct forward_unquantized calls on their original implementation.

C512 retains its original dot order and half residual. Only blocks without a
final pooling projection use the rounded store. The final encoder block and
explicit boundary probes keep their unquantized projection. Complete FP8 store
contracts allow the existing provenance analysis to eliminate the following
redundant conversion. No inferred numeric-range proof is used.

The bounded validation compares three separately owned sessions: repaired
baseline, branched fusion, and branched+C512 fusion. It checks fewer physical
calls with unchanged logical rounding; full eager/graph byte agreement; seven
rounds of captured temporal-body timing; and three rotated paired rounds over
13 resident 1080p motion samples. It then verifies all 243 face frames and final
reset against the accepted repair's full-output hashes and exact low arrays.
Constant versions, private histories, held outputs, inputs and scope restoration
are guarded. There is no new full-frame dump or video encoding.

The tests do not assume a speedup or automatically select a candidate. Body
times exclude the dynamic front and pipeline work; resident pipeline times
exclude decode, flow estimation, upload, encoding and compilation. Byte equality
means equality to this approved approximate route, not to the exact branch.

Data: `D:/Codex-NR-Experiments/nr-b580/reference/results/quantized-projection-store-v1/`.
The local B580 experiment was launched under lease
`nr-quantized-projection-store-v1` (1200-second limit), initial process session
80885. Existing Luna/max received submission
`01a08a91-86ba-7193-9e57-50b89fb70f8f` for monitoring and the prewritten audit;
any error goes to main without self-repair or retries. v1 failed during setup
as documented above; it produced no correctness or performance result.
No default change, publication or new quality approximation.
