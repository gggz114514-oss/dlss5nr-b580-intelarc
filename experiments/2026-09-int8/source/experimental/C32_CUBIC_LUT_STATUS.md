# Embedded C32 cubic lookup (2026-09-09)

The streaming C32 FP16 MLP now has an experimental implementation that uses the
existing authenticated 128 KiB cubic+FP8 lookup table inside the matrix kernel.
It preserves the four K32 contraction accumulations, expansion half rounding,
raw-input skip and final half boundary. This adds no new quantization error to
the previously approved FP16 path. It does not promote the FP16 path to native
NVIDIA byte equivalence or complete the migration.

## Measured benefit

Both sides use the complete current C32-scatter, ViT projection, fused MLP,
Swin, dynamic front and graph stack. Only the C32 MLP activation differs.
Both independently constructed models register the same extra constant before
sharing authenticated buffers and the serialized graph pool. Each keeps its
own history. Measurements alternate order across three 13-frame rounds.

Continuous-frame means exclude the first and final reset of each round:

| Path | Direct cubic | Embedded lookup | Time saved |
| --- | ---: | ---: | ---: |
| NR256 with 1080p residual composition | 27.3081 ms | 26.5635 ms | 0.7446 ms / 2.73% |
| Full 864x480 NR | 87.6769 ms | 83.6135 ms | 4.0634 ms / 4.63% |

The NR256 path includes GPU resize/preparation, NR, 1080p residual composition,
private history and completion. Full480 includes the full model, dynamic front,
history and completion. Both exclude flow estimation, upload, JIT, validation,
disk IO, presentation and game contention. NR256 is a padded 320-square body
with a 256x144 active source region, not full1080 NR. Its separate visual review
remains outstanding. Neither measurement demonstrates real-time game operation.

The six captured C32 primitive shapes improve by 1.89–2.13x with BM32,
four warps and one stage. These single-kernel ratios are not whole-model gains.
The reduced path's measured whole-pipeline gain is about 3%, so the next speed
work must cover other substantial costs rather than extrapolating a 2x gain.

## Correctness and ownership

- All 65,536 half bit patterns, including nonfinite encodings, match the current
  cubic helper and the previously authenticated table bit for bit.
- Six real operand sets plus 19-row and 65,539-row tail cases, four launch
  configurations each: 32 complete primitive comparisons pass.
- All four configurations also match the saved full NR256 temporal body output.
- Paired NR256 and full480 tests each pass 78 complete output comparisons,
  independent history, reset, dispatch, held-output and caller mutation checks.
- The table is an explicit int16 bit-pattern tensor registered as a model buffer
  before buffer sharing and graph construction. It remains outside transient
  graph pools. Its pointer, identity and version participate in session guards.
- Model calls with a replaced table or changed table version are rejected before
  graph replay and history/seed commit. The version test is a no-op write that
  leaves all table bytes unchanged. Normal tensor ownership is the contract;
  this does not claim to detect unsafe external memory writes.

The first reduced v1 run completed its byte comparisons but failed its final
test helper: cloning the replacement under inference mode made a tensor without
a version counter. Preserve that failed receipt and its frozen sources. The v2
helper creates the replacement outside inference mode and both paired v2 runs
pass all guards. The unused full v1 generated harness was removed before any run.

## Receipts

All local data is under `D:/Codex-NR-Experiments/nr-b580/reference`.
Each successful run has a return-code-zero lease and a separate saved-array audit.
The audit rereads complete saved arrays; it is not an independent CPU replay of
candidate arithmetic. Candidate/reference comparisons took place in the GPU runs.

| Relative validation report | SHA256 |
| --- | --- |
| experimental/c32-mlp-lut-v1/validation.json | cff24169c9c037ff243b33205a2f6ec2bc5c3c041e069af33c77c1d582400ea6 |
| results/c32-lut-residual256-v2/validation.json | 00156f5e5e01c2918f6fbca4c03449ffa5045feae45f111eec8e20e87e997782 |
| results/c32-lut-full480-v2/validation.json | 15bfcd683a21aceb505e79f612e3b9ffb421d8d921047809075645f8a9788615 |
| results/fp16_xmx-c32-lut-long-480-v1/validation.json | a1697052d1fc933fa693f58dde27c3e6e020d61fb5a2225bd8660fa8d210afea |
| results/fp16_xmx-c32-lut-long-1080-v1/validation.json | 2d28a8e4aff2aac39522df6e8e73cf5aa89eb36ca92fc9d3cb36138e1c2cde84 |

Full480 and full1080 long replay: all 243 + 390 = 633 frames equal the approved
FP16 outputs and private history, including reset reproduction and held-output
ownership. Both reuses of the previous visual approval follow the user's explicit
byte-identical waiver. All output artifacts are reused, with zero new encoding
or output-array copies. Long-run times are diagnostics, not paired speed tests.

## Reference machine and next work

SSH to `REFERENCE_USER@REFERENCE_HOST` is working. The reference GPU reports RTX 4060 Laptop,
driver 616.86 and 8188 MiB. Laptop E has 157,997,830,144 free bytes at the latest
check; no new native capture was needed for this internal arithmetic substitution.
Reference experiments stay in `E:/Codex-NR-Reference`; local logs/cache/results
stay on D. The 35 original backend arithmetic files remain unchanged on both
branches.

The next candidate is the same explicit owned lookup in the existing streaming
C64/C128/C256 branch-pair kernels and eight-group C512 feed-forward kernel.
Preserve their ordered reductions and every rounding boundary, then compare
captured real operands and complete paired pipelines. Do not globally replace
the cubic helper or assume larger kernels benefit equally. No such wider-kernel
implementation or performance claim is included here yet.
