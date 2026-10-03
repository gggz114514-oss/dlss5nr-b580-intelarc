# Wide MLP lookup: useful at full size, not selected for NR256 (2026-09-09)

The new experimental branch-pair and C512 group kernels replace their internal
cubic activation with the same authenticated half-bit lookup already used by
C32. Expansion/contraction order, half/FP8 boundaries, projection and sequential
half residual updates remain unchanged. These are FP16 execution changes, not
new INT8 arithmetic or a claim of NVIDIA byte equivalence for the fast branch.

## Complete-pipeline evidence

Three models have independent history and share authenticated immutable buffers
and a serialized transient graph pool. All three use the previously selected
C32 lookup and complete ViT/C32-scatter/Swin/front stack. Variants are:

- `previous`: C32 lookup only; prior branch-pair and C512 group implementations.
- `pairs`: add the measured C64/C128/C256 branch-pair lookup configurations.
- `wide`: add both branch-pair and C512 group lookup configurations.

Three rotated 13-frame rounds compare 117 complete outputs per workload. The
continuous-frame means below exclude the first and final reset in each round.

| Workload | Previous | Pairs | Wide |
| --- | ---: | ---: | ---: |
| NR256 with 1080p residual composition | 26.4968 ms | 26.7628 ms | 26.7074 ms |
| Full 864x480 NR | 84.1351 ms | 82.3046 ms | 81.6214 ms |

Select the wide candidate for further full-size use: the paired full480 saving
is 2.5137 ms, about 2.99%. Keep the previous C32-only stack for NR256; neither
wide candidate improves that workload. Do not compare these numbers with an old
run as if it were a paired test. No default production backend was changed.

Timing includes GPU model work, dynamic front, private history and completion;
the reduced workload also includes GPU resize/preparation and 1080p residual
composition. It excludes flow estimation, upload, JIT, validation, IO,
presentation and game contention. NR256 is not full1080 NR and still lacks its
own long-video quality approval. These modest gains do not establish real-time
operation.

## Primitive and state checks

Reuse 12 real full480 branch-MLP operand sets, 16 real NR256 C512 group sets, six
19/577-row branch tails, and three previously saved group tails/repeated sizes.
Four configurations per case give 148 complete primitive comparisons. Four
whole-body combinations also match the authenticated NR256 temporal output,
including an all-row variant used for correctness exploration only.

Launches selected from primitive measurements:

| Family | BM | Other configuration |
| --- | ---: | --- |
| C64 pair | 32 | 4 warps, 1 stage |
| C128 pair | 32 | 4 warps, 2 stages |
| C256 pair | 16 | 4 warps, 2 stages |
| C512 group | 32 | BN64, 4 warps, 1 stage |

C256 BM32 is slower than BM16. The tested small branch tails do not benefit
from lookup fusion; v2 wrappers keep the old 1024-row cutoff and original small
fallback. Large C512 primitive tuning uses explicitly labeled repeated operands;
the full480 pipeline validates the selected configuration on actual full frames.
The large-input pair tuning does not transfer to NR256, as its whole-pipeline
measurement demonstrates.

Both paired workloads pass complete output/private-history/reset checks,
logical dispatch accounting, held-output ownership, progress fallback and
constant replacement/version rejection before graph replay or history commit.
The lookup stays a shared model-owned buffer outside transient graph pools.
Both full480 (243 frames) and full1080 (390 frames) long replays match the
previously approved FP16 outputs and private history byte for byte. Reset, held
outputs, caller mutation isolation and logical dispatch checks pass. The user's
byte-identical waiver permits reusing those visual approvals. No new video is
encoded and identical full outputs reuse existing artifacts. These long runs
are correctness tests; their times are not paired performance measurements.

## Updated fastest NR256 body profile

This profiles the C32-only lookup stack that won the reduced workload, using
real temporal source frame 181 and all original model blocks. Thirteen isolated
graph segments are compared byte for byte against eager segment execution;
the assembled result and full graph match the authenticated complete output.
Seven rotated rounds of ten replays also leave history and seed unchanged.

The full static body median is **19.91583 ms**. Segment medians sum to 19.25130 ms;
isolated caches, extra launches and copies mean the sum is not an additive
decomposition of whole-frame latency. This excludes dynamic front/warp,
resize/composition, upload, flow and host session work.

| Stage | Median |
| --- | ---: |
| Model pre | 2.49747 ms |
| Encoder C32 / C64 / C128 / C256 / C512 | 0.88061 / 0.81978 / 0.99800 / 1.79276 / 1.56507 ms |
| ViT, all 8 blocks and 64 tokens | 2.75462 ms |
| Decoder C512 including input projection | 1.57738 ms |
| Decoder C256 / C128 / C64 / C32 | 1.79002 / 1.01146 / 0.80098 / 0.90785 ms |
| Model post | 1.85530 ms |

The next candidate should batch the independent branch expansion/activation/
contraction work across two/four/eight branches, followed by a kernel that
projects each branch and adds the half residual in its original sequence.
This targets repeated small launches, especially the C256 stages (3.58278 ms
together); the stage measurement alone does not attribute all that time to MLPs.
Preserve the quantized-input skip in BranchedMLP, which differs from C32's raw
skip. A single ordinary dense sum over all branches would remove half boundaries
and is not an equivalent implementation. No batched-branch implementation is
included in this change yet.

## Receipts and limits

All data is under `D:/Codex-NR-Experiments/nr-b580/reference`. Each completed
successful run has a return-code-zero lease and a separate saved-array audit.
The audits authenticate saved data and GPU comparison receipts; they are not
independent CPU evaluations of the entire candidate network.

| Relative report | SHA256 |
| --- | --- |
| experimental/wide-mlp-lut-v1/validation.json | 3e71f13778ea5e9348c53c950b689e2cc08c5398e0ad4c56414682f7ab0d2679 |
| results/wide-lut-residual256-v1/validation.json | a6cb7c35f9ccbebb359f22ccafd4c9568adc14d57aa212a7b2a69a3115fbd5cb |
| results/wide-lut-full480-v1/validation.json | cd96f20a70c54c7751d75b13a6e37dfa76932d3ab8fbce33b1767badf8fed335 |
| results/fp16_xmx-wide-lut-long-480-v1/validation.json | c3534289511675ee33db11ea71876f3b41d597ac21ddb821c2f06dae80e8a968 |
| results/fp16_xmx-wide-lut-long-1080-v1/validation.json | d894bc30b43a7082716718f1c3047ff3d962f02283391d9a315f3d5b25a9ac9a |
| experimental/c32-lut-body-stages-v1/validation.json | 7f299ccf6679c63524d7cafefae38415ab262d6f23ae7fb06958ba5be2f1cf9f |

Code stays on E, new logs/cache/results on D. Original arithmetic files and the
exact branch remain unchanged. The migration goal, production D3D12 integration,
real-time performance and remaining control/format coverage are not complete.
