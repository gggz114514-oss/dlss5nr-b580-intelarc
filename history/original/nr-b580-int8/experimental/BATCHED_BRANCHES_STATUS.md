# Batched independent branches (2026-09-09)

Experimental FP16 scheduling improvement. Goal remains active; no production
integration, native-exact arithmetic change, new INT8 arithmetic or real-time claim.

`batched_branched_mlp_v1` runs independent branch expansion/activation/contraction
in one grid. A second kernel performs projections in the original branch order,
rounding each skip addition to half before the next branch. The existing input
FP8 quantizer and outer output FP8 remain. This reduces per-branch launches without
collapsing the sequential residual sum. Both kernels contain compiled XMX DPAS.
The existing authenticated128KiB cubic lookup is an explicit model-owned constant.

`batched_branched_mlp_v2` selects six-config measurements separately for real
NR256 operands and full864x480 operands. C64 uses BM32/stages2 for small and
BM32/stages1 for full; C128 BM32/stages2 and C256 BM16/stages1 for both. Projection
BM16/BN64 throughout. These are measured workload selections, not general tuning
claims for every geometry.

## Correctness and paired speed

All36 actual temporal NR256 branch modules were captured from source181. The
capture preserves the entire existing body output. All36 real reduced cases,
12 real full480 cases and six synthetic19/577-row tails passed all six launch
configurations:324 complete primitive output comparisons, with matching logical
dispatch. Both selected configurations also preserve the whole NR256 body.
CPU receipt audits reread257 unique arrays totaling72,178,044 stored bytes.

Two workloads each ran two independent histories, three rotated13-frame rounds:
156 complete pipeline outputs match pinned previous references. Reset, seed,
dispatch, progress fallback, caller/held-output ownership and LUT replacement/
version invalidation guards pass. Transient graph storage is shared only under
serialized replay, with static inputs and outputs outside the shared pool.

| Paired temporal frames (11 per round) | Previous | Batched | Time saved |
| --- | ---: | ---: | ---: |
| NR256 +1080 residual | 26.47498 ms | 23.76597 ms | 2.70901 ms /10.23% |
| Full864x480 | 81.01874 ms | 79.52204 ms | 1.49670 ms /1.85% |

Reduced candidate rounds:23.63983,23.91568,23.74241 ms. Full candidate
rounds:79.29355,80.03585,79.23671 ms. The reduced rate is about42.1 fps for
this isolated measured pipeline, below60 fps and without game contention.
Timing includes GPU completion and history; reduced includes color/motion resize
and full1080 residual composition. Motion estimation, upload, JIT, validation,
disk/encoding, presentation and game contention are excluded.

Reduced baseline is the prior C32-only LUT stack with old direct pair/split.
Full baseline is the prior wide-LUT stack. Candidate replaces only C64/128/256
branches and retains the appropriate split implementation for each workload.
C32 LUT, chunk layout, VitProjectionv3, FusedSwin, StridedMatricesv2,
FusedDynamicFrontv1 and GraphFrontv6 remain. GraphHistoryWarpv2 is reduced-only.

## Next visual review

User explicitly requested: “下次出视频让我看一眼，避免你白忙活”. This review
must be displayed even if the scheduling change preserves bytes. Earlier approval
of full-size FP16/INT8 does not approve reduced NR256 residual quality.

`review_batched_residual_long1080_v1` compares previous and batched reduced
pipelines with independent uninterrupted histories and produces all390 source
frames in a four-panel video: original, existing exact, existing full FP16,
new reduced result. Playback is offline at60000/1001 fps,6.5065 seconds.
The reference panels are existing B580 results, not a new native4060 capture.
Only small lossless low-NR arrays, full-output hashes and the encoded video are
newly stored, to avoid another multi-gigabyte full-resolution dump.

Completed390 comparisons with uninterrupted independent reduced histories;
all complete full FP32 outputs and low NR arrays match between old and batched.
Reset reproduction, caller isolation and held outputs were also checked. The
lossless low-NR archive adds122,694,190 bytes. Encoded review is76,606,199 bytes.

The v1 run completed inference, encoding and post-sequence guards, then exited1
on missing H264 `color_transfer` metadata. Its frozen source/report are preserved.
`finalize_batched_residual_review_v2` copied the compressed stream and repaired
VUI tags, with no re-encoding or model rerun; complete compressed VCL payloads
match. Its CPU finalization run exited0. The v2 audit reread all390 low arrays,
decoded all video frames for frame-count verification and checked seven decoded
keyframes against pre-encoding images (RGB8 MAE0.18–1.35). This measures encode
fidelity, not NR quality. The video was displayed and an actual decoded frame
was inspected; raw comparisons were spot-inspected at source65 and195.

Video displayed inline and sent to the Codex file panel (open request queued):
`D:/Codex-NR-Experiments/nr-b580/reference/results/batched-residual-long1080-review-v2/comparison-full-59.94fps.mp4`.
Video SHA2564555f0066b1f2a4e51b0c1ae28ca21c872f9cd1ef610a0d1f499f1eede224edd.
Async visual review was requested and the user explicitly replied:
“可以接受，继续优化这条路线”. This approves the displayed complete NR256 residual
sequence and continuing this quality/performance route. It does not automatically
approve further changes to resolution, composition or arithmetic. A separate
`user-review-v1.json` stores the response with the exact report/video hashes;
the immutable inference and audit reports retain their original pending state.
Complete new full-size243/390 replays have not been run for
this batched candidate; prior wide-LUT long proofs alone do not prove those.

For future encodes with this bundled libx264, use explicit H264 VUI metadata or
the proven copy-remux finalizer: container colour flags alone were not retained.

## Immutable receipts (under D:/Codex-NR-Experiments/nr-b580/reference)

| validation.json folder | SHA256 |
| --- | --- |
| experimental/small-branched-mlp-operands-v1 | 0c3ee88b397831266e6359e5eb24db5d77f3279f5bd9c67f9bda8ae11df4c11b |
| experimental/batched-branches-v1 | 62c7e54e919302d398ff6cd37255e68ac66a6c4a91a4c7c86f4106b810d938ab |
| results/batched-branches-residual256-v1 | e0627317e6b1abdf15b25dca116ab93b3ce7bcad38e6bfbca9f117630dcecb42 |
| results/batched-branches-full480-v1 | e01e6ecf0119da87be5dd10622b22e1e501a8512f7f8079b0978f2f0a98dfc2a |
| results/batched-residual-long1080-review-v2 | d968f89ab835187d050a8b6805d9a11739492b954199ad090f1ab16f1f151751 |

The first four jobs returned0 and their four `saved-audit-v1.json` receipts passed.
The review has its separate v1 metadata-only failure and successful v2 finalizer
described above. Its `saved-audit-v1.json` also passed.
Auditor `audit_batched_branches_v1.py` SHA256
bcfb6be375ce9ec85b3f99124688dd297a7a880c0d558798b097c6d254a3b60a.
Review auditor SHA2567f9c6d664fab93ba8156ad0f61e13d003d2f3e4ccaf59b8c14e5d108d7fd4637.
Audits authenticate complete arrays and GPU run records; they do not independently
recompute the model arithmetic on CPU.

## Python versus a native host

The user asked whether compiling the Python prototype as C could transform speed.
Python owns host validation, scheduling and experiment infrastructure. Triton
already compiles the arithmetic kernels for XPU, with XMX DPAS confirmed in the
saved compiler output. GraphFrontv6 replays captured model work through XPUGraph.

The new paired report's diagnostic temporal body replay median is17.19697 ms
(previous20.11504 ms). This includes graph replay/completion and persistent output
copy; it is not a pure GPU timestamp or an exact lower bound. Whole reduced
pipeline temporal time is23.76597 ms. Their approximately6.57 ms gap also contains
real GPU preparation, motion/history processing, resizing/composition, copies and
synchronization. The different diagnostic timing scopes prevent attributing this
entire gap to Python.

C++ hosting remains useful for production integration with the user's stable
D3D12 GPU Block/DIS chain, shared resources and explicit lifetimes/fences. The
measured body work still needs optimization. No same-kernel Python/C++ A/B has
been measured, so there is no promised native-host speedup. Next investigate
dynamic-front/composition/submission costs and continue kernel work; do not
undertake a blanket rewrite on an assumed language-only performance gain.
