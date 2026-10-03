# Static hidden-range repair candidate

## Completed and reviewed

The bounded run completed successfully (87.40 seconds). Luna's prewritten audit
passed, checking 1,398 unique files. Main authenticated its report, log, lease,
video, seven PNGs and five critical source files after the user's feedback.
Report SHA256: bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa.
Video SHA256: faf816c9edf5ebda7c1e466a86117635c1cbb37df50603414dfa6bce3bf37af3.

Paired temporal resident-pipeline means (3 rounds, 33 calls per route) were
12.1311 ms for selected FP16, 11.5355 ms for original INT8 and 11.5258 ms for
the repaired INT8. The repair's -0.0845% difference from original INT8 is much
smaller than round-to-round variation; it supports no observed slowdown, not
a demonstrated speedup. All 39 calls including resets averaged 11.5472 versus
11.4976 ms for original/repaired INT8 (-0.4295%). Compiled INT8 kernels and
register/spill resources were identical. These are 1080p resident prepare,
NR256, history, residual and completion timings, not whole-tool frame rates.

On the 240 face frames not used for calibration, mean ROI RGB RMSE fell from
4.6655 to 1.1240 in 8-bit units (-75.91%); full-frame error fell 78.84% and
the temporal-error diagnostic fell 26.90%. This ROI includes background. The
two held-out probe frames had 0.067%-0.653% clipped hidden values per block;
clipping is reduced, not eliminated. All 64 CPU/GPU FFN checks passed.

User feedback: "可以的，色调看上去差不多了，这样会不会影响速度？"
The face-color approval is recorded in output `user-review-v1.json`, SHA256
8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214.
Approval applies to this face comparison, not unseen-video equivalence or a
default promotion. No new job is running and no default changed.

## Implementation and validation scope

The face-color intervention identified hidden clipping as a leading tested
source of immediate INT8 error. This candidate keeps the same integer kernels
and 40 owned constant buffers. It widens static per-channel hidden scales and
re-packs the contraction weights with the new scales during construction.
There is no runtime reduction pass or output color correction.

Calibration is predetermined: face frames 48/144/242 with the original INT8
history, and car samples 0/5/10 as reset inputs. Each supplies all 64 tokens for
all eight FFNs. The new scale is max(old scale, observed absmax * 1.25 / 127).
The original scale floor is preserved. Both original and repaired FFNs must
match the CPU integer oracle. Face frames 96/192 do not participate in fitting.
These are two existing clips; held-out frames remain correlated with the
calibration frames and do not establish unseen-video quality.

`experimental/repair_int8_ffn_range_v1.py` executes one bounded experiment:

- Collect six calibration samples with byte-equal graph/eager controls and 48
  CPU/GPU FFN checks; preserve only inputs, scales and small diagnostic records.
- Use three fresh sessions with shared immutable model data and a serialized
  transient pool. Time FP16/original INT8/repaired INT8 on 13 real-motion 1080p
  samples over three rotated paired rounds; verify frozen references and reset
  reproducibility. Resident pipeline timings exclude I/O, motion estimation,
  upload, encoding and compilation.
- Run the repaired session continuously for all 243 face frames. Reconstruct
  both frozen comparison routes from their authenticated low outputs and check
  full float RGB hashes. Save the repaired low outputs for reproducibility,
  seven PNGs and a 10.125-second video with identical face enlargement.
- Probe repaired frames 96/192 with their own newly generated histories and
  check all 16 FFNs against the CPU oracle. Report clipping, full/ROI color error
  and motion-compensated temporal-error diagnostics; do not assume improvement.
- Finalize MP4 tags through a separate FFmpeg9 remux. Verify SPS and container
  metadata, timestamps and every decoded YUV frame before/after finalization.

Outputs: `D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-range-repair-v1/`.
Luna/max owns monitoring and the prewritten audit, with immediate error handoff
and no self-repair or retries. Main owns corrections. No default promotion,
exact-branch changes or publication. The face-color review is recorded above.

Status: launched on the local B580 under lease `nr-int8-ffn-range-repair-v1`
(1800-second limit), initial process session 15387. Reassigned to existing
Luna/max with submission `01a08a74-d358-7532-a4a6-cbc4b50f1fb3`. Execution and
the prewritten audit completed; main authenticated the stored results after
the user's face-color feedback. No model rerun was needed.
