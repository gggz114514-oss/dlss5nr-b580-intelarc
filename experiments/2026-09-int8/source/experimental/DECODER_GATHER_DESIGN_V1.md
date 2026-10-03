# Decoder gather experiment — 2026-09-11

User rejects additional approximate QKV changes for their small absolute gain.
The completed QKV-full process is archived; its completion does not mean adoption
or visual approval. This experiment changes no arithmetic or quantization policy.

Reference: accepted floor16 C512 FFN + repaired ViT, using the frozen native-query
stack whose complete results are byte equal to that accepted route. This is not
a claim of NVIDIA/exact4060 parity for the already approximate fast route.

## Change and transferable reasoning

Decoder transitions allocate two successive nearest-neighbour repeat tensors,
crop, then merge with the encoder skip. Read projected[y//2,x//2,c] directly into
the merged output instead. Keep the original projection matrix, split-K order,
half conversion, compensated half FMA, and existing output FP8 conversion.

ViT-to-C512 and C256/C128/C64 write FP8-decoded half. C32 writes the unquantized
half merge into its original padded canvas, including +0 outside the valid area.
The following MLP needs that raw residual: marking it FP8 would be incorrect.
No attention window, skip, final RGB region, or temporal context is dropped.
Post-region and all existing body kernels are unchanged in this version.

This removes materialization across producer/consumer boundaries. It can be
ported to the exact backend after independent exact-reference validation. The
gather primitive accepts strides and odd crops; this screen integrates only the
NR256 route used inside the resident 1080p residual scaler. Native 1080p model
execution and other branches have not been validated by this screen.

## Fixed validation and timing

- Authenticate existing source/weight/result receipts and 1011 model constants.
- All 13 car frames: original low RGB and composed 1080p hashes match frozen
  results; candidate body uses the same inputs and must produce identical bytes.
  Reset and temporal input construction/history remain the original public path.
- Retain five real projected/skip/scale boundaries in GPU memory only. Compare
  old expansion/FMA/output conversion with the candidate byte for byte.
- All 63,488 finite half bit patterns appear in projected operands, with signed
  zero, tiny and extreme scales; also noncontiguous operands, odd crop sizes,
  all four C32 shifts, and quantized vs raw-padded output contracts.
- Reject any candidate compiled spill before dispatch; never run a GPU profiler.
- Four static graphs: original/gather intact temporal body and original/gather
  five merges. Persistent input/output outside shared graph pool. Live zero and
  negated inputs must change results, match eager execution, and restore bytes.
- 16 prewarm replays, 12 rotating paired rounds of 32 replays, host completion
  included. Local merge and intact-body timings are separate, not additive.
- No extra quantization, self-check removal, model promotion, full-video claim or
  new raw tensor/video files. A useful screen still needs complete temporal fast
  pipeline/face validation before adoption.

Main writes/freezes/launches; existing Luna/max only monitors and runs the fixed
stored-result audit. Exception, nonzero exit, stall or failed audit: report to
main immediately without repair or retry. All result data stays on D:.
