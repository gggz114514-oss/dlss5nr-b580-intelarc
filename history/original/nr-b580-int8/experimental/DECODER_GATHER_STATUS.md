# Decoder gather status — 2026-09-11

User chose byte-preserving optimizations and rejected further approximate QKV.
The old full-QKV process had already finished rc0; no further promotion or video
review is planned. Results are retained as experiments.

New frozen implementation b5b65f6a7bfa983443ca39ac59e2654affaaa464 directly
gathers coarse projected values when merging skips at five decoder boundaries.
Original half FMA/FP8 rounding and C32 raw residual/padding remain.
See DECODER_GATHER_DESIGN_V1.md for reasoning and limitations.

Static checks passed. GPU experiment actually launched via lease, session98117,
owner nr-decoder-gather-v1, timeout1800. Original Luna/max resumed after a
not_found response and reassigned monitoring/fixed audit. Main does not monitor.
No measured gain or byte-parity pass claimed yet. Default/exact/public unchanged.

Data: D:/Codex-NR-Experiments/nr-b580/reference/results/decoder-gather-v1/
Monitor manifest/audit handoff: sibling decoder-gather-v1-monitor-luna-v1/.
After completion authenticate stored results and judge intact-body gain. Local
merge-only gain is insufficient. Complete temporal fast pipeline/face validation
is still needed before adoption; this screen does not promote the candidate.

## Completed and authenticated by main

Luna rc0 / fixed audit passed. Main rehashed 898 files including handoff
(Luna897 excludes handoff). Result SHA
0f77ec5b4016fc1d52a95e47c341ab7de2fea7e78a8d0dce0b5f2f668580c7df;
handoff SHA319e97d312fbdff3b42b34592662dcccc4b5ed73836c8fb1f6addf0cc67c9420.
13 frozen car outputs, 5 propagated merge boundaries, 10 numeric/stride/pad
controls and 2 live-input cases passed byte equality. Constants unchanged.
Capture: 643->639 Triton, standalone FP8 140->136, logical quantization315->311,
elided175 unchanged. Candidate zero spill.

Paired whole body8.0323484375->7.9703921875ms, save0.06195625ms/0.771334%;
local five merges0.0647453125->0.0350375ms, save0.0297078125ms/45.884113%.
Both12/12 candidate wins. Medians are separate and must not be added.
Retain as useful byte-preserving candidate; not promoted, full temporal pipeline
and complete face clip validation remain. No new human review needed if bytes
continue to match the approved route. Main decision saved beside validation.
