# Continuous INT8 FFN: requested face review

The user requested face footage before deciding whether to accept the new
continuous INT8 route. The existing 480p clip contains a person lying down,
sitting up and looking at a phone. It has 243 frames at 24 fps (10.125 seconds).

`experimental/review_int8_face480_v1.py` was dispatched with the selected
FP16 NR256 stack and the continuous INT8 FFN NR256 candidate. Both receive
complete frames and the same authenticated pre-existing motion fields,
with independent temporal histories. A geometry adapter reuses the frozen
residual filter kernels for the 864x480 source, using 256x142 active pixels
inside the existing NR256 canvas. Exact-backend/default sources are unchanged.

The presentation has three columns: original, current selected FP16 NR256,
and new continuous INT8. Each shows the complete 864x480 picture, then the
same fixed output ROI `(256,16)-(640,400)` enlarged 2x with nearest sampling.
The model input is never cropped to the face. This is a quality review; no
performance claim is derived from encoding-time execution.

Results go to
`D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-face480-review-v1/`.
Expected video: `comparison-full-and-face-24fps.mp4` (2592x1352).
Only low-resolution model arrays and a few comparison PNGs are retained;
complete per-frame floating-point full-resolution outputs are not dumped.

The original run generated all 243 frames, then failed its final color-tag
assertion. H.264 SPS values were already BT.709 (1/1/1), but the MP4 `nclx`
atom stored unspecified primaries and transfer (2/2/1). FFprobe consequently
reported no primaries or transfer. Setting encoder flags and modifying the
H.264 bitstream alone did not make this container correct.

`experimental/finish_int8_face480_v1.py` creates a separate MP4 through an
explicit FFmpeg 9 stream-copy step. It checks both the MP4 atom and H.264
headers, validates metadata/timestamps and directly compares every decoded
YUV byte across all 243 frames. It also authenticates the frozen inference
evidence without rerunning the model. Original failed artifacts stay intact.
The recovery launch returned code 0. New artifacts are in
`D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-face480-finish-v1/`.
The final video keeps the name `comparison-full-and-face-24fps.mp4`.

Future exports should include this independent FFmpeg 9 stream-copy
finalization and check both metadata layers; encoder-side flags alone are
not sufficient evidence of a correctly tagged MP4 in this pipeline.

Luna/max completed the prewritten recovery audit successfully. It checked
800 source entries, 269 exact-gate entries and 1,536 unique files. Main
reauthenticated the completed report/log/lease/video and seven PNG hashes
after the user's face-quality feedback. All 243 decoded frames match across
the metadata repair; that equality does not apply to FP16 versus INT8.

The user's feedback was: "有脸的色调上有明显差别，画面似乎都可以接受？"
This is recorded as **provisionally acceptable on this clip, with visible
face-color differences**, not unconditional or general quality approval.
The separate `int8-ffn-face480-finish-v1/user-review-v1.json` preserves the
exact feedback and binds it to corrected video SHA256
`f9a6038f6293dffc1a316eddbfadaab59a46a170b5040117c212aec5f4c91465`.

Main inspected pre-encode PNG frames 96 and 192. FP16 appears warmer/darker
than the source; INT8 is cooler/brighter relative to FP16. Facial outlines
show no obvious collapse in these stills; this is not an independent
all-frame temporal judgment. The column differences are already present in
RGB PNGs before encoding. Source-to-FP16 appearance changes cannot be
attributed to scaling alone from this comparison.

Retain the INT8 route as a fast candidate and track skin-tone/luminance bias
alongside speed. No automatic color correction, default promotion, change to
the exact route, or publication was made. The car clip's pending review is
not implicitly accepted by this face feedback.
