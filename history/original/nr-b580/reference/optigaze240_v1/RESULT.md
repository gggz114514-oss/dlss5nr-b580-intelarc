# 240p input / 480p face-video comparison

Status: awaiting B580 smoke and 243-frame run. Do not claim visual acceptance or
speed from the implementation alone. The fixed contract, six comparison arms,
and measurement boundaries are in [PLAN.md](PLAN.md).

## Implementation

- `adapter.py`: 480p→240p RGB/motion preparation, residual extraction,
  480p source-guided reconstruction, and motion-vector temporal reprojection.
- `run_video.py`: frozen-input verification, separate baseline/NR256/240p arms,
  six-way review render retaining source audio, and synchronized timing.
- `launch.py`: isolated pinned runtime/toolchain launcher. Compiled artifacts and
  media are stored on D:, not in the source tree.

## Pending evidence

- Two-frame 240p geometry and GPU-kernel smoke.
- Three model arms at 243 frames, plus frozen source arm.
- Six-way MP4 at 243 frames with audio; user visual review.
- Full-size, NR256, and 240p model plus reconstruction timing.
