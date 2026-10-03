# Face-color cause investigation

## Completed diagnostic (2026-09-10)

The user's startup-status question authorized a one-time check. The local B580
job ran from 16:12:24 to 16:13:02 +08:00 (37.57 seconds), exit code 0. Luna's
startup receipt is 16:12:50; its prewritten audit completed successfully. Main
authenticated the completed report, log and lease against the audit hashes.
Report SHA256: eb53aaf9ab366965244032a7e1d0faa01d2e56ca188aa38542dc23f073367308.

Both reviewed frames were reproduced exactly, all 16 GPU FFNs matched the
CPU oracle, and both whole-body CPU replacements matched their GPU controls.
On frames 96/192, using INT8 prior history with an FP16 current body caused
larger ROI RGB RMSE than changing only the current body to INT8: 2.998/2.759
versus 0.957/1.192 in 8-bit units. These are separate interventions with a
nonzero interaction, not additive percentages of total error.

Removing hidden clipping while retaining the quantization step and other
INT8 rules reduced same-history ROI RGB RMSE from 0.957 to 0.188 and from
1.192 to 0.214 (about 80%/82%). Actual INT8-path hidden clipping ranged from
1.53% to 3.74% across these blocks/frames. Restoring expansion, internal FP8
rounding, or contraction weights did not produce comparable improvements.
The evidence identifies hidden-range clipping as the leading tested source
of immediate extra INT8 error, with previous-history differences carrying a
larger effect into these frames. It does not prove the entire sequence's
history error is due to clipping, nor explain all NR256-versus-full-size color
differences. The ROI includes background; this is not a skin-only measurement.

Widened hidden integers were a CPU diagnostic, not a deployable INT8 fix or a
speed result. Next useful work is a separately versioned calibration/range
experiment with diverse calibration samples and held-out sequential quality
checks. No new experiment is running, no defaults changed, and no new quality
approval is implied.

## Original dispatched design

The user requested the cause of visible face-tone differences. The changes
already appear in pre-encode RGB PNGs. Prior remux verification established
that correcting container tags preserved all decoded pixels.

A read-only comparison of saved source/NR256 PNG panels and full-resolution
exact/FP16 arrays found more than one contribution. Full480 FP16 is close to
the B580 exact route in the fixed review ROI's channel means; NR256 changes
those means further, and continuous INT8 adds a content-dependent difference.
The INT8-minus-FP16 brightness direction is not constant across frames.
These ROI statistics include background and are not a segmented skin metric.

The INT8 hidden scales were fitted to 48 correlated feature positions in one
other frame. This makes activation clipping a hypothesis worth testing, but
the calibration limitation alone does not establish the cause of the color
shift. Entry/weight quantization, replacement of the internal FP8 boundary,
folded contraction weights and changed accumulation can also contribute.

`experimental/diagnose_int8_face_color_v1.py` was dispatched for frames 96 and
192. It first restores each route's saved prior output/seed and must reproduce
the reviewed low RGB bytes and full-output hash. It then holds captured body
inputs fixed for these diagnostics:

- Cross FP16/INT8 current body with FP16/INT8 previous history, including an
  interaction term. This separates immediate arithmetic from prior-state
  effects on the selected frames without resetting away the history.
- Check the CPU INT8 oracle against all eight actual FFNs on both frames and
  against the whole GPU body; record hidden clipping and signed error.
- Restore all FP16 FFNs as a byte-equal control, then restore each FFN alone.
- Remove hidden clipping, restore FP16 expansion, restore the internal FP8
  boundary, or remove folded-weight quantization, one diagnostic at a time.

The widened-hidden and unrounded-weight probes are deliberately slow CPU
counterfactuals. Their purpose is attribution, not a deployable fast backend.
All interventions use fixed FP16 front/history inputs; each body result is
composited with the same residual scaler. Existing reviewed outputs, constants
and runtime defaults remain unchanged. No new video or weight files are made.

Data: `D:/Codex-NR-Experiments/nr-b580/reference/experimental/int8-face-color-cause-v1/`.
Luna/max monitors the 1200-second bounded job and runs only the prewritten
`experimental/audit_int8_face_color_v1.py`, reporting any error immediately
without repair or retry. The completed findings are recorded above. The
original report and executed experiment sources remain unchanged.
