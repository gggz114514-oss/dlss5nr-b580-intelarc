# Public tests and B580 offline reproduction review

The public tests remain pinned to the 571-file installed-current source manifest and the 260-file r18 constructor registry. After the metadata fix, the stdlib CPU suite passed 13/13, source validation passed for all 571 files, and the external asset bundle validated 527 files including 56 profile arrays. The model source, numerical kernels, and runtime behavior were not changed.

The first accepted precompile did execute GPU work and wrote all 13 synthetic outputs and histories. It failed when serializing the final result because `torch.__version__` was a `TorchVersion` string subclass and the durable serializer intentionally accepts only built-in scalar types. The public runner now converts this value with `str(...)` at capture; the serializer remains strict. The targeted CPU regression confirms that the subclass is still rejected directly while the captured built-in string is accepted. The failed attempt and its stack remain preserved as failed evidence; its `GPU_executed=false` fallback receipt does not mean that no GPU work occurred.

## GPU reproduction

The accepted current720 run completed precompile plus a fresh readonly process at fixed 1280x720 and 13 synthetic frames, with two capture routes, 11 initial replay routes, two warm cycles and three measured cycles. Both phases completed, all output/history values were finite, and their arrays were byte-identical. The fresh DiskOnly phase hit 137/137 actual compiler keys, reported zero readonly writes and verified graph retirement. Full `FullsizeGameModes.process` XPU timing was 45.330 ms mean over 39 samples (median 45.259 ms); raw graph-body timing was 39.077 ms mean over 50 samples (median 39.092 ms). These timing scopes are separate; neither is game FPS.

The bounded `fdp` pair used one measured cycle (`--candidate fdp --repeats 1`), with a matching accepted baseline first. Each arm completed its own precompile and fresh readonly phases over 13 frames; each had two capture routes and 11 initial replay routes, two warm cycles, finite output/history, byte-identical same-arm phase arrays, verified retirement, and zero readonly cache writes. DiskOnly hits were 139/139 for the paired baseline and 161/161 for FDP. The paired synthetic mean MAE was 0.000259, worst-frame MAE 0.000307, maximum absolute error 0.004761, and minimum PSNR 67.673 dB. No quality threshold or human visual review was applied.

| Scope | Matching baseline | FDP candidate | Candidate minus baseline |
|---|---:|---:|---:|
| Full process XPU events, 13 samples per arm | 47.581 ms | 44.672 ms | -2.909 ms |
| Raw body XPU events, 50 samples per arm | 40.067 ms | 38.284 ms | -1.783 ms |

This one-repeat smoke pair verifies public construction, finite values, actual cache routing and same-arm reproducibility. Its timings are descriptive only and are not performance qualification.

The compact receipt is at [`evidence/2026-10-03/public-review/PUBLIC_GPU_REPRODUCTION_RECEIPT.json`](../evidence/2026-10-03/public-review/PUBLIC_GPU_REPRODUCTION_RECEIPT.json). The external runtime/assets and raw NPY outputs are not included in the release tree. This synthetic test does not establish private face-video quality, 4060 byte identity, native game integration, manual visual acceptance or game FPS.
