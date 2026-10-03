# Public GPU reproduction review — 2026-10-03

The portable public `current720` entry completed on an Intel Arc B580 with synthetic fixed-seed input. The accepted route passed its three-repeat workflow, followed by a separate matching baseline/FDP pair using one measured repeat. This verifies the public offline workflow and its cache, replay, finite-output/history, and retirement checks. It does not qualify game performance or private-video quality.

The only code repair was an explicit `str(torch.__version__)` conversion in test-runner metadata capture. The durable serializer remains strict, and no current model or kernel source changed. CPU checks passed 13/13; source validation passed all 571 files; external assets validated at 527 files and 56 profile arrays.

## Accepted current720

The run processed 13 frames over an initial sequence, two warm sequences, and three measured sequences. It recorded two captures and eleven initial replay routes; all 13 outputs and histories were finite, phase arrays matched byte-for-byte, retirement passed, and the fresh read-only cache hit 137/137 keys with zero writes and unchanged inventory.

| Timing scope | Samples | Mean | Median |
|---|---:|---:|---:|
| Fullsize process, XPU events | 39 | 45.330 ms | 45.259 ms |
| Raw body, XPU events | 50 | 39.077 ms | 39.092 ms |

## Single-repeat matched FDP pair

Each arm used the same 13-frame controls, two warm cycles, and one measured cycle. Both arms passed finite-output/history, capture/replay, same-arm cold/read-only identity, retirement, and read-only cache checks. Read-only keys/hits were 139/139 for the accepted baseline and 161/161 for FDP; neither arm wrote to cache.

| Timing scope | Baseline mean | FDP mean | Difference |
|---|---:|---:|---:|
| Fullsize process, XPU events (13 samples/arm) | 47.581 ms | 44.672 ms | -2.909 ms |
| Raw body, XPU events (50 samples/arm) | 40.067 ms | 38.284 ms | -1.783 ms |

The paired synthetic outputs and histories were finite. Mean MAE was 0.0002585, worst-frame MAE 0.0003066, maximum absolute error 0.004761, mean PSNR 68.573 dB, and minimum PSNR 67.673 dB. No quality threshold was applied. The timing deltas are single-run observations and are **not** a repeatable performance qualification.

## Receipt failure and correction

The first accepted precompile did execute on the GPU and produced all 13 frame/history outputs. Its final result write failed because the strict JSON serializer rejected TorchVersion metadata; the fallback failure receipt's `GPU_executed=false` flag therefore did not mean GPU work had not run. The run and stack were preserved. The runner now converts only the captured Torch version to built-in `str` and retains strict serialization.

## Scope and evidence

No game was started, no game FPS is claimed, and no output arrays or absolute private paths are included in this review. Private-video/human visual review, NVIDIA/4060 byte identity, and native game integration remain outside this synthetic public test.

- Compact result receipt: [PUBLIC_GPU_REPRODUCTION_RECEIPT.json](../public-review/PUBLIC_GPU_REPRODUCTION_RECEIPT.json)
- Sealed tests READY SHA-256: `da31b237b40f06a145acfb4aca3fbd46e0598f3b3801761435436b27560402ae`
- Receipt SHA-256: `5db2e89c76134701ee8a0f9f1784ad5b591f054da2433d8f40c9d67e7ae17813`
- Review JSON SHA-256: `b10e5f77c1c3523b37aab98ba78a74bfdc363f0edb8360da76901862e74b469f`
