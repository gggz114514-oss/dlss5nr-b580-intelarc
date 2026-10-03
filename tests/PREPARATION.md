# External assets and runtime dependencies

Supply a dedicated Python with Torch XPU, NumPy and a matching Triton-XPU build,
plus an Intel driver. `--python` selects that interpreter. Record driver details
in the external asset receipt; actual device/Torch/Triton/library versions will
also be recorded by the GPU child. Do not use a global ComfyUI environment.

Supply `--assets /path/to/bundle`, containing `assets.json` with schema
`nr-release-assets-v1` and a `files` list. Each file row has `relative_path`,
`sha256` and optionally `bytes`; paths are relative to the bundle and describe
the same relative paths in an isolated portable runtime. No junctions or
symlinks are accepted. The bundle contains:

- `exact/model-assets/sf-v2/WEIGHTS_HT.bin`; actual SF-v2 weight bytes.
- Complete `noise-sm89-v2`, `sigmoid-sm89-v1`, style and reciprocal-dimension
  model asset directories, with their original manifests and arrays.
- `templates/profile-v1.json`, its 8 ViT hidden-scale arrays and 16 C512
  floor16 calibration records, including every referenced `.npyz` blob.
- Both `data/reference/cubic-fp8-fused-cpu-v3.json` and
  `data/reference/cubic-fp8-fused-xpu-v3.json`, and the referenced real LUT NPY.
- The matching `toolchain/triton` distribution, libdevice and native extension,
  `host-helpers/manifest.json`, the three actual host-helper binaries and
  `host-helpers/fast-cache-identity.json`.

Profile paths should be `${ROOT}/...` or relative paths into this bundle.
Absolute paths into a developer installation are rejected. Hashes in array
metadata are checked before rebinding paths. Only derived copies are written
under `--output`; no published source or external asset is modified.

The installed initializer is `game/fullsize_session_v1.py` ->
`nr_runtime_v1.Session.create` -> `nr256_product_stack_v1.Stack`;
`FullsizeGameModes` replaces the fixed rows with the real fullsize route. The
factory name contains NR256, but no downsampling/scaler is used for 720p here.
`MotionNR.from_assets`, `nr_runtime_v1.load_profile`,
`compressed_arrays_v1.load`, and `cubic_lut_constant_v1.register` require real
assets and validate their own bytes. These checks are retained.

The extracted current tree initially lacked the small portable root bootstrap
utilities, the compiled host helpers and the toolchain. The tests include
the actual reviewed bootstrap utilities as source support, while binary/runtime
dependencies remain user-supplied. First-use Triton kernel compilation is an
explicit offline precompile phase; readonly measurement never fills a cache.

The private fixed13 media and historical V4/PHASE1 receipts are not public input
fixtures. The public runner instead generates seed `20261003` RGB and
current-to-previous motion or accepts an external `nr-release-inputs-v1` NPY
manifest. It labels the scenario and records each frame's file/raw SHA, reset
and seed semantics. Missing original receipts cannot be replaced with old
claimed results.
