# B580 full-size GraphFront ABBA

- Status: **complete**
- Passed: **True**
- Input and NR network: **480×864**; direct full-size fast model call.
- Fixed source sequence: 243 frames; steady statistics use frames 4–242 (239 samples).
- Historical eager steady median: **42.9709 ms** (`materials_entry_v1`, original-size C512 INT8 + ViT INT8; experimental path, not product speed).
- A/B variable: whether the existing GraphFront component is installed.

## Gates and runtime identity

- Input capture: `315ccc7179b5c789205d003417896dfd20bc7fce8182e5a74cb0a57f4c7e64a6`
- Input manifest: `1e2b871175b768e2518f445e59aa6cc21d7d1b99b5734c403cfc7f2c56a5ebb2`; 243 files, stored FP16, RGB/motion passed to the model as FP32.
- Profile: `8c0994c72c95404caf8116ef0a2ac0580fd8330c8b8a0c0a5f856489b1a9bb21`
- Graph class: `graph_front_v6.GraphFront`
- Fast backend: `E:\ComfyUI-aki-v3-IntelArc_20260722\nr-b580-int8\backend\nr_backend`
- Cache mode: DiskOnly: no compile fallback; missing artifact raises
- Product flags: `{"PREPFAST_CUTS": "off", "SELECTFAST_CUTS": "off", "SYNCFAST_CUTS": "off", "NR_LAUNCH_FASTPATH": "run,getitem,launch,md,book", "NR_SELECT_CACHE": "on", "NR_FMA_SCALAR_CACHE": "on", "NR_ALLMERGE_GUARD": "on", "NR_QUANTTRIM_GUARD": "on", "NR_PADGUARD": "on", "NR_QKVCACHE": "on", "NR_K8_FP16_XMX": "on", "NR_MOTION_SCOPE": "on", "NR_BACKEND_SIDECARS": "off"}`
- Cache provenance: `{'path': 'D:\\Codex-NR-Experiments\\nr-b580\\graph-replay-480x864-abba-v1-20260923-run07\\baseline-cache-copy', 'source': 'D:\\fullsize-rows-v1\\r1\\triton-cache', 'copied_from_frozen': True, 'preparation_mode': False, 'prepared_receipt': 'D:\\Codex-NR-Experiments\\nr-b580\\graph-replay-480x864-abba-v1-20260923-run07\\prepare-cache\\cache_receipt.json', 'file_count': 2510, 'bytes': 79514064, 'manifest_sha256': '7d2634ea13a14b34bc4a5a792da3dc0c2227c8b96b23d25b4ecbbb12cf13b1d0'}`

## Earlier launch attempts
- run01: failed-before-model — embedded Python default GBK could not decode Triton driver.c; no measurements.
- run02: failed-before-smoke-completion — used a different D: shared cache and DiskOnly rejected missing _matmul; no JIT.

## Arithmetic and product-path distinction

- This experiment: `materials_entry_v1 full-size fast arithmetic: C512 FFN INT8 rows + ViT FFN INT8 rows; direct stack.model(480x864); graph excluded in OFF arms and replayed in ON arms; this is a standalone experimental path`
- Historical 42.9709 ms eager reference: `{'path': 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580\\reference\\materials_v1\\materials_entry_v1.py', 'sha256': '82146bc51259cde7004ef9247d8072234699ce4961a8b4a7e1f51cb86bbdc2d7', 'arithmetic': 'C512 INT8 rows + ViT INT8 rows', 'graph': 'excluded; eager', 'historical_median_ms': 42.9709, 'scope': 'original-size experimental model call; not product speed'}`
- Separate reviewed ViT-FP16 quality lane: `{'path': 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580\\reference\\fp16land_v1\\vitffn_fp16_land_v1.py', 'sha256': '50754b3a55f690b1879bc30672cd6712966a28020190e187997aae6f0b571a71', 'arithmetic': 'separately reviewed ViT-FP16 quality lane', 'included_in_42_9709_ms': False}`
- Product default for 480×864: `{'path': 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580-int8\\product\\nr_runtime_v1.py', 'sha256': 'eb32070cbbc9977f8b8114abd019a000c471d5af1b9d77ad9cd56ee185f3240d', 'runtime_import': 'E:\\ComfyUI-aki-v3-IntelArc_20260722\\nr-b580-int8\\product\\nr_runtime_v1.py', 'runtime_import_sha256': 'eb32070cbbc9977f8b8114abd019a000c471d5af1b9d77ad9cd56ee185f3240d', 'behavior': '480x864 product input uses Face480Scale to NR256', 'measured_by_this_experiment': False, 'product_adapter_bypassed': True}`
- The 42.9709 ms number is an original-size **INT8-ViT eager experimental-call** result, not product-node speed and not the separately reviewed ViT-FP16 lane. The old G: validation points to output frames that are now missing, so it is a numeric anchor only; OFF-A1/OFF-A2 below are the direct same-input controls for this ABBA.

## ABBA frame-wall results

| Arm | Graph | Samples | Median ms | Mean ms | p10 ms | p90 ms | Replay count | Pixel-equal to first eager |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off-a1 | False | 239 | 41.072299998631934 | 41.176245606921576 | 39.7912800006452 | 42.75986000066041 | 0 | True |
| on-b1 | True | 239 | 36.22910000194679 | 36.30107824251525 | 36.00418000132777 | 36.726499998621875 | 243 | True |
| on-b2 | True | 239 | 36.356200002046535 | 36.509280334703895 | 36.080760000186274 | 37.03673999843886 | 243 | True |
| off-a2 | False | 239 | 41.168799998558825 | 41.68024142242631 | 39.82959999993909 | 44.220620001578936 | 0 | True |

## Paired comparison

- Graph saving: **4.827899996598717 ms** (11.740844898143704%).
- Eager minus 42.9709 ms: -1.850350001404621 ms.
- Graph minus 42.9709 ms: -6.678249998003338 ms.

## Smoke gate

- Passed: True; graph entries: 2; graph replay delta: 8; pixel-identical: True.

## Measurement boundary

Clock covers the synchronized full-size model call and existing Python/scopes/guards. Frame file I/O, CPU-to-XPU upload, CPU readback, output comparison, graph build, and warmup are excluded. All smoke and timed calls run under DiskOnly; a missing kernel artifact is a hard failure, so the run does not compile.

The historical 42.9709 ms INT8-ViT reference reports a 239-frame steady median and excludes upload, readback, I/O, motion generation, and codec. Its referenced output frames are unavailable, so pixel comparison is not claimed. This experiment instead uses the fixed precomputed RGB/motion input for all four ABBA arms and excludes the same operations.

## Files changed

- `PLAN.md` — fixed experiment contract.
- `graph_replay_480x864_abba_v1.py` — isolated benchmark entry.
- `RESULT.md` — measured result and current run status.
- D: run outputs, logs, temporary files, and Triton cache.

## Final execution record

- Final reproducible run: `D:\Codex-NR-Experiments\nr-b580\graph-replay-480x864-abba-v1-20260923-run07`. Its `abba/report.json`, `prepare-cache/cache_receipt.json`, lease logs, GPU preflight, cache manifest verifier, and result verifier retain the raw evidence. The fresh cache copy matched the frozen 2,496-file source manifest before preparation; the prepared receipt and post-run manifest also matched.
- Cache preparation passed with JIT restricted to the isolated cache copy; the subsequent smoke passed under DiskOnly. Eager and GraphFront preparation outputs were byte-identical; the smoke created the expected reset/temporal entries, replayed all 8 check frames, and remained byte-identical.
- Each ABBA arm processed all 243 frames. Across all four arms, every frame output SHA-256 matched the eager anchor; private history matched each returned frame. The timed arms used frames 4–242 for 239-sample steady medians. Graph replay count was 243 per ON arm and zero per OFF arm.
- GPU preflight for the timed run found no active GPU engine above the sampling threshold, no compute workload, no ComfyUI/model process, no lease owner, and no STOP marker. Timing covers only the synchronized direct `stack.model(480x864)` call and its in-call guards; it is not product-node latency.
- Test-harness-only corrections are recorded under run04–run06 `BLOCKED.md`: the C512 HWC padding tuple now matches the existing six-value implementation, and the smoke gate now reads GraphFront's captured dispatch record and sums per-frame FFN counters. Neither correction changes model math or product code. The final harness SHA-256 is `edf8b8bb6dd6877afeca12f68d197f53faa9f35be4304098f38414681e2db87d`.
