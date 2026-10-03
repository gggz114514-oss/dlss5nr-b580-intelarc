# OptiScaler NR integration reconnaissance (2026-09-08)

This is source inspection and an adapter plan, not a working OptiScaler B580 plugin.
The active backend remains the validated tensor implementation. New actual GPU
roundtrips are documented in [GPU_RESOURCE_INTEROP_STATUS.md](GPU_RESOURCE_INTEROP_STATUS.md):
the real PyTorch Level Zero queue can import a D3D12 buffer and expose it to
PyTorch/Triton as the same pointer. FP32/FP16 each32 rounds pass with separate
single-producer timeline fences. Full NR now also passes on shared inputs/output:
real480 and real1080 each13 frames, including own-history recurrence and reset;
all D3D12-read RGB and private history bytes equal the authenticated4060 reference.
The new bridge matches SYCL and DXGI LUIDs. Its owned test devices/queues and CPU
fixture upload/readback are not yet a production caller-owned game-texture adapter.
Warm means2.677/11.011sec include test upload and D3D12 output readback; still far
from real time, and not directly comparable to earlier model-only timing.

Inspected upstream: Dagherbou/OptiScaler_DLSSNR, branch dlss-neural-rendering,
commit `973761621353b99bee3dc7d4bb27b117fef2644f`. Private research checkout:
reference/integration/optiscaler-nr-source-v1. Eight NR interface/implementation/
shader files were downloaded from that exact commit; `git hash-object` matches
their committed blob IDs. No upstream code was built or installed in a game.
GitHub API rate limit and partial-clone lazy fetch failed; raw files from the
pinned public commit supplied the independently hash-checked content instead.

The relevant call is `g_nr.evaluate` in
[DlssNr_Dx12.cpp](https://github.com/Dagherbou/OptiScaler_DLSSNR/blob/973761621353b99bee3dc7d4bb27b117fef2644f/OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp#L2189).
It receives a caller-owned D3D12 command list, model input/depth/motion/output,
working dimensions, guide dimensions, inverted depth flag, reset, tuning, and MV
scales. The surrounding code preserves its encode/downsample/resolve composition
and only attaches NR to upscaler evaluation. This is the intended substitution
boundary, together with backend-specific feature creation and lifetime handling.

Submission ordering is still an explicit integration gap. In the inspected
source, evaluate is called while recording the caller's graphics command list;
the nearby code does not submit that list. Our passing bridge instead owns and
submits its producer list before the SYCL queue waits on the producer fence.
A replacement that starts host-waiting NR input validation while the game has
not submitted its producer list could deadlock. Having a timingQueue pointer
does not establish permission to close/submit/reset the caller's recording list.
The adapter needs a verified submission boundary or another way to schedule the
same-frame dependency. Texture packing alone will not resolve this requirement.

Observed integration requirements:

| Host behavior | Current core / work required |
| --- | --- |
| Colour/output at working size; depth/MV may be render resolution | Core currently consumes same-sized motion tensors. Recover and verify original guide sampling/subrect behavior before adapting. |
| Dynamic valid guide region may be smaller than allocation | Track allocation and valid rect separately; do not read stale margins. Native reference measurements are needed. |
| MV scale passed from game, then multiplied by workWidth/displayWidth | Core expects current-to-previous pixel displacement. Verify units and the working-size conversion once; avoid applying the resolution ratio twice. |
| Reset from game plus resize/feature rebuild | Preserve per-session private history and deterministic seed; test cuts, control changes and resize against the pinned reference. |
| Encode/resolve surrounds model call, including HDR processing | Preserve host composition and independently verify model input colour contract. Current NR evidence is scoped SDR; no general HDR claim. |
| Commands recorded on caller's D3D12 list; textures may be in flight | Implement resource/state/fence ownership and actual Intel interoperability. Tensor API alone cannot satisfy this. |

The existing XeSS tool's GPU frame contract also separates valid rect, allocation,
colour metadata, motion direction/units, source-frame IDs, adapter identity and
producer/consumer completion fences. The NR core should expose these contracts
through per-tool adapters rather than relying on implicit frame conventions.

Intel's [SYCL bindless image specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc)
describes external memory and semaphore import, but support is queried per device
and backend. Locally installed oneAPI 2026.1 headers include Windows NT handles
and DX12 fences. That is API availability, not proof that B580's Windows Level
Zero driver can import those resources. The read-only probe was compiled and run
under the shared lease. This oneAPI 2026.1 runtime enumerated one B580 device,
driver32.0.101.8974, OpenCL backend (enum1). Bindless images, external memory import
and external semaphore import aspects are all false; no Level Zero device was
enumerated, despite its adapter DLLs being installed. This does not prove every
other Intel/OpenCL/D3D interop route is unavailable. This result is limited to
the standalone 2026.1 runtime. The later actual-PyTorch-queue probe found the
Python environment's 2026.0 runtime using Level Zero with all three import
aspects true, and its D3D12 buffer import now passes actual tensor roundtrips.
Evidence: reference/integration/sycl-interop-probe-v1/capabilities.json, with source,
executable and lease hashes. No queue, memory allocation or GPU kernel was created
by the capability probe. A real D3D12 texture/fence roundtrip and compatibility
with the tensor runtime remain separate gates.

A second read-only OpenCL device query completed under the shared lease. This
driver advertises cl_khr_d3d11_sharing, cl_khr_d3d10_sharing,
cl_khr_external_memory and cl_intel_unified_shared_memory, among other sharing
extensions. No semaphore extension or external-memory Windows-handle extension
appeared in the device extension string. These are capabilities to investigate,
not a validated D3D12 import route. In particular, D3D11 sharing alone does not
establish D3D12 texture/fence import or tensor-runtime compatibility. The query
created no context, queue, allocation or kernel. Evidence:
reference/integration/opencl-interop-capabilities-v1.json and
reference/integration/opencl-interop-query-v1.log.lease.json (return code 0).
