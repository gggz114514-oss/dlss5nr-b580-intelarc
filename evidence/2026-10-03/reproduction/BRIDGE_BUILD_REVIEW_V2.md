# Prepared public bridge CPU build review V2

CPU_BUILD: **PASS**. SDK pin 7534ad00bf9e590eedb99e8dd9fd8c89dae3654f; default
build completed=True, original five CPU tests passed=5,
historical live-six-ON bridge build completed=True. Both builds requested
16 parallel CPU jobs with a task-local MSVC/Windows SDK environment.

Main's public tools/prepare_bridge_build.py generated the external source.
All frozen 162 original files remain SHA-pinned and unchanged; only the
prepared src/asi.cpp disabled-diagnostic #else scope declaration differs. The
prepared change and compiler preprocessing proof are recorded in the receipts.
No frozen current, public tests, GPU model, SDK or G/runtime source was edited.

CPU tests were the original five source-reviewed contracts and required a PE
import audit before launch. ASI/mock loaders, GPU probes and all other CTest
entries were excluded from execution. The live ASI was built without loading
or running it. No GPU device/library/driver API, deployment, SYCL helper rebuild
or historical binary identity proof was performed.

Full commands, compiler dependency list, hashes and failure evidence:
${EXPERIMENTS_E}\cyberpunk-opt\public-bridge-build-20261003\attempt-03\CPU_BUILD.json
Local review: ${WORKSPACE}\nr-b580\reference\development-archive-20261003\BRIDGE_BUILD_REVIEW_V2.json
Path-free public small receipt: ${WORKSPACE}\nr-b580\reference\development-archive-20261003\BRIDGE_BUILD_REVIEW_V2_PUBLIC.json
Prior attempt-01 environment failure and attempt-02 source-compile failure retain
their original receipts/logs and are SHA-referenced by this attempt. Build trees
are outside Git. Failure, if any: none.

The worker stops after this bounded default/CPU/live build validation.
