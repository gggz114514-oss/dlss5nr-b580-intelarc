cmake_minimum_required(VERSION 3.24)
project(cyberpunk_nr_bridge LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The ASI bridge requires 64-bit Windows")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPTISCALER_SOURCE "" CACHE PATH "Read-only OptiScaler source checkout for SDK headers and Detours")
if(NOT EXISTS "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk/nvsdk_ngx.h" OR
   NOT EXISTS "${OPTISCALER_SOURCE}/OptiScaler/library/detours/detours.lib")
  message(FATAL_ERROR "Set OPTISCALER_SOURCE to a complete OptiScaler source checkout")
endif()
if(EXISTS "${OPTISCALER_SOURCE}/.git")
  execute_process(COMMAND git -C "${OPTISCALER_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE OPTISCALER_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT OPTISCALER_COMMIT STREQUAL "7534ad00bf9e590eedb99e8dd9fd8c89dae3654f")
    message(FATAL_ERROR "This bridge is audited against OptiScaler v0.9.4 commit 7534ad0")
  endif()
endif()

add_library(nr_core STATIC src/nr_core.cpp src/nr_ngx_adapter.cpp src/nr_diag.cpp src/nr_gpu_handoff.cpp
  src/nr_contract.cpp src/nr_sync_probe.cpp src/routes/dlss.cpp src/routes/dlss_identity.cpp
  src/routes/fsr.cpp src/routes/xess.cpp)
target_include_directories(nr_core PUBLIC include PRIVATE "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
target_compile_definitions(nr_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

add_library(nr_hdr_proxy STATIC src/nr_hdr_proxy.cpp)
target_compile_definitions(nr_hdr_proxy PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(nr_hdr_proxy PUBLIC d3d12 d3dcompiler bcrypt)

add_library(CyberpunkNRBridge SHARED src/asi.cpp src/nr_python.cpp)
target_include_directories(CyberpunkNRBridge PRIVATE include
  "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk"
  "${OPTISCALER_SOURCE}/OptiScaler/include")
target_link_libraries(CyberpunkNRBridge PRIVATE nr_core
  "${OPTISCALER_SOURCE}/OptiScaler/library/detours/detours.lib" d3d12 dxgi)
target_compile_definitions(CyberpunkNRBridge PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
option(NRB_XESS_IDENTITY_TEST "Diagnostic XeSS pre-SR identical-color relay (not a release feature)" OFF)
if(NRB_XESS_IDENTITY_TEST)
  target_sources(CyberpunkNRBridge PRIVATE src/xess_identity.cpp)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_XESS_IDENTITY_TEST=1)
endif()
option(NRB_DEFERRED_IDENTITY_TEST "Diagnostic split-batch XeSS identity relay (never for release)" OFF)
option(NRB_LIVE_NR_TEST "Run B580 NR in the validated Cyberpunk split-batch slot" OFF)
option(NRB_MULTI_ROUTE_LIVE "Allow independently verified FSR/XeSS split-batch NR" OFF)
if(NRB_DEFERRED_IDENTITY_TEST)
  if(NOT NRB_XESS_IDENTITY_TEST OR NOT NRB_TAIL_ORDER_PROBE)
    message(FATAL_ERROR "Deferred identity requires identity and tail-order probes")
  endif()
  target_sources(CyberpunkNRBridge PRIVATE src/deferred_identity.cpp)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_DEFERRED_IDENTITY_TEST=1)
  if(NRB_LIVE_NR_TEST)
    target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_LIVE_NR_TEST=1)
    target_link_libraries(CyberpunkNRBridge PRIVATE nr_hdr_proxy)
  endif()
  if(NRB_MULTI_ROUTE_LIVE)
    target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_MULTI_ROUTE_LIVE=1)
  endif()
endif()
option(NRB_TAIL_ORDER_PROBE "Bounded same-list command order probe for the diagnostic build" OFF)
if(NRB_TAIL_ORDER_PROBE)
  target_sources(CyberpunkNRBridge PRIVATE src/nr_tail_probe.cpp)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_TAIL_ORDER_PROBE=1)
endif()
option(NRB_ONE_FRAME_COLOR_READBACK "Diagnostic one-frame pre-SR color readback (never enable in release packages)" OFF)
if(NRB_ONE_FRAME_COLOR_READBACK)
  target_sources(CyberpunkNRBridge PRIVATE src/nr_one_frame_readback.cpp)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_ONE_FRAME_COLOR_READBACK=1)
endif()
set_target_properties(CyberpunkNRBridge PROPERTIES PREFIX "" SUFFIX ".asi")

# Reusable, read-only first pass for a new DLSS game. It records the NGX input
# contract and always forwards the original frame; no game-specific queue or
# render-tail assumptions are enabled in this binary.
option(NRB_BUILD_GENERIC_DLSS_PROBE "Build a passive DLSS input probe for a new game" OFF)
if(NRB_BUILD_GENERIC_DLSS_PROBE)
  add_library(GenericDLSSProbe SHARED src/generic_dlss_probe.cpp)
  target_include_directories(GenericDLSSProbe PRIVATE include
    "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk"
    "${OPTISCALER_SOURCE}/OptiScaler/include")
  target_link_libraries(GenericDLSSProbe PRIVATE nr_core
    "${OPTISCALER_SOURCE}/OptiScaler/library/detours/detours.lib" d3d12 dxgi)
  target_compile_definitions(GenericDLSSProbe PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN NRB_GENERIC_DLSS_PROBE=1)
  set_target_properties(GenericDLSSProbe PROPERTIES PREFIX "" SUFFIX ".asi")
endif()

add_custom_command(TARGET CyberpunkNRBridge POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/game/cyberpunk_nr_adapter.py"
  "$<TARGET_FILE_DIR:CyberpunkNRBridge>/cyberpunk_nr_adapter.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/game/cyberpunk_nr_web.py"
  "$<TARGET_FILE_DIR:CyberpunkNRBridge>/cyberpunk_nr_web.py")

add_executable(nr_offline_tests tests/offline.cpp)
target_link_libraries(nr_offline_tests PRIVATE nr_core)
target_include_directories(nr_offline_tests PRIVATE "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
enable_testing()
add_test(NAME nr_offline_tests COMMAND nr_offline_tests)
add_executable(nr_sync_probe_tests tests/sync_probe.cpp)
target_link_libraries(nr_sync_probe_tests PRIVATE nr_core)
target_compile_definitions(nr_sync_probe_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_sync_probe_tests COMMAND nr_sync_probe_tests)
add_executable(nr_tail_probe_tests tests/tail_probe.cpp src/nr_tail_probe.cpp)
target_include_directories(nr_tail_probe_tests PRIVATE include)
target_compile_definitions(nr_tail_probe_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(nr_tail_probe_tests PRIVATE d3d12)
add_test(NAME nr_tail_probe_tests COMMAND nr_tail_probe_tests)
add_executable(nr_hdr_proxy_shader_tests tests/hdr_proxy_shader.cpp)
target_link_libraries(nr_hdr_proxy_shader_tests PRIVATE d3dcompiler)
add_test(NAME nr_hdr_proxy_shader_tests COMMAND nr_hdr_proxy_shader_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/nr_hdr_proxy.hlsl")

add_library(OptiScalerMock SHARED tests/mock_optiscaler.cpp)
target_include_directories(OptiScalerMock PRIVATE "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
set_target_properties(OptiScalerMock PROPERTIES OUTPUT_NAME "OptiScaler")
if(NRB_BUILD_GENERIC_DLSS_PROBE)
  add_executable(nr_generic_probe_smoke tests/generic_probe_smoke.cpp)
  target_include_directories(nr_generic_probe_smoke PRIVATE
    "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
  add_dependencies(nr_generic_probe_smoke GenericDLSSProbe OptiScalerMock)
  add_test(NAME nr_generic_probe_smoke COMMAND nr_generic_probe_smoke)
endif()
add_executable(nr_hook_smoke tests/hook_smoke.cpp)
target_include_directories(nr_hook_smoke PRIVATE include "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
add_dependencies(nr_hook_smoke CyberpunkNRBridge OptiScalerMock)
add_test(NAME nr_hook_smoke COMMAND nr_hook_smoke)

add_library(StreamlineMock SHARED tests/mock_streamline.cpp)
target_include_directories(StreamlineMock PRIVATE "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
set_target_properties(StreamlineMock PROPERTIES OUTPUT_NAME "sl.interposer")
add_executable(nr_sl_smoke tests/sl_smoke.cpp)
target_include_directories(nr_sl_smoke PRIVATE include "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
add_dependencies(nr_sl_smoke CyberpunkNRBridge OptiScalerMock StreamlineMock)
add_test(NAME nr_sl_smoke COMMAND nr_sl_smoke)
set_tests_properties(nr_hook_smoke nr_sl_smoke PROPERTIES RESOURCE_LOCK bridge_log)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../re8-b580-nr-xess/game/nr_game_controls.py")
    add_test(NAME nr_re8_panel_smoke COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/re8_panel_smoke.py")
    set_tests_properties(nr_re8_panel_smoke PROPERTIES ENVIRONMENT
      "PYTHONDONTWRITEBYTECODE=1")
  endif()
  add_test(NAME nr_adapter_smoke COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/adapter_smoke.py")
  set_tests_properties(nr_adapter_smoke PROPERTIES ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME nr_web_smoke COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/web_smoke.py"
    "${CMAKE_CURRENT_BINARY_DIR}/web-smoke")
  set_tests_properties(nr_web_smoke PROPERTIES ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME nr_pixel_probe_smoke COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/pixel_probe_smoke.py"
    "${CMAKE_CURRENT_BINARY_DIR}")
  set_tests_properties(nr_pixel_probe_smoke PROPERTIES ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1")
endif()

# Main integration: the scoped recorded-state fix has a separate CPU test.
option(NRB_DLSS_COLOR_STATE_PROOF_V1 "Scoped DLSS recorded-color state fix" OFF)
if(NRB_DLSS_COLOR_STATE_PROOF_V1)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_DLSS_COLOR_STATE_PROOF_V1=1)
endif()
add_executable(nr_color_state_proof_tests tests/state_proof_cpu.cpp)
target_link_libraries(nr_color_state_proof_tests PRIVATE nr_core)
target_compile_definitions(nr_color_state_proof_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_color_state_proof_tests COMMAND nr_color_state_proof_tests)

# CPU fake owners exercise the same lease/proof/key implementation as HdrProxy.
add_executable(nr_hdr_resource_pool_cpu_tests tests/hdr_resource_pool_cpu.cpp)
target_link_libraries(nr_hdr_resource_pool_cpu_tests PRIVATE nr_hdr_proxy)
target_include_directories(nr_hdr_resource_pool_cpu_tests PRIVATE src include)
target_compile_definitions(nr_hdr_resource_pool_cpu_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_hdr_resource_pool_cpu_tests COMMAND nr_hdr_resource_pool_cpu_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/nr_hdr_proxy.hlsl")

# Compile only in this CPU sidecar. Never registered with CTest or auto-run.
add_executable(nr_hdr_resource_pool_gpu_probe EXCLUDE_FROM_ALL tests/hdr_resource_pool_gpu.cpp)
target_include_directories(nr_hdr_resource_pool_gpu_probe PRIVATE src)
target_compile_definitions(nr_hdr_resource_pool_gpu_probe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(nr_hdr_resource_pool_gpu_probe PRIVATE nr_hdr_proxy dxgi)

# Link the actual deferred owner, without loading ASI/Python or creating a GPU.
add_executable(nr_hdr_resource_pool_controls_cpu_tests
  tests/hdr_resource_pool_controls_cpu.cpp src/deferred_identity.cpp src/nr_tail_probe.cpp src/nr_gpu_handoff.cpp)
target_include_directories(nr_hdr_resource_pool_controls_cpu_tests PRIVATE src include
  "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
target_compile_definitions(nr_hdr_resource_pool_controls_cpu_tests PRIVATE
  NOMINMAX WIN32_LEAN_AND_MEAN NRB_LIVE_NR_TEST=1 NRB_MULTI_ROUTE_LIVE=1)
target_link_libraries(nr_hdr_resource_pool_controls_cpu_tests PRIVATE nr_hdr_proxy)
add_test(NAME nr_hdr_resource_pool_controls_cpu_tests COMMAND nr_hdr_resource_pool_controls_cpu_tests)

add_executable(nr_record_timing_tests tests/record_timing_cpu.cpp)
target_include_directories(nr_record_timing_tests PRIVATE include "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")
target_compile_definitions(nr_record_timing_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_record_timing_tests COMMAND nr_record_timing_tests)

# Pure CPU protocol and frozen ABI/proof checks; no ASI loading or GPU creation.
add_executable(nr_gpu_handoff_cpu_tests tests/gpu_handoff_cpu.cpp src/nr_gpu_handoff.cpp)
target_include_directories(nr_gpu_handoff_cpu_tests PRIVATE src include)
target_compile_definitions(nr_gpu_handoff_cpu_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_gpu_handoff_cpu_tests COMMAND nr_gpu_handoff_cpu_tests)
