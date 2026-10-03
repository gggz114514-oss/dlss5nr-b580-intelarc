"""CPU-only actual controller/marshalling/panel checks. No DLL/server/GPU load."""
import ctypes as C
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTROL_GAME = ROOT / "game"
if not (CONTROL_GAME / "nr_gpu_handoff_host_v1.py").is_file():
    CONTROL_GAME = ROOT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game"
if not all((CONTROL_GAME / name).is_file() for name in
           ("nr_gpu_handoff_host_v1.py", "nr_game_controls.py")):
    raise RuntimeError("CPU checks require the local bundle or the shared RE8 bridge sources")
sys.path.insert(0, str(CONTROL_GAME))
sys.path.insert(0, str(ROOT / "game"))
import cyberpunk_nr_web as web
from nr_gpu_handoff_host_v1 import HandoffController, source_handoff
# The standalone artifact intentionally bundles no installed model/profile tree.
# Only the panel's independent request transaction is exercised here.
profiles = ModuleType("numeric_game_profiles_720_v1")
profiles.EXPERIMENTS_720 = ("baseline",)
profiles.EXPERIMENT_LABELS = {"baseline": "baseline"}
profiles.FUSED_REPLAY_PROFILES = profiles.HIT_FIELDS = ()
profiles.PROFILES = {}
profiles.combined_mode_options = lambda *_: {}
profiles.conflicting_profiles = lambda *_: ()
sys.modules[profiles.__name__] = profiles
from nr_game_controls import ControlPanel

CHECKS = 0


def check(ok, label):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(label)


def raises(call, label):
    try:
        call()
    except (RuntimeError, ValueError):
        check(True, label)
    else:
        check(False, label)


class Native:
    def __init__(self):
        self.stats = web.GpuHandoffStats(abi_size=C.sizeof(web.GpuHandoffStats), version=1)
        self.point = web.ProducerPoint(abi_size=C.sizeof(web.ProducerPoint), version=1,
            device=0x1000, queue=0x2000, color=0x3000, motion=0x4000, fence=0x5000,
            value=17, frame_id=23, registry_epoch=1, prepared_cpu_waited=1,
            processor_context=0x6000, processor_process=0x7000, callback_cookie=1,
            owner_thread=threading.get_ident())
        self.last_cap = None
        self.scope = True

    def NRB_GetGpuHandoffStats(self, output):
        C.memmove(output, C.byref(self.stats), C.sizeof(self.stats)); return 1

    def NRB_SetGpuHandoffRequested(self, requested):
        self.stats.requested = requested
        self.stats.request_epoch += 1
        self.point.request_epoch = self.stats.request_epoch
        self.stats.armed = self.stats.cap_healthy = 0
        return 1

    def NRB_GetCurrentProducerPoint(self, output):
        if not self.scope or threading.get_ident() != self.point.owner_thread:
            return 0
        C.memmove(output, C.byref(self.point), C.sizeof(self.point)); return 1

    def NRB_ClearGpuHandoffCapability(self):
        self.stats.armed = self.stats.cap_healthy = 0; return 1

    def NRB_ArmGpuHandoffCapability(self, capability):
        cap = C.cast(capability, C.POINTER(web.HandoffCapability)).contents
        self.last_cap = web.HandoffCapability.from_buffer_copy(cap)
        self.stats.armed = self.stats.cap_healthy = int(bool(self.stats.requested))
        return self.stats.armed


class Bridge:
    def __init__(self):
        self.gpu_handoff = False
        self.thread = threading.get_ident()
        self.handle = 0x8000
        self.stream = SimpleNamespace(sycl_queue=0x9000)
        self.configure_calls = []
        self.info_reads = 0
        self.idle_values = []
        self.refs = [object()]
        self.bad_info = {}

    def _ready(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError("Wrong owning thread")

    def poll_idle(self):
        self._ready()
        idle = self.idle_values.pop(0) if self.idle_values else True
        if idle:
            self.refs.clear()
        return idle

    def configure_gpu_handoff(self, enabled, *, producer_fence_required):
        self._ready()
        self.configure_calls.append((enabled, producer_fence_required))
        self.gpu_handoff = enabled

    def gpu_handoff_info(self):
        self._ready(); self.info_reads += 1
        values = {name: 0 for name, _ in web.NativeHandoffInfo._fields_}
        values.update(abi_size=128, version=1, enabled=1, healthy=1, reuse_safe_idle=1,
            queue_context_equal=1, in_order=1, native_luid_matched=1, max_active_frames=1,
            private_records=2, forward_import_per_wait=1, producer_cpu_wait_bypass_supported=1,
            producer_cpu_wait_bypass_configured=1, producer_fence_required=1,
            borrowed_sycl_queue=self.stream.sycl_queue, native_context=0xa000,
            native_device=0xb000, d3d12_device=0x1000, d3d12_queue=0x2000)
        values.update(self.bad_info)
        return values


def source(controller, bridge):
    return controller.source(bridge, web, 0x1000, 0x2000, 0x3000, 0x4000, 23)


native = Native(); web._native = native
web._native = None
check(source_handoff(SimpleNamespace(gpu_handoff=False), 1, 2, 3, 4, 5) == {},
      "shared legacy/web-disabled host never loads an ASI")
raises(lambda: source_handoff(SimpleNamespace(gpu_handoff=True), 1, 2, 3, 4, 5),
       "active helper requires owning ASI binding")
web._native = native
check(C.sizeof(web.Controls) == 52 and C.sizeof(web.ProducerPoint) == 120 and
      C.sizeof(web.NativeHandoffInfo) == 128 and C.sizeof(web.HandoffCapability) == 264 and
      C.sizeof(web.GpuHandoffStats) == 192, "Win64 ctypes ABI sizes")
bridge = Bridge(); controller = HandoffController()
check(not web.get_gpu_handoff_state()["requested"], "web request default OFF")
check(source(controller, bridge) == {"producer_fence": 0x5000, "producer_value": 17} and
      not bridge.configure_calls and not bridge.info_reads, "OFF keeps old helper and actual source point")
web.set_gpu_handoff_requested(True)
check(not bridge.configure_calls, "web request performs no native/GPU transition")
check(source(controller, bridge)["producer_value"] == 17 and bridge.configure_calls == [(True, True)] and
      web.get_gpu_handoff_state()["armed"], "first CPU waited callback configures required producer point and arms")
check(native.last_cap.point.fence == 0x5000 and native.last_cap.point.value == 17 and
      native.last_cap.native_info.native_context == 0xa000 and native.last_cap.native_helper == bridge.handle,
      "actual capability marshalling")
native.point.prepared_cpu_waited = 0
source(controller, bridge)
check(bridge.info_reads == 1 and len(bridge.configure_calls) == 1,
      "steady frame avoids constant health/context checks")
web.get_gpu_handoff_state(); check(bridge.info_reads == 1, "web snapshot never queries native helper")
web.set_gpu_handoff_requested(False); web.set_gpu_handoff_requested(True)
source(controller, bridge)
check(not web.get_gpu_handoff_state()["armed"] and bridge.configure_calls == [(True, True)],
      "OFF/ON after committed bypass retains current frame and defers arm")
native.point.prepared_cpu_waited = 1
source(controller, bridge)
check(web.get_gpu_handoff_state()["armed"] and bridge.info_reads == 2,
      "next CPU waited frame re-arms after request race")
native.point.prepared_cpu_waited = 0
raises(lambda: web.current_producer_point(0x1000, 0x2008, 0x3000, 0x4000, 23), "queue source mismatch rejected")
raises(lambda: web.current_producer_point(0x1000, 0x2000, 0x3008, 0x4000, 23), "color source mismatch rejected")
native.scope = False
raises(lambda: source(controller, bridge), "missing scope cannot inherit an armed capability")
native.scope = True
wrong_thread = []
def worker():
    try:
        source(controller, bridge)
    except RuntimeError:
        wrong_thread.append(True)
thread = threading.Thread(target=worker); thread.start(); thread.join()
check(wrong_thread == [True] and len(bridge.configure_calls) == 2, "cross-thread native transition rejected")
bridge.refs = [object()]; bridge.idle_values = [False] * 100
raises(lambda: controller.ensure_idle(bridge, timeout=0), "busy previous frame times out")
check(len(bridge.refs) == 1 and len(bridge.configure_calls) == 2, "busy previous frame retains owners/config")
bridge.idle_values = [False, False, True]
controller.ensure_idle(bridge)
check(not bridge.refs and controller.snapshot()["native_idle_waits"] == 2,
      "actual idle releases prior refs with separate bounded CPU reuse counter")
web.set_gpu_handoff_requested(False)
check(bridge.gpu_handoff and not web.get_gpu_handoff_state()["armed"], "web OFF clears CPU capability without in-flight native mutation")
native.point.prepared_cpu_waited = 1
source(controller, bridge)
check(bridge.configure_calls[-1] == (False, False) and not bridge.gpu_handoff,
      "owner applies OFF only after actual idle")
for invalid in (1, "true", None):
    raises(lambda: web.set_gpu_handoff_requested(invalid), "strict boolean web request")
web.set_gpu_handoff_requested(True)
legacy = SimpleNamespace(gpu_handoff=False, _ready=lambda: None)
check(source(HandoffController(), legacy)["producer_fence"] == 0x5000 and
      not web.get_gpu_handoff_state()["armed"], "missing moving seam leaves unqualified CPU path")
for changed in ({"in_order": 0}, {"native_luid_matched": 0}, {"queue_context_equal": 0},
                {"d3d12_queue": 0x2008}, {"producer_fence_required": 0}, {"active": 1}, {"healthy": 0}):
    bad = Bridge(); bad.bad_info = changed
    raises(lambda: source(HandoffController(), bad), "native proof failure rejected")
    check(not web.get_gpu_handoff_state()["armed"], "invalid proof stays unarmed")
controller = HandoffController(); bridge = Bridge(); source(controller, bridge)
controller.invalidate(web)
check(not web.get_gpu_handoff_state()["armed"], "reinitialization disarms")
native.point.prepared_cpu_waited = 0
raises(lambda: source(controller, Bridge()), "replaced helper after prepared bypass rejected")
native.point.prepared_cpu_waited = 1
source(controller, Bridge())
check(web.get_gpu_handoff_state()["armed"], "new helper requalifies after CPU wait")
panel = ControlPanel(lambda *_: True,
    gpu_handoff_read=web.get_gpu_handoff_state, gpu_handoff_apply=web.set_gpu_handoff_requested,
    bridge_pool_read=lambda: {"enabled": False}, bridge_pool_apply=lambda *_: (_ for _ in ()).throw(AssertionError("pool touched")))
check(panel.update_gpu_handoff({"requested": False})["requested"] is False,
      "independent panel request does not touch pool/controls")
for values in ({"requested": 1}, {"enabled": True}, {"requested": True, "extra": 0}):
    raises(lambda: panel.update_gpu_handoff(values), "strict panel endpoint")
page = (CONTROL_GAME / "nr_game_controls.py").read_text(encoding="utf-8")
check('id="gpuHandoffRequested" type="checkbox">' in page and
      'id="bridgePoolEnabled" type="checkbox">' in page,
      "both independent checkboxes default unchecked")
for path in (ROOT / "game").glob("*.py"):
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
check(True, "all staged Python sources compile without import/GPU")
print(f"PASS: {CHECKS} CPU controller/ctypes/panel checks; no DLL/server/GPU/live API")
