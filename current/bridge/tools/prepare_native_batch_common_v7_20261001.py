"""Compose V7 cold switches from the reviewed V6 sources; no runtime/GPU I/O."""
from pathlib import Path
import hashlib
import json

PROJECT = Path("E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt")
SHARED = Path("E:/ComfyUI-aki-v3-IntelArc_20260722/.codex-worktrees/re8-fp8-unround-fast-20260928/game")
STAGE = PROJECT / "artifacts/native-batch-v7-20261001/common"
SOURCES = {
    "plugins/cyberpunk_nr_adapter.py": (PROJECT / "game/cyberpunk_nr_adapter.py",
        "3a40d6189f4cab2b1c232871135cdd07b3b56d76afefa1eb1d07b67192b292a2"),
    "game/nr_game_controls.py": (SHARED / "nr_game_controls.py",
        "1d45081a72f2015ad4f1877a63b89994e78fc1130e2e84b1ade594d1fb329079"),
    "game/numeric_cleanup_suite_720_v1.py": (SHARED / "numeric_cleanup_suite_720_v1.py",
        "4dfa514ed6172adc8c9756725bca9923163d9956f836215713f86c14f32cfb67"),
}


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError("Unexpected source span: " + old[:100])
    return text.replace(old, new)


def adapter(text):
    text = replace(text, "_lifecycle_audit_epoch = 0\n",
        "_lifecycle_audit_epoch = 0\n"
        "_native_batch_requested = (False, False, False, False)\n"
        "_native_batch_applied = (False, False, False, False)\n"
        "_native_batch_epoch = 0\n")
    text = replace(text, "def _apply_lifecycle_audit():\n", '''def native_batch_state():
    from native_batch_policy_720_v7 import FAMILIES, LABELS, flags_dict
    with _settings_lock:
        owner = getattr(getattr(host, "_modes", None), "numeric_cleanup_calls", None)
        actual = getattr(owner, "native_batch_flags", None)
        return {"requested": flags_dict(_native_batch_requested),
                "applied": flags_dict(_native_batch_applied),
                "pending": _native_batch_requested != _native_batch_applied,
                "epoch": _native_batch_epoch, "available": list(FAMILIES),
                "labels": dict(LABELS),
                "actual": flags_dict(actual) if actual is not None else None}


def set_native_batch(values):
    from native_batch_policy_720_v7 import normalize
    new = normalize(values)
    global _native_batch_requested
    # The HTTP thread submits only metadata, never waits on the process lock.
    with _settings_lock:
        _native_batch_requested = new
        return native_batch_state()


def _apply_lifecycle_audit():
    """Compatibility entry point; apply both cold requests in one retirement."""
    return _apply_execution_policies()


def _apply_execution_policies():
''')
    text = replace(text,
        "    global _lifecycle_audit_applied, _lifecycle_audit_epoch\n",
        "    global _lifecycle_audit_applied, _lifecycle_audit_epoch\n"
        "    global _native_batch_applied, _native_batch_epoch\n")
    text = replace(text,
        "        requested = _lifecycle_audit_requested\n"
        "    if requested == _lifecycle_audit_applied:\n",
        "        requested = _lifecycle_audit_requested\n"
        "        batch_requested = _native_batch_requested\n"
        "    lifecycle_changed = requested != _lifecycle_audit_applied\n"
        "    batch_changed = batch_requested != _native_batch_applied\n"
        "    if not lifecycle_changed and not batch_changed:\n")
    text = replace(text,
        "        lifecycle.TRIAL_ENABLED = requested\n",
        "        from native_batch_policy_720_v7 import configure, flags_dict\n"
        "        configure(flags_dict(batch_requested))\n"
        "        lifecycle.TRIAL_ENABLED = requested\n")
    text = replace(text,
        "            _lifecycle_audit_applied = requested\n"
        "            _lifecycle_audit_epoch += 1\n",
        "            _lifecycle_audit_applied = requested\n"
        "            _native_batch_applied = batch_requested\n"
        "            _lifecycle_audit_epoch += int(lifecycle_changed)\n"
        "            _native_batch_epoch += int(batch_changed)\n")
    text = replace(text,
        "                lifecycle_pending = _lifecycle_audit_requested != _lifecycle_audit_applied\n",
        "                lifecycle_pending = (_lifecycle_audit_requested != _lifecycle_audit_applied or\n"
        "                                     _native_batch_requested != _native_batch_applied)\n")
    text = replace(text,
        "            panel = CyberpunkPanel(apply_native,\n",
        "            native_batch_options = ({\"native_batch_read\": native_batch_state,\n"
        "                                     \"native_batch_apply\": set_native_batch}\n"
        "                                    if \"native_batch_read\" in inspect.signature(ControlPanel).parameters\n"
        "                                    else {})\n\n"
        "            panel = CyberpunkPanel(apply_native,\n")
    text = replace(text,
        "                **timing_options, **validation_options, **lifecycle_options)",
        "                **timing_options, **validation_options, **lifecycle_options,\n"
        "                **native_batch_options)")
    return text


def suite(text):
    return replace(text, "        self.identity = options.identity\n", '''        from native_batch_policy_720_v7 import flags_identity
        self.native_batch_flags = flags_identity()
        # OFF preserves the original V6 signature. ON is frozen for this suite.
        self.identity = (options.identity + (("native_batch_v7", self.native_batch_flags),)
                         if any(self.native_batch_flags) else options.identity)
''')


def controls(text):
    text = replace(text, "status[\"lifecycle_audit\"] = self.lifecycle_audit_state()\n",
        "status[\"lifecycle_audit\"] = self.lifecycle_audit_state()\n"
        "        status[\"native_batch_v7\"] = self.native_batch_state()\n")
    text = replace(text, "    def update_timing(self, values):\n", '''    def native_batch_state(self):
        return self._native_batch_read() if self._native_batch_read else None

    def update_native_batch(self, values):
        if self._native_batch_apply is None:
            raise ValueError("Native batch switch is unavailable on this route")
        if not isinstance(values, dict) or set(values) != {"flags"}:
            raise ValueError("Expected the native batch flags")
        self._native_batch_apply(values["flags"])
        return self.native_batch_state()

    def update_timing(self, values):
''')
    text = replace(text,
        '                elif self.path == "/api/lifecycle-audit":\n',
        '                elif self.path == "/api/native-batch":\n'
        '                    self._reply(200, json.dumps(panel.native_batch_state(), ensure_ascii=False))\n'
        '                elif self.path == "/api/lifecycle-audit":\n')
    text = replace(text,
        '"/api/validation", "/api/lifecycle-audit") or',
        '"/api/validation", "/api/lifecycle-audit", "/api/native-batch") or')
    text = replace(text,
        '                              else panel.update_lifecycle(values) if self.path == "/api/lifecycle-audit"\n',
        '                              else panel.update_lifecycle(values) if self.path == "/api/lifecycle-audit"\n'
        '                              else panel.update_native_batch(values) if self.path == "/api/native-batch"\n')
    text = replace(text,
        '<p id="lifecycleStatus" class="hint" hidden></p>\n',
        '<p id="lifecycleStatus" class="hint" hidden></p>\n'
        '<fieldset id="nativeBatchV7" hidden><legend>新一批实现优化（独立对照）</legend>\n'
        '<label><input type="checkbox" id="nativeBatch_matrix"> 矩阵与解码</label>\n'
        '<label><input type="checkbox" id="nativeBatch_vit"> ViT 注意力与投影</label>\n'
        '<label><input type="checkbox" id="nativeBatch_history"> 历史采样</label>\n'
        '<label><input type="checkbox" id="nativeBatch_post_front"> 输入噪声与输出合成</label>\n'
        '<p class="hint">仅优化下方已勾选选项的实现。取消勾选使用本轮之前的实现；切换会重建图并重置一次历史。</p>\n'
        '<p id="nativeBatchStatus" class="hint"></p></fieldset>\n')
    text = replace(text, 'let lifecycleBusy=false;\n',
        'let lifecycleBusy=false;\nlet nativeBatchBusy=false;\n')
    text = replace(text, '  updateLifecycleView(data);\n',
        '  updateLifecycleView(data);\n  updateNativeBatchView(data);\n', count=2)
    text = replace(text, "$('lifecycleEnabled').onchange=applyLifecycle;\n", '''$('lifecycleEnabled').onchange=applyLifecycle;
const nativeBatchFamilies=['matrix','vit','history','post_front'];
function updateNativeBatchView(data){
  const s=data.native_batch_v7;
  $('nativeBatchV7').hidden=s==null;
  if(s==null)return;
  for(const key of nativeBatchFamilies){
    const box=$('nativeBatch_'+key);
    if(!nativeBatchBusy)box.checked=Boolean(s.requested[key]);
    box.disabled=nativeBatchBusy||!(s.available??[]).includes(key)||data.settings.input_size!==720;
  }
  const names=nativeBatchFamilies.filter(key=>s.applied[key]).map(key=>s.labels[key]);
  $('nativeBatchStatus').textContent=s.pending?'等待下一帧重建图':
    names.length?'已应用：'+names.join('、'):'全部关闭，使用本轮之前的实现';
}
async function applyNativeBatch(){
  if(nativeBatchBusy)return;
  nativeBatchBusy=true;
  const flags=Object.fromEntries(nativeBatchFamilies.map(key=>[key,$('nativeBatch_'+key).checked]));
  for(const key of nativeBatchFamilies)$('nativeBatch_'+key).disabled=true;
  try{
    const response=await fetch('/api/native-batch',{method:'POST',
      headers:{'Content-Type':'application/json','X-NR-Token':token},
      body:JSON.stringify({flags})}), result=await response.json();
    if(!response.ok)throw Error(result.error);
    show('新一批实现设置已提交，下一帧重建图后生效');
  }catch(e){show('新一批实现未应用：'+e.message)}
  finally{nativeBatchBusy=false;await active()}
}
for(const key of nativeBatchFamilies)$('nativeBatch_'+key).onchange=applyNativeBatch;
''')
    # Constructor spans are patched separately after inspecting the exact signature.
    return text


def physical(path):
    if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Redirected path: " + str(path))


def main():
    rows = []
    for label, (source, pin) in SOURCES.items():
        physical(source)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise ValueError("Source changed: " + str(source))
        text = data.decode("utf-8").replace("\r\n", "\n")
        name = Path(label).name
        fn = (adapter if name == "cyberpunk_nr_adapter.py" else
              suite if name == "numeric_cleanup_suite_720_v1.py" else controls)
        text = fn(text)
        if name == "nr_game_controls.py":
            text = constructor(text)
        output = text.encode("utf-8")
        compile(output, label, "exec", dont_inherit=True)
        target = STAGE / "payload" / label
        physical(target)
        if target.exists() and target.read_bytes() != output:
            raise ValueError("Preserving staged edits: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(output)
        rows.append({"label": label, "source": str(source), "before_sha256": pin,
                     "target": str(target), "after_sha256": hashlib.sha256(output).hexdigest()})
    (STAGE / "common-source-receipt.json").write_text(json.dumps(
        {"completed": True, "writes_runtime": False, "gpu_used": False,
         "changes": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "changed_files": len(rows)}))


def constructor(text):
    raise NotImplementedError("Read the exact constructor span before composing")


if __name__ == "__main__":
    main()
