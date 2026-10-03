"""Cold ownership of the joint provider and its selected native-QKV route.

Only the registered before_numeric Complete scope may temporarily replace
provider.dense. Replay checks memory identities, never files or device state.
CPU AST/metadata tests are not GPU qualification; actual runs belong to Luna.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
from types import CodeType, MethodType


ENTRY = "audit_vit_c512_720_v1"
SCOPE = "audit_vit_c512_scope_720_v1"


def _cells(function):
    return {name: cell.cell_contents for name, cell in
            zip(function.__code__.co_freevars, function.__closure__ or ())}


def _nested(code, name):
    if code.co_qualname == name:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            found = _nested(value, name)
            if found is not None:
                return found
    return None


class DenseProviderOwner720:
    def __init__(self, child, original):
        self.modes, self.session, self.stack = child.modes, child.session, child.stack
        self.provider, self.graph, self.original = self.stack.provider, child.graph, original
        self.scope = None
        if (not isinstance(original, MethodType) or original.__self__ is not self.provider or
                self.provider.__dict__.get("dense") is not original or self.graph.entries or
                self.graph.replays or self.modes._session_frames):
            raise RuntimeError("Dense provider ownership must bind to the cold actual joint method")
        owners = getattr(self.modes, "implementation_calls_720", {})
        root = owners.get("before_numeric")
        scope = getattr(root, "children", {}).get(ENTRY)
        if scope is None or getattr(getattr(scope, "options", None), "c512_qkv_backend", None) != "native_dpas":
            return
        root_module, module = sys.modules.get("implementation_hooks_720_v1"), sys.modules.get(SCOPE)
        if (root_module is None or module is None or type(root) is not root_module.HookCounter or
                type(scope) is not module.CompleteCounter or root.stage != "before_numeric" or
                scope._in_frame or scope.original_dense is not original or scope.graph is not self.graph or
                scope.modes is not self.modes or scope.session is not self.session or scope.stack is not self.stack):
            raise RuntimeError("Native QKV replacement is not the cold registered Complete owner")
        selected = [h for h in root.hooks if h.module == ENTRY]
        if (len(selected) != 1 or selected[0].stage != "before_numeric" or
                selected[0].kwargs.get("profile") != "complete" or
                selected[0].kwargs.get("c512_qkv_backend") != "native_dpas"):
            raise RuntimeError("Native QKV replacement differs from the selected hook")
        serial = scope._serial_owner
        transfer = root.transfer_serial_thread
        if (serial is None or serial[0] is not root or serial[1] is not transfer.__func__ or
                serial[2] is not transfer.__func__.__code__ or serial[3] is not root_module or
                serial[4] != Path(root_module.__file__).resolve() or
                sha256(serial[4].read_bytes()).hexdigest() != serial[5]):
            raise RuntimeError("Native QKV replacement lost its actual main ownership binding")
        self.module_rows = []
        for name in (ENTRY, SCOPE):
            source = sys.modules.get(name)
            row = scope.sources.get(name)
            if (source is None or scope.source_modules.get(name) is not source or not isinstance(row, dict) or
                    Path(source.__file__).resolve() != Path(row["path"]) or
                    Path(row["path"]).parent != serial[4].parent or
                    sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]):
                raise RuntimeError("Native QKV replacement source differs from its cold receipt")
            self.module_rows.append((source, source.__file__, dict(row)))
        if root.sources.get(ENTRY) != scope.sources[ENTRY]:
            raise RuntimeError("Native QKV entry differs between main and Complete source receipts")
        installed = module.installed.__wrapped__
        self.wrapper = scope.wrapper
        frame = getattr(self.wrapper, "__wrapped__", None)
        code = _nested(installed.__code__, "installed.<locals>.wrapped_scope")
        self.route_code = _nested(installed.__code__, "installed.<locals>.wrapped_scope.<locals>.c512_dense")
        if (installed.__globals__ is not vars(module) or
                Path(installed.__code__.co_filename).resolve() != Path(scope.sources[SCOPE]["path"]) or
                frame is None or frame.__code__ is not code or frame.__globals__ is not vars(module) or
                self.route_code is None or _cells(frame).get("counter") is not scope or
                _cells(frame).get("options") is not scope.options):
            raise RuntimeError("Native QKV frame/route is outside its actual source and owner")
        self.root, self.scope, self.module, self.root_module = root, scope, module, root_module
        self.serial, self.options, self.option_values = serial, scope.options, dict(vars(scope.options))
        self.frame, self.frame_code = frame, code
        self.previous = self.wrapper.__nr_numeric_original_installed__
        self.installed, self.installed_code = installed, installed.__code__
        self.contains, self.contains_code = module.scope_contains, module.scope_contains.__code__
        self.interfaces = {name: (getattr(scope, name).__func__, getattr(scope, name).__func__.__code__)
                           for name in ("validate", "validate_frame_context", "_guard")}
        self.targets = scope.qkv_targets
        self.target_rows = []
        for family in ("encoder512", "decoder512"):
            blocks = getattr(self.stack.model, family)
            if len(blocks) != 8:
                raise RuntimeError("Native QKV replacement requires the actual sixteen block owners")
            for i, block in enumerate(blocks):
                self.target_rows.append((block.attention.qkv, f"qkv.{family}.{i}", tuple(block.window_shift)))
        self.current()

    def current(self):
        actual = self.provider.__dict__.get("dense")
        if self.scope is None:
            if actual is not self.original:
                raise RuntimeError("Unregistered dense provider replacement")
            return True
        root, scope, module = self.root, self.scope, self.module
        if (sys.modules.get(module.__name__) is not module or
                sys.modules.get(self.root_module.__name__) is not self.root_module or
                type(root) is not self.root_module.HookCounter or type(scope) is not module.CompleteCounter or
                self.modes.implementation_calls_720.get("before_numeric") is not root or
                root.children.get(ENTRY) is not scope or not root.active or root.retired or
                root.modes is not self.modes or root.session is not self.session or root.graph is not self.graph or
                root.hooks != self.modes.implementation_hooks_720 or root.stage != "before_numeric" or
                not scope.live or scope.retired or scope.modes is not self.modes or
                scope.session is not self.session or scope.stack is not self.stack or scope.graph is not self.graph or
                self.stack.provider is not self.provider or scope.original_dense is not self.original or
                self.session.__dict__.get("_audit_vit_c512_720") is not scope or scope._serial_owner is not self.serial or
                root.transfer_serial_thread.__func__ is not self.serial[1] or self.serial[1].__code__ is not self.serial[2] or
                scope.options is not self.options or scope._fixed_options is not self.options or
                vars(self.options) != self.option_values or scope.wrapper is not self.wrapper or
                self.wrapper.__wrapped__ is not self.frame or self.frame.__code__ is not self.frame_code or
                self.frame.__globals__ is not vars(module) or
                _cells(self.frame).get("counter") is not scope or _cells(self.frame).get("options") is not self.options or
                _cells(self.frame).get("previous_scope") is not self.previous or
                self.wrapper.__nr_numeric_original_installed__ is not self.previous or
                module.installed.__wrapped__ is not self.installed or self.installed.__code__ is not self.installed_code or
                module.scope_contains is not self.contains or self.contains.__code__ is not self.contains_code or
                not self.contains(self.session._installed, self.wrapper)):
            raise RuntimeError("Native QKV provider lifecycle/registration changed")
        for source, file, row in self.module_rows:
            if (sys.modules.get(source.__name__) is not source or source.__file__ != file or
                    scope.source_modules.get(source.__name__) is not source or scope.sources.get(source.__name__) != row):
                raise RuntimeError("Native QKV frozen source owner changed")
        if root.sources.get(ENTRY) != scope.sources[ENTRY]:
            raise RuntimeError("Native QKV main source receipt changed")
        for name, (function, code) in self.interfaces.items():
            action = getattr(scope, name)
            if action.__self__ is not scope or action.__func__ is not function or function.__code__ is not code:
                raise RuntimeError("Native QKV owner validation interface changed")
        if (scope.qkv_targets is not self.targets or len(self.targets) != 16 or
                len({id(w) for w, _, _ in self.target_rows}) != 16 or any(
                    not isinstance(self.targets.get(id(w)), tuple) or len(self.targets[id(w)]) != 3 or
                    self.targets[id(w)][0] is not w or self.targets[id(w)][1:] != (site, shift)
                    for w, site, shift in self.target_rows)):
            raise RuntimeError("Native QKV actual weight/site/shift ownership changed")
        if not scope._in_frame:
            if actual is not self.original or "c512_qkv_dense" in scope._frame_routes:
                raise RuntimeError("Native QKV scope did not restore the original joint provider")
            return True
        route = scope._frame_routes.get("c512_qkv_dense")
        if (not isinstance(route, tuple) or len(route) != 3 or route[0] is not self.provider or
                route[1] != "dense" or route[2] is not actual or not isinstance(actual, MethodType) or
                actual.__self__ is not self.provider or actual.__func__.__code__ is not self.route_code or
                actual.__func__.__globals__ is not vars(module)):
            raise RuntimeError("Native QKV current dense method differs from its actual frame route")
        cells = _cells(actual.__func__)
        if (cells.get("counter") is not scope or cells.get("old_dense") is not self.original or
                cells.get("options") is not self.options):
            raise RuntimeError("Native QKV route changed its actual owner/options/joint delegate")
        return True
