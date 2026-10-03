"""Static loaded-module inventory; dynamic Torch namespaces are not disk files."""
from pathlib import Path
from types import ModuleType


def static_file(module):
    namespace = vars(module)
    if '__file__' in namespace:
        return namespace['__file__']
    for cls in type(module).__mro__:
        if '__file__' in vars(cls):
            return vars(cls)['__file__']
    return None


def torch_virtual(name, module, file, modules, runtime, record, require):
    """Accept only the existing class owned by its real loaded Torch source."""
    cls = type(module)
    rules = {
        '_Ops': ('torch._ops', 'torch.ops', '_ops.py'),
        '_Classes': ('torch._classes', 'torch.classes', '_classes.py'),
        '_OpNamespace': ('torch._ops', 'torch.ops.', 'torch.ops'),
        '_HigherOrderNamespace': ('torch._ops', 'torch.ops.higher_order', 'torch.ops'),
    }
    rule = rules.get(cls.__name__)
    if rule is None or cls.__module__ != rule[0] or file != rule[2]:
        return None
    expected = rule[1]
    if not (name.startswith(expected) if expected.endswith('.') else name == expected):
        return None
    owner = modules.get(rule[0])
    if not isinstance(owner, ModuleType) or vars(owner).get(cls.__name__) is not cls:
        return None
    require(vars(module).get('__name__') == name, 'Foreign Torch virtual namespace identity')
    owner_file = vars(owner).get('__file__')
    require(isinstance(owner_file, str) and Path(owner_file).is_absolute(),
            'Torch namespace has no real source owner')
    owner_path = Path(owner_file).resolve(strict=True)
    require(owner_path.is_relative_to(runtime / 'python') and
            owner_path.name == rule[0].split('.')[-1] + '.py',
            'Foreign Torch namespace source owner')
    return {'virtual_filename': file, 'class_owner': rule[0] + '.' + cls.__name__,
            'physical_owner': record(owner_path), 'kind': 'torch_dynamic_namespace_not_disk_module'}


def collect(modules, runtime, here, allowed, front_modules, record, require):
    runtime, here = runtime.resolve(), here.resolve()
    result, virtual = {}, {}
    for name, module in list(modules.items()):
        if module is None:
            continue
        require(name not in front_modules, 'Offline operator runner imported a game/API/bridge frontend')
        if not isinstance(module, ModuleType):
            continue
        file = static_file(module)
        if not file:
            continue
        require(isinstance(file, str), 'Loaded module filename is not static text')
        pseudo = torch_virtual(name, module, file, modules, runtime, record, require)
        if pseudo is not None:
            virtual[name] = pseudo
            continue
        physical = Path(file)
        path = (physical if physical.is_absolute() else here / physical).resolve()
        if path.is_relative_to(runtime) or path.is_relative_to(here):
            require(path.stem not in front_modules,
                    'Offline operator runner imported a game/API/bridge frontend')
            row = record(path)  # Missing real protected modules still fail.
            if not path.is_relative_to(runtime / 'python') and not path.is_relative_to(here):
                require(str(path) in allowed and row['sha256'] == allowed[str(path)],
                        'Unpinned runtime module: ' + str(path))
            if path.is_relative_to(here / 'source'):
                require(str(path) in allowed and row['sha256'] == allowed[str(path)],
                        'Unpinned effective model source')
            result[name] = row
    return result, virtual
