"""Audit direct imports and installed pins without running data/model workflows.

Includes function-local imports, literal __import__ and importlib.import_module.
Only this developer audit script uses variable import targets; those targets are
exactly the modules discovered below. Requirements files contain exact pins.
"""
import argparse
import ast
from collections import defaultdict
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import re
import sys

IMPORT_TO_PACKAGE = {
    'torch': 'torch', 'torchvision': 'torchvision', 'torch_geometric': 'torch-geometric',
    'numpy': 'numpy', 'scipy': 'scipy', 'pydicom': 'pydicom', 'nibabel': 'nibabel',
    'PIL': 'Pillow', 'yaml': 'PyYAML', 'shapely': 'shapely',
    'openslide': 'openslide-python', 'pytest': 'pytest',
}
WSI_IMPORTS = {'openslide'}
DEV_IMPORTS = {'pytest'}


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_pins(path, seen=()):
    path = path.resolve()
    if path in seen:
        raise ValueError(f'Requirements include cycle: {path}')
    result = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('-r '):
            additions = read_pins(path.parent / line[3:].strip(), (*seen, path))
        else:
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
            if not match:
                raise ValueError(f'Expected exact verified pin in {path.name}: {line}')
            additions = {normalize(match[1]): match[2]}
        for name, version in additions.items():
            if name in result and result[name] != version:
                raise ValueError(f'Conflicting pin: {name}')
            result[name] = version
    return result


def scan(root):
    locations = defaultdict(list)
    files = []
    unresolved = []
    for folder in ('src', 'scripts', 'examples', 'tests'):
        for path in sorted((root / folder).rglob('*.py')):
            relative = str(path.relative_to(root))
            files.append(relative)
            # Check the declared Python 3.10 grammar even when auditing on newer Python.
            tree = ast.parse(path.read_text(), filename=relative, feature_version=(3, 10))
            for node in ast.walk(tree):
                names = []
                kind = 'static'
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                elif isinstance(node, ast.Call):
                    fn = node.func
                    dynamic = ((isinstance(fn, ast.Name) and fn.id in ('__import__', 'import_module'))
                               or (isinstance(fn, ast.Attribute) and fn.attr == 'import_module'))
                    if dynamic:
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            names = [node.args[0].value]
                            kind = 'dynamic_literal'
                        elif path.resolve() != Path(__file__).resolve():
                            unresolved.append(f'{relative}:{node.lineno}')
                for module in names:
                    top = module.split('.')[0]
                    if top in sys.stdlib_module_names or top == 'mcminet':
                        continue
                    locations[top].append(dict(module=module, file=relative, line=node.lineno, kind=kind))
    return files, locations, unresolved


def audit(root, *, with_wsi=False, with_dev=False):
    files, locations, unresolved = scan(root)
    groups = {name: read_pins(root / filename) for name, filename in (
        ('core', 'requirements.txt'), ('wsi', 'requirements-wsi.txt'), ('dev', 'requirements-dev.txt'))}
    errors = [f'Unresolved dynamic import: {item}' for item in unresolved]
    rows = []
    for module, occurrences in sorted(locations.items()):
        package = IMPORT_TO_PACKAGE.get(module)
        group = 'wsi' if module in WSI_IMPORTS else 'dev' if module in DEV_IMPORTS else 'core'
        row = dict(module=module, package=package, group=group, occurrences=occurrences)
        rows.append(row)
        if package is None:
            errors.append(f'Unmapped third-party import: {module}')
            continue
        version = groups[group].get(normalize(package))
        row['pin'] = version
        if version is None:
            errors.append(f'Undeclared {group} dependency: {module} -> {package}')
        enabled = group == 'core' or (group == 'wsi' and with_wsi) or (group == 'dev' and with_dev)
        if not enabled:
            row['runtime_status'] = 'optional_not_requested'
            continue
        try:
            row['installed_version'] = metadata.version(package)
            imported = import_module(module)
            row['module_version'] = getattr(imported, '__version__', None)
            row['runtime_status'] = 'imported'
            if row['installed_version'] != version:
                errors.append(f'{package}: installed {row["installed_version"]}, pinned {version}')
            if module == 'openslide':
                row['native_library_version'] = imported.__library_version__
        except Exception as exc:
            row['runtime_status'] = f'{type(exc).__name__}: {exc}'
            errors.append(f'{package}: {row["runtime_status"]}')
    imported_core = {normalize(row['package']) for row in rows if row['group'] == 'core' and row['package']}
    for name in set(groups['core']) - imported_core:
        errors.append(f'Core declaration without a direct import: {name}')
    # Include declared native provider and build tools even though the project does
    # not directly import them. Their presence has a separate installation reason.
    enabled_groups = ['core'] + (['wsi'] if with_wsi else []) + (['dev'] if with_dev else [])
    installed_pins = {}
    for group in enabled_groups:
        for name, version in groups[group].items():
            try:
                actual = metadata.version(name)
                installed_pins[name] = actual
                if actual != version:
                    errors.append(f'{name}: installed {actual}, pinned {version}')
            except metadata.PackageNotFoundError:
                errors.append(f'{name}: distribution not installed')
    return dict(purpose='code-execution environment audit; not paper training provenance',
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                scanned_file_count=len(files), files=files, python_310_syntax='passed',
                imports=rows, installed_pins=installed_pins, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-wsi', action='store_true')
    parser.add_argument('--with-dev', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], with_wsi=args.with_wsi, with_dev=args.with_dev)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for row in result['imports']:
        print(f"{row['module']} -> {row['package']} [{row['group']}]: {row.get('runtime_status', 'unmapped')}")
    print(f"Scanned {result['scanned_file_count']} files; {len(result['imports'])} third-party roots; {len(result['errors'])} errors")
    for error in result['errors']:
        print(error, file=sys.stderr)
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
