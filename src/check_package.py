"""Static syntax/import inventory and complete file hashes; no empirical claims."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = {}
imports = set()
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.name == 'package_inventory.json':
        continue
    if path.suffix == '.py':
        source = path.read_text(encoding='utf-8-sig')
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.Import):
                imports.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
    files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(files=files, import_inventory=sorted(imports), syntax_passed=True,
              uploaded=False, empirical_results_reproduced=False)
(ROOT / 'package_inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(files=len(files), syntax_passed=True, empirical_results_reproduced=False)))
