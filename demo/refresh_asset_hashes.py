"""Refresh hashes after intentional reviewed asset changes, not to hide corruption."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((root / 'dist').rglob('*')) if p.is_file()}
if not hashes:
    raise SystemExit('No runtime assets found')
(root / 'assets.sha256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
print(f'Recorded {len(hashes)} runtime hashes. Run validate.py next.')
