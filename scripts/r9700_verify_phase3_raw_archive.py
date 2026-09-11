from __future__ import annotations
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
inv=json.loads((ROOT/'docs/evidence/archive/r9700_phase3_raw_bundle_20260911_inventory.json').read_text())
b64=(ROOT/inv['encoded_path']).read_text().strip()
gz=base64.b64decode(b64,validate=True)
assert hashlib.sha256(gz).hexdigest()==inv['gzip_sha256']
raw=gzip.decompress(gz)
assert len(raw)==inv['raw_bundle_bytes']
assert hashlib.sha256(raw).hexdigest()==inv['raw_bundle_sha256']
bundle=json.loads(raw)
seen={x['name']:(x['bytes'],x['sha256']) for x in bundle['files']}
expected={x['name']:(x['bytes'],x['sha256']) for x in inv['files']}
assert seen==expected
assert not bundle['missing']
print(json.dumps({'pass':True,'raw_bundle_sha256':inv['raw_bundle_sha256'],'gzip_sha256':inv['gzip_sha256'],'files':len(seen)},sort_keys=True))
