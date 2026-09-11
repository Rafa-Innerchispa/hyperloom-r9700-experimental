#!/usr/bin/env python3
import hashlib, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
paths = [
"scripts/r9700_wna16_hybrid_patch.py",
"docs/evidence/r9700_force_stock_restore_20260909T043435Z.json",
"docs/evidence/r9700_force_stock_restore_20260909T043955Z.json",
"docs/evidence/r9700_same_process_ab_20260909T041312Z.json",
"docs/evidence/r9700_same_process_ab_20260909T042036Z.json",
"docs/evidence/r9700_same_process_c4_measure_20260909T043617Z.json",
"docs/evidence/r9700_signal_gate_probe_20260909T044325Z.json",
"docs/evidence/r9700_signal_gate_probe_20260909T044901Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T031745Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T032132Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T032312Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T032639Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T032837Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T034835Z.json",
"docs/evidence/r9700_stock_layout_e2e_single_20260909T035032Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T040419Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T040616Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T045450Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T045642Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T050017Z.json",
"docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T050218Z.json",
"docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json",
"docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json",
"docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json",
"docs/evidence/r9700_wna16_stock_layout_real_weight_smoke_20260909T040134Z.json",
"scripts/r9700_force_stock_restore.py",
"scripts/r9700_same_process_ab.py",
"scripts/r9700_same_process_c4_measure.py",
"scripts/r9700_signal_gate_probe.py",
"scripts/r9700_stock_layout_e2e_aggregate.py",
"scripts/r9700_stock_layout_e2e_single.py",
"scripts/r9700_stock_layout_e2e_v3_pair.py",
"scripts/r9700_tuned_config_live_smoke.py",
"scripts/r9700_vllm_tuned_config_override.py",
"scripts/r9700_vllm_wna16_bounded_tuner.py",
"r9700_sameproc_d07qtfk6.pth",
"scripts/r9700_refresh_audit.py",
"scripts/r9700_current_capability_probe.py"
]
out=[]; total=0
for rel in paths:
    p=ROOT/rel
    if not p.exists():
        out.append({"path":rel,"exists":False}); continue
    if p.is_dir():
        out.append({"path":rel,"exists":True,"is_dir":True}); continue
    b=p.read_bytes(); total += len(b)
    out.append({"path":rel,"exists":True,"is_dir":False,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()})
empty=ROOT/"r9700_sameproc_empty_ensx2s7h"
out.append({"path":"r9700_sameproc_empty_ensx2s7h/","exists":empty.exists(),"is_dir":empty.is_dir() if empty.exists() else False,"entries":sorted(x.name for x in empty.iterdir()) if empty.exists() and empty.is_dir() else []})
print(json.dumps({"schema":"hyperloom.r9700.recovery_inventory.v1","total_file_bytes":total,"files":out},indent=2,sort_keys=True))
