# Recovered uncommitted manifest — 2026-09-10

Source worktree on physical AMD node:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Checkpoint used for comparison:
`2a08585ca340936b9eea1edaccbcc44910100ae4`

This manifest records the exact modified/untracked state reported by Project Runtime bootstrap on 2026-09-10. It is preservation evidence. The files listed here must be classified and archived/versioned before cleanup.

## Modified tracked file

- `scripts/r9700_wna16_hybrid_patch.py`

## Recovered untracked evidence

- `docs/evidence/r9700_force_stock_restore_20260909T043435Z.json`
- `docs/evidence/r9700_force_stock_restore_20260909T043955Z.json`
- `docs/evidence/r9700_same_process_ab_20260909T041312Z.json`
- `docs/evidence/r9700_same_process_ab_20260909T042036Z.json`
- `docs/evidence/r9700_same_process_c4_measure_20260909T043617Z.json`
- `docs/evidence/r9700_signal_gate_probe_20260909T044325Z.json`
- `docs/evidence/r9700_signal_gate_probe_20260909T044901Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T031745Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T032132Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T032312Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T032639Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T032837Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T034835Z.json`
- `docs/evidence/r9700_stock_layout_e2e_single_20260909T035032Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T040419Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T040616Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T045450Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T045642Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T050017Z.json`
- `docs/evidence/r9700_stock_layout_e2e_v3_single_20260909T050218Z.json`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json`
- `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`
- `docs/evidence/r9700_wna16_stock_layout_real_weight_smoke_20260909T040134Z.json`

## Recovered untracked scripts

- `scripts/r9700_force_stock_restore.py`
- `scripts/r9700_same_process_ab.py`
- `scripts/r9700_same_process_c4_measure.py`
- `scripts/r9700_signal_gate_probe.py`
- `scripts/r9700_stock_layout_e2e_aggregate.py`
- `scripts/r9700_stock_layout_e2e_single.py`
- `scripts/r9700_stock_layout_e2e_v3_pair.py`
- `scripts/r9700_tuned_config_live_smoke.py`
- `scripts/r9700_vllm_tuned_config_override.py`
- `scripts/r9700_vllm_wna16_bounded_tuner.py`

## Temporary runtime artifacts recovered

- `r9700_sameproc_d07qtfk6.pth`
- `r9700_sameproc_empty_ensx2s7h/`

These temporary artifacts belong to the failed same-process/runtime-gate experiment. They must not be installed or treated as required runtime state. Preserve their existence and relevant contents/evidence, then quarantine them.

## Additional audit helper created during recovery

- `scripts/r9700_refresh_audit.py`

This helper was created only to extract the recovered evidence and inspect non-secret active-container state on the physical AMD node. Whether it is retained as a reproducibility tool or quarantined must be decided explicitly, not silently.

## Recovered tuner findings

`r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json` contains a bounded candidate with:

- `BLOCK_SIZE_M=16`
- `BLOCK_SIZE_N=64`
- `BLOCK_SIZE_K=32`
- `GROUP_SIZE_M=1`
- `num_warps=4`
- `num_stages=2`
- `waves_per_eu=4`
- `SPLIT_K=1`

Captured median isolated speedup across M=1,2,4,8,16: approximately `1.209256x`; minimum approximately `1.083843x`.

`r9700_tuned_config_live_smoke_20260909T052123Z.json`:

- overall `pass=false`
- `tuned_seen=false`
- candidate and stock output hash matched within the applicable comparison
- stock service restored healthy
- demonstrates the candidate override was not actually active, so no performance claim is allowed from that run.

`r9700_tuned_config_live_smoke_20260909T052523Z.json`:

- overall `pass=false`
- `tuned_seen=true`
- tuned config observed at M=1,2,4,8,16
- `only_small_m=true`
- correctness/hash gate failed (`hash_match_before_candidate=false`, `restore_hash_mismatch=true`)
- stock service restored healthy
- therefore the tuned config remains a lead, not a promoted result.

## Preservation status

At the time this manifest was written, the manifest and analysis are versioned in the refresh branch. The raw AMD-node recovered files still need to be copied/versioned or otherwise archived with checksums before cleanup. Until that happens, the physical AMD worktree is a protected evidence source and must not be reset, cleaned, or deleted.
