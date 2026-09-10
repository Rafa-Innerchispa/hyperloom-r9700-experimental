# Recovered WNA16 tuner result — 2026-09-10

This note preserves the interpretation of the recovered, previously uncommitted WNA16 tuning work from 2026-09-09. The original JSON files remain on the AMD project worktree and their SHA-256 values are recorded in `docs/recovery/20260910/RECOVERY_SHA256_INVENTORY.json`.

## Bounded WNA16 tuner

Source evidence: `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`

Recovered SHA-256: `f56942b75f6621ae22078a8173ce3f6ea01c8980d07f279d8456c3d2aaba6d50`

Winner configuration:

```text
BLOCK_SIZE_M=16
BLOCK_SIZE_N=64
BLOCK_SIZE_K=32
GROUP_SIZE_M=1
num_warps=4
num_stages=2
waves_per_eu=4
SPLIT_K=1
```

Recovered isolated-kernel score:

- median speedup across M=1,2,4,8,16: `1.2092561838928513x`
- minimum tested speedup: `1.083842659569329x`
- M1: `122.0649 us -> 112.6224 us`, `1.08384x`
- M2: `205.3677 us -> 134.3099 us`, `1.52906x`
- M4: `267.5773 us -> 221.2743 us`, `1.20926x`
- M8: `455.9464 us -> 356.2570 us`, `1.27982x`
- M16: `640.6282 us -> 578.2905 us`, `1.10780x`

Verdict for this bounded tuner: **MICROBENCH KEEP**. The result justifies retaining the gfx1201-specific WNA16 config as a candidate. It does not prove a serving-level gain.

## First live override smoke

Source evidence: `docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json`

Recovered SHA-256: `5d1891dbd8c9bf4a414542c68ee3e702f0c18e2c900eb5cbeaf92cba988239b9`

Observed:

- candidate health: PASS
- candidate decode: `22.3673 tok/s`
- stock before: `69.4568 tok/s`
- stock after: `22.9643 tok/s`
- deterministic hash matched stock-before
- `tuned_seen=false`
- temporary `.pth` removed
- test-level `pass=false`

Interpretation: the tuned config was not observed in the actual call path, so this run cannot validate the override. It also demonstrates the process-start bimodality directly: stock moved from the fast regime before the candidate to the slow regime after restore.

## Second live override smoke

Source evidence: `docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json`

Recovered SHA-256: `54e6a79cae81e21e39c34912008f66fb35f16e8e263faae84d4ef663eb428f8f`

Observed:

- candidate health: PASS
- candidate decode: `22.5802 tok/s`
- `tuned_seen=true`
- `only_small_m=true`
- observed M values: `1,2,4,8,16`
- all observed calls used the recovered winner configuration
- stock before: `22.3732 tok/s` (slow regime)
- stock after: `69.7573 tok/s` (fast regime)
- `hash_match_before_candidate=false`
- `restore_hash_mismatch=true`
- temporary `.pth` removed
- test-level `pass=false`

Interpretation: this run proves that the small-M override did execute, but it fails the promotion gate because correctness did not match and process-start state changed across the comparison. The ~1.21x isolated WNA16 config win therefore cannot be promoted to a model-level claim.

## Final recovered verdict

**BOUNDED TUNER: MICROBENCH KEEP**

**LIVE WNA16 OVERRIDE: REJECT / NOT PROMOTED**

The recovered work is useful because it establishes that gfx1201-specific WNA16 tuning has measurable isolated potential, while also showing that a naive live override is unsafe as a production or hackathon claim. Any reuse must be re-integrated through the real vLLM config path, revalidated for deterministic correctness, and benchmarked against a stable/fast RDNA4 baseline using independent process starts.

## Relationship to the 2026-09-10 Unified Attention campaign

The three-start Unified Attention campaign provides a much cleaner serving baseline than the 2026-09-09 live tuner smoke. Future WNA16/HyperLoom integration should be evaluated on top of that stable RDNA4 attention path rather than against an uncontrolled slow stock spawn.
