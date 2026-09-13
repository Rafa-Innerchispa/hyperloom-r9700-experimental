# HyperLoom R9700 Phase 5 — reversible canary runbook

Status: **PREPARATION ONLY until the canary unit and bundle pass offline validation.**

Phase 4 is already closed. Do not rerun Phase 2/3/4 to rebuild context.

## Objective

Turn the validated S3 + readiness-warmup experiment into a separately deployable canary without silently replacing the operational stock service.

The canary uses:

- GPU: AMD Radeon AI PRO R9700 / gfx1201
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- runtime patch SHA256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- `GPU_MAX_HW_QUEUES=1`
- stock `ROCM_ATTN`
- canary port `18018`
- canary container `inneros-vllm-hyperloom-s3-canary`
- canary user unit `inneros-vllm-hyperloom-s3-canary.service`

## Safety design

The canary service deliberately **does not** contain `Conflicts=` or any command that stops stock.

Its preflight refuses to start unless:

1. `inneros-vllm-canary-rocm10.service` is not active;
2. no prior canary container exists;
3. port 18018 is free;
4. R9700 VRAM is below 1 GB;
5. the Phase5 bundle manifest exists and says PASS;
6. every overlay file matches the selected Phase3/4 hashes;
7. the S3 config matches its canonical SHA256.

Therefore the runtime switch must be an explicit, auditable operation, not a side effect of starting the canary.

## Preparation while stock remains online

Run:

`python3 scripts/r9700_phase5_prepare_canary.py`

This is allowed while stock is serving. It does not touch the GPU serving process. It rebuilds the #43389 overlay through the pinned Phase3 builder, verifies the runtime patch SHA and all five selected file hashes, copies them into a fresh Phase5 bundle, copies the canonical S3 config, and emits:

`var/r9700_phase5_canary_bundle/manifest.json`

plus versionable evidence:

`docs/evidence/r9700_phase5_canary_bundle_manifest.json`

Do not start a canary if this step fails.

## Canary unit

Source-controlled unit:

`scripts/systemd/inneros-vllm-hyperloom-s3-canary.service`

The unit runs on `127.0.0.1:18018` and has an `ExecStartPost` readiness gate:

`r9700_readiness_warmup.py --base-url http://127.0.0.1:18018`

Systemd startup therefore fails if the endpoint never becomes ready or the deterministic warmup output does not match the stock canonical hash.

## Controlled canary window

Only after preparation evidence is committed/pushed:

1. Verify stock `/v1/models` is HTTP 200 and record current state.
2. Explicitly stop `inneros-vllm-canary-rocm10.service`.
3. Verify R9700 VRAM returns to the clean ~60 MB class.
4. Install/reload the Phase5 user unit from the source-controlled unit file.
5. Start `inneros-vllm-hyperloom-s3-canary.service`.
6. Wait for systemd start completion. This includes HTTP readiness and canonical readiness warmup.
7. Verify `/v1/models` on `127.0.0.1:18018`.
8. Run the Phase4 measurement gate against port 18018: correctness, C1, C4, long-context, and fresh-prefix long control.
9. Preserve all evidence before deciding whether to keep or roll back.

Do not route general traffic to the canary before steps 6–8 pass.

## Rollback

Rollback remains the default end-state of the first canary window unless a separate promotion decision is made.

1. Stop `inneros-vllm-hyperloom-s3-canary.service`.
2. Confirm `inneros-vllm-hyperloom-s3-canary` is not running.
3. Verify R9700 VRAM is back below 1 GB.
4. Start `inneros-vllm-canary-rocm10.service`.
5. Wait for `http://127.0.0.1:8000/v1/models` HTTP 200 with the expected Qwen model.
6. Record restore evidence.

If the canary fails readiness, correctness, output hashes, or exhibits material regression, rollback immediately. Do not debug while leaving the operational stock service unavailable longer than necessary.

## Promotion gate

A persistent promotion is a separate decision. Minimum evidence:

- deterministic bundle preparation PASS;
- systemd unit validation PASS;
- clean canary boot from a cold GPU;
- readiness hash gate PASS;
- first-user TTFT remains in the hot ~50 ms class after readiness;
- canonical correctness and C4 hashes remain stock-exact;
- C1/C4/fresh-long remain inside the validated Phase4 envelope;
- rollback to stock has been demonstrated.

Only then consider changing the operational default.

## Invalid Phase5 worktrees

Two Phase5 worktrees were created from stale local refs before the local Git cache problem was identified. They are **not valid sources** and must not be used:

- `chatgpt/r9700-phase5-canary-20260913`
- `chatgpt/r9700-phase5-canary-v2-20260913`

Canonical Phase5 development branch:

`chatgpt/r9700-phase5-canary-v3-20260913`

It was created directly from exact GitHub canonical SHA:

`a80f34149f4bb07fb29aecc0b1eb0e918ffe10e1`
