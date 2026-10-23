# ROCm 10.1 storage, memory, and rollback gate

This gate protects the Radeon AI PRO R9700 host from starting a second 30B runtime while storage cleanup, memory pressure, or rollback preparation is incomplete.

## Live checkpoint

Read-only Peer Ops snapshot observed on 2026-10-23 at approximately 03:22 GYT:

- root filesystem: 466 GiB total, 401 GiB used, 46 GiB available, 90% used;
- memory: 30 GiB total, approximately 10 GiB available;
- swap: 8 GiB total, approximately 8 GiB used;
- stock container `inneros-vllm-qwen3-coder-30b-awq`: running;
- stock endpoint `127.0.0.1:8000`: listening;
- candidate port `18029`: no listener observed;
- Radeon AI PRO R9700 / `gfx1201`: detected.

Memory availability improved materially from the earlier snapshot, but the disk gate is still closed. Full swap alone is not enough to prove active thrashing; the gate samples swap I/O and blocks only when full swap is also actively moving pages above the configured threshold.

## Mandatory launch thresholds

The 30B ROCm 10.1 candidate remains on hold unless all of these are true:

1. root free space is at least **70 GiB**;
2. root usage is at most **85%**;
3. available memory is at least **6 GiB** before opening the exclusive window;
4. storage cleanup is no longer active;
5. the exact ROCm 10.1 vLLM image already exists locally;
6. the stock container and `:8000` endpoint are healthy;
7. port `18029` is free;
8. swap I/O is below 1 MiB/s when swap usage is above 90%.

The gate never authorizes an image pull. If the image is missing, keep the candidate on hold until the disk situation is resolved and the missing artifact is reviewed explicitly.

## Read-only preflight

Run from the repository root:

```bash
python3 scripts/rocm10_1_host_gate.py \
  --cleanup-active \
  --output docs/evidence/rocm10_1_host_gate.json
```

While Cursor or another storage process is still cleaning, exit code `2` is expected.

After cleanup is complete:

```bash
python3 scripts/rocm10_1_host_gate.py \
  --output docs/evidence/rocm10_1_host_gate.json
```

Exit code `0` means only that the host is ready to prepare a bounded exclusive GPU window. It does **not** start, stop, pull, build, or promote anything.

## Stock rollback manifest

After the resource gate passes, but before stopping the stock 30B service, capture its exact redacted identity:

```bash
python3 scripts/rocm10_1_stock_rollback_manifest.py \
  --output docs/evidence/rocm10_1_stock_rollback_manifest.json
```

The command is read-only. It records:

- stock container ID and running state;
- exact image ID, tags, and available repository digests;
- command, entrypoint, ROCm device mappings, groups, network/IPC/security settings, and mounts;
- a redacted environment preserving non-secret runtime flags such as `ROCM_ATTN` and `GPU_MAX_HW_QUEUES`;
- the `:8000` TCP state and `/v1/models` response;
- the immediate rollback action `docker start inneros-vllm-qwen3-coder-30b-awq`;
- a stable SHA-256 over the manifest.

The stock container must remain present. The exclusive candidate window may stop it temporarily, but must never remove or recreate it as part of the experiment.

## Safe execution order

1. Finish the existing storage task `ops_f9983b081feb`; do not create a second cleanup task.
2. Record `df`, memory, swap I/O, Docker image identity, stock container identity, and `:8000` health.
3. Capture the stock rollback manifest and require its hash before any stop operation.
4. Reuse the existing ROCm 10.1 image. Do not download it again merely because the project resumed in another chat.
5. Open one bounded **exclusive GPU window**. A separate port is necessary but not sufficient: two 30B models cannot safely share the 32 GiB R9700.
6. Stop the stock model only after the rollback manifest and stock health evidence exist.
7. Launch the candidate on port `18029` using AMD ROCm device access:
   - `/dev/kfd`;
   - `/dev/dri`;
   - video/render group access;
   - `--ipc=host`;
   - `seccomp=unconfined` only where required by the validated ROCm image.
8. Never use `nvidia-smi`, `--gpus`, CUDA device flags, or NVIDIA container assumptions on this host.
9. Capture direct candidate evidence: runtime versions, exact 30B load, AWQ/WNA16 backend, correctness hash, tool calls, C1/C2/C4, long context, VRAM, GPU use, cold/warm TTFT, repeated starts, and bounded soak.
10. On any failed gate, stop the candidate, run the recorded stock start action, and verify `127.0.0.1:8000` plus the expected Qwen model before doing anything else.
11. Promotion to the canonical endpoint is a separate blue/green decision after all direct candidate evidence passes.

## Cleanup boundaries

Storage recovery must not remove:

- the active stock/S3 vLLM image or container;
- the already-downloaded ROCm 10.1 image if complete;
- Qwen/Ollama/Hugging Face models in use;
- ACT III, MC3, Match 4, or submission evidence;
- MongoDB, PostgreSQL, Qdrant, or active Docker volumes;
- active worktrees or unique backups.

Do not use `docker system prune -a`, `docker volume prune`, `git clean -fdx`, or broad package autoremove operations.

## Longer-term placement

After the candidate is stable, move only regenerable growth to the verified DATA filesystem through a separate migration plan with rollback:

- Docker build cache;
- model download cache;
- temporary experiment artifacts;
- terminal worktrees that have passed lock and evidence checks;
- package caches.

Production container storage, active model files, databases, and unique evidence require separate migration gates.
