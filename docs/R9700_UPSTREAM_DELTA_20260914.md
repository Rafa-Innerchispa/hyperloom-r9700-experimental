# Radeon AI PRO R9700 / gfx1201 upstream delta review — 2026-09-14

Status: evidence-backed review against the current upstream state. This review does **not** reopen Phases 2–6, does not rerun performance/soak/fallback/promotion gates, and does not mutate the live runtime.

## Scope and pinned identities

Exact local workload under review:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ + vLLM + ROCm 10 + Radeon AI PRO R9700/gfx1201 + MoE + AutoAWQ + WNA16`

Local canonical repository state at review start:

- repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- canonical `main`: `b442bf11b24d715ef17b26c26deccc86fa59ba22`
- validated S3 runtime image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- S3 runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`

Upstream snapshots used for this review:

- ROCm 10.0.0 compatibility/specification documentation, checked 2026-09-14.
- vLLM `main`: `e85c8826ce2a810367e8a70eedb987e31b140800` (2026-09-14).
- latest stable vLLM release: `v0.29.0`, published 2026-09-09.
- AMD-AGI/Hyperloom `main`: `0425bde3f6e76e1588400c37d056dfd3bb75ac11` (2026-09-14), README version 1.1.0.

## Decision matrix

| componente local | soporte upstream actual | evidencia exacta | solapamiento | riesgo | decisión |
|---|---|---|---|---|---|
| ROCm 10 base sobre Radeon AI PRO R9700 / RDNA4 / `gfx1201` | ROCm 10.0.0 ya enumera R9700/R9700S como RDNA4 `gfx1201`; la matriz de compatibilidad incluye la serie Radeon AI PRO R9000. Las release notes de ROCm 10 también listan vLLM 0.27.0 para `gfx1201`. | AMD ROCm 10 GPU specs: `https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html`; compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`; ROCm 10 release notes: `https://rocm.docs.amd.com/en/latest/about/release-notes.html` | alto para enablement básico de hardware, bajo para nuestro workload exacto | Confundir soporte de ROCm al dispositivo con soporte de HyperLoom o con equivalencia del camino AWQ MoE/WNA16. | retain |
| overlay S3 tipo vLLM #43389 para repack/interleave INT4/W4A16 MoE de GPTQ/AWQ en Triton | **No existe equivalencia upstream completa.** PR #43389 sigue `open`, `merged=false`; específicamente repackea AWQ/GPTQ para Triton y reporta pruebas en `gfx1201`. El `main` actual sí tiene WNA16/MoE más moderno, pero su oracle rechaza `AutoAWQConfig` para backend Triton con `the AutoAWQ weight layout is not supported`. | vLLM PR #43389, head `56ef89e1ff4a1552beb3b5c51c00b73ea44daca1`: `https://github.com/vllm-project/vllm/pull/43389`; current oracle at vLLM `e85c8826...`: `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`; current quant config: `vllm/model_executor/layers/quantization/moe_wna16.py` | parcial | Eliminar el overlay porque upstream “tiene WNA16/AWQ” rompería la equivalencia del camino exacto que S3 validó. Copiar el overlay viejo sin rebase tampoco es seguro porque la arquitectura WNA16 upstream cambió. | retain |
| R9700-specific `int4_w4a16` MoE tuning/config usado por S3 | Upstream tiene soporte genérico WNA16 y mejoras ROCm, pero no hay evidencia de una config upstream equivalente para `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` sobre R9700 que reproduzca nuestro punto operativo. | Local evidence: `FINAL_STATUS.md`, `docs/R9700_PROJECT_CONTINUITY.md`; upstream vLLM current `moe_wna16.py`; PR #52112 merged as `e8ad2855e7f5d40665250eb468bfd567e3a4b3c1` corrige varios errores ROCm int4/int8, pero no sustituye #43389 ni nuestro tuning. | medio | Sustituir tuning medido por defaults genéricos puede conservar corrección y perder el rendimiento/estabilidad demostrado. | retain |
| vLLM 0.27.0 ROCm10 runtime pin validado por S3 | ROCm 10 release notes validan vLLM 0.27.0 para `gfx1201`, pero upstream vLLM ya está en `v0.29.0` y `main` contiene refactors WNA16/MoE posteriores (#44120, #52112, #54809). | ROCm 10 release notes; vLLM release `v0.29.0`: `https://github.com/vllm-project/vllm/releases/tag/v0.29.0`; WNA16 history including #44120 and #52112. | medio/alto | Actualizar el runtime validado in-place invalidaría la evidencia S3 y podría romper el overlay #43389-style por cambios de layout/oracle. Permanecer indefinidamente en 0.27.0 acumula deuda. | rebase/adapt |
| stock `ROCM_ATTN` en la receta S3 | vLLM ya habilitó AITER/FP8 en gfx12 vía PR #43615, pero ese mismo PR deja AITER RMSNorm desactivado por defecto en gfx12 porque el kernel está roto allí. Esto no prueba que AITER sea un reemplazo seguro para nuestro workload AWQ S3. | vLLM PR #43615, merged as `f43e1d26e3e6b40398be27b218cf2f2786432028`: `https://github.com/vllm-project/vllm/pull/43615`; local S3 evidence in `FINAL_STATUS.md`. | parcial | Cambiar attention backend sólo por disponibilidad upstream puede introducir una regresión ajena al objetivo ya cerrado. | retain |
| `GPU_MAX_HW_QUEUES=1` en S3 | No se encontró una sustitución upstream que vuelva innecesario este knob para el stack exacto. Es parte del control local medido contra el baseline y del estado reproducible S3. | `FINAL_STATUS.md` y continuidad local; no existe evidencia upstream revisada que demuestre equivalencia para este workload. | bajo | Retirarlo sin una regresión concreta reabre tuning cerrado y cambia el entorno de medición. | retain |
| readiness-stage warmup determinista para absorber late prefix-prefill Triton JIT | Triton soporta RDNA4/gfx1201 y su backend AMD sigue evolucionando, pero soporte arquitectónico no implica ausencia de lazy/JIT latency en el stack vLLM exacto. El warmup local fue el gate que movió el primer-user TTFT a ~50 ms en la receta validada. | Triton AMD layout actual reconoce RDNA4 `gfx1200/gfx1201`: `https://github.com/triton-lang/triton/blob/main/python/triton/experimental/gluon/language/amd/_layouts.py`; evidencia local en `FINAL_STATUS.md`. | bajo/medio | Quitar el warmup por inferencia, sin prueba exacta, puede reintroducir el cold-user penalty ya mitigado. | retain |
| S3 fail-closed preflight, hashes, separación stock/canary, automatic fallback y rollback | Upstream HyperLoom tiene validación/KEEP-REVERT dentro de campañas y recibió mejoras recientes de integración/liveness, pero no sustituye esta capa operacional específica del servicio R9700. | Local Phase 5/6 continuity and evidence; AMD-AGI/Hyperloom current `main` `0425bde3f6e76e1588400c37d056dfd3bb75ac11`; upstream PR #1501. | bajo | Reemplazar una capa de despliegue ya probada por semántica interna de campañas perdería guards operacionales específicos. | retain |
| fork/port experimental de HyperLoom para Radeon R9700 | HyperLoom 1.1.0 upstream sigue listando como plataformas soportadas únicamente MI300X, MI325X y MI355X. Añadió xDiT/KernelForge y mejoras de controlador, pero no declara R9700/Radeon como plataforma soportada. | AMD-AGI/Hyperloom `README.md` at `0425bde3f6e76e1588400c37d056dfd3bb75ac11`: `https://github.com/AMD-AGI/Hyperloom/blob/0425bde3f6e76e1588400c37d056dfd3bb75ac11/README.md` | parcial en harness, nulo en soporte Radeon declarado | No incorporar fixes upstream acumula deuda; hacer un merge mayor directamente sobre el runtime validado puede desestabilizar la integración R9700. | rebase/adapt |
| afirmación histórica implícita de que `gfx1201` necesita un port porque ROCm no soporta el hardware | Ya no es correcta como descripción de ROCm 10. ROCm 10 soporta/lista oficialmente R9700/gfx1201. Lo experimental sigue siendo la integración HyperLoom + vLLM + AutoAWQ MoE/WNA16 + tuning/operación S3. | ROCm 10 compatibility/spec docs; HyperLoom 1.1.0 supported-platform table; vLLM #43389 sigue abierto. | alto en wording, no en implementación | Mantener una narrativa vieja debilita credibilidad técnica; afirmar que todo el stack es oficialmente soportado también sería falso. | remove |

## Findings that change the architecture decision

### 1. ROCm has caught up at the hardware-enablement layer

R9700 / `gfx1201` is no longer an unofficial ROCm device target. That removes any reason to preserve a *device-recognition-only* compatibility hack if one is later found in the fork. No such standalone hack is being removed in this change because none was identified as a distinct current component in the canonical evidence.

This does **not** make HyperLoom-on-R9700 an upstream-supported configuration: HyperLoom 1.1.0 still publishes only MI300X, MI325X and MI355X in its supported platform table.

### 2. The exact AutoAWQ → Triton WNA16 gap remains upstream

The most important result is negative: upstream vLLM has not made the S3 #43389-style overlay redundant.

PR #43389 remains open and unmerged. Its purpose is precisely the ROCm-only GPTQ/AWQ INT4 MoE repack/interleave path that S3 used conceptually, and its own benchmark table includes `gfx1201`.

Meanwhile, current vLLM `main` has a more mature WNA16 oracle and AWQ integration, but the Triton backend still returns an explicit incompatibility for `AutoAWQConfig`: `the AutoAWQ weight layout is not supported`.

Therefore the overlay decision is **retain**, not remove. A future refresh should port/rebase its behavior onto the current WNA16 modular-kernel/oracle architecture rather than replay the old patch blindly.

### 3. Upstream refactors justify an isolated migration lane, not a live upgrade

vLLM 0.29.0 and current `main` contain useful ROCm/WNA16 changes, including:

- #44120: migrate `MoeWNA16Method` to the modular-kernel oracle;
- #52112: merged ROCm int4/int8 fixes, including asymmetric Triton MoE support;
- #54809: later GPTQ activation-order cleanup;
- #43615: gfx12 AITER/FP8 enablement, while explicitly keeping AITER RMSNorm off by default on gfx12.

These are sufficient reason to open a separate **rebase/adapt** experiment later. They are not sufficient reason to mutate or promote over the validated S3 runtime.

### 4. HyperLoom upstream should be rebased selectively

Current upstream HyperLoom 1.1.0 has significant harness improvements, including the 2026-09-14 controller-patch integration and pump-liveness fixes in PR #1501. Those are attractive for maintainability but orthogonal to the validated R9700 kernel/runtime recipe.

The correct migration shape is selective rebase/adaptation in an isolated branch with existing correctness/fail-closed contracts preserved. A wholesale in-place merge is not justified by this review.

## Actions from this review

1. **Do not remove the #43389-style S3 overlay.** Current upstream does not provide an equivalent AutoAWQ→Triton WNA16 path.
2. **Do not upgrade the live/preserved S3 runtime from vLLM 0.27.0 in place.** Treat v0.29.0/current `main` as a separate migration candidate.
3. **Keep the R9700-specific W4A16 config, `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, deterministic warmup, correctness hashes, fail-closed preflight and fallback contracts.** No reviewed upstream evidence invalidates them.
4. **Future rebase lane:** adapt the local AWQ repack/interleave behavior onto current vLLM WNA16 modular-kernel/oracle architecture, then run only the new-version regression gates required by that change. This is a new migration test, not a repetition of Phases 2–6.
5. **Future HyperLoom lane:** selectively rebase harness fixes from upstream 1.1.0 while keeping Radeon-specific integration and deployment guards isolated.
6. **Public wording:** say that ROCm 10 supports R9700/gfx1201, while this project remains an experimental/bleeding-edge HyperLoom + vLLM integration for the exact AWQ MoE workload. Do not claim official HyperLoom R9700 support, universal acceleration, or “world first”.

## Change boundary

This delta review intentionally changes documentation only. It does not:

- start/stop/reconfigure the live vLLM service;
- promote or demote S3;
- run a benchmark/soak;
- repeat automatic fallback testing;
- alter correctness hashes or thresholds;
- change Python/runtime code;
- reopen Phases 2, 3, 4, 5 or 6.
