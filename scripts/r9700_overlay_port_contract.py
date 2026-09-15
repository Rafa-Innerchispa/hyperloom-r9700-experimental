#!/usr/bin/env python3
"""Validate the non-runtime R9700 vLLM AWQ->Triton overlay port manifest.

The contract is intentionally static and fail-closed. It does not patch vLLM,
start a model, touch S3, or authorize promotion. Its job is to keep the future
port narrow enough that the exact R9700 migration can be reviewed before any
GPU/runtime work begins.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "r9700-vllm-overlay-port-contract-v1"
MANIFEST_SCHEMA = "r9700-vllm-overlay-port-manifest-v1"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "r9700_vllm029_overlay_port_manifest_20260915.json"
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_EDIT_PATHS = {
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
    "vllm/model_executor/layers/fused_moe/config.py",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
    "vllm/model_executor/layers/fused_moe/fused_moe.py",
}

REQUIRED_NON_EDITS = {
    "vllm/model_executor/layers/quantization/auto_awq.py",
    "vllm/platforms/rocm.py",
    "S3 runtime image",
    "stock fallback",
    "ROCM_ATTN",
    "GPU_MAX_HW_QUEUES",
    "deterministic readiness warmup",
    "Phase 2-6 evidence",
}

REQUIRED_PRESERVE_FLAGS = {
    "keep_direct_autoawq_triton_rejection",
    "keep_s3_overlay",
    "keep_stock_fallback",
    "keep_rocm_attn",
    "keep_gpu_max_hw_queues_1",
    "keep_deterministic_warmup",
    "keep_fail_closed_hashes",
}

REQUIRED_FACTS = {
    "pr_43389_merged": False,
    "direct_autoawq_triton_rejection_present": True,
    "moe_wna16_utils_module_present": False,
    "rocm_r9700_device_id_present": True,
    "rocm_on_rdna_helper_present": True,
    "rdna3_native_backend_is_not_r9700_replacement": True,
}

REQUIRED_PATH_STEPS = (
    "AutoAWQConfig",
    "MoeWNA16Config fallback/normalization",
    "MoeWNA16Method",
    "select_wna16_moe_backend",
    "WNA16MoEBackend.TRITON",
    "convert_to_wna16_moe_kernel_format",
    "TritonWNA16Experts",
    "invoke_fused_moe_wna16_triton_kernel",
    "fused_moe_kernel_gptq_awq",
)


class ContractError(ValueError):
    """Controlled manifest validation failure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA_RE.fullmatch(value))


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("manifest:invalid_json") from exc
    _require(isinstance(payload, dict), "manifest:not_object")
    return payload


def evaluate_manifest(manifest: Any) -> dict[str, Any]:
    """Return a machine-readable PASS/FAIL verdict without mutating anything."""
    try:
        _validate(manifest)
    except ContractError as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "review_required": True,
            "reasons": [str(exc)],
            "runtime_mutation": False,
            "automatic_promotion_allowed": False,
        }

    return {
        "schema": SCHEMA,
        "ok": True,
        "review_required": False,
        "reasons": [],
        "runtime_mutation": False,
        "automatic_promotion_allowed": False,
        "decision": {
            "s3": "retain",
            "prototype": "rebase/adapt",
        },
        "edit_paths": sorted(EXPECTED_EDIT_PATHS),
    }


def _validate(manifest: Any) -> None:
    _require(isinstance(manifest, dict), "manifest:not_object")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest:schema")
    _require(manifest.get("status") == "prototype_only_not_applied", "manifest:status")
    _require(manifest.get("runtime_mutation") is False, "manifest:runtime_mutation")
    _require(_is_sha(manifest.get("hyperloom_base_sha")), "manifest:hyperloom_base_sha")

    refs = manifest.get("refs")
    _require(isinstance(refs, dict), "manifest:refs")
    for key in (
        "vllm_stable_sha",
        "rehearsal_baseline_main_sha",
        "prototype_current_main_sha",
        "vllm_pr_43389_head_sha",
    ):
        _require(_is_sha(refs.get(key)), f"manifest:refs:{key}")
    _require(refs.get("vllm_stable_tag") == "v0.29.0", "manifest:refs:vllm_stable_tag")
    _require(
        refs["prototype_current_main_sha"] != refs["rehearsal_baseline_main_sha"],
        "manifest:current_main_not_advanced",
    )

    blobs = manifest.get("source_blobs_at_prototype_current_main")
    _require(isinstance(blobs, dict) and bool(blobs), "manifest:source_blobs")
    for path, sha in blobs.items():
        _require(isinstance(path, str) and bool(path), "manifest:source_blob_path")
        _require(_is_sha(sha), f"manifest:source_blob_sha:{path}")

    facts = manifest.get("current_upstream_facts")
    _require(isinstance(facts, dict), "manifest:current_upstream_facts")
    _require(facts.get("pr_43389_state") == "open", "manifest:pr_43389_state")
    for key, expected in REQUIRED_FACTS.items():
        _require(facts.get(key) is expected, f"manifest:fact:{key}")

    path_steps = manifest.get("exact_path")
    _require(isinstance(path_steps, list), "manifest:exact_path")
    _require(tuple(path_steps) == REQUIRED_PATH_STEPS, "manifest:exact_path_changed")

    preserve = manifest.get("preserve_contracts")
    _require(isinstance(preserve, dict), "manifest:preserve_contracts")
    for flag in REQUIRED_PRESERVE_FLAGS:
        _require(preserve.get(flag) is True, f"manifest:preserve:{flag}")

    edits = manifest.get("prototype_edits")
    _require(isinstance(edits, list), "manifest:prototype_edits")
    edit_paths = {edit.get("path") for edit in edits if isinstance(edit, dict)}
    _require(edit_paths == EXPECTED_EDIT_PATHS, "manifest:edit_paths_changed")
    for edit in edits:
        _require(isinstance(edit, dict), "manifest:edit_not_object")
        _require(edit.get("scope") == "runtime_candidate_not_applied", "manifest:edit_scope")
        _require(edit.get("decision") == "rebase/adapt", "manifest:edit_decision")
        _require(edit.get("action") in {"add", "adapt"}, "manifest:edit_action")
        _require(isinstance(edit.get("design"), str) and edit["design"], "manifest:edit_design")

    non_edits = manifest.get("explicit_non_edits")
    _require(isinstance(non_edits, list), "manifest:explicit_non_edits")
    _require(REQUIRED_NON_EDITS.issubset(set(non_edits)), "manifest:required_non_edits")
    _require(
        "vllm/model_executor/layers/quantization/auto_awq.py" not in edit_paths,
        "manifest:auto_awq_must_not_be_edited",
    )
    _require(
        "vllm/platforms/rocm.py" not in edit_paths,
        "manifest:rocm_platform_must_not_be_edited",
    )

    forbidden = manifest.get("forbidden_manifest_tokens")
    _require(isinstance(forbidden, list) and bool(forbidden), "manifest:forbidden_tokens")
    design_text = "\n".join(str(edit.get("design", "")) for edit in edits)
    for token in forbidden:
        _require(isinstance(token, str) and bool(token), "manifest:forbidden_token_type")
        _require(token not in design_text, f"manifest:forbidden_design_token:{token}")

    gpu_gates = manifest.get("gpu_required_before_any_runtime_candidate")
    _require(isinstance(gpu_gates, list) and len(gpu_gates) >= 5, "manifest:gpu_gates")

    decision = manifest.get("decision")
    _require(isinstance(decision, dict), "manifest:decision")
    _require(decision.get("s3") == "retain", "manifest:decision:s3")
    _require(decision.get("vllm_v0_29_0") == "rebase/adapt", "manifest:decision:v0_29")
    _require(decision.get("vllm_current_main") == "rebase/adapt", "manifest:decision:main")
    _require(decision.get("remove_any_current_overlay") is False, "manifest:decision:remove_overlay")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        result = evaluate_manifest(manifest)
    except ContractError as exc:
        result = {
            "schema": SCHEMA,
            "ok": False,
            "review_required": True,
            "reasons": [str(exc)],
            "runtime_mutation": False,
            "automatic_promotion_allowed": False,
        }

    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
