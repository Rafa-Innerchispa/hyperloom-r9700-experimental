from __future__ import annotations

import copy

from scripts import r9700_overlay_port_contract as contract


def manifest():
    return contract.load_manifest(contract.DEFAULT_MANIFEST)


def test_repository_manifest_passes_contract():
    result = contract.evaluate_manifest(manifest())

    assert result["ok"] is True
    assert result["review_required"] is False
    assert result["runtime_mutation"] is False
    assert result["automatic_promotion_allowed"] is False
    assert result["decision"] == {"s3": "retain", "prototype": "rebase/adapt"}
    assert set(result["edit_paths"]) == contract.EXPECTED_EDIT_PATHS


def test_runtime_mutation_fails_closed():
    payload = manifest()
    payload["runtime_mutation"] = True

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:runtime_mutation"]


def test_current_main_must_be_distinct_from_rehearsal_baseline():
    payload = manifest()
    payload["refs"]["prototype_current_main_sha"] = payload["refs"]["rehearsal_baseline_main_sha"]

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:current_main_not_advanced"]


def test_pr_43389_merge_requires_review():
    payload = manifest()
    payload["current_upstream_facts"]["pr_43389_merged"] = True

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:fact:pr_43389_merged"]


def test_direct_autoawq_rejection_must_remain_in_prototype_contract():
    payload = manifest()
    payload["preserve_contracts"]["keep_direct_autoawq_triton_rejection"] = False

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:preserve:keep_direct_autoawq_triton_rejection"]


def test_auto_awq_source_must_not_become_a_prototype_edit():
    payload = manifest()
    payload["prototype_edits"].append(
        {
            "path": "vllm/model_executor/layers/quantization/auto_awq.py",
            "action": "adapt",
            "scope": "runtime_candidate_not_applied",
            "design": "broaden direct AutoAWQ Triton support",
            "decision": "rebase/adapt",
        }
    )

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:edit_paths_changed"]


def test_rocm_platform_source_must_not_become_a_prototype_edit():
    payload = manifest()
    payload["prototype_edits"][0]["path"] = "vllm/platforms/rocm.py"

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:edit_paths_changed"]


def test_removing_a_required_edit_fails_closed():
    payload = manifest()
    payload["prototype_edits"] = payload["prototype_edits"][:-1]

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:edit_paths_changed"]


def test_s3_must_remain_retain():
    payload = manifest()
    payload["decision"]["s3"] = "remove"

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:decision:s3"]


def test_gpu_gates_cannot_be_collapsed_to_empty_plan():
    payload = manifest()
    payload["gpu_required_before_any_runtime_candidate"] = []

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == ["manifest:gpu_gates"]


def test_forbidden_design_token_fails_closed():
    payload = manifest()
    token = payload["forbidden_manifest_tokens"][0]
    payload["prototype_edits"][0]["design"] += f" {token}"

    result = contract.evaluate_manifest(payload)

    assert result["ok"] is False
    assert result["reasons"] == [f"manifest:forbidden_design_token:{token}"]


def test_manifest_copy_does_not_mutate_repository_fixture():
    first = manifest()
    second = copy.deepcopy(first)
    second["decision"]["s3"] = "remove"

    assert contract.evaluate_manifest(first)["ok"] is True
    assert contract.evaluate_manifest(second)["ok"] is False
