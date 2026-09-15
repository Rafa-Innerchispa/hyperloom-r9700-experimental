from __future__ import annotations

import copy

from scripts import r9700_rebase_readiness as readiness


BASELINE = {
    "schema": readiness.SNAPSHOT_SCHEMA,
    "refs": dict(readiness.EXPECTED_REFS),
    "vllm": {
        "pr_43389": {
            "state": "open",
            "merged": False,
            "head_sha": readiness.EXPECTED_REFS["vllm_pr_43389_head"],
        },
        "stable": {
            "autoawq_triton_rejection_present": True,
            "triton_wna16_experts_present": True,
            "select_wna16_moe_backend_present": True,
            "convert_to_wna16_moe_kernel_format_present": True,
            "triton_conversion_branch_present": True,
            "process_weights_after_loading_present": True,
            "rdna3_backend_present": False,
            "overlay_oracle_repack_present": False,
            "overlay_fused_interleave_present": False,
            "overlay_config_layout_marker_present": False,
            "overlay_triton_layout_argument_present": False,
            "overlay_utility_module_present": False,
        },
        "main": {
            "autoawq_triton_rejection_present": True,
            "triton_wna16_experts_present": True,
            "select_wna16_moe_backend_present": True,
            "convert_to_wna16_moe_kernel_format_present": True,
            "triton_conversion_branch_present": True,
            "process_weights_after_loading_present": True,
            "rdna3_backend_present": True,
            "overlay_oracle_repack_present": False,
            "overlay_fused_interleave_present": False,
            "overlay_config_layout_marker_present": False,
            "overlay_triton_layout_argument_present": False,
            "overlay_utility_module_present": False,
        },
    },
    "hyperloom": {"declares_r9700_or_gfx1201_support": False},
}


def evaluate(snapshot):
    return readiness.evaluate_snapshot(snapshot)


def test_baseline_is_rebase_adapt_ready_but_never_authorizes_overlay_removal():
    result = evaluate(copy.deepcopy(BASELINE))

    assert result["review_required"] is False
    assert result["reasons"] == []
    assert result["runtime_mutation"] is False
    assert result["automatic_overlay_removal_allowed"] is False
    assert result["overlay_action"] == "retain"
    assert result["decisions"] == {
        "s3_overlay": "retain",
        "vllm_v0_29_0": "rebase/adapt",
        "vllm_current_main": "rebase/adapt",
        "hyperloom_current_main": "rebase/adapt",
    }
    assert result["hunk_status"] == {
        "stable_v0_29_0": "rebase_adapt",
        "current_main": "rebase_adapt",
    }


def test_missing_wna16_backbone_fails_closed():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["vllm"]["main"]["select_wna16_moe_backend_present"] = False

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "current_main:wna16_backbone_changed" in result["reasons"]
    assert result["hunk_status"]["current_main"] == "manual_review"


def test_upstream_overlay_like_code_requires_manual_review_not_auto_remove():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["vllm"]["main"]["overlay_oracle_repack_present"] = True

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "current_main:overlay_like_code_present_upstream" in result["reasons"]
    assert result["automatic_overlay_removal_allowed"] is False
    assert result["overlay_action"] == "retain"


def test_pr_merge_requests_revalidation():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["vllm"]["pr_43389"]["state"] = "closed"
    snapshot["vllm"]["pr_43389"]["merged"] = True

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "vllm_pr_43389_not_open" in result["reasons"]
    assert "vllm_pr_43389_merged" in result["reasons"]


def test_pr_head_change_requests_revalidation():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["vllm"]["pr_43389"]["head_sha"] = "deadbeef"

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "vllm_pr_43389_head_changed" in result["reasons"]


def test_ref_change_requests_revalidation():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["refs"]["vllm_main_sha"] = "deadbeef"

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "ref_changed:vllm_main_sha" in result["reasons"]


def test_hyperloom_support_declaration_requests_review():
    snapshot = copy.deepcopy(BASELINE)
    snapshot["hyperloom"]["declares_r9700_or_gfx1201_support"] = True

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert "hyperloom_r9700_support_declared" in result["reasons"]


def test_malformed_snapshot_fails_closed():
    snapshot = copy.deepcopy(BASELINE)
    del snapshot["vllm"]["stable"]["triton_wna16_experts_present"]

    result = evaluate(snapshot)

    assert result["review_required"] is True
    assert result["reasons"] == ["snapshot:stable:triton_wna16_experts_present"]
    assert result["automatic_overlay_removal_allowed"] is False


def test_repository_fixture_is_valid():
    fixture = readiness.load_snapshot(readiness.DEFAULT_SNAPSHOT)

    assert evaluate(fixture)["review_required"] is False


def test_live_collection_uses_exact_pinned_refs(monkeypatch):
    oracle = "\n".join(
        [
            readiness.AUTOAWQ_REJECTION,
            "TritonWNA16Experts",
            "def select_wna16_moe_backend(",
            "def convert_to_wna16_moe_kernel_format(",
            "elif backend == WNA16MoEBackend.TRITON:",
        ]
    )
    moe = "def process_weights_after_loading(\nconvert_to_wna16_moe_kernel_format("
    plain = "no overlay markers here"

    def fake_json(url, *, timeout):
        assert url == readiness.PR_URL
        assert timeout == 1
        return {
            "state": "open",
            "merged": False,
            "head": {"sha": readiness.EXPECTED_REFS["vllm_pr_43389_head"]},
        }

    def fake_text(url, *, timeout):
        assert timeout == 1
        if url.endswith("README.md"):
            return "Supported platforms: MI300X, MI325X, MI355X"
        if readiness.ORACLE_PATH in url:
            if readiness.EXPECTED_REFS["vllm_main_sha"] in url:
                return oracle + "\nWNA16MoEBackend.RDNA3"
            return oracle
        if readiness.MOE_WNA16_PATH in url:
            return moe
        return plain

    monkeypatch.setattr(readiness, "_fetch_json", fake_json)
    monkeypatch.setattr(readiness, "_fetch_text", fake_text)
    monkeypatch.setattr(
        readiness,
        "_fetch_optional_text",
        lambda url, timeout: None,
    )

    snapshot = readiness.collect_live_snapshot(timeout=1)
    result = readiness.evaluate_snapshot(snapshot, source="live")

    assert result["review_required"] is False
    assert snapshot["vllm"]["stable"]["rdna3_backend_present"] is False
    assert snapshot["vllm"]["main"]["rdna3_backend_present"] is True
