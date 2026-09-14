from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts import r9700_upstream_watch as watch


BASELINE = {
    "schema": watch.SNAPSHOT_SCHEMA,
    "captured_at": "2026-09-14T18:00:00Z",
    "vllm": {
        "pr_43389": {"state": "open", "merged": False},
        "latest_release": "v0.29.0",
        "autoawq_triton_rejection_present": True,
    },
    "hyperloom": {"declares_r9700_or_gfx1201_support": False},
}


def evaluate(current):
    return watch.evaluate_snapshot(current, BASELINE)


def test_baseline_does_not_request_review():
    result = evaluate(copy.deepcopy(BASELINE))

    assert result["review_required"] is False
    assert result["reasons"] == []
    assert result["automatic_overlay_removal_allowed"] is False
    assert result["overlay_action"] == "retain_until_manual_revalidation"


def test_merged_pr_requests_review():
    current = copy.deepcopy(BASELINE)
    current["vllm"]["pr_43389"] = {"state": "closed", "merged": True}

    result = evaluate(current)

    assert result["review_required"] is True
    assert "vllm_pr_43389_merged" in result["reasons"]
    assert result["automatic_overlay_removal_allowed"] is False


def test_closed_unmerged_pr_requests_review():
    current = copy.deepcopy(BASELINE)
    current["vllm"]["pr_43389"]["state"] = "closed"

    result = evaluate(current)

    assert result["review_required"] is True
    assert "vllm_pr_43389_state_changed" in result["reasons"]


def test_autoawq_triton_rejection_disappearing_requests_review():
    current = copy.deepcopy(BASELINE)
    current["vllm"]["autoawq_triton_rejection_present"] = False

    result = evaluate(current)

    assert result["review_required"] is True
    assert "autoawq_triton_rejection_disappeared" in result["reasons"]


def test_hyperloom_r9700_support_requests_review():
    current = copy.deepcopy(BASELINE)
    current["hyperloom"]["declares_r9700_or_gfx1201_support"] = True

    result = evaluate(current)

    assert result["review_required"] is True
    assert "hyperloom_r9700_support_declared" in result["reasons"]


def test_new_vllm_release_requests_review():
    current = copy.deepcopy(BASELINE)
    current["vllm"]["latest_release"] = "v0.30.0"

    result = evaluate(current)

    assert result["review_required"] is True
    assert "vllm_latest_release_changed" in result["reasons"]


def test_multiple_deltas_are_all_reported_without_authorizing_removal():
    current = copy.deepcopy(BASELINE)
    current["vllm"]["pr_43389"] = {"state": "closed", "merged": True}
    current["vllm"]["latest_release"] = "v0.30.0"
    current["vllm"]["autoawq_triton_rejection_present"] = False
    current["hyperloom"]["declares_r9700_or_gfx1201_support"] = True

    result = evaluate(current)

    assert result["review_required"] is True
    assert set(result["reasons"]) == {
        "vllm_pr_43389_merged",
        "vllm_latest_release_changed",
        "autoawq_triton_rejection_disappeared",
        "hyperloom_r9700_support_declared",
    }
    assert result["automatic_overlay_removal_allowed"] is False
    assert result["overlay_action"] == "retain_until_manual_revalidation"


def test_malformed_snapshot_fails_closed():
    current = copy.deepcopy(BASELINE)
    del current["vllm"]["pr_43389"]["merged"]

    result = evaluate(current)

    assert result["review_required"] is True
    assert result["reasons"] == ["snapshot:pr_merged"]
    assert result["automatic_overlay_removal_allowed"] is False


def test_live_collection_uses_vllm_oracle_and_hyperloom_readme(monkeypatch):
    json_payloads = {
        watch.PR_URL: {"state": "open", "merged": False},
        watch.RELEASE_URL: {"tag_name": "v0.29.0"},
    }
    text_payloads = {
        watch.WNA16_URL: f"reason = '{watch.AUTOAWQ_TRITON_REJECTION}'",
        watch.HYPERLOOM_README_URL: "Supported platforms: MI300X, MI325X, MI355X",
    }

    monkeypatch.setattr(
        watch,
        "_fetch_json",
        lambda url, timeout: json_payloads[url],
    )
    monkeypatch.setattr(
        watch,
        "_fetch_text",
        lambda url, timeout: text_payloads[url],
    )

    snapshot = watch.collect_live_snapshot(timeout=1)
    result = evaluate(snapshot)

    assert result["review_required"] is False
    assert snapshot["vllm"]["autoawq_triton_rejection_present"] is True
    assert snapshot["hyperloom"]["declares_r9700_or_gfx1201_support"] is False


def test_load_snapshot_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    try:
        watch.load_snapshot(path)
    except watch.SnapshotError as exc:
        assert str(exc) == "snapshot:invalid_json"
    else:
        raise AssertionError("invalid JSON must fail closed")


def test_repository_baseline_fixture_is_valid():
    baseline = watch.load_snapshot(watch.DEFAULT_BASELINE)

    assert baseline == json.loads(watch.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    assert evaluate(baseline)["review_required"] is False
