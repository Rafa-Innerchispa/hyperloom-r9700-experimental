from pathlib import Path


WORKFLOW = Path(".github/workflows/r9700-upstream-watch.yml")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_r9700_upstream_watch_has_daily_manual_and_scoped_pr_triggers():
    text = workflow_text()

    assert 'cron: "17 12 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "pull_request:" in text
    assert '      - ".github/workflows/r9700-upstream-watch.yml"' in text
    assert '      - "scripts/r9700_upstream_watch.py"' in text
    assert '      - "scripts/tests/test_r9700_upstream_watch_workflow.py"' in text
    assert '      - "docs/evidence/r9700_upstream_watch_baseline_20260914.json"' in text


def test_r9700_upstream_watch_is_read_only_and_uses_hosted_runner():
    text = workflow_text()

    assert "permissions:\n  contents: read\n" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "secrets." not in text

    forbidden_write_permissions = (
        "actions: write",
        "checks: write",
        "contents: write",
        "issues: write",
        "packages: write",
        "pull-requests: write",
        "id-token: write",
    )
    assert not any(permission in text for permission in forbidden_write_permissions)


def test_r9700_upstream_watch_runs_live_sentinel_and_preserves_evidence():
    text = workflow_text()

    assert "python scripts/r9700_upstream_watch.py --live" in text
    assert "r9700-upstream-watch.json" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "actions/upload-artifact@v7" in text
    assert "if-no-files-found: error" in text
    assert text.count("if: always()") >= 2


def test_r9700_upstream_watch_propagates_fail_closed_exit_code_last():
    text = workflow_text()

    assert 'echo "exit_code=$rc" >> "$GITHUB_OUTPUT"' in text
    assert "SENTINEL_EXIT_CODE: ${{ steps.sentinel.outputs.exit_code }}" in text
    assert 'exit "$SENTINEL_EXIT_CODE"' in text
    assert text.index("Upload sentinel evidence") < text.index("Enforce fail-closed sentinel verdict")


def test_r9700_upstream_watch_deduplicates_same_ref_runs():
    text = workflow_text()

    assert "group: r9700-upstream-watch-${{ github.ref }}" in text
    assert "cancel-in-progress: true" in text
