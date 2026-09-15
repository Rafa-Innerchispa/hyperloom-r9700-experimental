from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import r9700_vllm_source_apply_gate as gate


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _build_synthetic_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baselines = {
        "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": (
            f'AUTOAWQ_REASON = "{gate.AUTOAWQ_REJECTION}"\n'
        ).encode(),
        "vllm/model_executor/layers/fused_moe/config.py": b"BASE_CONFIG = True\n",
        "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": b"BASE_TRITON = True\n",
        "vllm/model_executor/layers/fused_moe/fused_moe.py": (
            "# (offs_k[:, None] // 2) * stride_bk\n# b_shifter = (offs_k[:, None] % 2) * 4\nBASE_FUSED = True\n"
        ).encode(),
    }
    monkeypatch.setattr(
        gate,
        "EXPECTED_BLOBS",
        {path: gate.git_blob_sha(data) for path, data in baselines.items()},
    )
    monkeypatch.setattr(
        gate,
        "EXPECTED_PATHS",
        frozenset((*baselines, gate.NEW_PATH)),
    )

    source = tmp_path / "source"
    for relative, data in baselines.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(["init", "-q"], source)
    _git(["add", "."], source)
    _git(
        [
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate-test@localhost",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        source,
    )

    (source / "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py").write_text(
        baselines["vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"].decode() + "NORMALIZED_TRITON = True\n",
        encoding="utf-8",
    )
    (source / "vllm/model_executor/layers/fused_moe/config.py").write_text(
        "BASE_CONFIG = True\nINTERLEAVED_CONFIG = True\n",
        encoding="utf-8",
    )
    (source / "vllm/model_executor/layers/fused_moe/experts/triton_moe.py").write_text(
        "BASE_TRITON = True\nINT4_LAYOUT = True\n",
        encoding="utf-8",
    )
    (source / "vllm/model_executor/layers/fused_moe/fused_moe.py").write_text(
        baselines["vllm/model_executor/layers/fused_moe/fused_moe.py"].decode()
        + "# use_int4_interleave\n# tl.interleave\n",
        encoding="utf-8",
    )
    utility = source / gate.NEW_PATH
    utility.parent.mkdir(parents=True, exist_ok=True)
    utility.write_text(
        "def repack_int4_to_int32(value):\n    return value\n",
        encoding="utf-8",
    )
    _git(["add", "-N", "."], source)
    patch = tmp_path / "candidate.patch"
    patch.write_text(_git(["diff", "--binary"], source), encoding="utf-8")
    monkeypatch.setattr(gate, "PATCH_SHA256", gate.sha256_file(patch))

    def fake_fetcher(url: str) -> bytes:
        for relative, data in baselines.items():
            if url.endswith(relative):
                return data
        raise AssertionError(f"unexpected URL: {url}")

    return patch, fake_fetcher


def test_actual_candidate_patch_identity_and_scope():
    assert gate.sha256_file(gate.DEFAULT_PATCH) == gate.PATCH_SHA256
    patch_text = gate.DEFAULT_PATCH.read_text(encoding="utf-8")
    assert gate.parse_patch_paths(patch_text) == gate.EXPECTED_PATHS


def test_git_blob_sha_matches_known_git_object_identity():
    assert gate.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_parse_patch_paths_rejects_rename():
    with pytest.raises(gate.GateError, match="patch:path_rename_not_allowed"):
        gate.parse_patch_paths("diff --git a/old.py b/new.py\n")


def test_materialize_baseline_verifies_blob_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = b"VALUE = 1\n"
    monkeypatch.setattr(gate, "EXPECTED_BLOBS", {"vllm/example.py": gate.git_blob_sha(data)})
    monkeypatch.setattr(gate, "NEW_PATH", "vllm/new.py")

    observed = gate._materialize_baseline(tmp_path, fetcher=lambda _url: data)

    assert observed == {"vllm/example.py": gate.git_blob_sha(data)}
    assert (tmp_path / "vllm/example.py").read_bytes() == data
    assert not (tmp_path / "vllm/new.py").exists()


def test_materialize_baseline_fails_closed_on_blob_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gate, "EXPECTED_BLOBS", {"vllm/example.py": "0" * 40})
    monkeypatch.setattr(gate, "NEW_PATH", "vllm/new.py")

    with pytest.raises(gate.GateError, match="source:blob_mismatch:vllm/example.py"):
        gate._materialize_baseline(tmp_path, fetcher=lambda _url: b"VALUE = 1\n")


def test_synthetic_clean_source_apply_gate_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    patch, fetcher = _build_synthetic_candidate(tmp_path, monkeypatch)

    result = gate.evaluate_live(patch, fetcher=fetcher)

    assert result["ok"] is True
    assert result["review_required"] is False
    assert result["apply_check"] == "PASS"
    assert result["apply"] == "PASS"
    assert result["diff_check"] == "PASS"
    assert result["py_compile"] == "PASS"
    assert result["changed_paths"] == sorted(gate.EXPECTED_PATHS)
    assert result["direct_autoawq_triton_rejection_preserved"] is True
    assert result["classic_wna16_path_preserved"] is True
    assert result["runtime_mutation"] is False
    assert result["gpu_runtime_tested"] is False
    assert result["automatic_promotion_allowed"] is False


def test_synthetic_gate_rejects_patch_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    patch, fetcher = _build_synthetic_candidate(tmp_path, monkeypatch)
    monkeypatch.setattr(gate, "PATCH_SHA256", "0" * 64)

    with pytest.raises(gate.GateError, match="patch:sha256_mismatch"):
        gate.evaluate_live(patch, fetcher=fetcher)


def test_fail_result_is_fail_closed():
    result = gate.fail_result("test_failure")

    assert result["ok"] is False
    assert result["review_required"] is True
    assert result["reason"] == "test_failure"
    assert result["runtime_mutation"] is False
    assert result["gpu_runtime_tested"] is False
    assert result["automatic_promotion_allowed"] is False


def test_main_without_live_writes_fail_closed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["gate", "--output", str(output)])

    assert gate.main() == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["reason"] == "live_mode_required"
    assert result["runtime_mutation"] is False


def test_patch_sha_constant_is_sha256_shape():
    assert len(gate.PATCH_SHA256) == hashlib.sha256().digest_size * 2
