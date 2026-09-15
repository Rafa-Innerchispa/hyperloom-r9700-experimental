#!/usr/bin/env python3
"""Fail-closed clean-source apply gate for the R9700 vLLM v0.29.0 candidate.

The gate downloads only the immutable upstream blobs touched by the candidate,
verifies their Git blob IDs, builds a disposable minimal source tree, executes
``git apply --check`` and then applies the patch. It never imports or mutates a
serving vLLM installation and it never authorizes promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

SCHEMA = "r9700-vllm029-source-apply-v1"
TARGET_VLLM_TAG = "v0.29.0"
TARGET_VLLM_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"
PATCH_SHA256 = "372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH = ROOT / "docs" / "evidence" / "vllm_v0_29_0_r9700_awq_triton_candidate.patch"
DEFAULT_OUTPUT = ROOT / "r9700-vllm-source-apply-result.json"

EXPECTED_BLOBS = {
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": "c56712ac58bc9933165ccf04753b9ee1a6b9d45a",
    "vllm/model_executor/layers/fused_moe/config.py": "8736b1393d82590c1adb9a8ca91841ab2337e555",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "9f982644b5691f8bccde0aa5229c404d394882f9",
    "vllm/model_executor/layers/fused_moe/fused_moe.py": "dc619b2b12d27d2955f38a656e7c524fd6042887",
}
NEW_PATH = "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py"
EXPECTED_PATHS = frozenset((*EXPECTED_BLOBS, NEW_PATH))
RAW_URL = "https://raw.githubusercontent.com/vllm-project/vllm/{sha}/{path}"
AUTOAWQ_REJECTION = "the AutoAWQ weight layout is not supported"
CLASSIC_WEIGHT_MARKERS = (
    "(offs_k[:, None] // 2) * stride_bk",
    "b_shifter = (offs_k[:, None] % 2) * 4",
)
INTERLEAVE_MARKERS = (
    "use_int4_interleave",
    "tl.interleave",
)


class GateError(RuntimeError):
    """Controlled fail-closed error."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise GateError(code)


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "hyperloom-r9700-vllm-source-apply/1"},
        method="GET",
    )


def fetch_bytes(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        data = response.read(MAX_SOURCE_BYTES + 1)
    _require(len(data) <= MAX_SOURCE_BYTES, "source:too_large")
    return data


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_patch_paths(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        parts = line.split()
        _require(len(parts) == 4, "patch:diff_header")
        old_path, new_path = parts[2], parts[3]
        _require(old_path.startswith("a/") and new_path.startswith("b/"), "patch:path_prefix")
        old_rel, new_rel = old_path[2:], new_path[2:]
        _require(old_rel == new_rel, "patch:path_rename_not_allowed")
        paths.add(new_rel)
    return paths


def _materialize_baseline(
    tree: Path,
    *,
    fetcher: Callable[[str], bytes],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected_blob in EXPECTED_BLOBS.items():
        url = RAW_URL.format(sha=TARGET_VLLM_SHA, path=relative)
        data = fetcher(url)
        blob = git_blob_sha(data)
        _require(blob == expected_blob, f"source:blob_mismatch:{relative}")
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        observed[relative] = blob
    (tree / NEW_PATH).parent.mkdir(parents=True, exist_ok=True)
    _require(not (tree / NEW_PATH).exists(), "source:new_path_already_exists")
    return observed


def _git_or_raise(args: list[str], *, cwd: Path, code: str) -> subprocess.CompletedProcess[str]:
    result = run_git(args, cwd=cwd)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = " | ".join(line[:300] for line in detail[-8:]) if detail else "git_failed"
        raise GateError(f"{code}:{suffix[:1800]}")
    return result


def _changed_paths(tree: Path) -> set[str]:
    _git_or_raise(["add", "-N", "."], cwd=tree, code="git:add_intent")
    result = _git_or_raise(["diff", "--name-only"], cwd=tree, code="git:diff_names")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _compile_paths(tree: Path) -> None:
    targets = [str(tree / path) for path in sorted(EXPECTED_PATHS)]
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", *targets],
        cwd=tree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = detail[-1][:300] if detail else "compile_failed"
        raise GateError(f"python:compile:{suffix}")


def evaluate_live(
    patch_path: Path = DEFAULT_PATCH,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    patch_path = patch_path.resolve()
    _require(patch_path.is_file(), "patch:not_found")
    patch_sha = sha256_file(patch_path)
    _require(patch_sha == PATCH_SHA256, f"patch:sha256_mismatch:{patch_sha}")
    patch_text = patch_path.read_text(encoding="utf-8")
    patch_paths = parse_patch_paths(patch_text)
    _require(patch_paths == EXPECTED_PATHS, "patch:scope_mismatch")

    if fetcher is None:
        fetcher = lambda url: fetch_bytes(url, timeout=timeout)

    with tempfile.TemporaryDirectory(prefix="r9700-vllm029-apply-") as tmp:
        tree = Path(tmp)
        source_blobs = _materialize_baseline(tree, fetcher=fetcher)

        _git_or_raise(["init", "-q"], cwd=tree, code="git:init")
        _git_or_raise(["add", "."], cwd=tree, code="git:add_baseline")
        _git_or_raise(
            [
                "-c",
                "user.name=R9700 Apply Gate",
                "-c",
                "user.email=r9700-apply-gate@localhost",
                "commit",
                "-q",
                "-m",
                "vllm-v0.29.0-minimal-baseline",
            ],
            cwd=tree,
            code="git:commit_baseline",
        )

        _git_or_raise(["apply", "--check", "--verbose", str(patch_path)], cwd=tree, code="git:apply_check")
        _git_or_raise(["apply", str(patch_path)], cwd=tree, code="git:apply")
        _git_or_raise(["diff", "--check"], cwd=tree, code="git:diff_check")

        changed = _changed_paths(tree)
        _require(changed == EXPECTED_PATHS, "apply:changed_paths_mismatch")

        oracle = (tree / "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py").read_text(encoding="utf-8")
        fused = (tree / "vllm/model_executor/layers/fused_moe/fused_moe.py").read_text(encoding="utf-8")
        utility = (tree / NEW_PATH).read_text(encoding="utf-8")
        _require(AUTOAWQ_REJECTION in oracle, "guard:autoawq_rejection_missing")
        _require(all(marker in fused for marker in CLASSIC_WEIGHT_MARKERS), "guard:classic_int4_path_missing")
        _require(all(marker in fused for marker in INTERLEAVE_MARKERS), "guard:interleave_path_missing")
        _require("def repack_int4_to_int32(" in utility, "guard:repack_utility_missing")
        _compile_paths(tree)

        return {
            "schema": SCHEMA,
            "ok": True,
            "review_required": False,
            "target_vllm_tag": TARGET_VLLM_TAG,
            "target_vllm_sha": TARGET_VLLM_SHA,
            "patch_sha256": patch_sha,
            "source_blobs": source_blobs,
            "apply_check": "PASS",
            "apply": "PASS",
            "diff_check": "PASS",
            "py_compile": "PASS",
            "changed_paths": sorted(changed),
            "direct_autoawq_triton_rejection_preserved": True,
            "classic_wna16_path_preserved": True,
            "runtime_mutation": False,
            "gpu_runtime_tested": False,
            "automatic_promotion_allowed": False,
        }


def fail_result(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "review_required": True,
        "reason": reason,
        "target_vllm_tag": TARGET_VLLM_TAG,
        "target_vllm_sha": TARGET_VLLM_SHA,
        "patch_sha256": PATCH_SHA256,
        "runtime_mutation": False,
        "gpu_runtime_tested": False,
        "automatic_promotion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Fetch immutable upstream blobs and run the clean apply gate")
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    if not args.live:
        result = fail_result("live_mode_required")
    else:
        try:
            result = evaluate_live(args.patch, timeout=args.timeout)
        except (GateError, OSError, UnicodeError, urllib.error.URLError, ValueError) as exc:
            reason = str(exc) if isinstance(exc, GateError) else "source_apply:unavailable"
            result = fail_result(reason)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
