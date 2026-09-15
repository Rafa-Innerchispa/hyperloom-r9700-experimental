#!/usr/bin/env python3
"""Static verifier for the isolated R9700 vLLM source patch candidate.

No torch/vLLM dependency is required. The module validates the unified patch
artifact and independently checks the int4/ZP bit-layout transformation. It
never applies a patch to a serving tree or authorizes promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "r9700-vllm-source-candidate-v1"
PATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "vllm_v0_29_0_r9700_awq_triton_candidate.patch"
)
TARGET_VLLM_TAG = "v0.29.0"
TARGET_VLLM_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"
COMPARISON_MAIN_SHA = "e6960af33b379d502f409e3e2241bbf2b2c2f68d"
PR_43389_HEAD_SHA = "56ef89e1ff4a1552beb3b5c51c00b73ea44daca1"

TARGET_BLOBS = {
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": "c56712ac58bc9933165ccf04753b9ee1a6b9d45a",
    "vllm/model_executor/layers/fused_moe/config.py": "8736b1393d82590c1adb9a8ca91841ab2337e555",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "9f982644b5691f8bccde0aa5229c404d394882f9",
    "vllm/model_executor/layers/fused_moe/fused_moe.py": "dc619b2b12d27d2955f38a656e7c524fd6042887",
}
EXPECTED_PATHS = {
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
    "vllm/model_executor/layers/fused_moe/config.py",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
    "vllm/model_executor/layers/fused_moe/fused_moe.py",
}
FORBIDDEN_PATHS = {
    "vllm/model_executor/layers/quantization/auto_awq.py",
    "vllm/platforms/rocm.py",
}
DIRECT_AUTOAWQ_REJECTION = "the AutoAWQ weight layout is not supported"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")
DIFF_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")


class CandidateError(ValueError):
    """Controlled patch validation failure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CandidateError(code)


def _count_value(raw: str | None) -> int:
    return 1 if raw is None else int(raw)


def hunk_count_mismatches(text: str) -> list[dict[str, Any]]:
    """Return every hunk whose declared old/new counts do not match its body."""
    lines = text.splitlines()
    mismatches: list[dict[str, Any]] = []
    current_path = "<unknown>"
    i = 0
    while i < len(lines):
        diff_match = DIFF_RE.match(lines[i])
        if diff_match:
            current_path = diff_match.group(1)
            i += 1
            continue
        if not lines[i].startswith("@@ "):
            i += 1
            continue
        header = lines[i]
        match = HUNK_RE.match(header)
        if match is None:
            mismatches.append({"path": current_path, "header": header, "error": "invalid_header"})
            i += 1
            continue
        old_expected = _count_value(match.group(2))
        new_expected = _count_value(match.group(4))
        old_actual = 0
        new_actual = 0
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git "):
            body_line = lines[i]
            if body_line and not body_line.startswith("\\ No newline"):
                prefix = body_line[0]
                if prefix in {" ", "-"}:
                    old_actual += 1
                if prefix in {" ", "+"}:
                    new_actual += 1
            i += 1
        if old_actual != old_expected or new_actual != new_expected:
            mismatches.append(
                {
                    "path": current_path,
                    "header": header,
                    "old_expected": old_expected,
                    "old_actual": old_actual,
                    "new_expected": new_expected,
                    "new_actual": new_actual,
                }
            )
    return mismatches


def parse_unified_patch(text: str) -> dict[str, Any]:
    """Parse paths and enforce syntactically consistent unified-diff hunks."""
    mismatches = hunk_count_mismatches(text)
    _require(
        not mismatches,
        "patch:hunk_count_mismatch:" + json.dumps(mismatches, sort_keys=True),
    )
    files: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = DIFF_RE.match(line)
        if match:
            _require(match.group(1) == match.group(2), "patch:rename_not_allowed")
            files.append({"path": match.group(1)})
    _require(bool(files), "patch:no_files")
    paths = [item["path"] for item in files]
    _require(len(paths) == len(set(paths)), "patch:duplicate_file")
    return {"files": files, "paths": paths}


def patch_sha256(path: Path = PATCH_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_patch(path: Path = PATCH_PATH) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        parsed = parse_unified_patch(text)
        paths = set(parsed["paths"])
        _require(paths == EXPECTED_PATHS, "patch:paths_changed")
        _require(not (paths & FORBIDDEN_PATHS), "patch:forbidden_path")
        _require(DIRECT_AUTOAWQ_REJECTION not in text, "patch:direct_autoawq_guard_touched")
        required_markers = (
            "def repack_int4_to_int32",
            "def unpack_zp_int4_to_fp16",
            "def is_int4_w4a16_interleaved",
            "if isinstance(quant_config, MoeWNA16Config):",
            "if num_bits == 4 and on_rdna():",
            "use_int4_interleave = use_int4_w4a16 and B.dtype == torch.int32",
            "b = tl.interleave(b, b)",
            "if use_int4_w4a16 and not use_int4_interleave:",
            "int4_packed_as_int32=_interleave",
        )
        for marker in required_markers:
            _require(marker in text, f"patch:missing_marker:{marker}")
        _require(
            "(offs_k[:, None] // 2) * stride_bk" in text
            and "b_shifter = (offs_k[:, None] % 2) * 4" in text,
            "patch:classic_int4_path_missing",
        )
    except (OSError, UnicodeDecodeError, CandidateError) as exc:
        code = str(exc) if isinstance(exc, CandidateError) else "patch:unavailable"
        return {
            "schema": SCHEMA,
            "ok": False,
            "review_required": True,
            "reasons": [code],
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
        "patch_sha256": patch_sha256(path),
        "target": {
            "tag": TARGET_VLLM_TAG,
            "sha": TARGET_VLLM_SHA,
            "comparison_main_sha": COMPARISON_MAIN_SHA,
            "pr_43389_head_sha": PR_43389_HEAD_SHA,
            "source_blobs": TARGET_BLOBS,
        },
        "paths": sorted(EXPECTED_PATHS),
    }


def repack_int4_reference(w: list[list[list[int]]]) -> list[list[list[int]]]:
    """Pure-Python [E,N,K/2] bytes -> [E,K,N/8] packed 32-bit values."""
    E = len(w)
    _require(E > 0, "repack:empty_e")
    N = len(w[0])
    _require(N > 0 and N % 8 == 0, "repack:n_not_divisible_by_8")
    K_half = len(w[0][0])
    _require(K_half > 0, "repack:empty_k")
    K = K_half * 2
    out = [[[0 for _ in range(N // 8)] for _ in range(K)] for _ in range(E)]
    for e in range(E):
        _require(len(w[e]) == N, "repack:ragged_n")
        for n in range(N):
            _require(len(w[e][n]) == K_half, "repack:ragged_k")
            for kh, byte in enumerate(w[e][n]):
                _require(isinstance(byte, int) and 0 <= byte <= 255, "repack:byte_range")
                for nibble_index, value in enumerate((byte & 0xF, (byte >> 4) & 0xF)):
                    k = kh * 2 + nibble_index
                    out[e][k][n // 8] |= value << ((n % 8) * 4)
    return out


def unpack_repacked_int4_reference(
    packed: list[list[list[int]]], N: int
) -> list[list[list[int]]]:
    E = len(packed)
    K = len(packed[0])
    out = [[[0 for _ in range(K)] for _ in range(N)] for _ in range(E)]
    for e in range(E):
        for k in range(K):
            for n in range(N):
                out[e][n][k] = (packed[e][k][n // 8] >> ((n % 8) * 4)) & 0xF
    return out


def unpack_original_int4_reference(w: list[list[list[int]]]) -> list[list[list[int]]]:
    E = len(w)
    N = len(w[0])
    K_half = len(w[0][0])
    out = [[[0 for _ in range(K_half * 2)] for _ in range(N)] for _ in range(E)]
    for e in range(E):
        for n in range(N):
            for kh, byte in enumerate(w[e][n]):
                out[e][n][kh * 2] = byte & 0xF
                out[e][n][kh * 2 + 1] = (byte >> 4) & 0xF
    return out


def unpack_zp_reference(zp: list[list[list[int]]]) -> list[list[list[int]]]:
    """Pure-Python [E,N/2,K_groups] bytes -> [E,K_groups,N] int4 values."""
    E = len(zp)
    N_half = len(zp[0])
    K_groups = len(zp[0][0])
    out = [[[0 for _ in range(N_half * 2)] for _ in range(K_groups)] for _ in range(E)]
    for e in range(E):
        for nh in range(N_half):
            for kg, byte in enumerate(zp[e][nh]):
                _require(isinstance(byte, int) and 0 <= byte <= 255, "zp:byte_range")
                out[e][kg][nh * 2] = byte & 0xF
                out[e][kg][nh * 2 + 1] = (byte >> 4) & 0xF
    return out


def expected_real_shapes(N: int, K: int, group_size: int) -> dict[str, tuple[int, ...]]:
    _require(N % 8 == 0, "shape:n_not_divisible_by_8")
    _require(K % group_size == 0, "shape:k_not_divisible_by_group")
    return {
        "stock_uint8": (1, N, K // 2),
        "repacked_int32": (1, K, N // 8),
        "packed_zp_uint8": (1, N // 2, K // group_size),
        "unpacked_zp": (1, K // group_size, N),
        "scale": (1, K // group_size, N),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=Path, default=PATCH_PATH)
    parser.add_argument("--diagnose-hunks", action="store_true")
    args = parser.parse_args()
    if args.diagnose_hunks:
        text = args.patch.read_text(encoding="utf-8")
        print(json.dumps(hunk_count_mismatches(text), sort_keys=True))
        return 0
    result = evaluate_patch(args.patch)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
