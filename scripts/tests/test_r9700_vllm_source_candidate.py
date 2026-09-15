from __future__ import annotations

from pathlib import Path

from scripts import r9700_vllm_source_candidate as candidate


def test_repository_patch_is_structurally_valid():
    result = candidate.evaluate_patch()

    assert result["ok"] is True, result
    assert result["review_required"] is False
    assert result["runtime_mutation"] is False
    assert result["automatic_promotion_allowed"] is False
    assert set(result["paths"]) == candidate.EXPECTED_PATHS
    assert len(result["patch_sha256"]) == 64
    assert result["patch_sha256"] == ("6e8aba7839dfd4bc1ba060ab2dad576d42f91ed4df0a7038bfb735d0d9f06b3c")


def test_target_is_pinned_vllm_029():
    result = candidate.evaluate_patch()

    assert result["ok"] is True, result
    assert result["target"]["tag"] == "v0.29.0"
    assert result["target"]["sha"] == "98dff2a81d747d1dba01a47f939f48c3526d4206"
    assert result["target"]["comparison_main_sha"] == ("e6960af33b379d502f409e3e2241bbf2b2c2f68d")
    assert result["target"]["pr_43389_head_sha"] == ("56ef89e1ff4a1552beb3b5c51c00b73ea44daca1")


def test_exact_five_file_scope_and_no_broad_autoawq_or_rocm_edit():
    parsed = candidate.parse_unified_patch(candidate.PATCH_PATH.read_text(encoding="utf-8"))
    paths = set(parsed["paths"])

    assert paths == candidate.EXPECTED_PATHS
    assert paths.isdisjoint(candidate.FORBIDDEN_PATHS)


def test_direct_autoawq_triton_rejection_is_not_touched():
    text = candidate.PATCH_PATH.read_text(encoding="utf-8")

    assert candidate.DIRECT_AUTOAWQ_REJECTION not in text
    assert "vllm/model_executor/layers/quantization/auto_awq.py" not in text


def test_classic_scalar_shift_path_is_preserved_in_candidate():
    text = candidate.PATCH_PATH.read_text(encoding="utf-8")

    assert "(offs_k[:, None] // 2) * stride_bk" in text
    assert "b_shifter = (offs_k[:, None] % 2) * 4" in text
    assert "if use_int4_w4a16 and not use_int4_interleave:" in text


def test_bit_repack_round_trip_matches_original_nibbles():
    original = [
        [
            [0x10, 0x32, 0x54],
            [0x76, 0x98, 0xBA],
            [0xDC, 0xFE, 0x01],
            [0x23, 0x45, 0x67],
            [0x89, 0xAB, 0xCD],
            [0xEF, 0x10, 0x32],
            [0x54, 0x76, 0x98],
            [0xBA, 0xDC, 0xFE],
        ]
    ]

    repacked = candidate.repack_int4_reference(original)
    expected = candidate.unpack_original_int4_reference(original)
    recovered = candidate.unpack_repacked_int4_reference(repacked, N=8)

    assert recovered == expected
    assert len(repacked) == 1
    assert len(repacked[0]) == 6
    assert len(repacked[0][0]) == 1


def test_zero_point_unpack_preserves_low_high_order():
    zp = [[[0x10, 0x32], [0x54, 0x76], [0x98, 0xBA], [0xDC, 0xFE]]]

    unpacked = candidate.unpack_zp_reference(zp)

    assert len(unpacked) == 1
    assert len(unpacked[0]) == 2
    assert len(unpacked[0][0]) == 8
    assert unpacked[0][0] == [0, 1, 4, 5, 8, 9, 12, 13]
    assert unpacked[0][1] == [2, 3, 6, 7, 10, 11, 14, 15]


def test_real_gate_proj_shapes_match_phase3_evidence():
    shapes = candidate.expected_real_shapes(N=768, K=2048, group_size=128)

    assert shapes == {
        "stock_uint8": (1, 768, 1024),
        "repacked_int32": (1, 2048, 96),
        "packed_zp_uint8": (1, 384, 16),
        "unpacked_zp": (1, 16, 768),
        "scale": (1, 16, 768),
    }


def test_real_down_proj_shapes_match_phase3_evidence():
    shapes = candidate.expected_real_shapes(N=2048, K=768, group_size=128)

    assert shapes == {
        "stock_uint8": (1, 2048, 384),
        "repacked_int32": (1, 768, 256),
        "packed_zp_uint8": (1, 1024, 6),
        "unpacked_zp": (1, 6, 2048),
        "scale": (1, 6, 2048),
    }


def test_corrupt_hunk_count_fails_closed(tmp_path: Path):
    text = candidate.PATCH_PATH.read_text(encoding="utf-8")
    broken = text.replace("@@ -0,0 +1,33 @@", "@@ -0,0 +1,34 @@", 1)
    path = tmp_path / "broken.patch"
    path.write_text(broken, encoding="utf-8")

    result = candidate.evaluate_patch(path)

    assert result["ok"] is False
    assert result["review_required"] is True
    assert result["reasons"][0].startswith("patch:hunk_count_mismatch:")


def test_forbidden_path_fails_closed(tmp_path: Path):
    text = candidate.PATCH_PATH.read_text(encoding="utf-8")
    extra = (
        "\ndiff --git a/vllm/platforms/rocm.py b/vllm/platforms/rocm.py\n"
        "--- a/vllm/platforms/rocm.py\n"
        "+++ b/vllm/platforms/rocm.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    path = tmp_path / "forbidden.patch"
    path.write_text(text + extra, encoding="utf-8")

    result = candidate.evaluate_patch(path)

    assert result["ok"] is False
    assert result["reasons"] == ["patch:paths_changed"]
