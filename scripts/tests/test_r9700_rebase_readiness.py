from __future__ import annotations

from scripts import r9700_rebase_readiness as readiness


def _autoawq() -> str:
    return "\n".join(readiness.REQUIRED_AUTOAWQ_MARKERS)


def _oracle(*, blocked: bool = True) -> str:
    text = "\n".join(readiness.REQUIRED_ORACLE_MARKERS)
    if blocked:
        text += f"\n{readiness.AUTOAWQ_TRITON_REJECTION}\n"
    return text


def _hyperloom(*, r9700: bool = False) -> str:
    text = " ".join(readiness.HYPERLOOM_SUPPORTED_ACCELERATORS)
    if r9700:
        text += " Radeon AI PRO R9700 gfx1201"
    return text


def test_current_shape_requires_rebase_adapt_but_keeps_overlay():
    result = readiness.assess_sources(
        autoawq=_autoawq(),
        oracle=_oracle(blocked=True),
        hyperloom_readme=_hyperloom(),
    )

    assert result["runtime_mutation"] is False
    assert result["decision"] == "rebase-adapt"
    assert result["overlay_equivalent_upstream"] is False
    assert result["observed"]["autoawq_triton_blocked"] is True
    assert "autoawq_triton_still_blocked" in result["reasons"]


def test_upstream_equivalence_requires_triton_block_to_disappear():
    result = readiness.assess_sources(
        autoawq=_autoawq(),
        oracle=_oracle(blocked=False),
        hyperloom_readme=_hyperloom(),
    )

    assert result["decision"] == "rebase-adapt"
    assert result["overlay_equivalent_upstream"] is True
    assert result["observed"]["autoawq_triton_blocked"] is False


def test_missing_autoawq_api_fails_closed_to_retain():
    result = readiness.assess_sources(
        autoawq="class AutoAWQMoEMethod",
        oracle=_oracle(blocked=True),
        hyperloom_readme=_hyperloom(),
    )

    assert result["decision"] == "retain"
    assert result["overlay_equivalent_upstream"] is False
    assert "autoawq_api_surface_changed" in result["reasons"]


def test_missing_oracle_api_fails_closed_to_retain():
    result = readiness.assess_sources(
        autoawq=_autoawq(),
        oracle=readiness.AUTOAWQ_TRITON_REJECTION,
        hyperloom_readme=_hyperloom(),
    )

    assert result["decision"] == "retain"
    assert "wna16_oracle_api_surface_changed" in result["reasons"]


def test_hyperloom_r9700_declaration_is_observed_without_claiming_equivalence():
    result = readiness.assess_sources(
        autoawq=_autoawq(),
        oracle=_oracle(blocked=True),
        hyperloom_readme=_hyperloom(r9700=True),
    )

    assert result["observed"]["hyperloom_declares_r9700_or_gfx1201"] is True
    assert result["overlay_equivalent_upstream"] is False


def test_hyperloom_supported_platform_table_drift_requests_review():
    result = readiness.assess_sources(
        autoawq=_autoawq(),
        oracle=_oracle(blocked=True),
        hyperloom_readme="MI300X only",
    )

    assert result["review_required"] is True
    assert "hyperloom_supported_platform_table_changed" in result["reasons"]
