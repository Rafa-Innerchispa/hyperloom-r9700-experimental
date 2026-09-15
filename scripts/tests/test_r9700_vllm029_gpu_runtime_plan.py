from __future__ import annotations

from scripts import r9700_vllm029_gpu_runtime_plan as plan


def test_default_plan_is_exact_isolated_and_non_promoting():
    payload = plan.build_plan()

    assert payload["pass"] is True, payload
    assert payload["errors"] == []
    assert payload["schema"] == "hyperloom.r9700.vllm029.gpu_runtime_plan.v1"
    assert payload["candidate"]["vllm_sha"] == "98dff2a81d747d1dba01a47f939f48c3526d4206"
    assert payload["candidate"]["patch_sha256"] == (
        "372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9"
    )
    assert payload["candidate"]["observed_patch_sha256"] == payload["candidate"]["patch_sha256"]
    assert payload["candidate"]["arch"] == "gfx1201"
    assert payload["candidate"]["rocm_generation"] == "10.0"
    assert payload["candidate"]["port"] == 18029
    assert payload["candidate"]["port"] not in plan.FORBIDDEN_PORTS
    assert payload["preserved_runtime"]["must_not_be_mutated"] is True

    stages = {stage["id"]: stage for stage in payload["stages"]}
    assert stages["load"]["requirements"]["aiter"] is False
    assert stages["load"]["requirements"]["attention_backend"] == "ROCM_ATTN"
    assert stages["path_evidence"]["requirements"]["interleave_marker"] == "tl.interleave"
    assert stages["warmup_correctness"]["requirements"]["exact_hash_match_required"] is True
    assert stages["promotion"]["requirements"]["automatic_promotion"] is False
    assert stages["promotion"]["requirements"]["stock_fallback_remains_default"] is True


def test_stage_order_requires_correctness_before_performance_and_promotion():
    payload = plan.build_plan()
    order = [stage["id"] for stage in payload["stages"]]

    assert order == [
        "source_identity",
        "build",
        "gpu_preflight",
        "isolation",
        "import_abi",
        "load",
        "path_evidence",
        "warmup_correctness",
        "bounded_performance",
        "promotion",
    ]
    assert order.index("path_evidence") < order.index("bounded_performance")
    assert order.index("warmup_correctness") < order.index("bounded_performance")
    assert order.index("bounded_performance") < order.index("promotion")


def test_preserved_ports_are_rejected():
    for port in sorted(plan.FORBIDDEN_PORTS):
        payload = plan.build_plan(candidate_port=port)
        assert payload["pass"] is False
        assert "candidate_port_collides_with_preserved_runtime" in payload["errors"]


def test_preserved_container_is_rejected():
    payload = plan.build_plan(candidate_container=plan.PRODUCTION_CONTAINER)

    assert payload["pass"] is False
    assert "candidate_container_collides_with_preserved_runtime" in payload["errors"]


def test_invalid_port_and_nonisolated_root_fail_closed():
    payload = plan.build_plan(candidate_port=0, candidate_root=".")

    assert payload["pass"] is False
    assert "candidate_port_out_of_range" in payload["errors"]
    assert "candidate_root_not_isolated" in payload["errors"]


def test_truth_boundary_does_not_claim_gpu_success():
    payload = plan.build_plan()

    assert "does not mean the GPU runtime works" in payload["truth_boundary"]
    assert payload["stages"][2]["gate"] == "prove_physical_target_identity"
