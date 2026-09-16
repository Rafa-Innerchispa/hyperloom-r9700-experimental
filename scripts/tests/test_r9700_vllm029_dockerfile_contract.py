from __future__ import annotations

from pathlib import Path

from scripts import r9700_vllm029_gpu_runtime_plan as plan

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile.r9700-vllm029-rocm10"


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_pins_exact_rocm10_build_identity():
    text = dockerfile_text()

    assert f"ARG BASE_IMAGE={plan.BUILD_BASE_REF}" in text
    assert plan.BUILD_BASE_DIGEST in text
    assert f"ARG VLLM_SHA={plan.VLLM_SHA}" in text
    assert f"ARG TRITON_REPO={plan.TRITON_REPO}" in text
    assert f"ARG TRITON_SHA={plan.TRITON_SHA}" in text
    assert f"ARG PYTORCH_ROCM_ARCH={plan.GPU_ARCH}" in text
    assert "torch == 2.13.0" in text
    assert "rustup toolchain install 1.95" in text
    assert 'git clone "${TRITON_REPO}" rocm-triton' in text
    assert 'git checkout --detach "${TRITON_SHA}"' in text
    assert "import triton" in text


def test_dockerfile_verifies_exact_candidate_patch_before_applying():
    text = dockerfile_text()

    assert plan.PATCH_SHA256 in text
    assert "sha256sum --check --strict" in text
    assert "git apply --check --verbose /tmp/r9700-vllm029.patch" in text
    assert "git diff --check" in text
    assert 'test "$(git diff --name-only | sort | wc -l)" -eq 5' in text


def test_dockerfile_builds_source_instead_of_installing_vllm_wheel():
    text = dockerfile_text()

    assert "git clone https://github.com/vllm-project/vllm.git vllm" in text
    assert "python3 setup.py develop" in text
    assert "pip install vllm==" not in text
    assert "VLLM_TARGET_DEVICE=rocm" in text


def test_dockerfile_does_not_enable_aiter_or_touch_runtime_services():
    text = dockerfile_text().lower()

    assert "vllm_rocm_use_aiter=1" not in text
    assert "systemctl" not in text
    assert "docker rm" not in text
    assert "--port 8000" not in text
    assert "inneros-vllm-hyperloom-s3-production" not in text


def test_dockerfile_remains_passive_until_runtime_gate():
    text = dockerfile_text()

    assert 'ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]' in text
    assert "physical gfx1201 identity" in text
    assert "canonical correctness" in text
