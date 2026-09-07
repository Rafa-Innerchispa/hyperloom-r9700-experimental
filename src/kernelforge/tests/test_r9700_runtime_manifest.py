from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "r9700_runtime_manifest.py"
SPEC = importlib.util.spec_from_file_location("r9700_runtime_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def test_redact_removes_common_token_shapes() -> None:
    text = "cursor-agent --api-key crsr_abc123 token=secretvalue Authorization BearerValue"
    redacted = MANIFEST.redact(text)
    assert "crsr_abc123" not in redacted
    assert "secretvalue" not in redacted
    assert "BearerValue" not in redacted
    assert "REDACTED" in redacted


def test_build_manifest_records_truth_boundary_and_runtime_identity() -> None:
    def fake_request(_url: str):
        return {
            "data": [
                {
                    "id": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
                    "root": "/models/qwen",
                    "max_model_len": 8192,
                }
            ]
        }

    def fake_run(argv: list[str]):
        if argv[:3] == ["docker", "ps", "--format"]:
            return {
                "returncode": 0,
                "stdout": '{"ID":"abc","Image":"rocm/vllm:rocm10","Names":"inneros-vllm-canary-rocm10","Command":"python3 -m vllm.entrypoints.openai.api_server","Ports":""}\n',
            }
        if argv[:2] == ["docker", "exec"]:
            return {"returncode": 0, "stdout": '{"python":"3.14.7","torch_hip":"7.15.26333","torch_version":"2.12.0+rocm10.0.0","vllm_version":"0.27.1"}'}
        if argv and argv[0] == "rocm-smi":
            return {
                "returncode": 0,
                "stdout": '{"card0":{"Card Series":"AMD Radeon AI PRO R9700","GFX Version":"gfx1201"}}',
            }
        if argv and argv[0] == "ps":
            return {
                "returncode": 0,
                "stdout": "123 1 python3 -m vllm.entrypoints.openai.api_server --model /models/qwen --api-key crsr_abc\n",
            }
        raise AssertionError(argv)

    manifest = MANIFEST.build_manifest(base_url="http://127.0.0.1:8000/v1", request_json=fake_request, run=fake_run)
    assert manifest["schema"] == MANIFEST.SCHEMA
    assert manifest["model"]["selected_model"].endswith("-AWQ")
    assert manifest["truth_boundary"]["hardware_label"] == "AMD Radeon AI PRO R9700 workstation GPU, gfx1201"
    assert manifest["truth_boundary"]["concurrency_result_class"].startswith("serving/concurrency")
    assert manifest["runtime_versions"]["payload"]["torch_version"] == "2.12.0+rocm10.0.0"
    assert "crsr_abc" not in manifest["launch_processes"][0]["cmd"]
    assert manifest["manifest_sha256"] == MANIFEST.stable_json_sha256(manifest)
