from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r9700_awq_backend_probe as probe


class R9700AWQBackendProbeTests(unittest.TestCase):
    def test_cli_flag_reads_value_and_default(self) -> None:
        argv = ["python3", "-m", "vllm", "--model", "/models/demo", "--dtype", "float16"]
        self.assertEqual(probe.cli_flag(argv, "--model"), "/models/demo")
        self.assertEqual(probe.cli_flag(argv, "--dtype"), "float16")
        self.assertEqual(probe.cli_flag(argv, "--max-model-len", "8192"), "8192")

    def test_last_json_object_ignores_vllm_log_lines(self) -> None:
        text = "INFO startup\nWARNING something\n{\"backend\":\"EMULATION\"}\n"
        self.assertEqual(probe.last_json_object(text), {"backend": "EMULATION"})

    def test_classify_linear_backend_custom_op(self) -> None:
        result = probe.classify_linear_backend(
            use_triton_awq=False,
            awq_gemm_source="return torch.ops._C.awq_gemm(input, qweight, scales, qzeros, split_k_iters)",
        )
        self.assertEqual(result["backend"], "VLLM_CUSTOM_OP__C_AWQ")
        self.assertEqual(result["kernel_entry"], "torch.ops._C.awq_gemm")
        self.assertTrue(result["proved"])

    def test_classify_linear_backend_triton(self) -> None:
        result = probe.classify_linear_backend(
            use_triton_awq=True,
            awq_gemm_source="from somewhere import awq_gemm_triton\nreturn awq_gemm_triton(...)\n",
        )
        self.assertEqual(result["backend"], "TRITON_AWQ")
        self.assertTrue(result["proved"])

    def test_stable_hash_ignores_existing_hash_field(self) -> None:
        payload = {"schema": "x", "value": 7}
        first = probe.stable_json_sha256(payload)
        payload["probe_sha256"] = first
        self.assertEqual(probe.stable_json_sha256(payload), first)

    def test_redact_hides_common_credentials(self) -> None:
        redacted = probe.redact("Authorization=Bearer123 api_key=secret sk-abcdef")
        self.assertNotIn("Bearer123", redacted)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("sk-abcdef", redacted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
