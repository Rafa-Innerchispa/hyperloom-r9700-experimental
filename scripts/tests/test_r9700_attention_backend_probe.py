import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r9700_attention_backend_probe import (  # noqa: E402
    choose_backend,
    cli_flag,
    last_json_object,
    redact,
    stable_json_sha256,
)


class AttentionBackendProbeTests(unittest.TestCase):
    def test_choose_backend_uses_lowest_priority_number(self):
        rows = [
            {"name": "TRITON_ATTN", "priority": 3, "path": "triton.path"},
            {"name": "ROCM_ATTN", "priority": 2, "path": "rocm.path"},
        ]
        self.assertEqual(choose_backend(rows)["name"], "ROCM_ATTN")

    def test_choose_backend_ignores_invalid_rows(self):
        self.assertIsNone(choose_backend([{"name": "X"}, "bad"]))

    def test_cli_flag(self):
        argv = ["python3", "server.py", "--dtype", "float16"]
        self.assertEqual(cli_flag(argv, "--dtype", "auto"), "float16")
        self.assertEqual(cli_flag(argv, "--model", "none"), "none")

    def test_last_json_object(self):
        text = "noise\n" + json.dumps({"a": 1}) + "\nmore\n" + json.dumps({"b": 2})
        self.assertEqual(last_json_object(text), {"b": 2})

    def test_stable_hash_ignores_own_hash(self):
        a = {"schema": "x", "value": 1}
        b = {"schema": "x", "value": 1, "probe_sha256": "junk"}
        self.assertEqual(stable_json_sha256(a), stable_json_sha256(b))

    def test_redact(self):
        text = "Authorization BearerSecret token=abcd password hunter2 sk-abcdefghijklmnop"
        out = redact(text)
        self.assertNotIn("abcd", out)
        self.assertNotIn("hunter2", out)
        self.assertNotIn("sk-abcdefghijklmnop", out)


if __name__ == "__main__":
    unittest.main()
