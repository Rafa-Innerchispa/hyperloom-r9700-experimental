import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "r9700_phase6_readiness.py"
spec = importlib.util.spec_from_file_location("p6readiness", MODULE_PATH)
p6r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p6r)


def sample(ok: bool, digest: str = "x") -> dict:
    return {
        "ok": ok,
        "canonical_match": ok,
        "status": 200,
        "text_sha256": digest,
    }


class Phase6ReadinessTests(unittest.TestCase):
    def test_models_endpoint_failure_fails_closed(self):
        with mock.patch.object(p6r, "wait_models", return_value={"ok": False, "error": "down"}), \
             mock.patch.object(p6r, "warmup") as warmup:
            out = p6r.gate("http://127.0.0.1:8000", "model", 1, 4, 2)
        self.assertFalse(out["pass"])
        self.assertEqual(out["reason"], "models_endpoint_not_ready")
        warmup.assert_not_called()

    def test_cold_mismatch_then_two_canonical_passes(self):
        with mock.patch.object(p6r, "wait_models", return_value={"ok": True, "status": 200}), \
             mock.patch.object(p6r, "warmup", side_effect=[sample(False, "cold"), sample(True, "canonical"), sample(True, "canonical")]):
            out = p6r.gate("http://127.0.0.1:8000", "model", 1, 4, 2)
        self.assertTrue(out["pass"])
        self.assertEqual(out["reason"], "canonical_steady_state_proven")
        self.assertEqual(len(out["attempts"]), 3)
        self.assertEqual(out["consecutive_canonical"], 2)

    def test_nonconsecutive_canonical_samples_fail(self):
        sequence = [sample(True, "a"), sample(False, "b"), sample(True, "a"), sample(False, "b")]
        with mock.patch.object(p6r, "wait_models", return_value={"ok": True, "status": 200}), \
             mock.patch.object(p6r, "warmup", side_effect=sequence):
            out = p6r.gate("http://127.0.0.1:8000", "model", 1, 4, 2)
        self.assertFalse(out["pass"])
        self.assertEqual(out["reason"], "canonical_steady_state_not_proven")

    def test_two_immediate_canonical_samples_pass_without_extra_attempts(self):
        with mock.patch.object(p6r, "wait_models", return_value={"ok": True, "status": 200}), \
             mock.patch.object(p6r, "warmup", side_effect=[sample(True, "canonical"), sample(True, "canonical")]) as warmup:
            out = p6r.gate("http://127.0.0.1:8000", "model", 1, 4, 2)
        self.assertTrue(out["pass"])
        self.assertEqual(warmup.call_count, 2)


if __name__ == "__main__":
    unittest.main()
