import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "r9700_phase6_control.py"
spec = importlib.util.spec_from_file_location("p6", MODULE_PATH)
p6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p6)


class Phase6ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict("os.environ", {"HYPERLOOM_PHASE6_STATE": str(pathlib.Path(self.tmp.name) / "state.json"), "XDG_RUNTIME_DIR": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_default_is_stock(self):
        state = p6.load_state()
        self.assertEqual(state["active_backend"], "stock")
        self.assertEqual(state["desired_backend"], "stock")
        self.assertFalse(state["validated"])

    def test_promote_dry_run_never_mutates(self):
        with mock.patch.object(p6, "set_service") as mutate:
            result = p6.promote(dry_run=True)
        mutate.assert_not_called()
        self.assertEqual(result["plan"]["from"], "stock")
        self.assertEqual(result["plan"]["to"], "hyperloom_s3")

    def test_promotion_success_commits_s3_only_after_readiness(self):
        calls = []
        def set_service(action, name, **kwargs):
            calls.append((action, name))
        with mock.patch.object(p6, "verify_backend", side_effect=[{"pass": True}, {"pass": True, "service":"s3"}]), \
             mock.patch.object(p6, "set_service", side_effect=set_service), \
             mock.patch.object(p6, "wait_service_inactive"), \
             mock.patch.object(p6, "wait_vram_clean"):
            state = p6.promote()
        self.assertEqual(state["active_backend"], "hyperloom_s3")
        self.assertTrue(state["validated"])
        self.assertIn(("stop", p6.STOCK_SERVICE), calls)
        self.assertIn(("start", p6.S3_SERVICE), calls)
        self.assertIn(("start", p6.GUARD_TIMER), calls)

    def test_promotion_failure_falls_back_to_stock(self):
        def verify(service):
            if service == p6.STOCK_SERVICE:
                return {"pass": True, "service": service}
            return {"pass": False, "service": service}
        with mock.patch.object(p6, "verify_backend", side_effect=verify), \
             mock.patch.object(p6, "set_service"), \
             mock.patch.object(p6, "wait_service_inactive"), \
             mock.patch.object(p6, "wait_vram_clean"):
            with self.assertRaises(RuntimeError):
                p6.promote()
        state = p6.load_state()
        self.assertEqual(state["active_backend"], "stock")
        self.assertTrue(state["validated"])
        self.assertEqual(state["reason"], "automatic_fallback")

    def test_guard_falls_back_immediately_if_s3_service_dies(self):
        state = p6.default_state()
        state.update({"active_backend": "hyperloom_s3", "desired_backend": "hyperloom_s3", "validated": True})
        p6.save_state(state)
        with mock.patch.object(p6, "service_state", return_value="failed"), \
             mock.patch.object(p6, "fallback_locked", return_value={"active_backend": "stock"}) as fallback:
            out = p6.guard_once()
        self.assertEqual(out["action"], "fallback")
        fallback.assert_called_once()

    def test_guard_waits_three_http_failures(self):
        state = p6.default_state()
        state.update({"active_backend": "hyperloom_s3", "desired_backend": "hyperloom_s3", "validated": True})
        p6.save_state(state)
        with mock.patch.object(p6, "service_state", return_value="active"), \
             mock.patch.object(p6, "model_ready", return_value=(False, {"error":"boom"})), \
             mock.patch.object(p6, "fallback_locked", return_value={"active_backend":"stock"}) as fallback:
            self.assertEqual(p6.guard_once(failure_threshold=3)["action"], "degraded")
            self.assertEqual(p6.guard_once(failure_threshold=3)["action"], "degraded")
            self.assertEqual(p6.guard_once(failure_threshold=3)["action"], "fallback")
        fallback.assert_called_once()

    def test_lock_is_exclusive(self):
        with p6.exclusive_lock():
            with self.assertRaisesRegex(RuntimeError, "phase6_lock_busy"):
                with p6.exclusive_lock():
                    pass


if __name__ == "__main__":
    unittest.main()
