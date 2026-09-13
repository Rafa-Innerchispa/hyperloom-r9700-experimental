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
        with mock.patch.object(p6, "set_service") as mutate, \
             mock.patch.object(p6, "start_guard_scheduler") as guard:
            result = p6.promote(dry_run=True)
        mutate.assert_not_called()
        guard.assert_not_called()
        self.assertEqual(result["plan"]["from"], "stock")
        self.assertEqual(result["plan"]["to"], "hyperloom_s3")
        self.assertIn("wait_port_8000_free", result["plan"]["steps"])

    def test_promotion_success_commits_s3_only_after_readiness(self):
        calls = []

        def set_service(action, name, **kwargs):
            calls.append((action, name))

        with mock.patch.object(p6, "verify_backend", side_effect=[{"pass": True}, {"pass": True, "service": "s3"}]), \
             mock.patch.object(p6, "set_service", side_effect=set_service), \
             mock.patch.object(p6, "wait_service_inactive"), \
             mock.patch.object(p6, "wait_vram_clean"), \
             mock.patch.object(p6, "wait_port_free") as wait_port, \
             mock.patch.object(p6, "start_guard_scheduler") as guard:
            state = p6.promote()
        self.assertEqual(state["active_backend"], "hyperloom_s3")
        self.assertTrue(state["validated"])
        self.assertIn(("stop", p6.STOCK_SERVICE), calls)
        self.assertIn(("start", p6.S3_SERVICE), calls)
        wait_port.assert_called_once_with(timeout=60)
        guard.assert_called_once_with()

    def test_start_guard_scheduler_uses_transient_timer(self):
        with mock.patch.object(p6, "stop_guard_scheduler") as stop, \
             mock.patch.object(p6, "run") as runner, \
             mock.patch.object(p6, "wait_for") as waiter:
            p6.start_guard_scheduler()
        stop.assert_called_once_with()
        argv = runner.call_args.args[0]
        self.assertEqual(argv[0], "systemd-run")
        self.assertIn("--on-active=30s", argv)
        self.assertIn("--on-unit-active=30s", argv)
        self.assertIn("--timer-property=AccuracySec=2s", argv)
        self.assertIn(p6.GUARD_SERVICE, argv)
        waiter.assert_called_once()

    def test_fallback_waits_for_port_and_stock_readiness(self):
        with mock.patch.object(p6, "set_service"), \
             mock.patch.object(p6, "wait_service_inactive"), \
             mock.patch.object(p6, "wait_vram_clean"), \
             mock.patch.object(p6, "wait_port_free") as wait_port, \
             mock.patch.object(p6, "wait_backend_ready", return_value={"pass": True, "service": p6.STOCK_SERVICE}) as wait_ready:
            state = p6.fallback_locked("test")
        wait_port.assert_called_once_with(timeout=60)
        wait_ready.assert_called_once_with(p6.STOCK_SERVICE, timeout=600)
        self.assertEqual(state["active_backend"], "stock")
        self.assertTrue(state["validated"])

    def test_reconcile_stock_requires_ready_stock_and_inactive_s3(self):
        verification = {"pass": True, "service": p6.STOCK_SERVICE}
        with mock.patch.object(p6, "service_state", return_value="inactive"), \
             mock.patch.object(p6, "wait_backend_ready", return_value=verification) as wait_ready, \
             mock.patch.object(p6, "stop_guard_scheduler"), \
             mock.patch.object(p6, "set_service"):
            state = p6.reconcile_stock_state()
        wait_ready.assert_called_once_with(p6.STOCK_SERVICE, timeout=600)
        self.assertEqual(state["active_backend"], "stock")
        self.assertEqual(state["desired_backend"], "stock")
        self.assertTrue(state["validated"])
        self.assertEqual(state["reason"], "stock_reconciled_after_interrupted_promotion")

    def test_reconcile_stock_refuses_active_s3(self):
        with mock.patch.object(p6, "service_state", return_value="active"):
            with self.assertRaisesRegex(RuntimeError, "cannot_reconcile_stock_while_s3_active"):
                p6.reconcile_stock_state()

    def test_promotion_failure_falls_back_to_stock(self):
        def verify(service):
            if service == p6.STOCK_SERVICE:
                return {"pass": True, "service": service}
            return {"pass": False, "service": service}

        with mock.patch.object(p6, "verify_backend", side_effect=verify), \
             mock.patch.object(p6, "set_service"), \
             mock.patch.object(p6, "wait_service_inactive"), \
             mock.patch.object(p6, "wait_vram_clean"), \
             mock.patch.object(p6, "wait_port_free"), \
             mock.patch.object(p6, "wait_backend_ready", return_value={"pass": True, "service": p6.STOCK_SERVICE}):
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
             mock.patch.object(p6, "model_ready", return_value=(False, {"error": "boom"})), \
             mock.patch.object(p6, "fallback_locked", return_value={"active_backend": "stock"}) as fallback:
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
