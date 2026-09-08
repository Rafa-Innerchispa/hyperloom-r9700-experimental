import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r9700_independent_process_harness import aggregate, audit_report, build_plan  # noqa: E402


def row(started, baseline, candidate, gain, p95, decision="KEEP", baseline_ttft=50.0, candidate_ttft=70.0):
    return {
        "restart_returncode": 0,
        "health": {"ok": True},
        "identity_after_restart": {"started_at": started},
        "runner_returncode": 0,
        "runner_summary": {
            "ok": True,
            "baseline_median_output_tok_s": baseline,
            "candidate_median_output_tok_s": candidate,
            "baseline_median_p95_ttft_ms": baseline_ttft,
            "candidate_median_p95_ttft_ms": candidate_ttft,
            "gain_percent": gain,
            "p95_ratio": p95,
            "decision": decision,
        },
    }


class IndependentProcessHarnessTests(unittest.TestCase):
    def test_plan_bounds(self):
        self.assertEqual(build_plan(process_count=3)["process_count"], 3)
        with self.assertRaises(ValueError):
            build_plan(process_count=1)
        with self.assertRaises(ValueError):
            build_plan(process_count=6)

    def test_audit_accepts_unique_process_spawns(self):
        report = {"process_count": 2, "processes": [row("a", 20, 36, 80, 1.0), row("b", 22, 35, 59, 1.1)]}
        self.assertTrue(audit_report(report)["ok"])

    def test_audit_rejects_duplicate_started_at(self):
        report = {"process_count": 2, "processes": [row("same", 20, 36, 80, 1.0), row("same", 22, 35, 59, 1.1)]}
        self.assertEqual(audit_report(report)["reason"], "duplicate_process_started_at")

    def test_aggregate_is_paired(self):
        rows = [row("a", 20, 36, 80, 1.0, baseline_ttft=50, candidate_ttft=65), row("b", 25, 37.5, 50, 1.1, baseline_ttft=55, candidate_ttft=70), row("c", 22, 35.2, 60, 1.2, "REJECT", baseline_ttft=52, candidate_ttft=68)]
        result = aggregate(rows)
        self.assertEqual(result["baseline_output_tok_s_median_across_processes"], 22.0)
        self.assertEqual(result["candidate_output_tok_s_median_across_processes"], 36.0)
        self.assertEqual(result["baseline_ttft_p95_ms_median_across_processes"], 52.0)
        self.assertEqual(result["candidate_ttft_p95_ms_median_across_processes"], 68.0)
        self.assertEqual(result["paired_gain_percent_median"], 60.0)
        self.assertEqual(result["keep_count"], 2)
        self.assertEqual(result["reject_count"], 1)
        self.assertTrue(result["all_processes_gain_ge_10pct"])

    def test_aggregate_accepts_runner_verdict_field(self):
        sample = row("a", 20, 36, 80, 1.0)
        sample["runner_summary"]["verdict"] = sample["runner_summary"].pop("decision")
        result = aggregate([sample])
        self.assertEqual(result["decisions"], ["KEEP"])
        self.assertEqual(result["keep_count"], 1)
        self.assertEqual(result["reject_count"], 0)


if __name__ == "__main__":
    unittest.main()
