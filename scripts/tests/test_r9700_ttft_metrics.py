#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r9700_upstream_agent_e2e as mod


class FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def __iter__(self):
        return iter(self.lines)


class TTFTMetricsTests(unittest.TestCase):
    def test_streaming_request_measures_ttft_and_usage(self):
        lines = [
            'data: ' + json.dumps({"choices":[{"delta":{"role":"assistant"}}],"usage":None}) + '\n',
            'data: ' + json.dumps({"choices":[{"delta":{"content":"AMD test"}}],"usage":None}) + '\n',
            'data: ' + json.dumps({"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}) + '\n',
            'data: [DONE]\n',
        ]
        fake = FakeResponse(lines)
        with patch.object(mod.urllib.request, "urlopen", return_value=fake), patch.object(mod.time, "perf_counter", side_effect=[10.0, 10.05, 10.2]):
            row = mod._one_request("model", 1)
        self.assertTrue(row["ok"])
        self.assertAlmostEqual(row["ttft_sec"], 0.05, places=6)
        self.assertAlmostEqual(row["elapsed_sec"], 0.2, places=6)
        self.assertEqual(row["prompt_tokens"], 7)
        self.assertEqual(row["completion_tokens"], 2)

    def test_aggregate_rounds_includes_ttft(self):
        rounds = [
            {"requests":1,"passed":1,"failed":0,"output_tok_s":10.0,"total_tok_s":20.0,"mean_e2e_ms":200.0,"p95_e2e_ms":200.0,"mean_ttft_ms":50.0,"p95_ttft_ms":50.0},
            {"requests":1,"passed":1,"failed":0,"output_tok_s":12.0,"total_tok_s":22.0,"mean_e2e_ms":220.0,"p95_e2e_ms":220.0,"mean_ttft_ms":60.0,"p95_ttft_ms":60.0},
            {"requests":1,"passed":1,"failed":0,"output_tok_s":11.0,"total_tok_s":21.0,"mean_e2e_ms":210.0,"p95_e2e_ms":210.0,"mean_ttft_ms":55.0,"p95_ttft_ms":55.0},
        ]
        agg = mod._aggregate_rounds(rounds)
        self.assertEqual(agg["median_output_tok_s"], 11.0)
        self.assertEqual(agg["median_p95_ttft_ms"], 55.0)
        self.assertEqual(agg["median_mean_ttft_ms"], 55.0)

    def test_verdict_rejects_missing_ttft_metric(self):
        counts = {
            "round_count": mod.MEASUREMENT_ROUNDS,
            "requests": mod.MEASUREMENT_ROUNDS * mod.REQUESTS_PER_ARM,
            "passed": mod.MEASUREMENT_ROUNDS * mod.REQUESTS_PER_ARM,
            "failed": 0,
        }
        base = {**counts, "median_output_tok_s":10.0,"median_p95_e2e_ms":100.0,"median_p95_ttft_ms":50.0}
        candidate = {**counts, "median_output_tok_s":12.0,"median_p95_e2e_ms":110.0,"median_p95_ttft_ms":None}
        verdict, gate = mod._verdict(base, candidate)
        self.assertEqual(verdict, "REJECT")
        self.assertEqual(gate.get("reason"), "invalid_metric")
        self.assertEqual(gate.get("field"), "median_p95_ttft_ms")


if __name__ == "__main__":
    unittest.main()
