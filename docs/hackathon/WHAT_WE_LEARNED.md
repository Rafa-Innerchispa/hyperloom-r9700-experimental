# What We Learned

- A local R9700 can host an OpenAI-compatible vLLM server for bounded agentic experiments.
- Small local models need compact tool catalogs and strict JSON fallback paths.
- Throughput gains from concurrency must not be confused with kernel optimization.
- Evidence needs request-level samples, deterministic replay, and fail-closed auditors.
- Upstream contributions should be small: device identity, benchmark harness, docs, and reproducibility checks before broad RDNA4 claims.

