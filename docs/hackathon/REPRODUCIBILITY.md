# Reproducibility

Reproduce the runtime manifest:

```bash
python3 scripts/r9700_runtime_manifest.py --base-url http://127.0.0.1:8000/v1
```

Reproduce the single-spawn autonomous E2E:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
python3 scripts/r9700_upstream_agent_e2e.py
```

Create the multi-spawn benchmark plan:

```bash
python3 scripts/r9700_multispawn_harness.py --spawn-count 5
```

Run live multi-spawn only in an explicit benchmark window:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
python3 scripts/r9700_multispawn_harness.py --spawn-count 5 --execute
```

Do not restart the resident vLLM service for these commands unless a separate benchmark window requires a clean launch.

