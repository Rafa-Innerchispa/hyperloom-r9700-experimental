# Technical Architecture

The demo path is:

1. Resident vLLM OpenAI-compatible server on AMD node `.5`.
2. `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` loaded from local disk.
3. KernelForge `local-openai` backend with compact JSON tool mode.
4. A bounded agent writes only one candidate file.
5. A deterministic harness benchmarks baseline and candidate.
6. A gate records KEEP/REJECT with request-level evidence.
7. Offline auditors verify saved evidence without network or model access.

Safety boundaries:

- No shell is exposed to the local model in the R9700 E2E.
- Candidate values are allowlisted.
- The model cannot mark its own work as PASS.
- Evidence serialization rejects non-finite values.
- Public claims are separated into PROVEN, PARTIAL, and NOT PROVEN.

