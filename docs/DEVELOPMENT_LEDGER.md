
## Qwen MoE W1/W2 pair benchmark — PROVEN

Evidence: `docs/evidence/r9700_wna16_moe_pair_algebraic_bench_20260908T215643Z.json`

Evidence SHA-256: `99d9b87cc5ac7e50dd1039f918f0502878306bda48c1ca52b27353baee34df45`

A separate shape benchmark tested both Qwen MoE GEMMs using the same algebraic kernel against a pre-dequantized BF16 micro-reference.

W1 (`K=2048 → N=1536`) remained numerically correct but was slightly slower in the isolated single-expert kernel benchmark: median `0.94284x`, range `0.93065x–0.97105x`. This does not contradict the routed-MoE benchmark, where the candidate can win at very small token counts because routing/block geometry and the reference launch path differ.

W2 (`K=768 → N=2048`) was numerically correct but slower for every tested effective row count: median `0.69999x`; at the decode-like `M=8` point it reached `0.79693x` versus the BF16 micro-reference.

**Integration decision:** do not replace both GEMMs. The first production-style candidate must be hybrid and conservative:

- consider the algebraic packed kernel only for W1 at a strongly proven small-token operating point;
- retain/fallback to the existing BF16 path for W2;
- retain fallback for larger W1 token blocks;
- re-benchmark the final hybrid through the complete vLLM MoE pipeline before any serving claim.

The strongest currently measured routed point is W1 `M=1`, `1.25808x` versus the routed BF16 reference. M=2 is only `1.02926x`, M=4 `1.00327x`, so an initial dispatch policy should prefer the clearly separated M=1 case rather than treating marginal wins as universal.
