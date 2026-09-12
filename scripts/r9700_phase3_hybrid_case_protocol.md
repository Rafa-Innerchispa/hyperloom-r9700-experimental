# Phase 3 hybrid independent-start protocol

Use after the selective M1 R9700 INT4 config has passed the paired kernel gate.

## Contract

- stock `inneros-vllm-canary-rocm10.service` must be inactive;
- R9700 VRAM must be below 1 GiB before each fresh start;
- vLLM PR #43389 runtime overlay hashes must match the Phase 3 manifest;
- selective hybrid config SHA must be recorded per launch;
- each independent start uses a distinct loopback port to avoid residual TCP/socket state (`18011`, `18012`, `18013` in the canonical campaign);
- no process restart is allowed merely because startup is slow;
- first serving measurement is diagnostic/cold, second is the canonical hot measurement;
- correctness hash and all four deterministic C4 output hashes must match the stock references;
- record C1, C4 and ~6K-context decode for every hot start;
- remove the candidate and verify VRAM is clean before the next start;
- restore stock systemd and verify `/v1/models` HTTP 200 at campaign end.

## Canonical stock references

- C1 clean queue1 starts: `62.964`, `64.822 tok/s` (median ~`63.893`).
- C4 clean queue1: `158.567959`, `158.412033 tok/s` (median `158.489996`).
- ~6K clean queue1: `58.913`, `58.825 tok/s` (median ~`58.869`).
- healthy-fast stock C4 same-day reference: `162.102053 tok/s`.
- historical best-stock C4 reference: `165.699 tok/s`.

## Hash gates

Correctness SHA-256:
`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

C4 vector:
1. `891b5901302b3d6901fd310b055c7f3fa03ed1ea6a4cf1c1deebd0e5fc372e68`
2. `f3164647b31f3f554d6d6f1be77d63f87736ffb85b89bc3e3b59fcdab5ac6c5c`
3. `fbe90e7b9c033267a5017a845902599a1c0b6fead46d2a709ecad63ddb97a407`
4. `fbe90e7b9c033267a5017a845902599a1c0b6fead46d2a709ecad63ddb97a407`

The protocol exists to prevent a port/TIME_WAIT artifact, slow startup, or spawn lottery from being mistaken for a model result.
