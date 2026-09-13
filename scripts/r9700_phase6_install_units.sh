#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$ROOT/scripts/systemd"
UNIT_DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$UNIT_DST"
install -m 0644 "$UNIT_SRC/inneros-vllm-hyperloom-s3-production.service" "$UNIT_DST/inneros-vllm-hyperloom-s3-production.service"
install -m 0644 "$UNIT_SRC/inneros-vllm-hyperloom-phase6-guard.service" "$UNIT_DST/inneros-vllm-hyperloom-phase6-guard.service"
systemctl --user daemon-reload

# Deliberately do not enable/start anything here. Stock remains default until
# r9700_phase6_control.py promote passes its explicit live gates. The 30-second
# guard scheduler is a transient systemd timer created by the controller only
# after S3 readiness is proven, so a reboot cannot silently preserve promotion.
printf '%s\n' 'Phase 6 units installed only; no service routing was changed.'
systemctl --user is-active inneros-vllm-canary-rocm10.service || true
