#!/usr/bin/env bash

set -euo pipefail

INTERVAL="${MONITOR_INTERVAL:-2}"

while true; do
  date --iso-8601=seconds
  rocm-smi --showuse --showmeminfo vram
  sleep "$INTERVAL"
done

