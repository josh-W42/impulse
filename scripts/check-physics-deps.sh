#!/usr/bin/env bash
# Enforces the core architecture rule: impulse-physics may depend on glam only.
set -euo pipefail
cd "$(dirname "$0")/.."

deps="$(cargo tree -p impulse-physics -e normal --depth 1 --prefix none \
  | tail -n +2 | awk '{print $1}' | sort -u)"
bad="$(printf '%s\n' "$deps" | grep -vxE 'glam|' || true)"

if [ -n "$bad" ]; then
  echo "impulse-physics must depend only on glam; found: $bad" >&2
  exit 1
fi
echo "impulse-physics dependencies OK: ${deps:-none}"
