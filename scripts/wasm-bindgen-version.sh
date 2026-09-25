#!/usr/bin/env bash
# Prints the wasm-bindgen crate version locked in Cargo.lock.
# The wasm-bindgen CLI must match it exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3; exit }' Cargo.lock)"
if [ -z "$version" ]; then
  echo "wasm-bindgen not found in Cargo.lock" >&2
  exit 1
fi
echo "$version"
