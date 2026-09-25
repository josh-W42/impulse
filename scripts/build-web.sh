#!/usr/bin/env bash
# Builds the browser demo into web/dist/ for local serving or Firebase Hosting.
set -euo pipefail
cd "$(dirname "$0")/.."

expected="$(scripts/wasm-bindgen-version.sh)"
if ! actual="$(wasm-bindgen --version 2>/dev/null)"; then
  echo "wasm-bindgen CLI not found. Install it with:" >&2
  echo "  cargo install --locked wasm-bindgen-cli --version $expected" >&2
  exit 1
fi
if [ "$actual" != "wasm-bindgen $expected" ]; then
  echo "wasm-bindgen CLI is '$actual' but Cargo.lock has $expected. Reinstall with:" >&2
  echo "  cargo install --locked wasm-bindgen-cli --version $expected --force" >&2
  exit 1
fi

cargo build --release -p impulse-demo --target wasm32-unknown-unknown

rm -rf web/dist
mkdir -p web/dist
wasm-bindgen --target web --no-typescript --out-dir web/dist --out-name impulse \
  target/wasm32-unknown-unknown/release/impulse-demo.wasm
cp web/index.html web/dist/
echo "Built web/dist/"
