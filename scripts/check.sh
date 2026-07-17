#!/usr/bin/env bash
# Local source-of-truth gate for trit-vsa (no GitHub Actions required).
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo check (default features)"
cargo check

echo "==> cargo test"
cargo test

echo "OK: trit-vsa checks passed"