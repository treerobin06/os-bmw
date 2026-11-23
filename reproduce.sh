#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${S1_OUT_DIR:-${ROOT_DIR}/out}"
mkdir -p "${OUT_DIR}"

echo "=== [1/3] OS-BMW Core Rust Crate Unit & Semilattice Tests (s1_v2_rust: 19 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_rust"
  cargo test --release --lib
  cargo test --release --bin s1_v2_equal_error_hip
)

echo ""
echo "=== [2/3] Apache DataFusion 55.3 Sparse+Flat-Slab Wire Framing Tests (s1_v2_datafusion: 2 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_datafusion"
  cargo test --release
)

echo ""
echo "=== [3/3] Exact Mathematical Theorem Verification (Theorems 1-4) ==="
python3 "${ROOT_DIR}/scripts/verify_proofs.py" "${OUT_DIR}/proofs_verification.json"

echo ""
echo "=== ALL CLEAN-ROOM CHECKS PASSED (21/21 Rust tests + Theorems 1-4 verified) ==="
echo "Output written to: ${OUT_DIR}/proofs_verification.json"
