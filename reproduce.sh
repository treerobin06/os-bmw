#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${S1_OUT_DIR:-${ROOT_DIR}/out}"
REPRO_DIR="${S1_REPRO_DIR:-${ROOT_DIR}/reproduced}"
TAB_DIR="${S1_TAB_DIR:-${REPRO_DIR}/tables}"
FIG_DIR="${S1_FIG_DIR:-${REPRO_DIR}/figures}"

mkdir -p "${OUT_DIR}" "${TAB_DIR}" "${FIG_DIR}"

echo "=== [1/4] OS-BMW Core Rust Crate Unit & Semilattice Tests (s1_v2_rust: 30 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_rust"
  cargo test --release --lib
  cargo test --release --bin s1_v2_equal_error_hip
)

echo ""
echo "=== [2/4] Apache DataFusion 55.3 Symmetric Sparse+Flat-Slab & SQL Tests (s1_v2_datafusion: 4 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_datafusion"
  cargo test --release
)

echo ""
echo "=== [3/4] Exact Mathematical Theorem Verification (Theorems 1-4) & W9 Converter Self-Test ==="
python3 "${ROOT_DIR}/scripts/verify_proofs.py" "${REPRO_DIR}/proofs_verification.json"
python3 "${ROOT_DIR}/scripts/prepare_w9_clickbench.py" --self-test
S1_OUT_DIR="${OUT_DIR}" python3 "${ROOT_DIR}/scripts/dump_protocol_and_constants.py" --verify

echo ""
echo "=== [4/4] Reproducing All Manuscript Tables (9 .tex) & Figures (4 .pdf) from Raw TSV/JSON ==="
S1_OUT_DIR="${OUT_DIR}" S1_TAB_DIR="${TAB_DIR}" S1_FIG_DIR="${FIG_DIR}" \
  python3 "${ROOT_DIR}/scripts/build_tables_and_figs.py"

echo ""
echo "=== ALL CLEAN-ROOM CHECKS PASSED (34/34 Rust tests + Theorems 1-4 + W9 self-test + m5 verify + 9 tables + 4 PDF figures) ==="
echo "Proof output: ${REPRO_DIR}/proofs_verification.json"
echo "Tables:       ${TAB_DIR}"
echo "Figures:      ${FIG_DIR}"
