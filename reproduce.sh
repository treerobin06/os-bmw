#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${S1_OUT_DIR:-${ROOT_DIR}/out}"
REPRO_DIR="${S1_REPRO_DIR:-${ROOT_DIR}/reproduced}"
TAB_DIR="${S1_TAB_DIR:-${REPRO_DIR}/tables}"
FIG_DIR="${S1_FIG_DIR:-${REPRO_DIR}/figures}"

mkdir -p "${OUT_DIR}" "${TAB_DIR}" "${FIG_DIR}"

echo "=== [1/4] OS-BMW Core Rust Crate Unit & Semilattice Tests (s1_v2_rust: 32 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_rust"
  cargo test --release --offline --lib
  cargo test --release --offline --bin s1_v2_equal_error_hip
)

echo ""
echo "=== [2/4] Apache DataFusion 55.3 Symmetric Sparse+Flat-Slab & SQL Tests (s1_v2_datafusion: 4 tests) ==="
(
  cd "${ROOT_DIR}/s1_v2_datafusion"
  cargo test --release --offline
)

echo ""
echo "=== [3/4] Exact Mathematical Theorem Verification (Theorems 1-3 + Rank Codec), W9 Self-Test, Protocol & Wrapper Audits ==="
python3 "${ROOT_DIR}/scripts/verify_proofs.py" "${REPRO_DIR}/proofs_verification.json"
python3 "${ROOT_DIR}/scripts/prepare_w9_clickbench.py" --self-test
S1_OUT_DIR="${OUT_DIR}" python3 "${ROOT_DIR}/scripts/dump_protocol_and_constants.py" --verify
python3 "${ROOT_DIR}/scripts/audit_closest_cells_and_wrappers.py" --out-dir "${OUT_DIR}" --dest "${REPRO_DIR}/m2_closest_cells_and_wrappers.json"

echo ""
echo "=== [4/4] Reproducing Manuscript Macros, Tables (5 .tex: v2_mc, v2_multishard, v2_micro, v2_df, v2_supp) & Figures (4 .pdf) from Raw TSV/JSON ==="
python3 "${ROOT_DIR}/scripts/build_story_macros.py" --out-dir "${OUT_DIR}" --dest "${TAB_DIR}/story_macros.tex"
python3 "${ROOT_DIR}/scripts/build_story_tables.py" --out-dir "${OUT_DIR}" --dest-dir "${TAB_DIR}"
python3 "${ROOT_DIR}/scripts/build_story_figs.py" --out-dir "${OUT_DIR}" --fig-dir "${FIG_DIR}"

echo ""
echo "=== ALL CLEAN-ROOM CHECKS PASSED (36/36 Rust tests + Theorems 1-3 & Rank + W9 self-test + m5/E-11 verify + macros + 5 tables + 4 PDF figures) ==="
echo "Proof output: ${REPRO_DIR}/proofs_verification.json"
echo "Tables:       ${TAB_DIR}"
echo "Figures:      ${FIG_DIR}"
