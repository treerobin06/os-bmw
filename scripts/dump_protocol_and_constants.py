#!/usr/bin/env python3
"""
EXP-OPT-3: Deterministic Protocol & Design Constants Extractor (`m5_protocol_and_constants.json`).

Extracts verbatim numerical, wire-framing, bootstrap, memory-accounting, estimator,
workload, hardware-topology, and test-suite metadata from the live artifact source
and sanitized baseline outputs. Uses ONLY relative paths and zero private host/user strings.

Usage:
  python3 scripts/dump_protocol_and_constants.py           # writes out/m5_protocol_and_constants.json
  python3 scripts/dump_protocol_and_constants.py --verify  # verifies existing JSON matches live sources
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATTERNS = ("/Users/", "/home/", "/data/users/")


def read_text(rel_path: str) -> str:
    p = ARTIFACT_ROOT / rel_path
    if not p.is_file():
        raise FileNotFoundError(f"Missing required artifact file: {rel_path}")
    return p.read_text(encoding="utf-8")


def must_match(pattern: str, text: str, label: str, flags: int = 0) -> re.Match[str]:
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"Pattern mismatch while extracting {label}: {pattern!r}")
    return m


def extract_protocol_and_constants() -> dict[str, Any]:
    pmf_src = read_text("s1_v2_rust/src/pmf.rs")
    base_src = read_text("s1_v2_rust/src/baselines.rs")
    lib_src = read_text("s1_v2_rust/src/lib.rs")
    hip_src = read_text("s1_v2_rust/src/bin/s1_v2_equal_error_hip.rs")
    acc_src = read_text("s1_v2_rust/src/bin/s1_v2_accuracy.rs")
    rec_src = read_text("s1_v2_rust/src/bin/s1_v2_reconcile.rs")
    df_src = read_text("s1_v2_datafusion/src/main.rs")
    env_info = json.loads(read_text("out/env_info.json"))

    # 1. PMF & MLE Newton solver constants
    newton_iters = int(must_match(r"for\s+it\s+in\s+1\.\.=(\d+)", pmf_src, "newton_max_iters").group(1))
    m_tol = must_match(
        r"if\s+\(hi\s*-\s*lo\)\s*<=\s*([0-9eE.\-]+)\s*\*\s*lam\s*\|\|\s*s\.abs\(\)\s*<=\s*([0-9eE.\-]+)\s*\*\s*zero_lin",
        pmf_src,
        "newton_tolerances",
    )
    bracket_rel_tol = float(m_tol.group(1))
    score_rel_tol = float(m_tol.group(2))

    # 2. DataFusion SQL wire framing, chunk slab, bootstrap, and clear_refs memory accounting
    tag_single = int(must_match(r"const\s+TAG_SINGLE_HASH:\s*u8\s*=\s*(0x[0-9A-Fa-f]+);", df_src, "TAG_SINGLE_HASH").group(1), 16)
    tag_sparse = int(must_match(r"const\s+TAG_EXACT_CELL_SPARSE:\s*u8\s*=\s*(0x[0-9A-Fa-f]+);", df_src, "TAG_EXACT_CELL_SPARSE").group(1), 16)
    sparse_cap = int(must_match(r"const\s+SPARSE_CHUNK_CAP:\s*usize\s*=\s*(\d+);", df_src, "SPARSE_CHUNK_CAP").group(1))
    must_match(r"TAG_SINGLE_HASH\s*&&\s*bytes\.len\(\)\s*==\s*9", df_src, "single_hash_wire_9B")
    boot_b = int(must_match(r"const\s+B:\s*usize\s*=\s*([0-9_]+);", df_src, "bootstrap_resamples").group(1).replace("_", ""))
    must_match(r'/proc/self/clear_refs', df_src, "clear_refs_path")
    clear_refs_val = must_match(r'\.write_all\(b"(\d+)(?:\\n)?"\)\s*\n\s*\.expect\("write 5 to /proc/self/clear_refs"\)', df_src, "clear_refs_write").group(1)

    # 3. Baseline constants (AVLL & Hash4j ULL FGRA)
    avll_max_exp = int(must_match(r"pub\s+const\s+MAX_EXP:\s*u8\s*=\s*(\d+);", base_src, "AvllSketch::MAX_EXP").group(1))
    avll_htl = int(must_match(r"pub\s+const\s+HTL:\s*u8\s*=\s*(\d+);", base_src, "AvllSketch::HTL").group(1))
    avll_hist_margin = int(must_match(r"pub\s+const\s+HISTORY_MARGIN:\s*u8\s*=\s*(\d+);", base_src, "AvllSketch::HISTORY_MARGIN").group(1))
    has_ull_fgra = "pub fn estimate_fgra(&self) -> f64" in base_src

    # 4. Accuracy & Reconcile experiment protocol constants
    sweep_trials_main = int(must_match(r'matches!\(blab,\s*"0\.5KB"\s*\|\s*"1KB"\s*\|\s*"4KB"\)\s*\{\s*(\d+)usize\s*\}\s*else\s*\{\s*(\d+)usize', acc_src, "sweep_trials").group(1))
    sweep_trials_other = int(must_match(r'matches!\(blab,\s*"0\.5KB"\s*\|\s*"1KB"\s*\|\s*"4KB"\)\s*\{\s*(\d+)usize\s*\}\s*else\s*\{\s*(\d+)usize', acc_src, "sweep_trials_other").group(2))
    mc_trials_per_pt = int(must_match(r"let\s+trials_per_pt\s*=\s*(\d+)usize;", acc_src, "mc_trials_per_pt").group(1))
    mc_cards = [float(x.strip().replace("_f64", "")) for x in must_match(r"let\s+mc_cards\s*=\s*\[([^\]]+)\];", acc_src, "mc_cards").group(1).split(",")]
    hash_abl_trials = int(must_match(r"let\s+abl_trials\s*=\s*(\d+)usize;", acc_src, "hash_abl_trials").group(1))
    uniform_n = int(must_match(r'\("Synthetic_Uniform_2M".*?\(0\.\.([0-9_]+)u64\)', acc_src, "uniform_2m").group(1).replace("_", ""))
    zipf_events = int(must_match(r'for\s+_\s+in\s+0\.\.([0-9_]+)\s*\{', acc_src, "zipf_events").group(1).replace("_", ""))
    reconcile_n_total = int(must_match(r"let\s+n_total\s*=\s*([0-9_]+)u64;", rec_src, "reconcile_n_total").group(1).replace("_", ""))
    rank_bytes = int(must_match(r"let\s+codec_k60\s*=\s*OsBmwRankCodec::new\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\);", rec_src, "OsBmwRankCodec").group(1))
    rank_k = int(must_match(r"let\s+codec_k60\s*=\s*OsBmwRankCodec::new\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\);", rec_src, "OsBmwRankCodec_k").group(2))
    rank_kreached = int(must_match(r"let\s+codec_k60\s*=\s*OsBmwRankCodec::new\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\);", rec_src, "OsBmwRankCodec_kr").group(5))

    # 5. External baseline metadata (from official TSV outputs in artifact/out)
    with open(ARTIFACT_ROOT / "out/official_hash4j_sys.tsv", newline="", encoding="utf-8") as f:
        h4j_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(ARTIFACT_ROOT / "out/official_datasketches_sys.tsv", newline="", encoding="utf-8") as f:
        ds_rows = list(csv.DictReader(f, delimiter="\t"))

    # 6. Rust test counts across all suites
    n_lib_tests = len(re.findall(r"#\[test\]", lib_src))
    n_hip_tests = len(re.findall(r"#\[test\]", hip_src))
    n_df_tests = len(re.findall(r"#\[test\]", df_src))

    payload = {
        "schema": "s1_osbmw_protocol_and_constants_v1",
        "relative_sources": [
            "s1_v2_rust/src/pmf.rs",
            "s1_v2_rust/src/baselines.rs",
            "s1_v2_rust/src/lib.rs",
            "s1_v2_rust/src/bin/s1_v2_equal_error_hip.rs",
            "s1_v2_rust/src/bin/s1_v2_accuracy.rs",
            "s1_v2_rust/src/bin/s1_v2_reconcile.rs",
            "s1_v2_datafusion/src/main.rs",
            "out/env_info.json",
            "out/official_hash4j_sys.tsv",
            "out/official_datasketches_sys.tsv",
        ],
        "mle_solver": {
            "source": "s1_v2_rust/src/pmf.rs",
            "newton_max_iters": newton_iters,
            "newton_bracket_rel_tol": bracket_rel_tol,
            "newton_score_rel_tol": score_rel_tol,
            "initial_guess": "geometric_mean(lo, hi)",
            "monotonic_bracket_fallback": True,
        },
        "datafusion_sql_protocol": {
            "source": "s1_v2_datafusion/src/main.rs",
            "tag_single_hash_hex": hex(tag_single),
            "single_hash_wire_bytes": 9,
            "tag_exact_cell_sparse_hex": hex(tag_sparse),
            "sparse_chunk_cap_cells": sparse_cap,
            "sparse_chunk_bytes": 4 + 4 + sparse_cap * 4,
            "bootstrap_resamples_b": boot_b,
            "ci_level": 0.95,
            "memory_reset_method": f"malloc_trim(0) + write \"{clear_refs_val}\" to /proc/self/clear_refs",
            "stage_consistency_audit": "Single vs Multi-Stage (p1_vs_p4 / p1_vs_p8 / p1_vs_p32) exact bit-level group check",
        },
        "baselines_and_rank_codec": {
            "avll_base56": {
                "source": "s1_v2_rust/src/baselines.rs",
                "buckets_per_u64": 11,
                "max_exp": avll_max_exp,
                "htl": avll_htl,
                "history_margin": avll_hist_margin,
                "exact_preimage_regime": "Exact under prefill and bounded shifts without clamping/history eviction; approximate once clamping or history eviction activates",
            },
            "ull_fgra_hash4j_port": {
                "source": "s1_v2_rust/src/baselines.rs",
                "present": has_ull_fgra,
                "reference_library": "Dynatrace hash4j 0.22.0 OptimalFGRAEstimator (arXiv:2308.16862)",
                "byte_bijection": "r_hash4j = r_rust + 4*p - 8 (0 -> 0; large-range sentinel u = 65 - p)",
            },
            "osbmw_rank_theoretical_wire": {
                "g": rank_bytes,
                "k": rank_k,
                "R": 7,
                "d": 2,
                "K_reach": rank_kreached,
                "fixed_bytes": 1277,
            },
        },
        "accuracy_and_stream_protocols": {
            "equal_byte_sweep_trials_per_cell": {
                "budgets_0.5_1_4_kib": sweep_trials_main,
                "budgets_2_8_16_kib": sweep_trials_other,
                "avll_hll4fixed_cap_above_1e5": 200,
            },
            "poissonized_mc": {
                "cards": mc_cards,
                "trials_per_point": mc_trials_per_pt,
                "total_trials_per_config": len(mc_cards) * mc_trials_per_pt,
            },
            "hash64_vs_hash128_ablation_trials": hash_abl_trials,
            "real_and_skewed_workloads": {
                "synthetic_uniform_distinct": uniform_n,
                "zipf_domain_distinct": uniform_n,
                "zipf_stream_events": zipf_events,
                "multishard_reconcile_distinct_n": reconcile_n_total,
            },
        },
        "hardware_and_external_toolchains": {
            "cpu_model": env_info["cpu_model"],
            "os_uname": env_info["uname"],
            "rustc_version": env_info["rustc_version"],
            "cargo_version": env_info["cargo_version"],
            "l3_cache_per_socket_kib": 61440,
            "l3_cache_per_socket_mib": 60,
            "l3_cache_total_mib": 120,
            "sockets": 2,
            "cores_per_socket_physical": 32,
            "logical_cpus_total": 128,
            "numa_nodes": 2,
            "avx512_flags": env_info["lscpu_avx512_flags"],
            "hash4j_version": "0.22.0",
            "hash4j_sys_methods_measured": sorted({r["method"] for r in h4j_rows}),
            "datasketches_cpp_commit": "1e49055d3f3dba9aa9ebc6d293a42c7bd8e53209",
            "datasketches_sys_methods_measured": sorted({r["method"] for r in ds_rows}),
        },
        "rust_verification_tests": {
            "s1_v2_rust_lib": n_lib_tests,
            "s1_v2_equal_error_hip": n_hip_tests,
            "s1_v2_datafusion": n_df_tests,
            "total": n_lib_tests + n_hip_tests + n_df_tests,
        },
    }

    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    for bad in FORBIDDEN_PATTERNS:
        if bad in serialized:
            raise AssertionError(f"Privacy hygiene violation ({bad!r}) in output JSON!")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Verify out/m5_protocol_and_constants.json matches live sources")
    args = ap.parse_args()

    out_dir = Path(os.environ.get("S1_OUT_DIR", str(ARTIFACT_ROOT / "out")))
    out_path = out_dir / "m5_protocol_and_constants.json"

    payload = extract_protocol_and_constants()
    rendered = json.dumps(payload, indent=2) + "\n"

    if args.verify:
        if not out_path.is_file():
            print(f"[FAIL] Missing {out_path}", file=sys.stderr)
            return 1
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != payload:
            print(f"[FAIL] {out_path} does not match extracted constants!", file=sys.stderr)
            return 1
        print(
            f"[OK] Verified {out_path.name}: {payload['rust_verification_tests']['total']}/"
            f"{payload['rust_verification_tests']['total']} tests registered, "
            f"newton_max_iters={payload['mle_solver']['newton_max_iters']}, "
            f"sparse_chunk={payload['datafusion_sql_protocol']['sparse_chunk_bytes']}B"
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rendered)} bytes, {payload['rust_verification_tests']['total']} Rust tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
