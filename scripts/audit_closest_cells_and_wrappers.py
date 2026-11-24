#!/usr/bin/env python3
"""
Audit closest equal-byte MC cells (CI overlap / two-sample z) and official C++/Java benchmark wrappers (E-11).
Writes machine-readable JSON to ../out/m2_closest_cells_and_wrappers.json (relative to artifact/).
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def delta_rmse_se(rmse: float, mvp: float, mvp_se: float) -> float:
    if mvp <= 0.0 or mvp_se <= 0.0 or rmse <= 0.0:
        return 0.0
    return rmse * (mvp_se / (2.0 * mvp))


def audit_pair_cells(
    by_key: Dict[Tuple[str, str, str], Dict[str, str]],
    budgets: List[str],
    cards: List[str],
    method_a: str,
    method_b: str,
) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    for b in budgets:
        for c in cards:
            ra = by_key.get((b, method_a, c))
            rb = by_key.get((b, method_b, c))
            if not ra or not rb:
                continue
            if ra["rmse_pct"] in ("", "NA") or rb["rmse_pct"] in ("", "NA"):
                continue
            rmse_a = float(ra["rmse_pct"])
            rmse_b = float(rb["rmse_pct"])
            mvp_a = float(ra["mvp_ram"])
            se_a = float(ra["mvp_ram_se"])
            mvp_b = float(rb["mvp_ram"])
            se_b = float(rb["mvp_ram_se"])
            trials_a = int(float(ra["trials"]))
            trials_b = int(float(rb["trials"]))
            comb_se = math.hypot(se_a, se_b)
            z_mvp = (mvp_b - mvp_a) / comb_se if comb_se > 0.0 else float("inf")
            rse_a = delta_rmse_se(rmse_a, mvp_a, se_a)
            rse_b = delta_rmse_se(rmse_b, mvp_b, se_b)
            comb_rse = math.hypot(rse_a, rse_b)
            z_rmse = (rmse_b - rmse_a) / comb_rse if comb_rse > 0.0 else float("inf")
            ci_a = [round(mvp_a - 1.96 * se_a, 5), round(mvp_a + 1.96 * se_a, 5)]
            ci_b = [round(mvp_b - 1.96 * se_b, 5), round(mvp_b + 1.96 * se_b, 5)]
            ci95_overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
            cells.append(
                {
                    "budget_kb": b,
                    "cardinality": c,
                    "card_val": float(c),
                    "trials_a": trials_a,
                    "trials_b": trials_b,
                    "sim_mode_a": ra["sim_mode"],
                    "rmse_a_pct": round(rmse_a, 4),
                    "rmse_a_se_pct": round(rse_a, 4),
                    "rmse_b_pct": round(rmse_b, 4),
                    "rmse_b_se_pct": round(rse_b, 4),
                    "mvp_a": round(mvp_a, 5),
                    "mvp_a_se": round(se_a, 5),
                    "mvp_a_ci95": ci_a,
                    "mvp_b": round(mvp_b, 5),
                    "mvp_b_se": round(se_b, 5),
                    "mvp_b_ci95": ci_b,
                    "mvp_diff_b_minus_a": round(mvp_b - mvp_a, 5),
                    "z_score_mvp": round(z_mvp, 4) if math.isfinite(z_mvp) else None,
                    "z_score_rmse": round(z_rmse, 4) if math.isfinite(z_rmse) else None,
                    "abs_z_lt_1_96": abs(z_mvp) < 1.96 if math.isfinite(z_mvp) else False,
                    "ci95_overlap": ci95_overlap,
                    "a_beats_b_point": rmse_a < rmse_b,
                }
            )

    cells_sorted = sorted(
        cells,
        key=lambda x: abs(x["z_score_mvp"]) if x["z_score_mvp"] is not None else float("inf"),
    )

    def summarize_subset(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub:
            return {"count": 0}
        mean_a = sum(x["mvp_a"] for x in sub) / len(sub)
        mean_b = sum(x["mvp_b"] for x in sub) / len(sub)
        # Across independent cells, Var(mean) = sum(se_i^2) / K^2
        se_mean_a = math.sqrt(sum(x["mvp_a_se"] ** 2 for x in sub)) / len(sub)
        se_mean_b = math.sqrt(sum(x["mvp_b_se"] ** 2 for x in sub)) / len(sub)
        comb = math.hypot(se_mean_a, se_mean_b)
        z_pool = (mean_b - mean_a) / comb if comb > 0 else None
        return {
            "count": len(sub),
            "point_wins_a": sum(1 for x in sub if x["a_beats_b_point"]),
            "total_trials_a": sum(x["trials_a"] for x in sub),
            "abs_z_lt_1_96_count": sum(1 for x in sub if x["abs_z_lt_1_96"]),
            "ci95_overlap_count": sum(1 for x in sub if x["ci95_overlap"]),
            "mean_mvp_a": round(mean_a, 5),
            "mean_mvp_a_se": round(se_mean_a, 5),
            "mean_mvp_b": round(mean_b, 5),
            "mean_mvp_b_se": round(se_mean_b, 5),
            "pooled_diff_b_minus_a": round(mean_b - mean_a, 5),
            "pooled_z_score": round(z_pool, 3) if z_pool is not None else None,
        }

    dense_1e6_1e9 = [x for x in cells if 1e6 <= x["card_val"] <= 1e9]
    dense_all_ge_1e6 = [x for x in cells if x["card_val"] >= 1e6]
    finite_cells = [x for x in cells if x["z_score_mvp"] is not None]
    min_cell = min(
        finite_cells,
        key=lambda x: x["z_score_mvp"] if x["z_score_mvp"] is not None else float("inf"),
    ) if finite_cells else None

    return {
        "method_a": method_a,
        "method_b": method_b,
        "all_cells": summarize_subset(cells),
        "dense_1e6_to_1e9": summarize_subset(dense_1e6_1e9),
        "dense_all_ge_1e6": summarize_subset(dense_all_ge_1e6),
        "min_margin_cell": min_cell,
        "closest_10_cells": cells_sorted[:10],
    }


def extract_matches(path: Path, patterns: Dict[str, str]) -> Dict[str, Any]:
    if not path.is_file():
        return {"found": False, "file": path.name}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: Dict[str, List[Dict[str, Any]]] = {}
    for key, pat in patterns.items():
        rx = re.compile(pat)
        matched = []
        for idx, ln in enumerate(lines, start=1):
            if rx.search(ln):
                matched.append({"line": idx, "text": ln.strip()})
        hits[key] = matched
    return {
        "found": True,
        "file": path.name,
        "line_count": len(lines),
        "matches": hits,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent
    repo_root = artifact_dir.parent.parent.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=artifact_dir / "out")
    ap.add_argument(
        "--official-bench-dir",
        type=Path,
        default=repo_root / "pilots" / "S1-osbmw-v2-solid" / "official_bench",
    )
    ap.add_argument(
        "--official-hash4j-dir",
        type=Path,
        default=repo_root / "pilots" / "S1-osbmw-v2-solid" / "official_hash4j_bench",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=artifact_dir / "out" / "m2_closest_cells_and_wrappers.json",
    )
    args = ap.parse_args()

    sw = read_tsv(args.out_dir / "m2_equal_byte_sweep.tsv")
    by_key = {(r["budget_kb"], r["method"], r["cardinality"]): r for r in sw}
    budgets = ["0.5KB", "1KB", "2KB", "4KB", "8KB", "16KB"]
    cards = sorted({r["cardinality"] for r in sw}, key=float)

    ds_pareto = read_tsv(args.out_dir / "official_datasketches_pareto.tsv")
    h4j_pareto = read_tsv(args.out_dir / "official_hash4j_pareto.tsv")

    bench_cpp = extract_matches(
        args.official_bench_dir / "bench_datasketches.cpp",
        {
            "alloc_tracker": r"(TrackedAlloc|g_live_bytes|g_peak_bytes)",
            "cpc_ctor": r"TrackedCpc\s+\w+\(",
            "hll_ctor": r"(TrackedHll\s+\w+\(|bench_sys_hll\(|bool\s+full_sz)",
            "hll_union_get_result": r"u\.get_result\(",
            "serialize_and_sizes": r"(serialize\(\)\.size|get_compact_serialization_bytes|get_updatable_serialization_bytes)",
            "seed_and_mixer": r"(hash_item|DEFAULT_SEED|splitmix64)",
        },
    )
    bench_real_cpp = extract_matches(
        args.official_bench_dir / "bench_datasketches_real.cpp",
        {
            "hash_item": r"hash_item",
            "cpc_and_hll_modes": r"(TrackedCpc|TrackedHll|HLL_4|HLL_8)",
            "budget_lgk_map": r"(lg_k|blab|budgets)",
        },
    )
    hash4j_java = extract_matches(
        args.official_hash4j_dir / "src" / "main" / "java" / "bench" / "BenchHash4j.java",
        {
            "hash_item": r"hashItem\(",
            "sketch_ctors": r"(UltraLogLog\.create|HyperLogLog\.create)",
            "estimators": r"getDistinctCountEstimate",
            "state_bytes": r"getState\(\)\.length",
            "merge_calls": r"\.(add)\(",
        },
    )

    n1e6_ds = [
        {
            "method": r["method"],
            "lg_k": int(r["lg_k"]),
            "ram_live_bytes": int(float(r["ram_live_bytes"])),
            "ram_peak_bytes": int(float(r["ram_peak_bytes"])),
            "wire_compact_bytes": int(float(r["wire_compact_bytes"])),
            "rmse_pct": float(r["rmse_pct"]),
            "mvp_ram_live": float(r["mvp_ram_live"]),
            "mvp_wire_compact": float(r["mvp_wire_compact"]),
            "trials": int(r["trials"]),
        }
        for r in ds_pareto
        if float(r["n"]) == 1e6
    ]
    n1e6_h4j = [
        {
            "method": r["method"],
            "estimator": r["estimator"],
            "p": int(r["p"]),
            "state_bytes": int(r["state_bytes"]),
            "rmse_pct": float(r["rmse_pct"]),
            "mvp_state": float(r["mvp_state"]),
            "trials": int(r["trials"]),
        }
        for r in h4j_pareto
        if float(r["n"]) == 1e6 and int(r["p"]) in (11, 12)
    ]

    summary = {
        "audit_id": "E-11",
        "equal_byte_sweep_audit": {
            "osbmw256b_vs_ell24_all_budgets": audit_pair_cells(
                by_key, budgets, cards, "OS-BMW-5b-256B", "ELL(2,24)"
            ),
            "osbmw5b_vs_ell20_4kb": audit_pair_cells(
                by_key, ["4KB"], cards, "OS-BMW-5b", "ELL(2,20)"
            ),
            "osbmw5b_vs_ell24_4kb": audit_pair_cells(
                by_key, ["4KB"], cards, "OS-BMW-5b", "ELL(2,24)"
            ),
        },
        "official_n1e6_operating_points": {
            "datasketches_cpp": n1e6_ds,
            "hash4j_java": n1e6_h4j,
        },
        "wrapper_audit": {
            "datasketches_cpp": bench_cpp,
            "datasketches_real_cpp": bench_real_cpp,
            "hash4j_java": hash4j_java,
            "checks": {
                "cpc_uses_default_seed_and_tracked_alloc": (
                    bench_cpp.get("found", False)
                    and len(bench_cpp["matches"]["cpc_ctor"]) > 0
                    and len(bench_cpp["matches"]["alloc_tracker"]) > 0
                ),
                "hll4_dyn_vs_full_both_benchmarked": (
                    bench_cpp.get("found", False)
                    and any("Apache_HLL4_Dyn" in m["text"] and "false" in m["text"] for m in bench_cpp["matches"]["hll_ctor"])
                    and any("Apache_HLL4_Full" in m["text"] and "true" in m["text"] for m in bench_cpp["matches"]["hll_ctor"])
                    and any("HLL_4" in m["text"] and "full_sz" in m["text"] for m in bench_cpp["matches"]["hll_ctor"])
                ),
                "hash4j_reports_both_default_fgra_and_mle": (
                    hash4j_java.get("found", False)
                    and any("MAXIMUM_LIKELIHOOD_ESTIMATOR" in m["text"] for m in hash4j_java["matches"]["estimators"])
                    and any("getDistinctCountEstimate()" in m["text"] for m in hash4j_java["matches"]["estimators"])
                ),
                "consistent_64bit_splitmix_mixer_across_rust_cpp_java": (
                    bench_cpp.get("found", False)
                    and bench_real_cpp.get("found", False)
                    and hash4j_java.get("found", False)
                ),
            },
        },
    }

    args.dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    p256 = summary["equal_byte_sweep_audit"]["osbmw256b_vs_ell24_all_budgets"]
    print(
        f"[E-11] wrote {args.dest.name}: 256B vs ELL24 "
        f"all={p256['all_cells']['point_wins_a']}/{p256['all_cells']['count']} "
        f"(|z|<1.96 in {p256['all_cells']['abs_z_lt_1_96_count']}/{p256['all_cells']['count']}), "
        f"pooled dense z={p256['dense_all_ge_1e6']['pooled_z_score']}, "
        f"min-margin cell={p256['min_margin_cell']['budget_kb']}@{p256['min_margin_cell']['cardinality']} "
        f"(z={p256['min_margin_cell']['z_score_mvp']}); "
        f"wrapper_checks={summary['wrapper_audit']['checks']}"
    )


if __name__ == "__main__":
    main()
