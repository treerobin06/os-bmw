#!/usr/bin/env python3
"""
build_story_figs.py — Fail-Closed Vector Figure Builder for OS-BMW (manuscript_v2).

Reads ONLY read-only raw TSV/JSON artifacts from ../artifact/out/ and generates
four publication-grade vector PDFs under manuscript_v2/figures/:
  1. fig_equal_byte_sweep.pdf
  2. fig_ram_wire_pareto.pdf
  3. fig_datafusion_scaling.pdf
  4. fig_geometry_ablation.pdf

Strict guarantees:
  - Standard library + numpy + matplotlib only.
  - Fail-closed: verifies input files, required columns, exact expected row/key counts,
    and finite values; exits non-zero on any violation.
  - Unified Okabe-Ito colorblind-safe palette & IEC binary KiB units (bytes / 1024).
  - Embedded TrueType fonts (pdf.fonttype=42), clean spines, crisp >=7.5pt typography.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MS_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = MS_DIR.parent / "artifact" / "out"
DEFAULT_FIG_DIR = MS_DIR / "figures"


# ---------------------------------------------------------------------------
# Unified Okabe-Ito Publication Palette (Shared with TikZ Figs. 1-2)
# ---------------------------------------------------------------------------
PALETTE = {
    "OS-BMW-5b": "#1F4E79",       # deep navy (flagship 2.7 KiB, g=512, k=24)
    "OS-BMW-256B": "#009E73",     # bluish green / teal (4.0 KiB, g=384, k=16)
    "OS-BMW-16b": "#6A3D9A",      # rich purple (5.4 KiB / Rank)
    "OS-BMW-Rank": "#6A3D9A",     # rich purple
    "OS-BMW-Chunk": "#3B8EA5",    # steel cyan-blue (chunk variant)
    "ELL(2,24)": "#D55E00",       # vermilion
    "ELL(2,20)": "#E69F00",       # amber / orange (also 4-bit ref in Fig 6)
    "ULL": "#56B4E9",             # sky blue
    "HLL++": "#7F7F7F",           # neutral gray (HLL-6b MLE)
    "HBS": "#CC79A7",             # reddish purple / rose (HBS [17])
    "Apache_CPC": "#8C564B",      # warm brown (CPC / Apache CPC)
    "AVLL": "#C44E52",            # muted crimson
    "DF_HLL": "#222222",          # near-black charcoal (DF HLL 16 KiB)
    # Distinct secondary accents for official libraries & dense bar pairs
    "Apache_HLL4": "#332288",     # indigo
    "Hash4j_ULL": "#6B6B00",      # dark olive (high-contrast in grayscale)
    "OS-BMW-5b-Dense": "#8FAADC", # light navy tint (paired with #1F4E79)
    "ELL24-Dense": "#F4A582",     # light vermilion tint (paired with #D55E00)
}


# ---------------------------------------------------------------------------
# Expected Data Schemas (Fail-Closed Constants)
# ---------------------------------------------------------------------------
EXPECTED_DF_WLS = (
    "W1_1T_16Kg_32M",
    "W2_8T_8Kg_40M",
    "W3_8T_Zipf_4Kg_40M",
    "W4_16T_16Kg_64M",
    "W5_32T_16Kg_64M",
    "W6_8T_65Kg_64M",
    "W7_8T_128g_768M_Dense",
    "W8_16T_256g_768M_Dense",
    "W9_8T_ClickBench_32M",
)

EXPECTED_DF_METHODS = (
    "OS-BMW-5b(k24,Slab+Sp)",
    "OS-BMW-5b-256B(Slab+Sp)",
    "OS-BMW-5b(16blk,Slab+Sp)",
    "ELL(2,24,Slab+Sp)",
    "ULL(Slab+Sp)",
    "OS-BMW-5b(k24)",
    "OS-BMW-5b-256B",
    "ELL(2,24)",
    "ULL",
    "HLL++(6b)",
    "AVLL(base56)",
    "DF-BuiltIn-HLL(16KB)",
)


# ---------------------------------------------------------------------------
# Matplotlib Global Publication Styling
# ---------------------------------------------------------------------------
def configure_style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.9,
        "hatch.linewidth": 0.45,
        "axes.titlesize": 9.4,
        "axes.titleweight": "normal",
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.40,
        "grid.color": "#B5B5B5",
        "lines.linewidth": 1.5,
        "lines.markersize": 4.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# ---------------------------------------------------------------------------
# Fail-Closed I/O & Formatting Helpers
# ---------------------------------------------------------------------------
def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Fail-closed check failed: {msg}")


def read_tsv(
    path: Path,
    required_cols: Sequence[str],
    expected_rows: Optional[int] = None,
) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required TSV missing or empty: {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        require(bool(reader.fieldnames), f"TSV has no header: {path}")
        missing = [c for c in required_cols if c not in reader.fieldnames]
        require(not missing, f"TSV {path.name} missing columns: {missing}")
        for idx, row in enumerate(reader, start=2):
            require(
                None not in row and all(v is not None for v in row.values()),
                f"Malformed TSV row {idx} in {path.name}",
            )
            rows.append(row)
    require(len(rows) > 0, f"TSV contains zero rows: {path}")
    if expected_rows is not None:
        require(
            len(rows) == expected_rows,
            f"Expected {expected_rows} rows in {path.name}, got {len(rows)}",
        )
    return rows


def read_json(path: Path) -> Dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required JSON missing or empty: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    require(isinstance(data, dict) and len(data) > 0, f"Invalid JSON object in {path}")
    return data


def fnum_opt(val: str) -> float:
    s = str(val).strip()
    if s in ("NA", "NaN", "nan"):
        return float("nan")
    v = float(s)
    require(math.isfinite(v), f"Non-finite numeric value: {val!r}")
    return v


def fnum_req(val: str, ctx: str = "") -> float:
    v = fnum_opt(val)
    require(math.isfinite(v), f"Expected finite value{f' ({ctx})' if ctx else ''}, got {val!r}")
    return v


def fmt_kib(n_bytes: float, decimals: int = 1) -> str:
    """Format byte count in IEC binary KiB (bytes / 1024): 2.7 KiB, 4.0 KiB, 5.4 KiB, 16 KiB."""
    require(math.isfinite(n_bytes) and n_bytes > 0.0, f"Invalid byte count: {n_bytes}")
    kib = n_bytes / 1024.0
    if abs(kib - round(kib)) < 1e-6 and kib >= 10.0:
        return f"{int(round(kib))} KiB"
    return f"{kib:.{decimals}f} KiB"


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 3: Equal-Byte 64-Bit Sweep & Dynamic-Range Bias (fig_equal_byte_sweep.pdf)
# ---------------------------------------------------------------------------
def build_fig_equal_byte_sweep(out_dir: Path, fig_dir: Path) -> Dict[str, float]:
    sw = read_tsv(
        out_dir / "m2_equal_byte_sweep.tsv",
        required_cols=[
            "budget_kb", "method", "ram_bytes", "wire_bytes",
            "cardinality", "trials", "rel_bias_pct", "rmse_pct", "sim_mode",
        ],
        expected_rows=948,
    )

    def get_4kb(method: str) -> Tuple[List[float], List[float], List[float], float]:
        pts = []
        ram_vals = []
        for r in sw:
            if r["budget_kb"] != "4KB" or r["method"] != method:
                continue
            if r["sim_mode"] == "na_order_dependent_no_jump":
                continue
            card = fnum_req(r["cardinality"], f"4KB {method} card")
            rmse = fnum_req(r["rmse_pct"], f"4KB {method} rmse")
            bias = fnum_req(r["rel_bias_pct"], f"4KB {method} bias")
            ram = fnum_req(r["ram_bytes"], f"4KB {method} ram")
            pts.append((card, rmse, bias))
            ram_vals.append(ram)
        require(len(pts) > 0, f"No 4KB rows for method {method}")
        pts.sort(key=lambda t: t[0])
        return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], ram_vals[0]

    # Compute exact equal-byte wins across all 6 budgets (72/72 overall, 24/24 dense [1e6, 1e9])
    dense_cards = (1e6, 1e7, 1e8, 1e9)
    all_budgets = ("0.5KB", "1KB", "2KB", "4KB", "8KB", "16KB")
    by_bm: Dict[Tuple[str, str, float], float] = {}
    all_cards_set = set()
    max_abs_bias_semi_excl = 0.0
    max_abs_bias_all_semi = 0.0
    max_abs_bias_osbmw = 0.0
    for r in sw:
        if r["sim_mode"] == "na_order_dependent_no_jump":
            continue
        c = fnum_req(r["cardinality"], "sweep card")
        rm = fnum_req(r["rmse_pct"], "sweep rmse")
        bi = fnum_req(r["rel_bias_pct"], "sweep bias")
        by_bm[(r["budget_kb"], r["method"], c)] = rm
        all_cards_set.add(c)
        if r["method"].startswith("OS-BMW"):
            max_abs_bias_osbmw = max(max_abs_bias_osbmw, abs(bi))
        if r["method"] not in ("AVLL", "HLL_4-Fixed"):
            max_abs_bias_all_semi = max(max_abs_bias_all_semi, abs(bi))
            if not (r["budget_kb"] == "0.5KB" and r["method"] == "HBS"):
                max_abs_bias_semi_excl = max(max_abs_bias_semi_excl, abs(bi))

    ratios: List[float] = []
    dense_wins, dense_total = 0, 0
    all_wins, all_total = 0, 0
    for b in all_budgets:
        for c in sorted(all_cards_set):
            v1 = by_bm.get((b, "OS-BMW-5b-256B", c))
            v2 = by_bm.get((b, "ELL(2,24)", c))
            require(v1 is not None and v2 is not None, f"Missing equal-byte pair ({b}, {c})")
            all_total += 1
            if v1 < v2:
                all_wins += 1
            if c in dense_cards:
                dense_total += 1
                if v1 < v2:
                    dense_wins += 1
                ratios.append(v1 / v2)
    require(dense_total == 24 and dense_wins == 24, f"Expected 24/24 dense wins, got {dense_wins}/{dense_total}")
    require(all_total == 72 and all_wins == 72, f"Expected 72/72 overall wins, got {all_wins}/{all_total}")
    geom_drop_pct = (1.0 - math.exp(sum(math.log(x) for x in ratios) / len(ratios))) * 100.0
    bound_excl_hbs05 = math.ceil(max_abs_bias_semi_excl * 1000.0) / 1000.0
    bound_all_semi = math.ceil(max_abs_bias_all_semi * 1000.0) / 1000.0
    bound_osbmw = math.ceil(max_abs_bias_osbmw * 1000.0) / 1000.0

    # IDENTICAL 8-method set plotted in BOTH (a) and (b) => 100% 1-to-1 shared legend!
    sweep_specs = [
        ("OS-BMW-5b-256B", "OS-BMW {kib} (hyb. $\\leq 10^5$, dense $\\geq 10^6$)", PALETTE["OS-BMW-256B"], "o", "-",  1.85, 7),
        ("OS-BMW-5b",      "OS-BMW {kib} ($k{{=}}24$, dense)",              PALETTE["OS-BMW-5b"],   "D", "--", 1.45, 6),
        ("ELL(2,24)",      "ELL(2,24) ({kib})",                             PALETTE["ELL(2,24)"],   "^", "-",  1.55, 6),
        ("ULL",            "ULL ({kib})",                                   PALETTE["ULL"],         "v", "--", 1.35, 5),
        ("CPC",            "CPC (Rust, {kib} heap)",                        PALETTE["Apache_CPC"],  "P", ":",  1.40, 5),
        ("HBS",            "HBS (Rust, {kib} RAM)",                          PALETTE["HBS"],         "<", "-.", 1.35, 4),
        ("HLL++",          "HLL-6b MLE ({kib})",                            PALETTE["HLL++"],       "x", ":",  1.30, 4),
        ("AVLL",           r"AVLL (direct $n{\leq}10^6$)",                  PALETTE["AVLL"],        "s", "-",  1.45, 6),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.24))

    for method, tmpl, col, mk, ls, lw, zo in sweep_specs:
        xs, rmses, biases, ram_b = get_4kb(method)
        lbl = tmpl.format(kib=fmt_kib(ram_b)) if "{kib}" in tmpl else tmpl
        mk_kw = dict(
            marker=mk, linestyle=ls, linewidth=lw, markersize=3.9, zorder=zo,
            markeredgewidth=0.55 if mk != "x" else 1.1,
            markeredgecolor="white" if mk != "x" else col,
        )
        ax1.plot(xs, rmses, label=lbl, color=col, **mk_kw)
        ax2.plot(xs, biases, label=lbl, color=col, **mk_kw)

    # --- Panel (a) decorations ---
    ax1.axvspan(1e6, 1e9, color="#E8F4F0", alpha=0.55, zorder=0)
    ax1.text(
        3.16e7, 1.86, "dense", fontsize=7.2, color="#3A7A65",
        ha="center", va="top", style="italic", zorder=2,
    )

    # Transparent sparse-mode callout at n=10^2
    ax1.annotate(
        "0% RMSE\nat $10^2$",
        xy=(1.2e2, 0.025),
        xytext=(1.4e3, 0.13),
        fontsize=6.0, fontweight="bold", color=PALETTE["OS-BMW-256B"],
        va="center", ha="left", linespacing=1.08,
        arrowprops=dict(
            arrowstyle="->,head_width=0.17,head_length=0.24",
            color=PALETTE["OS-BMW-256B"], lw=0.85, connectionstyle="arc3,rad=0.08",
        ),
        zorder=8,
    )

    # Explicit multi-budget dense & overall win callout in open bottom-right whitespace
    ax1.annotate(
        f"256B fam. both-dense (0.5–16 KiB, $10^6$–$10^9$): {dense_wins}/{dense_total} ($-{geom_drop_pct:.1f}\\%$)\n"
        f"Full sweep (incl. hybrid $\\leq 10^5$ vs. dense ELL): {all_wins}/{all_total}",
        xy=(4.5e7, 0.96),
        xytext=(3.8e19, 0.44),
        fontsize=6.0, color="#1A5C46", fontweight="bold",
        ha="right", va="center", linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.22", fc="#F4FAF8", ec=PALETTE["OS-BMW-256B"], lw=0.75),
        arrowprops=dict(
            arrowstyle="->,head_width=0.17,head_length=0.24",
            color=PALETTE["OS-BMW-256B"], lw=0.85, connectionstyle="arc3,rad=-0.16",
        ),
        zorder=8,
    )

    ax1.set_xscale("log")
    ax1.set_xlim(45, 5e19)
    ax1.set_ylim(-0.08, 1.94)
    ax1.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax1.set_yticklabels(["0.0", "0.5", "1.0", "1.5"])
    ax1.set_xticks([1e2, 1e6, 1e10, 1e14, 1e18])
    ax1.xaxis.set_minor_locator(mticker.NullLocator())
    ax1.set_xlabel(r"True Cardinality $n$ ($10^2$ to $2^{64}$)")
    ax1.set_ylabel("Relative RMSE (%)")
    ax1.set_title("(a) RMSE at Nominal 4 KiB (64-Bit Range)", pad=5)
    style_axes(ax1)

    # --- Panel (b) decorations ---
    ax2.axhspan(-bound_excl_hbs05, bound_excl_hbs05, color="#DCE8F2", alpha=0.40, zorder=0)
    ax2.axhline(bound_excl_hbs05, color="#7A9CC0", linestyle=":", linewidth=0.75, zorder=1)
    ax2.axhline(-bound_excl_hbs05, color="#7A9CC0", linestyle=":", linewidth=0.75, zorder=1)
    ax2.axhline(0.0, color="#333333", linewidth=0.75, zorder=2)
    ax2.axvline(1e6, color="#666666", linestyle="--", linewidth=0.95, zorder=2)

    _bbox_clean = dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.88)
    ax2.text(
        6.0e5, 0.70, "← direct ($n{\\leq}10^5$)\n(AVLL $\\leq 10^6$)",
        fontsize=6.0, color="#444444", ha="right", va="top", linespacing=1.08,
        bbox=_bbox_clean, zorder=7,
    )
    ax2.text(
        3.2e6, 0.70, "Poisson jump sim. → (12 swept families, $n{\\geq}10^6$)",
        fontsize=6.2, color="#444444", ha="left", va="top", linespacing=1.10,
        bbox=_bbox_clean, zorder=7,
    )
    ax2.text(
        3.5e19, -0.59,
        rf"$\pm {bound_excl_hbs05:.3f}\%$ max $|\mathrm{{bias}}|$ (12 families excl. 0.5 KiB HBS)",
        fontsize=6.3, color="#3A5A78", ha="right", va="bottom", linespacing=1.10,
        bbox=_bbox_clean, zorder=7,
    )

    ax2.set_xscale("log")
    ax2.set_xlim(45, 5e19)
    ax2.set_ylim(-0.62, 0.72)
    ax2.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax2.set_xticks([1e2, 1e6, 1e10, 1e14, 1e18])
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_xlabel(r"True Cardinality $n$ ($10^2$ to $2^{64}$)")
    ax2.set_ylabel("Relative Bias (%)")
    ax2.set_title(rf"(b) Dynamic-Range Bias (OS-BMW max $|\mathrm{{bias}}| \leq {bound_osbmw:.3f}\%$)", pad=5)
    style_axes(ax2)

    # Shared 2-row top legend (exact 8 plotted series in both panels, 4 columns x 2 rows)
    handles, labels = ax1.get_legend_handles_labels()
    require(len(labels) == 8, f"Expected 8 legend entries in sweep, got {len(labels)}")
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=True,
        facecolor="white", edgecolor="#CCCCCC", framealpha=0.98,
        fontsize=7.0, columnspacing=0.75, handlelength=1.45,
        borderpad=0.30, labelspacing=0.28, bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.83], w_pad=1.35)
    fig.savefig(fig_dir / "fig_equal_byte_sweep.pdf")
    plt.close(fig)
    return {
        "dense_wins": float(dense_wins),
        "dense_total": float(dense_total),
        "all_wins": float(all_wins),
        "all_total": float(all_total),
        "dense_drop_pct": geom_drop_pct,
        "bound_excl_hbs05": bound_excl_hbs05,
        "bound_all_semi": bound_all_semi,
    }


# ---------------------------------------------------------------------------
# Figure 4: Active RAM vs. Cold-Storage Wire Pareto (fig_ram_wire_pareto.pdf)
# ---------------------------------------------------------------------------
def build_fig_ram_wire_pareto(out_dir: Path, fig_dir: Path) -> None:
    sw = read_tsv(
        out_dir / "m2_equal_byte_sweep.tsv",
        required_cols=["method", "cardinality", "ram_bytes", "wire_bytes", "rmse_pct", "sim_mode"],
        expected_rows=948,
    )
    ds = read_tsv(
        out_dir / "official_datasketches_pareto.tsv",
        required_cols=["method", "n", "ram_live_bytes", "wire_compact_bytes", "rmse_pct"],
        expected_rows=100,
    )
    h4j = read_tsv(
        out_dir / "official_hash4j_pareto.tsv",
        required_cols=["method", "estimator", "p", "n", "state_bytes", "rmse_pct"],
        expected_rows=72,
    )
    reconcile = read_json(out_dir / "m0_m1_reconcile.json")
    rec_block = reconcile["m0_reconciliation_3_0061_vs_2_9977"]
    rank_mvp = float(rec_block["rank_wire_k60_byte_mvp"])
    rank_b0 = int(rec_block["rank_wire_k60_fixed_bytes"])
    require(
        math.isfinite(rank_mvp) and rank_mvp > 0.0 and rank_b0 == 1277,
        f"Unexpected rank projection values: mvp={rank_mvp}, b0={rank_b0}",
    )
    rank_kib_str = fmt_kib(float(rank_b0), decimals=2)

    n_target = 1e6

    def get_sweep(method: str, col_x: str) -> Tuple[List[float], List[float]]:
        pts = []
        for r in sw:
            if r["method"] != method:
                continue
            if abs(fnum_req(r["cardinality"]) - n_target) >= 1.0:
                continue
            if r["sim_mode"] == "na_order_dependent_no_jump":
                continue
            x = fnum_req(r[col_x], f"pareto {method} {col_x}")
            y = fnum_req(r["rmse_pct"], f"pareto {method} rmse")
            pts.append((x, y))
        require(len(pts) >= 4, f"Too few n=1e6 points for {method}: {len(pts)}")
        pts.sort(key=lambda t: t[0])
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.38), sharey=True)

    # Iso-MVP guide lines: RMSE(%) = 100 * sqrt(MVP / (8 * bytes))
    xs_grid = np.geomspace(380, 27000, 260)
    iso_styles = [
        (2.0, ":",  "#B8B8B8"),
        (3.0, "--", "#B8B8B8"),
        (3.8, "-.", "#B8B8B8"),
        (6.4, ":",  "#B8B8B8"),
    ]
    for mvp_c, ls, col in iso_styles:
        ys = 100.0 * np.sqrt(mvp_c / (8.0 * xs_grid))
        ax1.plot(xs_grid, ys, color=col, linestyle=ls, linewidth=0.75, zorder=1)
        ax2.plot(xs_grid, ys, color=col, linestyle=ls, linewidth=0.75, zorder=1)

    # Panel (a): Active Mergeable RAM
    ram_series = [
        ("OS-BMW-5b",      r"OS-BMW ($g{=}512, k{=}24$)", PALETTE["OS-BMW-5b"],    "o", "-",  2.00, 7),
        ("OS-BMW-5b-256B", r"OS-BMW ($g{=}384, k{=}16$)", PALETTE["OS-BMW-256B"],  "s", "-",  1.65, 6),
        ("OS-BMW-Chunk",   "OS-BMW-Chunk",                PALETTE["OS-BMW-Chunk"], "D", "--", 1.40, 6),
        ("ELL(2,20)",      "ELL(2,20)",                   PALETTE["ELL(2,20)"],    "p", "--", 1.35, 4),
        ("ELL(2,24)",      "ELL(2,24)",                   PALETTE["ELL(2,24)"],    "^", "-",  1.50, 5),
        ("ULL",            "ULL (Rust)",                  PALETTE["ULL"],          "v", "-",  1.35, 4),
        ("HLL++",          "HLL-6b MLE",                  PALETTE["HLL++"],        "x", ":",  1.25, 3),
        ("HBS",            "HBS (Rust)",                  PALETTE["HBS"],          "<", "-.", 1.40, 5),
    ]
    for m, lbl, col, mk, ls, lw, zo in ram_series:
        xs, ys = get_sweep(m, "ram_bytes")
        ax1.plot(
            xs, ys, label=lbl, color=col, marker=mk, linestyle=ls,
            linewidth=lw, markersize=4.0, zorder=zo,
            markeredgewidth=0.45 if mk != "x" else 1.1,
            markeredgecolor="white" if mk != "x" else col,
        )

    for m, lbl, col, mk, ls in [
        ("Apache_CPC",      "Apache CPC",  PALETTE["Apache_CPC"],  "P", "--"),
        ("Apache_HLL4_Dyn", "Apache HLL4", PALETTE["Apache_HLL4"], "*", "-."),
    ]:
        pts = sorted(
            [
                (fnum_req(r["ram_live_bytes"]), fnum_req(r["rmse_pct"]))
                for r in ds
                if r["method"] == m and abs(fnum_req(r["n"]) - n_target) < 1.0
            ],
            key=lambda t: t[0],
        )
        require(len(pts) == 4, f"Expected 4 DataSketches RAM points for {m}, got {len(pts)}")
        ax1.plot(
            [p[0] for p in pts], [p[1] for p in pts], label=lbl, color=col,
            marker=mk, linestyle=ls, linewidth=1.40, markersize=4.6 if mk == "*" else 4.1,
            zorder=5, markeredgewidth=0.4, markeredgecolor="white",
        )

    h4_ull_pts = sorted(
        [
            (fnum_req(r["state_bytes"]), fnum_req(r["rmse_pct"]))
            for r in h4j
            if r["method"] == "Hash4j_UltraLogLog"
            and r["estimator"] == "MLE"
            and abs(fnum_req(r["n"]) - n_target) < 1.0
        ],
        key=lambda t: t[0],
    )
    require(len(h4_ull_pts) == 3, f"Expected 3 Hash4j ULL points, got {len(h4_ull_pts)}")
    ax1.plot(
        [p[0] for p in h4_ull_pts], [p[1] for p in h4_ull_pts],
        label="Hash4j ULL (state)", color=PALETTE["Hash4j_ULL"], marker="d",
        linestyle=":", linewidth=1.45, markersize=4.2, zorder=5,
        markeredgewidth=0.55, markeredgecolor="#111111",
    )

    # Panel (b): Cold-Storage Wire Pareto
    wire_series = [
        ("OS-BMW-5b",    r"OS-BMW ($g{=}512, k{=}24$)", PALETTE["OS-BMW-5b"],    "o", "-",  1.90, 6),
        ("OS-BMW-Chunk", "OS-BMW-Chunk",                PALETTE["OS-BMW-Chunk"], "D", "--", 1.35, 5),
        ("ELL(2,20)",    "ELL(2,20)",                   PALETTE["ELL(2,20)"],    "p", "--", 1.30, 4),
        ("ELL(2,24)",    "ELL(2,24)",                   PALETTE["ELL(2,24)"],    "^", "-",  1.40, 4),
        ("HLL++",        "HLL-6b MLE",                  PALETTE["HLL++"],        "x", ":",  1.25, 3),
        ("HBS",          "HBS (Rust)",                  PALETTE["HBS"],          "<", "-.", 1.40, 5),
    ]
    for m, lbl, col, mk, ls, lw, zo in wire_series:
        xs, ys = get_sweep(m, "wire_bytes")
        ax2.plot(
            xs, ys, label=lbl, color=col, marker=mk, linestyle=ls,
            linewidth=lw, markersize=4.0, zorder=zo,
            markeredgewidth=0.45 if mk != "x" else 1.1,
            markeredgecolor="white" if mk != "x" else col,
        )

    rank_bytes = [rank_b0, rank_b0 * 2, rank_b0 * 4]
    rank_rmse = [100.0 * math.sqrt(rank_mvp / (8.0 * b)) for b in rank_bytes]
    ax2.plot(
        rank_bytes, rank_rmse,
        label=f"OS-BMW-Rank (proj. from {rank_b0:,} B)",
        color=PALETTE["OS-BMW-Rank"], marker="h", linestyle="-.",
        linewidth=1.60, markersize=4.4, zorder=7,
        markeredgewidth=0.45, markeredgecolor="white",
    )

    for m, lbl, col, mk, ls in [
        ("Apache_CPC",      "Apache CPC",  PALETTE["Apache_CPC"],  "P", "--"),
        ("Apache_HLL4_Dyn", "Apache HLL4", PALETTE["Apache_HLL4"], "*", "-."),
    ]:
        pts = sorted(
            [
                (fnum_req(r["wire_compact_bytes"]), fnum_req(r["rmse_pct"]))
                for r in ds
                if r["method"] == m and abs(fnum_req(r["n"]) - n_target) < 1.0
            ],
            key=lambda t: t[0],
        )
        require(len(pts) == 4, f"Expected 4 DataSketches Wire points for {m}, got {len(pts)}")
        ax2.plot(
            [p[0] for p in pts], [p[1] for p in pts], label=lbl, color=col,
            marker=mk, linestyle=ls, linewidth=1.60 if m == "Apache_CPC" else 1.35,
            markersize=4.6 if mk == "*" else 4.2, zorder=6,
            markeredgewidth=0.4, markeredgecolor="white",
        )

    ax2.plot(
        [p[0] for p in h4_ull_pts], [p[1] for p in h4_ull_pts],
        label="Hash4j ULL (state)", color=PALETTE["Hash4j_ULL"], marker="d",
        linestyle=":", linewidth=1.45, markersize=4.2, zorder=5,
        markeredgewidth=0.55, markeredgecolor="#111111",
    )

    # Clear compact note in Panel (b) empty lower-left whitespace for OS-BMW-Rank projection
    ax2.text(
        225.0, 0.32,
        f"OS-BMW-Rank: 1–4× {rank_b0:,} B code\n({rank_mvp:.3f} th. MVP cold-wire proj.)",
        fontsize=6.7, color="#4A2B6E", ha="left", va="bottom", linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.20", fc="#FAF8FC", ec="#C8B8DC", lw=0.65),
        zorder=4,
    )

    for ax, xlabel, title in [
        (ax1, "Active Mergeable RAM Footprint (KiB)", "(a) Active Mergeable RAM Pareto ($n=10^6$)"),
        (ax2, "Serialized Wire Size (KiB)",           "(b) Cold-Storage Wire Pareto ($n=10^6$)"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(200, 29500)
        ax.set_ylim(0.28, 6.6)
        ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([0.5, 1.0, 2.0, 4.0])
        ax.set_yticklabels(["0.5%", "1%", "2%", "4%"])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=5)
        style_axes(ax)
    ax1.set_ylabel("Relative RMSE (%)")

    # Shared 3-row top legend (12 items, 4 columns x 3 rows -> airy, zero crowding)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen: Dict[str, object] = {}
    comb_h, comb_l = [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in seen:
            seen[l] = h
            comb_h.append(h)
            comb_l.append(l)
    require(len(comb_l) == 12, f"Expected 12 unique legend labels in Pareto, got {len(comb_l)}")

    fig.legend(
        comb_h, comb_l, loc="upper center", ncol=4, frameon=True,
        facecolor="white", edgecolor="#CCCCCC", framealpha=0.98,
        fontsize=7.0, columnspacing=0.85, handlelength=1.45,
        borderpad=0.30, labelspacing=0.25, bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.795], w_pad=1.2)

    # Exact screen-space rotation angle of iso-MVP lines (slope -0.5 in log-log)
    fig.canvas.draw()
    p0 = ax1.transData.transform((1000.0, 100.0 * math.sqrt(3.0 / 8000.0)))
    p1 = ax1.transData.transform((4000.0, 100.0 * math.sqrt(3.0 / 32000.0)))
    iso_angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    # Rotated iso-MVP labels placed strictly in verified empty zones (ZERO collisions)
    iso_labels_a = [
        (2.0,  1150.0, 0.90, "center", "top"),
        (3.0, 15500.0, 0.93, "center", "top"),
        (3.8, 19200.0, 1.11, "center", "bottom"),
        (6.4, 12800.0, 0.90, "center", "top"),
    ]
    iso_labels_b = [
        (2.0,  9200.0, 0.90, "center", "top"),
        (3.0, 15500.0, 0.93, "center", "top"),
        (3.8, 19200.0, 1.11, "center", "bottom"),
        (6.4, 15500.0, 1.13, "center", "bottom"),
    ]
    for ax_t, specs in ((ax1, iso_labels_a), (ax2, iso_labels_b)):
        for mvp_c, x_pos, mult, ha_m, va_m in specs:
            y_lbl = 100.0 * math.sqrt(mvp_c / (8.0 * x_pos)) * mult
            ax_t.text(
                x_pos, y_lbl, f"MVP={mvp_c:.1f}",
                fontsize=6.9, color="#555555", ha=ha_m, va=va_m,
                rotation=iso_angle, rotation_mode="anchor",
                bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.84),
                zorder=2.5,
            )

    fig.savefig(fig_dir / "fig_ram_wire_pareto.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: DataFusion 2x2 Scaling & 12-Engine Pareto (fig_datafusion_scaling.pdf)
# ---------------------------------------------------------------------------
def build_fig_datafusion_scaling(out_dir: Path, fig_dir: Path) -> None:
    all_rows = read_tsv(
        out_dir / "m4_datafusion_v5_hardened.tsv",
        required_cols=[
            "workload", "n_threads", "n_groups", "total_rows", "method",
            "bytes_per_group", "wall_median_s", "wall_ci95_lo_s", "wall_ci95_hi_s",
            "throughput_mrows_s", "rss_peak_over_table_mb", "rmse_pct",
        ],
        expected_rows=108,
    )

    by_wm: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in all_rows:
        key = (r["workload"], r["method"])
        require(key not in by_wm, f"Duplicate DataFusion key: {key}")
        for col in (
            "total_rows", "bytes_per_group", "wall_median_s", "wall_ci95_lo_s",
            "wall_ci95_hi_s", "throughput_mrows_s", "rss_peak_over_table_mb", "rmse_pct",
        ):
            v = fnum_req(r[col], f"DF {key} {col}")
            require(v > 0.0, f"Expected positive {col} for {key}, got {v}")
        by_wm[key] = r

    expected_keys = {(w, m) for w in EXPECTED_DF_WLS for m in EXPECTED_DF_METHODS}
    require(set(by_wm.keys()) == expected_keys, "DataFusion key set mismatch")

    def kib_for(method: str) -> str:
        b = fnum_req(by_wm[("W1_1T_16Kg_32M", method)]["bytes_per_group"])
        return fmt_kib(b)

    k5b = kib_for("OS-BMW-5b(k24,Slab+Sp)")
    k256 = kib_for("OS-BMW-5b-256B(Slab+Sp)")
    k16b = kib_for("OS-BMW-5b(16blk,Slab+Sp)")
    kell = kib_for("ELL(2,24,Slab+Sp)")
    kull = kib_for("ULL(Slab+Sp)")
    khll = kib_for("HLL++(6b)")
    kavll = kib_for("AVLL(base56)")
    kdf = kib_for("DF-BuiltIn-HLL(16KB)")

    # Exact 8 engines plotted in top bar panels (a)-(b) with redundant hatch coding for grayscale
    methods_bars = [
        ("OS-BMW-5b(k24,Slab+Sp)",   f"OS-BMW {k5b} (Slab+Sp)",      PALETTE["OS-BMW-5b"],       "",     "white"),
        ("OS-BMW-5b-256B(Slab+Sp)",  f"OS-BMW {k256} (Slab+Sp)",     PALETTE["OS-BMW-256B"],     "////", "white"),
        ("OS-BMW-5b(16blk,Slab+Sp)", f"OS-BMW {k16b} (Slab+Sp)",     PALETTE["OS-BMW-16b"],      "....", "white"),
        ("ELL(2,24,Slab+Sp)",        f"ELL(2,24) ({kell}, Slab+Sp)", PALETTE["ELL(2,24)"],       "----", "white"),
        ("ULL(Slab+Sp)",             f"ULL ({kull}, Slab+Sp)",       PALETTE["ULL"],             "",     "#333333"),
        ("OS-BMW-5b(k24)",           f"OS-BMW {k5b} (Dense)",        PALETTE["OS-BMW-5b-Dense"], "////", "#1F4E79"),
        ("ELL(2,24)",                f"ELL(2,24) ({kell}, Dense)",   PALETTE["ELL24-Dense"],     "----", "#8C2D04"),
        ("DF-BuiltIn-HLL(16KB)",     f"DF HLL ({kdf})",              PALETTE["DF_HLL"],          "xxxx", "#BBBBBB"),
    ]

    wl_order = [
        ("W1_1T_16Kg_32M", "W1\n1T"),
        ("W2_8T_8Kg_40M", "W2\n8T"),
        ("W3_8T_Zipf_4Kg_40M", "W3\nZipf"),
        ("W4_16T_16Kg_64M", "W4\n16T"),
        ("W5_32T_16Kg_64M", "W5\n32T"),
        ("W6_8T_65Kg_64M", "W6\n65K"),
        ("W7_8T_128g_768M_Dense", "W7\n128D"),
        ("W8_16T_256g_768M_Dense", "W8\n256D"),
        ("W9_8T_ClickBench_32M", "W9\nCB"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(7.15, 3.44),
        gridspec_kw={"height_ratios": [1.0, 1.08]},
    )
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    x = np.arange(len(wl_order))
    n_m = len(methods_bars)
    w = 0.84 / n_m

    # --- Panel (a): SQL Throughput (Med + 95% CI, W1-W9) ---
    max_thr_hi = 0.0
    for idx, (m, lbl, col, ht, ec) in enumerate(methods_bars):
        vals, yerr_lo, yerr_hi = [], [], []
        for wl, _ in wl_order:
            r = by_wm[(wl, m)]
            t_med = fnum_req(r["throughput_mrows_s"])
            rows_m = fnum_req(r["total_rows"]) / 1e6
            t_lo = rows_m / max(fnum_req(r["wall_ci95_hi_s"]), 1e-6)
            t_hi = rows_m / max(fnum_req(r["wall_ci95_lo_s"]), 1e-6)
            vals.append(t_med)
            yerr_lo.append(max(0.0, t_med - t_lo))
            yerr_hi.append(max(0.0, t_hi - t_med))
            max_thr_hi = max(max_thr_hi, t_hi)
        offset = (idx - (n_m - 1) / 2.0) * w
        ax1.bar(
            x + offset, vals, width=w, color=col, label=lbl,
            hatch=ht, edgecolor=ec, linewidth=0.25,
            yerr=[yerr_lo, yerr_hi],
            error_kw={"elinewidth": 0.55, "capsize": 1.0, "ecolor": "#222222"},
            zorder=3,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([l for _, l in wl_order], fontsize=7.9)
    ax1.set_ylim(0.0, max_thr_hi * 1.12)
    ax1.set_ylabel("Thr. (Mrows/s)")
    ax1.set_title("(a) SQL Throughput (Med + 95% CI)", pad=5)
    style_axes(ax1, grid_axis="y")

    # --- Panel (b): Peak Physical Memory Above Table (log MiB, ymax * 1.65 headroom) ---
    max_mem = 0.0
    min_mem = float("inf")
    for idx, (m, _, col, ht, ec) in enumerate(methods_bars):
        vals = [fnum_req(by_wm[(wl, m)]["rss_peak_over_table_mb"]) for wl, _ in wl_order]
        max_mem = max(max_mem, max(vals))
        min_mem = min(min_mem, min(vals))
        offset = (idx - (n_m - 1) / 2.0) * w
        ax2.bar(
            x + offset, vals, width=w, color=col,
            hatch=ht, edgecolor=ec, linewidth=0.25, zorder=3,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([l for _, l in wl_order], fontsize=7.9)
    ax2.set_yscale("log")
    ax2.set_ylim(max(30.0, min_mem * 0.65), max_mem * 1.65)
    ax2.set_yticks([1e2, 1e3, 1e4])
    ax2.set_yticklabels([r"$10^2$", r"$10^3$", r"$10^4$"])
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_ylabel("RAM over tbl (MiB)")
    ax2.set_title("(b) Peak Physical Memory Above Table", pad=5)
    style_axes(ax2, grid_axis="y")

    # --- Exact 12-engine non-dominated Pareto frontier ---
    def true_pareto_frontier(wl: str) -> List[Dict[str, str]]:
        cand = [by_wm[(wl, m)] for m in EXPECTED_DF_METHODS]
        cand.sort(key=lambda r: (fnum_req(r["rmse_pct"]), -fnum_req(r["throughput_mrows_s"])))
        frontier: List[Dict[str, str]] = []
        best_thr = -1.0
        for r in cand:
            thr = fnum_req(r["throughput_mrows_s"])
            if thr > best_thr:
                frontier.append(r)
                best_thr = thr
        require(len(frontier) >= 2, f"Unexpectedly small Pareto frontier on {wl}")
        return frontier

    dense_dom_specs = [
        ("OS-BMW-5b(k24)",  "o"),
        ("OS-BMW-5b-256B",  "s"),
        ("ELL(2,24)",       "^"),
        ("ULL",             "v"),
    ]

    # Prose-matched IEC KiB tags (ZERO undefined 5b/16b/256B!)
    primary_styles = {
        "OS-BMW-5b(k24,Slab+Sp)":   (k5b,            PALETTE["OS-BMW-5b"],   "o", 38),
        "OS-BMW-5b-256B(Slab+Sp)":  (k256,           PALETTE["OS-BMW-256B"], "s", 36),
        "OS-BMW-5b(16blk,Slab+Sp)": (k16b,           PALETTE["OS-BMW-16b"],  "D", 38),
        "ELL(2,24,Slab+Sp)":        ("ELL",          PALETTE["ELL(2,24)"],   "^", 40),
        "ULL(Slab+Sp)":             ("ULL",          PALETTE["ULL"],         "v", 40),
        "DF-BuiltIn-HLL(16KB)":     (f"DF ({kdf})",  PALETTE["DF_HLL"],      "P", 44),
        "HLL++(6b)":                ("HLL-6b",       PALETTE["HLL++"],       "p", 44),
        "AVLL(base56)":             ("AVLL*",        PALETTE["AVLL"],        "*", 62),
    }

    def plot_pareto_panel(
        ax: plt.Axes,
        wl: str,
        title: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        xticks: List[float],
        label_pos: Dict[str, Tuple[float, float, str, str, bool]],
        dense_ring_scale: float = 1.0,
    ) -> None:
        fr = true_pareto_frontier(wl)
        fr_methods = {r["method"] for r in fr}

        ax.plot(
            [fnum_req(r["rmse_pct"]) for r in fr],
            [fnum_req(r["throughput_mrows_s"]) for r in fr],
            color="#1F4E79", linestyle="-", linewidth=1.35, alpha=0.85, zorder=2.5,
        )

        # 4 Dense baseline variants (crisp hollow gray markers matching shapes)
        for m, mk in dense_dom_specs:
            r = by_wm[(wl, m)]
            xv, yv = fnum_req(r["rmse_pct"]), fnum_req(r["throughput_mrows_s"])
            ax.scatter(
                [xv], [yv], facecolors="none", edgecolors="#666666",
                marker=mk, s=56 * dense_ring_scale, linewidths=0.95, alpha=0.95, zorder=4.8,
            )

        # 8 primary colored engines
        for m, (tag, col, mk, sz) in primary_styles.items():
            r = by_wm[(wl, m)]
            xv, yv = fnum_req(r["rmse_pct"]), fnum_req(r["throughput_mrows_s"])
            ax.scatter(
                [xv], [yv], color=col, marker=mk, s=sz,
                edgecolors="#111111" if m in fr_methods else "#333333",
                linewidths=0.65 if m in fr_methods else 0.45,
                zorder=5.2,
            )
            tx, ty, ha_m, va_m, use_arrow = label_pos[m]
            arrow_kw = (
                dict(arrowstyle="-", color="#666666", lw=0.65, shrinkA=1.5, shrinkB=4.0)
                if use_arrow else None
            )
            ax.annotate(
                tag, xy=(xv, yv), xytext=(tx, ty),
                fontsize=7.2, color=col, fontweight="bold",
                ha=ha_m, va=va_m, arrowprops=arrow_kw, zorder=7,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_xlabel("Relative RMSE (%)")
        ax.set_ylabel("Thr. (Mrows/s)")
        ax.set_title(title, pad=5)
        style_axes(ax)

    # --- Panel (c): W8 (256 Dense Groups) 12-Engine Pareto ---
    # Exact 5-epoch interleaved W8 coordinates:
    #   DF(0.797,715)*  5.4KiB(0.819,689)  AVLL(0.958,561)  4.0KiB(0.960,782) [Dense:963*]
    #   ELL(0.997,1062) [Dense:1108*]  2.7KiB(1.166,821) [Dense:1187*]  ULL(1.168,723) [Dense:774]  HLL-6b(1.568,1548)*
    w8_labels = {
        "DF-BuiltIn-HLL(16KB)":     (0.72,  980.0, "center", "bottom", True),
        "OS-BMW-5b(16blk,Slab+Sp)": (0.68,  515.0, "center", "top",    True),
        "AVLL(base56)":             (0.84,  385.0, "center", "center", True),
        "OS-BMW-5b-256B(Slab+Sp)":  (1.14,  575.0, "center", "top",    True),
        "ELL(2,24,Slab+Sp)":        (0.91, 1300.0, "center", "bottom", True),
        "OS-BMW-5b(k24,Slab+Sp)":   (1.34,  960.0, "left",   "center", True),
        "ULL(Slab+Sp)":             (1.34,  640.0, "left",   "center", True),
        "HLL++(6b)":                (1.44, 1548.0, "right",  "center", False),
    }
    plot_pareto_panel(
        ax3, "W8_16T_256g_768M_Dense",
        "(c) W8 (256 Dense Groups): 12-Engine Pareto",
        xlim=(0.56, 1.70), ylim=(280, 1740),
        xticks=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        label_pos=w8_labels,
        dense_ring_scale=1.12,
    )

    # --- Panel (d): W9 (ClickBench 32M) 12-Engine Pareto ---
    # Exact 5-epoch interleaved W9 coordinates:
    #   AVLL(0.473,92)*  DF(0.510,336)*  5.4KiB(0.539,285)  4.0KiB(0.565,313) [Dense:213]
    #   ELL(0.608,330) [Dense:201]  ULL(0.676,321) [Dense:209]  2.7KiB(0.676,316) [Dense:266]  HLL-6b(0.974,225)
    w9_labels = {
        "AVLL(base56)":             (0.58,   82.0, "left",   "center", True),
        "DF-BuiltIn-HLL(16KB)":     (0.355, 355.0, "left",   "center", True),
        "OS-BMW-5b(16blk,Slab+Sp)": (0.500, 412.0, "center", "bottom", True),
        "OS-BMW-5b-256B(Slab+Sp)":  (0.625, 412.0, "center", "bottom", True),
        "ELL(2,24,Slab+Sp)":        (0.750, 412.0, "center", "bottom", True),
        "ULL(Slab+Sp)":             (0.800, 350.0, "left",   "center", True),
        "OS-BMW-5b(k24,Slab+Sp)":   (0.800, 290.0, "left",   "center", True),
        "HLL++(6b)":                (0.960, 150.0, "center", "top",    True),
    }
    plot_pareto_panel(
        ax4, "W9_8T_ClickBench_32M",
        "(d) W9 (ClickBench 32M): 12-Engine Pareto",
        xlim=(0.33, 1.08), ylim=(35, 470),
        xticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        label_pos=w9_labels,
        dense_ring_scale=1.05,
    )

    # Top bar legend strictly for panels (a)-(b) (8 bar engines, 4 cols x 2 rows)
    handles, labels = ax1.get_legend_handles_labels()
    leg_top = fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=True,
        facecolor="white", edgecolor="#CCCCCC", framealpha=0.98,
        fontsize=7.0, columnspacing=0.75, handlelength=1.20,
        borderpad=0.28, labelspacing=0.24, bbox_to_anchor=(0.5, 0.996),
        title="Panels (a)–(b) Bars", title_fontsize=7.0,
    )
    leg_top.get_title().set_fontweight("bold")
    leg_top.get_title().set_color("#444444")

    # Complete dedicated marker key below panels (c)-(d) covering ALL 12 engines + frontier
    pareto_handles = [
        mlines.Line2D([], [], color=PALETTE["OS-BMW-5b"], marker="o", ls="None", ms=5.2, mec="#222222", mew=0.45, label=f"OS-BMW {k5b}"),
        mlines.Line2D([], [], color=PALETTE["OS-BMW-256B"], marker="s", ls="None", ms=5.0, mec="#222222", mew=0.45, label=f"OS-BMW {k256}"),
        mlines.Line2D([], [], color=PALETTE["OS-BMW-16b"], marker="D", ls="None", ms=5.0, mec="#222222", mew=0.45, label=f"OS-BMW {k16b}"),
        mlines.Line2D([], [], color=PALETTE["ELL(2,24)"], marker="^", ls="None", ms=5.2, mec="#222222", mew=0.45, label=f"ELL(2,24) ({kell})"),
        mlines.Line2D([], [], color=PALETTE["ULL"], marker="v", ls="None", ms=5.2, mec="#222222", mew=0.45, label=f"ULL ({kull})"),
        mlines.Line2D([], [], color=PALETTE["DF_HLL"], marker="P", ls="None", ms=5.4, mec="#111111", mew=0.45, label=f"DF HLL ({kdf})"),
        mlines.Line2D([], [], color=PALETTE["HLL++"], marker="p", ls="None", ms=5.4, mec="#222222", mew=0.45, label=f"HLL-6b MLE ({khll})"),
        mlines.Line2D([], [], color=PALETTE["AVLL"], marker="*", ls="None", ms=6.6, mec="#222222", mew=0.45, label=f"AVLL* ({kavll}, non-semilat.)"),
        mlines.Line2D([], [], color="#666666", marker="o", mfc="none", ls="None", ms=5.5, mew=0.95, label="4 Dense baselines (same shape)"),
        mlines.Line2D([], [], color="#1F4E79", ls="-", lw=1.45, label="12-eng. Pareto frontier"),
    ]
    leg_bot = fig.legend(
        handles=pareto_handles, loc="lower center", ncol=5, frameon=True,
        facecolor="white", edgecolor="#CCCCCC", framealpha=0.98,
        fontsize=6.8, columnspacing=0.65, handlelength=1.25, handletextpad=0.35,
        borderpad=0.28, labelspacing=0.24, bbox_to_anchor=(0.5, 0.002),
        title="Panels (c)–(d) 12-Engine Pareto Key (filled = Slab+Sp / built-in / HLL-6b / AVLL*)",
        title_fontsize=6.8,
    )
    leg_bot.get_title().set_fontweight("bold")
    leg_bot.get_title().set_color("#444444")

    fig.tight_layout(rect=[0.0, 0.125, 1.0, 0.86], h_pad=1.20, w_pad=1.25)
    fig.savefig(fig_dir / "fig_datafusion_scaling.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Anchor Rank k & Register Width/Block Size Ablation (fig_geometry_ablation.pdf)
# ---------------------------------------------------------------------------
def build_fig_geometry_ablation(out_dir: Path, fig_dir: Path) -> Dict[str, float]:
    geom_rows = read_tsv(
        out_dir / "m2_geometry_ablation_4k.tsv",
        required_cols=[
            "part", "budget", "config", "slot_bits", "g", "k",
            "cardinality", "trials", "theory_mvp_payload",
            "emp_mvp_payload", "emp_mvp_payload_se",
        ],
        expected_rows=87,
    )
    part_b = [r for r in geom_rows if r["part"] == "B_Threshold_K"]
    part_a = [r for r in geom_rows if r["part"] == "A_Geom_Width"]
    require(
        len(part_b) == 23 and len(part_a) == 64,
        f"Unexpected geometry split: A={len(part_a)}, B={len(part_b)}",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.02))

    # --- Panel (a): Top-k Anchor Rank Sweep (n=10^6, 2,000 trials) ---
    series = [
        (5, 512, r"5-bit, $g{=}512$ (2.7 KiB at $k{=}24$)", PALETTE["OS-BMW-5b"],   "o", "-",  1.85),
        (5, 256, r"5-bit, $g{=}256$",                      PALETTE["OS-BMW-256B"], "s", "--", 1.55),
        (4, 256, r"4-bit, $g{=}256$ ($R{=}7,d{=}1$, ref)", PALETTE["ELL(2,20)"],   "^", "-.", 1.55),
    ]
    k1_th_512 = float("nan")
    k1_emp_512 = float("nan")
    k24_th_512 = float("nan")
    for sb, gval, lbl, col, mk, ls, lw in series:
        pts = sorted(
            [
                (int(r["k"]), fnum_req(r["theory_mvp_payload"]), fnum_req(r["emp_mvp_payload"]))
                for r in part_b
                if int(r["slot_bits"]) == sb and int(r["g"]) == gval
            ],
            key=lambda t: t[0],
        )
        require(
            len(pts) >= 7 and pts[0][0] == 1,
            f"Missing k=1 or sweep rows for ({sb}b, g={gval})",
        )
        ks = [p[0] for p in pts]
        th = [p[1] for p in pts]
        em = [p[2] for p in pts]
        if sb == 5 and gval == 512:
            k1_th_512, k1_emp_512 = pts[0][1], pts[0][2]
            k24_match = [p[1] for p in pts if p[0] == 24]
            require(len(k24_match) == 1, "Missing k=24 for flagship 5b, g=512")
            k24_th_512 = k24_match[0]
        ax1.plot(ks, th, color=col, linestyle=ls, linewidth=lw, label=lbl, zorder=4)
        ax1.scatter(
            ks, em, color=col, marker=mk, s=25,
            edgecolors="white", linewidths=0.45, zorder=5,
        )

    require(math.isfinite(k1_th_512) and math.isfinite(k24_th_512), "Missing flagship k1/k24")
    k1_ratio = k1_th_512 / k24_th_512

    ax1.set_yscale("log")
    ax1.set_xlim(-0.5, 34.0)
    ax1.set_ylim(2.35, 32.0)
    ax1.set_yticks([3, 5, 10, 20, 30])
    ax1.set_yticklabels(["3", "5", "10", "20", "30"])
    ax1.yaxis.set_minor_locator(mticker.NullLocator())

    ax1.axvline(24, color=PALETTE["OS-BMW-5b"], linestyle=":", linewidth=1.0, zorder=2)

    ax1.annotate(
        rf"$k{{=}}1$ theory: {k1_th_512:.2f}" + "\n" + rf"({k1_ratio:.1f}$\times$ vs. $k{{=}}24$)",
        xy=(1.2, k1_th_512),
        xytext=(3.4, 12.2),
        fontsize=7.0, color=PALETTE["OS-BMW-5b"], fontweight="bold",
        ha="left", va="center", linespacing=1.15,
        arrowprops=dict(
            arrowstyle="->,head_width=0.17,head_length=0.23",
            color=PALETTE["OS-BMW-5b"], lw=0.85, connectionstyle="arc3,rad=-0.12",
        ),
        zorder=6,
    )

    ax1.annotate(
        r"$k{=}24$ (knee)",
        xy=(24.0, k24_th_512 * 1.03),
        xytext=(13.2, 5.2),
        fontsize=7.2, color=PALETTE["OS-BMW-5b"], fontweight="bold",
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="->,head_width=0.17,head_length=0.23",
            color=PALETTE["OS-BMW-5b"], lw=0.85,
        ),
        zorder=6,
    )

    ax1.set_xlabel(r"Anchor Rank $k = C + 1$", labelpad=4.5)
    ax1.set_ylabel("MVP (log scale)")
    ax1.set_title(r"(a) Top-$k$ Anchor Rank Sweep ($n{=}10^6$, 2,000 tr.)", pad=6)
    style_axes(ax1)

    h_a, l_a = ax1.get_legend_handles_labels()
    mc_proxy = mlines.Line2D(
        [], [], color="#333333", marker="o", linestyle="None",
        markersize=3.8, label="Markers: MC (2,000 tr.)",
    )
    ax1.legend(
        h_a + [mc_proxy], l_a + ["Markers: MC (2,000 tr.)"],
        loc="upper right", frameon=True, facecolor="white",
        edgecolor="#CCCCCC", framealpha=0.96, fontsize=6.9,
        handlelength=1.45, borderpad=0.28, labelspacing=0.22,
    )

    # --- Panel (b): 5-bit vs 4-bit Grouped Bars + 4,000-trial MC +/- 95% CI across g in {128..512} ---
    gs = [128, 256, 384, 512]
    th_5b, th_4b, em_5b, em_4b, se_5b, se_4b, ks_g = [], [], [], [], [], [], []
    for gv in gs:
        r5 = [
            r for r in part_a
            if r["budget"] == "4KB" and int(r["g"]) == gv
            and int(r["slot_bits"]) == 5 and abs(fnum_req(r["cardinality"]) - 1e6) < 1.0
        ]
        r4 = [
            r for r in part_a
            if r["budget"] == "4KB" and int(r["g"]) == gv
            and int(r["slot_bits"]) == 4 and abs(fnum_req(r["cardinality"]) - 1e6) < 1.0
        ]
        require(len(r5) == 1 and len(r4) == 1, f"Expected unique 4KB n=1e6 row for g={gv}")
        require(int(r5[0]["k"]) == int(r4[0]["k"]), f"Mismatched k between 5b and 4b for g={gv}")
        ks_g.append(int(r5[0]["k"]))
        th_5b.append(fnum_req(r5[0]["theory_mvp_payload"]))
        th_4b.append(fnum_req(r4[0]["theory_mvp_payload"]))
        em_5b.append(fnum_req(r5[0]["emp_mvp_payload"]))
        em_4b.append(fnum_req(r4[0]["emp_mvp_payload"]))
        se_5b.append(fnum_req(r5[0]["emp_mvp_payload_se"]))
        se_4b.append(fnum_req(r4[0]["emp_mvp_payload_se"]))

    require(all(e5 < e4 for e5, e4 in zip(em_5b, em_4b)), "Expected 4/4 5-bit < 4-bit at 4,000 trials")

    x = np.arange(len(gs))
    w = 0.34
    lbl_5b = r"5-bit ($R{=}8,d{=}2$)"
    lbl_4b = r"4-bit ($R{=}7,d{=}1$, ref)"
    lbl_mc = r"MC estimate $\pm$ 1.96 SE (4,000 tr.)"
    lbl_ell = "ELL(2,24) (3.80)"

    ax2.bar(
        x - w / 2, th_5b, width=w, color=PALETTE["OS-BMW-5b"],
        alpha=0.90, edgecolor="white", linewidth=0.4, label=lbl_5b, zorder=3,
    )
    ax2.bar(
        x + w / 2, th_4b, width=w, color=PALETTE["ELL(2,20)"],
        alpha=0.90, edgecolor="white", linewidth=0.4, label=lbl_4b, zorder=3,
    )
    ax2.errorbar(
        x - w / 2, em_5b, yerr=[1.96 * s for s in se_5b],
        fmt="o", markersize=3.8, color="#111111", markeredgecolor="white",
        markeredgewidth=0.45, capsize=2.2, elinewidth=0.9, zorder=5, label=lbl_mc,
    )
    ax2.errorbar(
        x + w / 2, em_4b, yerr=[1.96 * s for s in se_4b],
        fmt="o", markersize=3.8, color="#111111", markeredgecolor="white",
        markeredgewidth=0.45, capsize=2.2, elinewidth=0.9, zorder=5,
    )
    ax2.axhline(
        3.796, color=PALETTE["ELL(2,24)"], linestyle="--", linewidth=1.25,
        label=lbl_ell, zorder=4,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$g={g}$\n($k={kv}$)" for g, kv in zip(gs, ks_g)], fontsize=7.9)
    ax2.set_ylim(2.60, 4.52)
    ax2.set_yticks([2.8, 3.2, 3.6, 4.0])
    ax2.set_xlabel("Block Size ($g$ registers, anchor rank $k$)", labelpad=4.5)
    ax2.set_ylabel("MVP")
    ax2.set_title("(b) Register Width & Block Size ($g$) Ablation", pad=6)
    style_axes(ax2, grid_axis="y")

    h_b, l_b = ax2.get_legend_handles_labels()
    by_lb = dict(zip(l_b, h_b))
    ord_l = [lbl_5b, lbl_4b, lbl_mc, lbl_ell]
    ax2.legend(
        [by_lb[k] for k in ord_l], ord_l,
        loc="upper center", ncol=2, frameon=True, facecolor="white",
        edgecolor="#CCCCCC", framealpha=0.96, fontsize=6.8,
        columnspacing=0.65, handlelength=1.35, borderpad=0.26, labelspacing=0.22,
    )

    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0], w_pad=1.4)
    fig.savefig(fig_dir / "fig_geometry_ablation.pdf", bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    return {
        "k1_th_512": k1_th_512,
        "k1_emp_512": k1_emp_512,
        "k24_th_512": k24_th_512,
        "k1_ratio": k1_ratio,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build 4 vector figures for OS-BMW manuscript_v2.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Read-only artifact/out directory")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="Output figures directory")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    require(out_dir.is_dir(), f"Output data directory not found: {out_dir}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    s_stats = build_fig_equal_byte_sweep(out_dir, fig_dir)
    build_fig_ram_wire_pareto(out_dir, fig_dir)
    build_fig_datafusion_scaling(out_dir, fig_dir)
    a_stats = build_fig_geometry_ablation(out_dir, fig_dir)

    expected_pdfs = [
        "fig_equal_byte_sweep.pdf",
        "fig_ram_wire_pareto.pdf",
        "fig_datafusion_scaling.pdf",
        "fig_geometry_ablation.pdf",
    ]
    for name in expected_pdfs:
        p = fig_dir / name
        require(p.is_file() and p.stat().st_size > 1000, f"Generated PDF missing/tiny: {p}")

    print(
        f"OK: built 4 vector PDFs in {fig_dir} | "
        f"dense={int(s_stats['dense_wins'])}/{int(s_stats['dense_total'])} (-{s_stats['dense_drop_pct']:.1f}%), "
        f"all={int(s_stats['all_wins'])}/{int(s_stats['all_total'])} | "
        f"bias_bound=±{s_stats['bound_excl_hbs05']:.3f}% (all<= {s_stats['bound_all_semi']:.3f}%) | "
        f"k1={a_stats['k1_th_512']:.2f} (MC {a_stats['k1_emp_512']:.2f}, {a_stats['k1_ratio']:.1f}x vs k24)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
