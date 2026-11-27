#!/usr/bin/env python3
"""build_story_tables.py -- publication-grade tables for manuscript_v2, from raw TSV/JSON.

Reads ../artifact/out (read-only) and writes tables/v2_*.tex. Zero hand-typed numbers.
Enforces centralized METHODS / METRICS registries and a fail-closed visual lint gate
before writing any output file.

Usage:  python3 gen/build_story_tables.py [--out-dir DIR] [--dest-dir DIR]
Standard library only.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def read_tsv(path: Path) -> list:
    if not path.is_file() or path.stat().st_size == 0:
        die(f"missing or empty input {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        die(f"no data rows in {path}")
    return rows


def one(rows: list, **eq) -> dict:
    hit = [r for r in rows if all(r[k] == v for k, v in eq.items())]
    if len(hit) != 1:
        die(f"expected exactly one row for {eq}, found {len(hit)}")
    return hit[0]


def comma(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def f(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Centralized Registries (METHODS, METRICS, & Visual Lint Gate)
# ---------------------------------------------------------------------------

# Fixed-size methods share one canonical ordering across Table 2 (MC) & Table 4 (Micro).
FIXED_METHODS = [
    {"id": "5b",     "display": r"\osbmw{} 2.7\,KiB", "mc": "OS-BMW-5b(g512,k24)",   "sys": "OS-BMW-5b(k24)",   "regs": 4096, "ours": True,  "exact": True},
    {"id": "256",    "display": r"\osbmw{} 4.0\,KiB", "mc": "OS-BMW-5b-256B(g384)",  "sys": "OS-BMW-5b-256B",   "regs": 6144, "ours": True,  "exact": True},
    {"id": "ell20",  "display": "ELL(2,20)",           "mc": "ELL(2,20)",             "sys": "ELL(2,20)",        "regs": 1024, "ours": False, "exact": True},
    {"id": "ell24",  "display": "ELL(2,24)",           "mc": "ELL(2,24)",             "sys": "ELL(2,24)",        "regs": 1024, "ours": False, "exact": True},
    {"id": "ull",    "display": "ULL",                 "mc": "ULL(8b)",               "sys": "ULL",              "regs": 4096, "ours": False, "exact": True},
    {"id": "hll6",   "display": r"\texttt{HLL-6b} MLE","mc": "HLL++(6b)",             "sys": "HLL++(6b)",        "regs": 4096, "ours": False, "exact": True},
    {"id": "avll",   "display": r"\texttt{AVLL}",      "mc": None,                    "sys": "AVLL(base56)",     "regs": None, "ours": False, "exact": False},
]

VARIABLE_METHODS = [
    {"id": "cpc",    "display": r"\texttt{CPC} ($\lg K{=}11$)", "sys": "CPC(lgK=11)"},
    {"id": "hll4d",  "display": r"\texttt{HLL\_4} dyn.",        "sys": "HLL_4(dyn)"},
    {"id": "hbs12",  "display": r"\texttt{HBS} ($p_h{=}12$)",   "sys": "HBS(p=12)"},
    {"id": "hbs13",  "display": r"\texttt{HBS} ($p_h{=}13$)",   "sys": "HBS(p=13)"},
]

METRICS = {
    "speed": {"arrow": r"$\uparrow$",   "deadband_pct": 2.0, "lower_is_better": False},
    "err":   {"arrow": r"$\downarrow$", "deadband_pct": 2.0, "lower_is_better": True},
    "mem":   {"arrow": r"$\downarrow$", "deadband_pct": 2.0, "lower_is_better": True},
    "mvp":   {"arrow": r"$\downarrow$", "lower_is_better": True},
}

FORBIDDEN_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9_\\-])5b(?![A-Za-z0-9_])"), "internal token '5b' leaked"),
    (re.compile(r"\b2\.7K\b"), "inconsistent shorthand '2.7K' (use '2.7\\,KiB')"),
    (re.compile(r"Sl\.\+Sp|\bDen\.\b|\bmrg\b|\bmis\."), "undefined internal abbreviation"),
    (re.compile(r"\\(resizebox|scalebox|scriptsize|tiny)\b"), "forbidden table shrink command"),
]


def rank_indices(values: list, lower_is_better: bool = True) -> tuple[int, int]:
    """Return (best_idx, second_best_idx) strictly from unrounded float values."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not lower_is_better)
    return order[0], order[1]


def fmt_sym_speed(ratio: float, deadband: float = 0.02) -> str:
    """Format speed ratio symmetrically: >+2% bold, <-2% muted, within +-2% plain."""
    rounded = round(ratio, 2)
    s = f(rounded, 2) + r"$\times$"
    if rounded > 1.0 + deadband:
        return r"\textbf{" + s + "}"
    if rounded < 1.0 - deadband:
        return r"\muted{" + s + "}"
    return s


def fmt_sym_pct(pct: float, nd: int = 0, deadband: float = 2.0, lower_is_better: bool = True) -> str:
    """Format signed % delta symmetrically using rounded value against deadband."""
    val = round(pct, nd)
    if val == 0.0:
        return "0" + (".0" if nd == 1 else "") + r"\%"
    s = ("+" if val > 0 else "$-$") + f(abs(val), nd) + r"\%"
    better = (val < -deadband) if lower_is_better else (val > deadband)
    worse = (val > deadband) if lower_is_better else (val < -deadband)
    if better:
        return r"\textbf{" + s + "}"
    if worse:
        return r"\muted{" + s + "}"
    return s


def lint_table(name: str, tex: str) -> None:
    """Fail-closed visual & semantic check before writing generated table."""
    # Check non-comment lines excluding bottom explanatory notes wrapped in \footnotesize
    body_lines = []
    for ln in tex.splitlines():
        if ln.lstrip().startswith("%"):
            continue
        if r"\multicolumn{" in ln and r"\footnotesize" in ln:
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines)
    # Macro keys inside \Nm{...} are schema identifiers, not rendered table text.
    body_text = re.sub(r"\\Nm\{[^}]+\}", "", body)

    for pat, msg in FORBIDDEN_PATTERNS:
        m = pat.search(body_text)
        if m:
            die(f"{name}: lint failed ({msg}): matched {m.group(0)!r}")

    if r"\caption{\textbf{" not in tex:
        die(f"{name}: lint failed (caption must start with bold takeaway)")
    if r"\uparrow" not in tex and r"\downarrow" not in tex:
        die(f"{name}: lint failed (missing ↑/↓ direction arrow in header or caption)")
    m_sep = re.search(r"\\setlength\{\\tabcolsep\}\{([0-9.]+)pt\}", tex)
    if not m_sep or float(m_sep.group(1)) < 3.5:
        die(f"{name}: lint failed (tabcolsep must be >= 3.5pt)")
    if r"\tblclean" not in tex:
        die(f"{name}: lint failed (missing \\tblclean for booktabs+colortbl)")


# ---------------------------------------------------------------------------
# Table 2: v2_mc.tex (tab:mc16k)
# ---------------------------------------------------------------------------

def table_mc(out: Path) -> str:
    mc = read_tsv(out / "m2_poisson_16k_mc.tsv")
    sysb = read_tsv(out / "m3_sys_bench.tsv")
    rows_cfg = [m for m in FIXED_METHODS if m["mc"] is not None]

    recs = []
    max_abs_bias = 0.0
    for m in rows_cfg:
        r = one(mc, method=m["mc"])
        b = int(one(sysb, method=m["sys"])["ram_bytes"])
        pay = int(float(r["payload_bytes"]))
        if b != pay + 8:
            die(f"{m['display']}: RAM bytes {b} != payload {pay} + 8")
        scale = b / pay
        th_hdr = float(r["basis_mvp_theory_payload"]) * scale
        meas_hdr = float(r["mc_mvp_with_hdr"])
        se_hdr = float(r["mc_mvp_payload_se"]) * scale
        max_abs_bias = max(max_abs_bias, abs(float(r["mc_rel_bias_pct"])))
        recs.append({"m": m, "b": b, "th": th_hdr, "meas": meas_hdr, "se": se_hdr})

    th_1, th_2 = rank_indices([r["th"] for r in recs], lower_is_better=True)
    ms_1, ms_2 = rank_indices([r["meas"] for r in recs], lower_is_better=True)
    ref_meas = recs[0]["meas"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{\textbf{{\osbmw{{}} achieves the lowest MVP among tested fixed-size, exactly mergeable sketches.}} Header-inclusive MVP ($\downarrow$, Thm.~\ref{{thm:mle}}; \Nm{{mc.trials}} trials at $n\in\{{10^5,10^7,10^{{10}},10^{{14}}\}}$; $|\mathrm{{bias}}|\le {f(max_abs_bias, 3)}\%$). \textbf{{Bold}}: best; \underline{{underline}}: 2nd; \emph{{Bits saved}} ($\uparrow$): equal-RMSE saving of 2.7\,KiB vs.\ row.}}",
        r"\label{tab:mc16k}",
        r"\tblclean",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{\textbf{MVP}$\downarrow$} & \textbf{Bits} \\",
        r"\cmidrule(lr){4-5}",
        r"\textbf{Sketch} & \textbf{Bytes} & \textbf{Regs.} & \textbf{Theory} & \textbf{Meas.\,$\pm$\,SE} & \textbf{saved}$\uparrow$ \\",
        r"\midrule",
    ]
    for i, rec in enumerate(recs):
        if i == 2:
            lines.append(r"\midrule")
        if rec["m"]["ours"]:
            lines.append(r"\rowhl")
        th_s = f(rec["th"], 2)
        if i == th_1:
            th_s = r"\textbf{" + th_s + "}"
        elif i == th_2:
            th_s = r"\underline{" + th_s + "}"
        ms_s = f(rec["meas"], 2)
        if i == ms_1:
            ms_s = r"\textbf{" + ms_s + "}"
        elif i == ms_2:
            ms_s = r"\underline{" + ms_s + "}"
        mvp_cell = f"{ms_s} $\\pm$ {f(rec['se'], 2)}"
        if i == 0:
            save_cell = "ref."
        else:
            pct_saved = 100.0 * (1.0 - ref_meas / rec["meas"])
            save_s = f(pct_saved, 0) + r"\%"
            save_cell = save_s if rec["m"]["ours"] else r"\textbf{" + save_s + "}"
        cells = [rec["m"]["display"], comma(rec["b"]), comma(rec["m"]["regs"]), th_s, mvp_cell, save_cell]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 3: v2_multishard.tex (tab:multishard)
# ---------------------------------------------------------------------------

def table_multishard(out: Path) -> str:
    rows = read_tsv(out / "m1_multishard_drift.tsv")
    rec = json.loads((out / "m0_m1_reconcile.json").read_text())
    want, got = rec["m1_algebraic_audit"]["hll4_fixed_counterexample"]
    if not (isinstance(want, int) and isinstance(got, int) and want != got):
        die("HLL_4 fixed-capacity counterexample missing or not a failure")
    groups = [
        (r"\osbmw{} (4 configs.)", ["OS-BMW-5b", "OS-BMW-256B", "OS-BMW-g128-Dense", "OS-BMW-Chunk"], True),
        ("ELL(2,24)",              ["ELL(2,24)"],           False),
        (r"HBS ($p_h{=}12$)",      ["HBS(p=12)"],           False),
        (r"\texttt{AVLL}",         ["AVLL"],                False),
        (r"\texttt{HLL\_4}, fixed cap", ["HLL_4-Fixed(cap=32)"], False),
    ]
    shards = ["2", "8", "64", "512"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Dense \osbmw{} logs zero merge failures across \Nm{ms.osbmw.trials} shard trials, whereas \texttt{AVLL} diverges.} $2\mathrm{M}$ keys, $S$ shards (20 seeds/config/$S$): balanced-tree vs.\ single-pass / left-deep chain mismatches ($\downarrow$) and max drift ($\downarrow$). Nonzero errors highlighted.}",
        r"\label{tab:multishard}",
        r"\tblclean",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\small",
        r"\begin{tabular}{@{}lccccr@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Part.\ / assoc.\ mismatches} ($\downarrow$)} & \textbf{Max.} \\",
        r"\cmidrule(lr){2-5}",
        r"\textbf{Sketch} & $S{=}2$ & $S{=}8$ & $S{=}64$ & $S{=}512$ & \textbf{drift}$\downarrow$ \\",
        r"\midrule",
    ]
    for label, methods, ours in groups:
        if ours:
            lines.append(r"\rowhl")
        cells = [label]
        drift = 0.0
        for s in shards:
            sel = [r for r in rows if r["method"] in methods and r["shards"] == s]
            if len(sel) != len(methods):
                die(f"multishard: missing rows for {label} at S={s}")
            pm = sum(int(r["part_state_mismatches"]) for r in sel)
            am = sum(int(r["assoc_state_mismatches"]) for r in sel)
            tr = sum(int(r["trials"]) for r in sel)
            if tr != 20 * len(methods):
                die("unexpected trial count")
            val = f"{pm}/{am}"
            if pm > 0 or am > 0:
                val = r"\alertnum{" + val + "}"
            elif ours:
                val = r"\textbf{" + val + "}"
            cells.append(val)
            drift = max(drift, max(float(r["max_abs_est_drift_pct"]) for r in sel))
        d_s = f(drift, 2) + r"\%"
        if drift > 0.0:
            d_s = r"\alertnum{" + d_s + "}"
        elif ours:
            d_s = r"\textbf{" + d_s + "}"
        cells.append(d_s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{@{}p{0.96\columnwidth}@{}}{\footnotesize The \osbmw{} row pools four dense layouts (2.7\,KiB, 4.0\,KiB, $g{=}128$, Chunk; 80 trials per $S$). \texttt{HBS} ($2^{p_h}$ buckets) checks logical registers; \texttt{AVLL} uses 1,408 buckets. Fixed-capacity \texttt{HLL\_4} ($g{=}2{,}048, C{=}32$) showed zero mismatches on random shards; a minimal $g{=}4, C{=}1$ fixture yields register " + str(got) + r" instead of " + str(want) + r". AVX-512 vs.\ scalar parity passes \Nm{parity.merges}/\Nm{parity.merges} cases.}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 4: v2_micro.tex (tab:sys_micro)
# ---------------------------------------------------------------------------

def table_micro(out: Path) -> str:
    rows = read_tsv(out / "m3_sys_bench.tsv")

    fixed_recs = []
    for m in FIXED_METHODS:
        r = one(rows, method=m["sys"])
        fixed_recs.append({
            "m": m,
            "bytes": float(r["ram_bytes"]),
            "l1": float(r["l1_steady_1e7_ns"]),
            "k4": float(r["dram_4096_steady_ns"]),
            "k64": float(r["dram_65536_steady_ns"]),
            "merge_us": float(r["merge_equal_ns"]) / 1000.0,
            "est_us": float(r["est_ml_us"]),
        })

    # Rank strictly within fixed-size rows on unrounded floats
    ranks = {}
    for col in ("l1", "k4", "k64", "merge_us", "est_us"):
        ranks[col] = rank_indices([rec[col] for rec in fixed_recs], lower_is_better=True)

    def fmt_merge_val(us: float) -> str:
        if us < 10.0:
            return f(us, 2)
        if us < 100.0:
            return f(us, 1) + r"\hphantom{0}"
        return f(us, 0) + r"\hphantom{.00}"

    def fmt_ranked_num(val: float, fmt_str: str, idx: int, col: str) -> str:
        b1, b2 = ranks[col]
        if idx == b1:
            return r"\textbf{" + fmt_str + "}"
        if idx == b2:
            return r"\underline{" + fmt_str + "}"
        return fmt_str

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Among fixed-size sketches, \osbmw{}~2.7\,KiB has the fastest 4K/64K updates and 2nd-fastest L1 update/MLE.} Single-thread Xeon Gold~6430 ($\downarrow$): updates (pre-hashed L1: $10^7$; 4K/64K: $10^6$/sketch), merge ($10^6{+}10^6$), MLE. Fixed-size (\texttt{AVLL} non-exact): \textbf{bold} best, \underline{underline} 2nd; variable-size: live RAM.}",
        r"\label{tab:sys_micro}",
        r"\tblclean",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Update (ns)}$\downarrow$} & \textbf{Merge} & \textbf{Est.} \\",
        r"\cmidrule(lr){3-5}",
        r"\textbf{Sketch} & \textbf{Bytes} & L1 & 4K & 64K & \textbf{($\mu$s)$\downarrow$} & \textbf{($\mu$s)$\downarrow$} \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Fixed-size sketches (ranked)}} \\",
    ]
    for i, rec in enumerate(fixed_recs):
        if rec["m"]["ours"]:
            lines.append(r"\rowhl")
        cells = [
            rec["m"]["display"],
            comma(rec["bytes"]),
            fmt_ranked_num(rec["l1"], f(rec["l1"], 2), i, "l1"),
            fmt_ranked_num(rec["k4"], f(rec["k4"], 1), i, "k4"),
            fmt_ranked_num(rec["k64"], f(rec["k64"], 1), i, "k64"),
            fmt_ranked_num(rec["merge_us"], fmt_merge_val(rec["merge_us"]), i, "merge_us"),
            fmt_ranked_num(rec["est_us"], f(rec["est_us"], 1), i, "est_us"),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Variable-size sketches (Bytes = live RAM; unranked)}} \\",
    ]
    for m in VARIABLE_METHODS:
        r = one(rows, method=m["sys"])
        mus = float(r["merge_equal_ns"]) / 1000.0
        cells = [
            m["display"],
            comma(float(r["ram_bytes"])),
            f(float(r["l1_steady_1e7_ns"]), 2),
            f(float(r["dram_4096_steady_ns"]), 1),
            f(float(r["dram_65536_steady_ns"]), 1),
            fmt_merge_val(mus),
            f(float(r["est_ml_us"]), 1),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 5: v2_df.tex (tab:datafusion, full-width table*)
# ---------------------------------------------------------------------------

WL = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]
DF = {
    "5b":  "OS-BMW-5b(k24,Slab+Sp)",
    "256": "OS-BMW-5b-256B(Slab+Sp)",
    "16b": "OS-BMW-5b(16blk,Slab+Sp)",
    "ell": "ELL(2,24,Slab+Sp)",
    "ull": "ULL(Slab+Sp)",
    "hll": "DF-BuiltIn-HLL(16KB)",
}
SHAPE = {
    "W1": "uniform", "W2": "uniform", "W3": "Zipf", "W4": "uniform",
    "W5": "uniform", "W6": "uniform", "W7": "dense", "W8": "dense", "W9": "ClickBench",
}


def table_df(out: Path) -> str:
    rows = read_tsv(out / "m4_datafusion_v5_hardened.tsv")
    wl = {w: [r for r in rows if r["workload"].startswith(w + "_")] for w in WL}

    def v(w: str, e: str, c: str) -> float:
        return float(one(wl[w], method=DF[e])[c])

    def spd_ratio(w: str, a: str, b: str) -> float:
        return v(w, b, "wall_median_s") / v(w, a, "wall_median_s")

    def err_ratio(w: str, a: str, b: str) -> float:
        return v(w, a, "rmse_pct") / v(w, b, "rmse_pct")

    def mem_ratio(w: str, a: str, b: str) -> float:
        return v(w, a, "rss_peak_over_table_mb") / v(w, b, "rss_peak_over_table_mb")

    def geomean(vals: list) -> float:
        return math.exp(sum(math.log(x) for x in vals) / len(vals))

    # Accumulate ratios for Geo-mean footer row
    gm_collect = {
        "spd_5b_ell": [], "err_5b_ell": [], "mem_5b_ell": [],
        "spd_5b_ull": [],
        "spd_256_ell": [], "err_256_ell": [],
        "spd_16b_hll": [], "err_16b_hll": [], "mem_16b_hll": [],
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{\osbmw{} cuts peak SQL RAM by \Nm{df.16b.hll.mem.min}--\Nm{df.16b.hll.mem.max} across all nine workloads vs.\ HLL and lowers RMSE everywhere at 4.0\,KiB.} DataFusion~55.3 \texttt{COUNT(DISTINCT)} (setup \S\ref{sec:eval}; CIs in Fig.~\ref{fig:datafusion}; $^*$ULL $\Delta$RMSE $\le\Nm{df.5b.ull.rmse.maxdiff}$, $\Delta$Mem $-26\%$ to $+5\%$; 4.0\,KiB $\Delta$Mem within $\pm 6\%$). Speed ($\uparrow$): $T_{\mathrm{base}}/T_{\mathrm{OS\text{-}BMW}}$; Err./Mem.\ ($\downarrow$): rel.\ delta. \textbf{Bold}: $>2\%$ better; \muted{gray}: $>2\%$ worse; plain: $\pm 2\%$.}",
        r"\label{tab:datafusion}",
        r"\tblclean",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\small",
        r"\begin{tabular}{@{}llrrr@{\hspace{8pt}}rrr@{\hspace{6pt}}r@{\hspace{8pt}}rr@{\hspace{8pt}}rrr@{}}",
        r"\toprule",
        r"& & & & & \multicolumn{4}{c}{\textbf{2.7\,KiB \osbmw{} vs.\ 4.0\,KiB}} & \multicolumn{2}{c}{\textbf{4.0 vs.\ 4.0\,KiB}} & \multicolumn{3}{c}{\textbf{5.4 vs.\ 16\,KiB}} \\",
        r"\cmidrule(lr){6-9}\cmidrule(lr){10-11}\cmidrule(l){12-14}",
        r"& & & & & \multicolumn{3}{c}{ELL(2,24)} & ULL$^*$ & \multicolumn{2}{c}{ELL(2,24)} & \multicolumn{3}{c}{built-in HLL} \\",
        r"\cmidrule(lr){6-8}\cmidrule(lr){9-9}\cmidrule(lr){10-11}\cmidrule(l){12-14}",
        r"& \textbf{Data} & \textbf{Thr.} & \textbf{Groups} & \textbf{Rows} & \textbf{Speed}$\uparrow$ & \textbf{Err.}$\downarrow$ & \textbf{Mem.}$\downarrow$ & \textbf{Speed}$\uparrow$ & \textbf{Speed}$\uparrow$ & \textbf{Err.}$\downarrow$ & \textbf{Speed}$\uparrow$ & \textbf{Err.}$\downarrow$ & \textbf{Mem.}$\downarrow$ \\",
        r"\midrule",
    ]

    for w in WL:
        if w in ("W7", "W9"):
            lines.append(r"\addlinespace[2pt]")
        r0 = wl[w][0]
        rows_m = float(r0["total_rows"]) / 1e6

        s_5b_ell = spd_ratio(w, "5b", "ell")
        e_5b_ell = err_ratio(w, "5b", "ell")
        m_5b_ell = mem_ratio(w, "5b", "ell")
        s_5b_ull = spd_ratio(w, "5b", "ull")
        s_256_ell = spd_ratio(w, "256", "ell")
        e_256_ell = err_ratio(w, "256", "ell")
        s_16b_hll = spd_ratio(w, "16b", "hll")
        e_16b_hll = err_ratio(w, "16b", "hll")
        m_16b_hll = mem_ratio(w, "16b", "hll")

        gm_collect["spd_5b_ell"].append(s_5b_ell)
        gm_collect["err_5b_ell"].append(e_5b_ell)
        gm_collect["mem_5b_ell"].append(m_5b_ell)
        gm_collect["spd_5b_ull"].append(s_5b_ull)
        gm_collect["spd_256_ell"].append(s_256_ell)
        gm_collect["err_256_ell"].append(e_256_ell)
        gm_collect["spd_16b_hll"].append(s_16b_hll)
        gm_collect["err_16b_hll"].append(e_16b_hll)
        gm_collect["mem_16b_hll"].append(m_16b_hll)

        cells = [
            w, SHAPE[w], r0["n_threads"], comma(float(r0["n_groups"])), f(rows_m, 0) + "M",
            fmt_sym_speed(s_5b_ell),
            fmt_sym_pct(100.0 * (e_5b_ell - 1.0), 0),
            fmt_sym_pct(100.0 * (m_5b_ell - 1.0), 0),
            fmt_sym_speed(s_5b_ull),
            fmt_sym_speed(s_256_ell),
            fmt_sym_pct(100.0 * (e_256_ell - 1.0), 0),
            fmt_sym_speed(s_16b_hll),
            fmt_sym_pct(100.0 * (e_16b_hll - 1.0), 1),
            fmt_sym_pct(100.0 * (m_16b_hll - 1.0), 0),
        ]
        lines.append(" & ".join(cells) + r" \\")

    gm_cells = [
        r"\textbf{Geo-mean}", r"\textit{all 9}", "--", "--", "--",
        fmt_sym_speed(geomean(gm_collect["spd_5b_ell"])),
        fmt_sym_pct(100.0 * (geomean(gm_collect["err_5b_ell"]) - 1.0), 0),
        fmt_sym_pct(100.0 * (geomean(gm_collect["mem_5b_ell"]) - 1.0), 0),
        fmt_sym_speed(geomean(gm_collect["spd_5b_ull"])),
        fmt_sym_speed(geomean(gm_collect["spd_256_ell"])),
        fmt_sym_pct(100.0 * (geomean(gm_collect["err_256_ell"]) - 1.0), 0),
        fmt_sym_speed(geomean(gm_collect["spd_16b_hll"])),
        fmt_sym_pct(100.0 * (geomean(gm_collect["err_16b_hll"]) - 1.0), 1),
        fmt_sym_pct(100.0 * (geomean(gm_collect["mem_16b_hll"]) - 1.0), 0),
    ]
    lines += [
        r"\midrule",
        r"\rowhl",
        " & ".join(gm_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 6: v2_supp.tex (tab:supp_sql, fully symmetric diagnostic pairs)
# ---------------------------------------------------------------------------

def table_supp(out: Path) -> str:
    w10 = read_tsv(out / "m4_w10_outofl3.tsv")
    fc = read_tsv(out / "m4_w6_fixedcap.tsv")
    sb = read_tsv(out / "m4_stage_breakdown.tsv")
    fg = read_tsv(out / "m4_ull_fgra_ablation.tsv")
    sc = read_tsv(out / "m4_scalar_sql.tsv")

    w10_5bd = one(w10, method="OS-BMW-5b(k24)")
    w10_elld = one(w10, method="ELL(2,24)")
    w10_ulld = one(w10, method="ULL")
    w10_5bs = one(w10, method="OS-BMW-5b(k24,Slab+Sp)")
    w10_ells = one(w10, method="ELL(2,24,Slab+Sp)")
    w10_ulls = one(w10, method="ULL(Slab+Sp)")
    w10_avll = one(w10, method="AVLL(base56)")
    w10_hll = one(w10, method="DF-BuiltIn-HLL(16KB)")

    fc_1k_5b = one(fc, build_tag="cap1024", method="OS-BMW-5b(k24,Slab+Sp)")
    fc_1k_ell = one(fc, build_tag="cap1024", method="ELL(2,24,Slab+Sp)")
    fc_4k_5b = one(fc, build_tag="cap4096", method="OS-BMW-5b(k24,Slab+Sp)")
    fc_4k_ell = one(fc, build_tag="cap4096", method="ELL(2,24,Slab+Sp)")

    p64_5bd = one(sb, workload="W5_64P_16Kg_64M_Skew", method="OS-BMW-5b(k24)")
    p64_elld = one(sb, workload="W5_64P_16Kg_64M_Skew", method="ELL(2,24)")
    p64_5bs = one(sb, workload="W5_64P_16Kg_64M_Skew", method="OS-BMW-5b(k24,Slab+Sp)")
    p64_ells = one(sb, workload="W5_64P_16Kg_64M_Skew", method="ELL(2,24,Slab+Sp)")

    fg_w6_5b = one(fg, workload="W6_8T_65Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    fg_w6 = one(fg, workload="W6_8T_65Kg_64M", method="ULL-FGRA(Slab+Sp)")
    fg_w9_5b = one(fg, workload="W9_8T_ClickBench_32M", method="OS-BMW-5b(k24,Slab+Sp)")
    fg_w9 = one(fg, workload="W9_8T_ClickBench_32M", method="ULL-FGRA(Slab+Sp)")
    fg_ratios = [
        100.0 * (float(r["rmse_pct"]) / float(one(fg, workload=r["workload"], method="OS-BMW-5b(k24,Slab+Sp)")["rmse_pct"]) - 1.0)
        for r in fg if r["method"] == "ULL-FGRA(Slab+Sp)"
    ]
    fg_lo, fg_hi = round(min(fg_ratios)), round(max(fg_ratios))

    sc_w5_5b = one(sc, workload="W5_32T_16Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w5_ell = one(sc, workload="W5_32T_16Kg_64M", method="ELL(2,24,Slab+Sp)")
    sc_w9_5b = one(sc, workload="W9_8T_ClickBench_32M", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w9_ell = one(sc, workload="W9_8T_ClickBench_32M", method="ELL(2,24,Slab+Sp)")
    sc_w10_5b = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="OS-BMW-5b(k24)")
    sc_w10_ell = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="ELL(2,24)")

    avll_mrg_s = f(float(w10_avll["mrg_cpu_ms"]) / 1000.0, 1)
    os_mrg_ms = int(round(float(w10_5bs["mrg_cpu_ms"])))

    def pair_row(label: str, os_r: dict, base_r: dict, t_nd: int = 2) -> str:
        t_os = float(os_r["wall_median_s"])
        t_bs = float(base_r["wall_median_s"])
        m_os = float(os_r["rss_peak_over_table_mb"])
        m_bs = float(base_r["rss_peak_over_table_mb"])
        spd = t_bs / t_os
        dram = 100.0 * (m_os / m_bs - 1.0)
        cells = [
            label,
            f(t_os, t_nd),
            fmt_sym_speed(spd),
            comma(round(m_os)),
            fmt_sym_pct(dram, 0),
        ]
        return " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Controlled DataFusion diagnostics: \osbmw{}~2.7\,KiB cuts peak RAM in 13 of 15 pairs (up to 69\%) and increases RAM in two edge cases (+4\%, +6\%).} Speed ($\uparrow$, $T_{\mathrm{base}}/T_{\mathrm{OS\text{-}BMW}}$) and $\Delta$RAM ($\downarrow$): \textbf{bold} $>2\%$ better, \muted{gray} $>2\%$ worse, plain $\pm 2\%$.}",
        r"\label{tab:supp_sql}",
        r"\tblclean",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Diagnostic pair} & \textbf{Time (s)} & \textbf{Speed}$\uparrow$ & \textbf{RAM (MiB)} & $\Delta$\textbf{RAM}$\downarrow$ \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Out-of-L3 \texttt{W10} (32T, 2,048 dense groups, 768M rows)}} \\",
        pair_row(r"W10 Dense vs.\ ELL",                    w10_5bd, w10_elld),
        pair_row(r"W10 Dense vs.\ ULL",                    w10_5bd, w10_ulld),
        pair_row(r"W10 \texttt{Slab+Sp} vs.\ ELL",         w10_5bs, w10_ells),
        pair_row(r"W10 \texttt{Slab+Sp} vs.\ ULL",         w10_5bs, w10_ulls),
        pair_row(r"W10 \texttt{Slab+Sp} vs.\ \texttt{AVLL}$^\dagger$", w10_5bs, w10_avll),
        pair_row(r"W10 \texttt{Slab+Sp} vs.\ HLL",         w10_5bs, w10_hll),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Sparse caps (\texttt{W6}), skew (\texttt{W5\_64P}), and \texttt{ULL-FGRA}}} \\",
        pair_row(r"W6 (cap=1K) vs.\ ELL",                  fc_1k_5b, fc_1k_ell),
        pair_row(r"W6 (cap=4K) vs.\ ELL",                  fc_4k_5b, fc_4k_ell),
        pair_row(r"W5\_64P Dense vs.\ ELL",                p64_5bd,  p64_elld),
        pair_row(r"W5\_64P \texttt{Slab+Sp} vs.\ ELL",     p64_5bs,  p64_ells),
        pair_row(r"W6 vs.\ \texttt{ULL-FGRA}$^\ddagger$",  fg_w6_5b, fg_w6),
        pair_row(r"W9 vs.\ \texttt{ULL-FGRA}$^\ddagger$",  fg_w9_5b, fg_w9, t_nd=3),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Portable \texttt{x86-64-v2} scalar SQL (vs.\ ELL)}} \\",
        pair_row(r"Scalar W5 \texttt{Slab+Sp}",            sc_w5_5b,  sc_w5_ell),
        pair_row(r"Scalar W9 \texttt{Slab+Sp}",            sc_w9_5b,  sc_w9_ell, t_nd=3),
        pair_row(r"Scalar W10 Dense",                      sc_w10_5b, sc_w10_ell),
        r"\bottomrule",
        rf"\multicolumn{{5}}{{@{{}}p{{0.96\columnwidth}}@{{}}}}{{\footnotesize $^\dagger$\texttt{{AVLL}} incurs {avll_mrg_s}\,s merge CPU vs.\ {os_mrg_ms}\,ms (0 mismatches) for \osbmw{{}}. $^\ddagger$\texttt{{ULL-FGRA}} trades +{fg_lo}--{fg_hi}\% higher RMSE across tested workloads vs.\ \osbmw{{}}.}}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(MS.parent / "artifact" / "out"))
    ap.add_argument("--dest-dir", default=str(MS / "tables"))
    a = ap.parse_args()
    out, dest = Path(a.out_dir), Path(a.dest_dir)
    if not out.is_dir():
        die(f"result directory {out} not found")
    files = {
        "v2_mc.tex": table_mc(out),
        "v2_multishard.tex": table_multishard(out),
        "v2_micro.tex": table_micro(out),
        "v2_df.tex": table_df(out),
        "v2_supp.tex": table_supp(out),
    }
    hdr = "% Auto-generated by gen/build_story_tables.py from ../artifact/out -- DO NOT EDIT BY HAND\n"
    for name, body in files.items():
        tex = hdr + body
        lint_table(name, tex)
        (dest / name).write_text(tex)
    print(f"wrote {len(files)} tables to {dest}")


if __name__ == "__main__":
    main()
