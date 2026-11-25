#!/usr/bin/env python3
"""build_story_macros.py -- every number that appears in the running text of manuscript_v2.

Reads the raw result files (read-only) and writes tables/story_macros.tex. The text refers to a
number as \\Nm{key}; an unknown key is a LaTeX error, so the text cannot cite a number that this
script did not derive from a raw file.

The script also checks the qualitative statements the text makes about the data (who is faster on
which workload, which counts are zero, ...). If a statement no longer holds for the given result
files, the script prints it and exits non-zero without writing the macro file.

Usage:  python3 gen/build_story_macros.py [--out-dir DIR] [--dest FILE]
        (defaults: ../artifact/out relative to manuscript_v2, tables/story_macros.tex)
Standard library only.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent

M: dict = {}          # key -> LaTeX text
FAILED: list = []     # violated statements


# ----------------------------------------------------------------------------- helpers
def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def read_tsv(path: Path) -> list:
    if not path.is_file():
        die(f"missing input {path}")
    if path.stat().st_size == 0:
        die(f"empty input {path}")
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


def put(key: str, text: str) -> None:
    if key in M:
        die(f"duplicate key {key}")
    M[key] = text


def fx(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def num(key: str, x: float, nd: int) -> None:
    put(key, fx(x, nd))


def pct(key: str, x: float, nd: int = 0) -> None:
    """x is already in percent units."""
    put(key, fx(x, nd) + r"\%")


def times(key: str, x: float, nd: int = 2) -> None:
    put(key, fx(x, nd) + r"$\times$")


def integer(key: str, n: int) -> None:
    put(key, f"{int(n):,}")


def check(cond: bool, statement: str) -> None:
    if not cond:
        FAILED.append(statement)


def geomean(xs: list) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ----------------------------------------------------------------------------- sections
def monte_carlo(out: Path) -> None:
    """16,000-trial Monte Carlo memory-variance products (Table tab:mc16k)."""
    rows = read_tsv(out / "m2_poisson_16k_mc.tsv")
    name = {
        "5b": "OS-BMW-5b(g512,k24)",
        "256": "OS-BMW-5b-256B(g384)",
        "chunk": "OS-BMW-Chunk(656B)",
        "ell20": "ELL(2,20)",
        "ell24": "ELL(2,24)",
        "ull": "ULL(8b)",
        "hll": "HLL++(6b)",
    }
    r = {k: one(rows, method=v) for k, v in name.items()}
    pay = {k: float(v["mc_mvp_payload"]) for k, v in r.items()}
    hdr = {k: float(v["mc_mvp_with_hdr"]) for k, v in r.items()}
    th = {k: float(v["basis_mvp_theory_payload"]) for k, v in r.items()}
    for k in name:
        num(f"mvp.{k}.pay", pay[k], 2)
        num(f"mvp.{k}.hdr", hdr[k], 2)
        num(f"mvp.{k}.theory", th[k], 2)
    num("mvp.5b.pay.x", pay["5b"], 3)
    num("mvp.5b.theory.x", th["5b"], 3)
    num("mvp.5b.se", float(r["5b"]["mc_mvp_payload_se"]), 3)
    scale_256 = (float(r["256"]["payload_bytes"]) + 8.0) / float(r["256"]["payload_bytes"])
    th_256_hdr = th["256"] * scale_256
    se_256_hdr = float(r["256"]["mc_mvp_payload_se"]) * scale_256
    num("mvp.256.theory.hdr", th_256_hdr, 2)
    num("mvp.256.se.hdr", se_256_hdr, 2)
    num("mvp.256.zse", abs(hdr["256"] - th_256_hdr) / se_256_hdr, 1)
    integer("bytes.chunk", int(float(r["chunk"]["payload_bytes"])) + 8)
    pct("mvp.5b.gap.theory", float(r["5b"]["max_abs_err_vs_theory_pct"]), 1)
    integer("mc.trials", int(r["5b"]["mc_trials"]))
    # memory saved at equal error = 1 - MVP ratio (header-inclusive)
    for k in ("ell20", "ell24", "ull", "hll"):
        pct(f"save.5b.{k}", 100 * (1 - hdr["5b"] / hdr[k]), 0)
        pct(f"save.5b.{k}.x", 100 * (1 - hdr["5b"] / hdr[k]), 1)
    # error reduction at equal memory = 1 - sqrt(MVP ratio)
    pct("errdrop.5b.ell20", 100 * (1 - math.sqrt(hdr["5b"] / hdr["ell20"])), 1)
    pct("save.chunk.ell20", 100 * (1 - hdr["chunk"] / hdr["ell20"]), 0)
    # bits per register (payload) and the information kept relative to ULL
    regs_5b = 8 * 512
    bits_5b = 8 * float(r["5b"]["payload_bytes"]) / regs_5b
    num("bits.5b.reg", bits_5b, 1)
    num("bits.5b.reg.x", bits_5b, 3)
    j_5b = bits_5b / th["5b"]
    j_ull = 8.0 / th["ull"]
    pct("info.5b.vs.ull", 100 * j_5b / j_ull, 1)
    check(pay["5b"] < pay["ell20"] < pay["ell24"] < pay["ull"] < pay["hll"],
          "MC MVP order OS-BMW-5b < ELL(2,20) < ELL(2,24) < ULL < HLL++")
    check(all(abs(float(v["mc_rel_bias_pct"])) < 0.07 for v in r.values()),
          "all Monte Carlo biases below 0.07%")
    check(float(r["5b"]["mc_trials"]) == 16000, "Monte Carlo uses 16,000 trials")


def sizes(out: Path) -> None:
    rows = read_tsv(out / "m3_sys_bench.tsv")
    b = {
        "5b": int(one(rows, method="OS-BMW-5b(k24)")["ram_bytes"]),
        "256": int(one(rows, method="OS-BMW-5b-256B")["ram_bytes"]),
        "ell24": int(one(rows, method="ELL(2,24)")["ram_bytes"]),
        "ell20": int(one(rows, method="ELL(2,20)")["ram_bytes"]),
        "ull": int(one(rows, method="ULL")["ram_bytes"]),
        "hll": int(one(rows, method="HLL++(6b)")["ram_bytes"]),
    }
    for k, v in b.items():
        integer(f"bytes.{k}", v)
    pct("bytes.5b.less.ull", 100 * (1 - b["5b"] / b["ull"]), 0)
    pct("bytes.5b.less.ull.x", 100 * (1 - b["5b"] / b["ull"]), 1)
    num("kb.5b", b["5b"] / 1024, 1)
    num("kb.256", b["256"] / 1024, 1)
    num("kb.ell24", b["ell24"] / 1024, 1)
    num("kb.ell20", b["ell20"] / 1024, 1)
    num("kb.ull", b["ull"] / 1024, 1)
    num("kb.hll", b["hll"] / 1024, 1)
    check(b["256"] == b["ell24"] == b["ull"], "256B variant, ELL(2,24) and ULL have equal byte size")


def multishard(out: Path) -> None:
    """Merge audit over 2/8/64/512 shards (Table tab:multishard)."""
    rows = read_tsv(out / "m1_multishard_drift.tsv")
    ours = ["OS-BMW-5b", "OS-BMW-256B", "OS-BMW-g128-Dense", "OS-BMW-Chunk"]
    o = [r for r in rows if r["method"] in ours]
    a = [r for r in rows if r["method"] == "AVLL"]
    check(len(o) == 16 and len(a) == 4, "merge audit has 4 shard counts for each OS-BMW variant and for AVLL")
    integer("ms.osbmw.trials", sum(int(r["trials"]) for r in o))
    integer("ms.osbmw.viol", sum(int(r["part_state_mismatches"]) + int(r["assoc_state_mismatches"]) for r in o))
    integer("ms.avll.trials", sum(int(r["trials"]) for r in a))
    integer("ms.avll.part", sum(int(r["part_state_mismatches"]) for r in a))
    integer("ms.avll.assoc", sum(int(r["assoc_state_mismatches"]) for r in a))
    num("ms.avll.drift.max", max(float(r["max_abs_est_drift_pct"]) for r in a), 2)
    check(M["ms.osbmw.viol"] == "0", "no OS-BMW variant shows a partition or associativity mismatch")
    check(all(float(r["max_abs_est_drift_pct"]) == 0.0 for r in o), "OS-BMW estimates do not drift under merging")
    exact = [r for r in rows if r["method"] in ("ELL(2,24)", "HBS(p=12)")]
    check(all(int(r["part_state_mismatches"]) + int(r["assoc_state_mismatches"]) == 0 for r in exact),
          "ELL(2,24) and HBS show no merge mismatch")
    hi = [r for r in a if int(r["shards"]) >= 64]
    check(all(int(r["part_state_mismatches"]) == int(r["trials"]) for r in hi),
          "AVLL fails partition invariance in every trial at 64 and 512 shards")


WL = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]
DF = {
    "5b": "OS-BMW-5b(k24,Slab+Sp)",
    "256": "OS-BMW-5b-256B(Slab+Sp)",
    "16b": "OS-BMW-5b(16blk,Slab+Sp)",
    "ell": "ELL(2,24,Slab+Sp)",
    "ull": "ULL(Slab+Sp)",
    "5b.dense": "OS-BMW-5b(k24)",
    "256.dense": "OS-BMW-5b-256B",
    "ell.dense": "ELL(2,24)",
    "ull.dense": "ULL",
    "hllpp": "HLL++(6b)",
    "avll": "AVLL(base56)",
    "hll": "DF-BuiltIn-HLL(16KB)",
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def datafusion(out: Path) -> None:
    """End-to-end SQL runs (Table tab:datafusion)."""
    rows = read_tsv(out / "m4_datafusion_v5_hardened.tsv")
    check(all(r["build_tag"] == "avx512" for r in rows), "all SQL rows must be avx512")
    wl = {w: [r for r in rows if r["workload"].startswith(w + "_")] for w in WL}
    check(all(len(v) == 12 for v in wl.values()), "12 engines on each of the 9 SQL workloads")

    freqs = [float(r["freq_med_mhz"]) for r in rows]
    num("df.freq.min", min(freqs) / 1000.0, 2)
    num("df.freq.max", max(freqs) / 1000.0, 2)

    def v(w: str, eng: str, col: str) -> float:
        return float(one(wl[w], method=DF[eng])[col])

    def speed(w: str, ours: str, other: str) -> float:  # > 1: ours is faster
        return v(w, other, "wall_median_s") / v(w, ours, "wall_median_s")

    def rmse_rel(w: str, ours: str, other: str) -> float:  # percent, > 0: ours has higher error
        return 100 * (v(w, ours, "rmse_pct") / v(w, other, "rmse_pct") - 1)

    def mem_less(w: str, ours: str, other: str) -> float:  # percent less peak memory over the table
        return 100 * (1 - v(w, ours, "rss_peak_over_table_mb") / v(w, other, "rss_peak_over_table_mb"))

    integer("bytes.16b", int(v("W1", "16b", "bytes_per_group")))
    num("kb.16b", v("W1", "16b", "bytes_per_group") / 1024, 1)
    check(int(v("W1", "5b", "bytes_per_group")) == 2760 and int(v("W1", "ell", "bytes_per_group")) == 4104,
          "SQL runs use 2,760 B (OS-BMW) and 4,104 B (ELL, ULL, 256B) per group")

    # --- exactness under repartitioning
    groups = sum(int(float(wl[w][0]["n_groups"])) for w in WL)
    exact_engines = [k for k in DF if k != "avll"]
    integer("df.groups.per", groups)
    integer("df.groups.total", groups * len(exact_engines))
    integer("df.runs.exact", len(WL) * len(exact_engines))
    integer("df.engines", len(DF))
    mism = {w: int(float(one(wl[w], method=DF["avll"])["stage_mismatch_groups"])) for w in WL}
    integer("df.avll.mismatch", sum(mism.values()))
    for w in WL:
        integer(f"df.avll.mismatch.{w}", mism[w])
    check(all(int(float(r["stage_mismatch_groups"])) == 0 for r in rows if r["method"] != DF["avll"]),
          "no exactly mergeable engine has a cross-partition mismatch in any SQL run")

    # --- operating point 1: 2,760 B OS-BMW vs 4,104 B ELL / ULL (same flat-slab mode)
    s_ell = {w: speed(w, "5b", "ell") for w in WL}
    win = [w for w in WL if s_ell[w] > 1.0]
    times("df.5b.ell.spd.min", min(s_ell[w] for w in win))
    times("df.5b.ell.spd.max", max(s_ell[w] for w in win))
    put("df.5b.ell.nwin.word", WORDS[len(win)])
    check(win == ["W2", "W3", "W4", "W5", "W6"], "OS-BMW-5b(Slab+Sp) is faster than ELL on W2-W6")
    times("df.5b.ell.lose.min", min(s_ell[w] for w in WL if w not in win))
    times("df.5b.ell.lose.max", max(s_ell[w] for w in WL if w not in win))
    times("df.5b.ell.spd.W1", s_ell["W1"])
    times("df.5b.ell.spd.W8", s_ell["W8"])
    pct("df.5b.ell.slower.W7", 100 * (1 - s_ell["W7"]), 0)
    pct("df.5b.ell.slower.W8", 100 * (1 - s_ell["W8"]), 0)
    times("df.5b.ell.spd.W9", s_ell["W9"])
    s_ull = {w: speed(w, "5b", "ull") for w in WL}
    win_ull = [w for w in WL if s_ull[w] > 1.0]
    check(win_ull == ["W2", "W3", "W4", "W5", "W6", "W8"],
          "OS-BMW-5b(Slab+Sp) is faster than ULL on W2-W6 and W8")
    put("df.5b.ull.nwin.word", WORDS[len(win_ull)])
    times("df.5b.ull.spd.min", min(s_ull[w] for w in win_ull))
    times("df.5b.ull.spd.max", max(s_ull[w] for w in win_ull))
    times("df.5b.ull.lose.min", min(s_ull[w] for w in WL if w not in win_ull))
    times("df.5b.ull.lose.max", max(s_ull[w] for w in WL if w not in win_ull))
    d_ull = {w: rmse_rel(w, "5b", "ull") for w in WL}
    same = [w for w in WL if v(w, "5b", "rmse_pct") == v(w, "ull", "rmse_pct")]
    put("df.5b.ull.same.word", WORDS[len(same)])
    check(same == ["W1", "W2", "W4", "W5", "W6"], "OS-BMW-5b and ULL report identical RMSE on W1, W2, W4, W5, W6")
    worst = max(abs(x) for x in d_ull.values())
    pct("df.5b.ull.rmse.maxdiff", math.ceil(worst * 100) / 100, 2)
    d_ell = {w: rmse_rel(w, "5b", "ell") for w in WL}
    pct("df.5b.ell.rmse.min", min(d_ell.values()), 0)
    pct("df.5b.ell.rmse.max", max(d_ell.values()), 0)
    check(min(d_ell.values()) > 0, "at 2,760 B OS-BMW-5b has higher RMSE than the 4,104 B ELL on every workload")
    mem_slab_ell = [mem_less(w, "5b", "ell") for w in WL if w != "W6"]
    check(min(mem_slab_ell) > 0 and mem_less("W6", "5b", "ell") < 0,
          "Slab+Sp 5b uses less peak RAM than ELL on all workloads except W6")
    pct("df.5b.ell.mem.slab.min", min(mem_slab_ell), 1)
    pct("df.5b.ell.mem.slab.max", max(mem_slab_ell), 1)
    pct("df.5b.ell.mem.slab.w6.over", -mem_less("W6", "5b", "ell"), 1)
    mem_dense_ell = [mem_less(w, "5b.dense", "ell.dense") for w in WL]
    check(min(mem_dense_ell) > 0, "dense OS-BMW-5b uses less peak RAM than dense ELL on all nine workloads")
    pct("df.dense.5b.ell.mem.min", min(mem_dense_ell), 1)
    pct("df.dense.5b.ell.mem.max", max(mem_dense_ell), 1)

    # --- operating point 2: equal 4,104 B
    d256 = {w: -rmse_rel(w, "256", "ell") for w in WL}   # percent lower error
    check(min(d256.values()) > 0, "at equal bytes the 256B variant has lower RMSE than ELL on all nine workloads")
    pct("df.256.ell.rmse.min", min(d256.values()), 0)
    pct("df.256.ell.rmse.max", max(d256.values()), 0)
    s256 = {w: speed(w, "256", "ell") for w in WL}
    num("df.256.ell.spd.min", min(s256.values()), 2)
    num("df.256.ell.spd.max", max(s256.values()), 2)
    f256 = [w for w in WL if s256[w] > 1.0]
    put("df.256.ell.nfaster.word", WORDS[len(f256)])
    check(f256 == ["W2", "W3", "W4", "W5"], "the 256B variant is faster than ELL exactly on W2-W5")
    times("df.256.ell.win.min", min(s256[w] for w in f256))
    times("df.256.ell.win.max", max(s256[w] for w in f256))
    times("df.256.ell.lose.min", min(s256[w] for w in WL if w not in f256))
    times("df.256.ell.lose.max", max(s256[w] for w in WL if w not in f256))

    # --- operating point 3: 5,512 B OS-BMW vs the built-in 16 KB HyperLogLog
    d16 = {w: rmse_rel(w, "16b", "hll") for w in WL}
    pct("df.16b.hll.rmse.maxabs", math.ceil(max(abs(x) for x in d16.values())), 0)
    lower = [w for w in WL if d16[w] < 0]
    check(lower == ["W1", "W2", "W3", "W4", "W5", "W6", "W7"], "the 16-block sketch has lower RMSE than built-in HLL on W1-W7")
    pct("df.16b.hll.rmse.drop.min", min(-d16[w] for w in lower), 1)
    pct("df.16b.hll.rmse.drop.max", max(-d16[w] for w in lower), 1)
    pct("df.16b.hll.rmse.W8", d16["W8"], 1)
    pct("df.16b.hll.rmse.W9", d16["W9"], 1)
    m16 = {w: mem_less(w, "16b", "hll") for w in WL}
    check(min(m16.values()) > 0, "the 16-block sketch uses less memory over the table than built-in HLL everywhere")
    pct("df.16b.hll.mem.min", min(m16.values()), 0)
    pct("df.16b.hll.mem.max", max(m16.values()), 0)
    s16 = {w: speed(w, "16b", "hll") for w in WL}
    f16 = [w for w in WL if s16[w] > 1.0]
    check(f16 == ["W2", "W3", "W4", "W5", "W6"], "the 16-block sketch is faster than built-in HLL exactly on W2-W6")
    put("df.16b.hll.nfaster.word", WORDS[len(f16)])
    times("df.16b.hll.spd.min", min(s16[w] for w in f16))
    times("df.16b.hll.spd.max", max(s16[w] for w in f16))
    pct("df.16b.hll.slower.min", 100 * (1 - max(s16[w] for w in WL if w not in f16)), 0)
    pct("df.16b.hll.slower.max", 100 * (1 - min(s16[w] for w in WL if w not in f16)), 0)

    # --- absolute figures quoted for ClickBench (W9) and the dense workloads
    num("df.W9.5b.mrows", v("W9", "5b", "throughput_mrows_s"), 0)
    pct("df.W9.5b.rmse", v("W9", "5b", "rmse_pct"), 2)
    integer("df.W9.5b.dense.mrows", int(round(v("W9", "5b.dense", "throughput_mrows_s"))))
    integer("df.W9.ell.dense.mrows", int(round(v("W9", "ell.dense", "throughput_mrows_s"))))
    times("df.W9.5b.dense.ell.spd", speed("W9", "5b.dense", "ell.dense"))
    pct("df.W9.5b.ell.mem", mem_less("W9", "5b", "ell"), 0)
    integer("df.W8.hllpp.mrows", int(round(v("W8", "hllpp", "throughput_mrows_s"))))
    pct("df.W8.hllpp.rmse", v("W8", "hllpp", "rmse_pct"), 2)
    integer("df.W8.16b.mrows", int(round(v("W8", "16b", "throughput_mrows_s"))))
    pct("df.W8.16b.rmse", v("W8", "16b", "rmse_pct"), 2)
    integer("df.W8.5b.dense.mrows", int(round(v("W8", "5b.dense", "throughput_mrows_s"))))
    pct("df.W8.5b.dense.rmse", v("W8", "5b.dense", "rmse_pct"), 2)
    integer("df.W8.256.dense.mrows", int(round(v("W8", "256.dense", "throughput_mrows_s"))))
    pct("df.W8.256.dense.rmse", v("W8", "256.dense", "rmse_pct"), 2)
    integer("df.W8.ell.dense.mrows", int(round(v("W8", "ell.dense", "throughput_mrows_s"))))
    times("df.W8.5b.dense.ell.spd", speed("W8", "5b.dense", "ell.dense"))
    times("df.W9.5b.hll.spd", speed("W9", "5b", "hll"))
    pct("df.W9.5b.hll.mem", mem_less("W9", "5b", "hll"), 0)
    times("df.W9.5b.ull.spd", speed("W9", "5b", "ull"))
    # W7: small, cache-resident group counts favour full-range registers and 6-bit HLL
    check(v("W7", "hllpp", "wall_median_s") < min(v("W7", e, "wall_median_s") for e in ("5b", "256", "16b", "ell", "ull", "hll")),
          "6-bit HLL++ is the fastest engine on W7")
    times("df.W7.hllpp.rmse.ratio", v("W7", "hllpp", "rmse_pct") / v("W7", "16b", "rmse_pct"), 1)
    # flat slab vs dense accumulators (same sketch)
    slab = {e: [speed(w, e, e + ".dense") for w in ("W2", "W3", "W4", "W5", "W6")] for e in ("5b", "ell", "ull")}
    times("df.slab.gain.min", min(min(x) for x in slab.values()), 1)
    times("df.slab.gain.max", max(max(x) for x in slab.values()), 1)
    slab_256 = [speed(w, "256", "256.dense") for w in ("W2", "W3", "W4", "W5", "W6")]
    times("df.slab.256.max", max(slab_256), 2)
    w9_paired = ("5b", "256", "ell", "ull")
    w9_slab_spd = [speed("W9", e, e + ".dense") for e in w9_paired]
    w9_mem_ratios = [
        v("W9", e + ".dense", "rss_peak_over_table_mb") / v("W9", e, "rss_peak_over_table_mb")
        for e in w9_paired
    ]
    w9_mem_drops = [
        100 * (1 - v("W9", e, "rss_peak_over_table_mb") / v("W9", e + ".dense", "rss_peak_over_table_mb"))
        for e in w9_paired
    ]
    check(all(x > 1.0 for x in w9_slab_spd),
          "on W9 all four Slab+Sp engines run faster than dense baselines")
    check(v("W9", "5b", "rss_peak_over_table_mb") < min(v("W9", e, "rss_peak_over_table_mb") for e in DF if e != "5b"),
          "on W9 OS-BMW-5b(Slab+Sp) has the lowest peak RAM of all 12 engines")
    check(v("W9", "5b.dense", "wall_median_s") < min(v("W9", e, "wall_median_s") for e in ("256.dense", "ell.dense", "ull.dense", "hllpp", "avll")),
          "on W9 dense OS-BMW-5b beats all other dense baselines")
    times("df.W9.slab.spd.min", min(w9_slab_spd), 2)
    times("df.W9.slab.spd.max", max(w9_slab_spd), 2)
    pct("df.W9.slab.memdrop.min", min(w9_mem_drops), 0)
    pct("df.W9.slab.memdrop.max", max(w9_mem_drops), 0)
    times("df.W9.slab.mem.min", min(w9_mem_ratios), 1)
    times("df.W9.slab.mem.max", max(w9_mem_ratios), 1)
    num("df.W9.16b.s", v("W9", "16b", "wall_median_s"), 4)
    num("df.W9.ell.s", v("W9", "ell", "wall_median_s"), 4)
    num("df.W9.ell.dense.s", v("W9", "ell.dense", "wall_median_s"), 4)
    # AVLL cost
    s_avll = {w: speed(w, "5b.dense", "avll") for w in WL}
    times("df.avll.slower.min", min(s_avll[w] for w in WL if w != "W7"))
    times("df.avll.slower.max", max(s_avll[w] for w in WL if w != "W7"))
    check(all(s_avll[w] > 1.5 for w in WL if w != "W7"), "dense OS-BMW-5b is >1.5x faster than AVLL on eight workloads")
    # extra operating-point and dense-mode summaries
    times("df.W7.16b.ell.spd", speed("W7", "16b", "ell"))
    pct("df.W7.16b.ell.rmse.drop", -rmse_rel("W7", "16b", "ell"), 0)
    times("df.W8.16b.hll.spd", speed("W8", "16b", "hll"))
    pct("df.W8.16b.hll.mem", mem_less("W8", "16b", "hll"), 0)
    times("df.W9.5b.ell.spd", speed("W9", "5b", "ell"))
    times("df.W9.16b.ell.spd", speed("W9", "16b", "ell"))
    pct("df.W9.16b.ell.rmse.drop", -rmse_rel("W9", "16b", "ell"), 0)
    pct("df.W9.16b.hll.mem", mem_less("W9", "16b", "hll"), 0)
    num("df.W1.hllpp.s", v("W1", "hllpp", "wall_median_s"), 2)
    num("df.W1.5b.dense.s", v("W1", "5b.dense", "wall_median_s"), 2)
    num("df.W1.ell.dense.s", v("W1", "ell.dense", "wall_median_s"), 2)
    s_dense_ell = {w: speed(w, "5b.dense", "ell.dense") for w in WL}
    f_dense = [w for w in WL if s_dense_ell[w] > 1.0]
    check(f_dense == ["W1", "W2", "W3", "W4", "W5", "W6", "W8", "W9"],
          "dense OS-BMW-5b is faster than dense ELL on 8/9 workloads")
    put("df.dense.5b.ell.nwin.word", WORDS[len(f_dense)])
    times("df.dense.5b.ell.win.min", min(s_dense_ell[w] for w in f_dense))
    times("df.dense.5b.ell.win.max", max(s_dense_ell[w] for w in f_dense))
    times("df.dense.5b.ell.w16.min", min(s_dense_ell[w] for w in ("W1", "W2", "W3", "W4", "W5", "W6")))
    times("df.dense.5b.ell.w16.max", max(s_dense_ell[w] for w in ("W1", "W2", "W3", "W4", "W5", "W6")))
    integer("df.W9.groups", int(float(wl["W9"][0]["n_groups"])))


def micro(out: Path) -> None:
    """Single-thread microbenchmarks (Table tab:sys_micro)."""
    rows = read_tsv(out / "m3_sys_bench.tsv")
    r = {
        "5b": one(rows, method="OS-BMW-5b(k24)"),
        "256": one(rows, method="OS-BMW-5b-256B"),
        "ell24": one(rows, method="ELL(2,24)"),
        "ell20": one(rows, method="ELL(2,20)"),
        "ull": one(rows, method="ULL"),
        "hll": one(rows, method="HLL++(6b)"),
        "avll": one(rows, method="AVLL(base56)"),
    }
    f = lambda k, c: float(r[k][c])
    pct("micro.ee", f("5b", "ee_rate_1e7_pct"), 1)
    for tag, col in (("4k", "dram_4096_steady_ns"), ("64k", "dram_65536_steady_ns"), ("l1", "l1_steady_1e7_ns"), ("cold", "l1_cold_1e4_ns")):
        num(f"micro.upd.{tag}.5b", f("5b", col), 1 if tag in ("4k", "64k") else 2)
        num(f"micro.upd.{tag}.ell24", f("ell24", col), 1 if tag in ("4k", "64k") else 2)
        for other in ("ell24", "ell20", "ull", "hll"):
            times(f"micro.upd.{tag}.vs.{other}.x", f(other, col) / f("5b", col))
    times("micro.upd.4k.vs.ell24", f("ell24", "dram_4096_steady_ns") / f("5b", "dram_4096_steady_ns"), 1)
    times("micro.upd.64k.vs.ell24", f("ell24", "dram_65536_steady_ns") / f("5b", "dram_65536_steady_ns"), 1)
    times("micro.cold.ell24.faster", f("5b", "l1_cold_1e4_ns") / f("ell24", "l1_cold_1e4_ns"))
    check(f("5b", "dram_4096_steady_ns") < min(f(o, "dram_4096_steady_ns") for o in ("ell24", "ell20", "ull", "hll")),
          "OS-BMW-5b has the fastest 4K-group update among fixed-size exact sketches")
    check(f("5b", "dram_65536_steady_ns") < min(f(o, "dram_65536_steady_ns") for o in ("ell24", "ell20", "ull", "hll")),
          "OS-BMW-5b has the fastest 64K-group update among fixed-size exact sketches")
    check(f("5b", "l1_cold_1e4_ns") > f("ell24", "l1_cold_1e4_ns"), "ELL(2,24) is faster on a cold single sketch")
    num("micro.merge.5b.eq", f("5b", "merge_equal_ns") / 1000, 2)
    num("micro.merge.5b.skew", f("5b", "merge_skew_ns") / 1000, 2)
    num("micro.merge.ell24.eq", f("ell24", "merge_equal_ns") / 1000, 2)
    num("micro.merge.ull.eq", f("ull", "merge_equal_ns") / 1000, 2)
    times("micro.merge.ell24.faster", f("5b", "merge_equal_ns") / f("ell24", "merge_equal_ns"), 1)
    times("micro.merge.256.ell24.faster", f("256", "merge_equal_ns") / f("ell24", "merge_equal_ns"), 2)
    times("micro.merge.256.ull.faster", f("256", "merge_equal_ns") / f("ull", "merge_equal_ns"), 2)
    num("micro.upd.l1.256", f("256", "l1_steady_1e7_ns"), 2)
    times("micro.merge.avll.slower", f("avll", "merge_equal_ns") / f("5b", "merge_equal_ns"), 0)
    num("micro.merge.avll", f("avll", "merge_equal_ns") / 1000, 1)
    num("micro.ser.5b", f("5b", "ser_ns"), 0)
    num("micro.est.5b", f("5b", "est_ml_us"), 1)
    num("micro.est.ell24", f("ell24", "est_ml_us"), 1)
    times("micro.est.vs.ell24", f("ell24", "est_ml_us") / f("5b", "est_ml_us"), 1)
    times("micro.merge.ull.faster", f("5b", "merge_equal_ns") / f("ull", "merge_equal_ns"), 1)
    num("micro.merge.cpc11", float(one(rows, method="CPC(lgK=11)")["merge_equal_ns"]) / 1000, 2)
    num("micro.merge.hll4dyn", float(one(rows, method="HLL_4(dyn)")["merge_equal_ns"]) / 1000, 1)
    num("micro.merge.hbs12", float(one(rows, method="HBS(p=12)")["merge_equal_ns"]) / 1000, 1)
    num("micro.merge.hbs13", float(one(rows, method="HBS(p=13)")["merge_equal_ns"]) / 1000, 0)
    # merge cost of the variable-size designs, Rust and official C++ implementations together
    var = [float(one(rows, method=m)["merge_equal_ns"]) / 1000 for m in ("CPC(lgK=11)", "HLLL(p=12)", "HLL_4(dyn)", "HBS(p=12)", "HBS(p=13)")]
    off = read_tsv(out / "official_datasketches_sys.tsv")
    picked = [x for x in off if x["method"] in ("Apache_CPC", "Apache_HLL4_Dyn")]
    check(len(picked) == 3, "official DataSketches timings contain CPC (lgK 11, 12) and HLL_4 (lgK 13)")
    var += [float(x["merge_equal_us"]) for x in picked]
    num("micro.merge.var.min", min(var), 1)
    num("micro.merge.var.max", max(var), 0)
    m5_us = f("5b", "merge_equal_ns") / 1000
    times("micro.merge.var.ratio.min", min(var) / m5_us, 1)
    times("micro.merge.var.ratio.max", max(var) / m5_us, 1)
    num("micro.upd.l1.hll4dyn", float(one(rows, method="HLL_4(dyn)")["l1_steady_1e7_ns"]), 2)
    check(min(var) > 3 * m5_us, "every variable-size design merges at least 3x slower than OS-BMW-5b")


def onset(out: Path) -> None:
    """Analytic: cardinality at which the 2,760 B configuration first drops a register.

    With the floor-clamped anchor nothing is dropped while the k-th largest register of every block is
    at most R. Under Poisson arrivals with rate lam per register and level probabilities 2^-j, a register
    exceeds level R with probability q = 1 - exp(-lam * 2^-R); a block keeps everything iff fewer than k
    of its g registers do. This is a formula, not a measurement."""
    g, k, R, n_blk = 512, 24, 8, 8
    m = g * n_blk

    def p_keep_all(n: float) -> float:
        q = 1.0 - math.exp(-(n / m) * 2.0 ** (-R))
        # P[Bin(g, q) <= k-1]
        term = (1 - q) ** g
        acc = term
        for i in range(1, k):
            term *= (g - i + 1) / i * q / (1 - q)
            acc += term
        return acc ** n_blk

    def solve(target: float) -> float:
        lo, hi = 1.0, 1e7
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if p_keep_all(mid) > target:
                lo = mid
            else:
                hi = mid
        return lo

    n50, n99 = solve(0.5), solve(0.99)
    put("onset.n.half", f"${n50 / 1e4:.1f}\\times 10^4$")
    put("onset.n.99", f"${n99 / 1e4:.1f}\\times 10^4$")
    integer("onset.rule", n_blk * k * 2 ** R)
    check(2e4 < n99 < n50 < 5e4, "truncation onset of the flagship lies between 2e4 and 5e4 distinct items")


def sweep(out: Path) -> None:
    """Equal-byte sweep over 6 budgets x 12 cardinalities (Figure fig:sweep)."""
    rows = read_tsv(out / "m2_equal_byte_sweep.tsv")

    def get(b, m, n):
        h = [r for r in rows if r["budget_kb"] == b and r["method"] == m and float(r["cardinality"]) == n]
        return h[0] if len(h) == 1 and h[0]["rmse_pct"] not in ("", "NA") else None

    budgets = sorted({r["budget_kb"] for r in rows}, key=lambda x: float(x[:-2]))
    cards = sorted({float(r["cardinality"]) for r in rows})
    cells, wins, ratios = 0, 0, []
    dense_cells, dense_wins = 0, 0
    for b in budgets:
        for n in cards:
            a, e = get(b, "OS-BMW-5b-256B", n), get(b, "ELL(2,24)", n)
            if a is None or e is None:
                continue
            cells += 1
            ra, re_ = float(a["rmse_pct"]), float(e["rmse_pct"])
            wins += ra < re_
            if 1e6 <= n <= 1e9:
                dense_cells += 1
                dense_wins += ra < re_
                ratios.append(ra / re_)
    integer("sweep.measured", sum(1 for r in rows if r["rmse_pct"] not in ("", "NA")))
    integer("sweep.na", sum(1 for r in rows if r["rmse_pct"] in ("", "NA")))
    integer("sweep.cells", cells)
    integer("sweep.wins", wins)
    integer("sweep.dense.cells", dense_cells)
    integer("sweep.dense.wins", dense_wins)
    integer("sweep.budgets", len(budgets))
    integer("sweep.cards", len(cards))
    pct("sweep.dense.drop", 100 * (1 - geomean(ratios)), 1)
    check(cells == wins == 72, "hybrid OS-BMW beats ELL(2,24) in all 72 sweep cells (48/48 both-dense for n >= 10^6)")
    check(dense_cells == dense_wins == 24, "in the dense range 10^6..10^9 the equal-byte variant wins all 24 cells")
    ours = [r for r in rows if r["method"] in ("OS-BMW-5b", "OS-BMW-5b-256B") and r["rel_bias_pct"] not in ("", "NA")]
    allos = [r for r in rows if r["method"].startswith("OS-BMW") and r["rel_bias_pct"] not in ("", "NA")]
    pct("sweep.bias.main.max", max(abs(float(r["rel_bias_pct"])) for r in ours), 2)
    pct("sweep.bias.all.max", max(abs(float(r["rel_bias_pct"])) for r in allos), 2)
    check(max(abs(float(r["rel_bias_pct"])) for r in allos) < 0.2, "every OS-BMW configuration has |bias| < 0.2% in the sweep")
    non_semi = {"AVLL", "HLL_4-Fixed"}
    semi_no_hbs05 = [
        r for r in rows
        if r["method"] not in non_semi
        and not (r["method"] == "HBS" and r["budget_kb"] == "0.5KB")
        and r["rel_bias_pct"] not in ("", "NA")
    ]
    max_no_hbs05 = max(abs(float(r["rel_bias_pct"])) for r in semi_no_hbs05)
    pct("sweep.semi.bias.nohbs05", math.ceil(max_no_hbs05 * 1000) / 1000, 3)
    check(max_no_hbs05 <= 0.367, "semilattice bias excl. 0.5KB HBS stays <= 0.367%")
    hbs05_rows = [
        r for r in rows
        if r["method"] == "HBS" and r["budget_kb"] == "0.5KB" and r["rel_bias_pct"] not in ("", "NA")
    ]
    hbs05_max_row = max(hbs05_rows, key=lambda r: abs(float(r["rel_bias_pct"])))
    hbs05_max = abs(float(hbs05_max_row["rel_bias_pct"]))
    pct("sweep.hbs05.bias.max", hbs05_max, 3)
    check(float(hbs05_max_row["cardinality"]) == 1e5, "0.5KB HBS attains its max bias at n=10^5")
    hbs05_2e64_row = max(hbs05_rows, key=lambda r: float(r["cardinality"]))
    hbs05_2e64 = abs(float(hbs05_2e64_row["rel_bias_pct"]))
    pct("sweep.hbs05.bias.2e64", hbs05_2e64, 3)
    check(hbs05_2e64 > 0.367, "0.5KB HBS bias at 2^64 also exceeds 0.367%")
    avll_max = max(
        abs(float(r["rel_bias_pct"])) for r in rows
        if r["method"] == "AVLL" and r["rel_bias_pct"] not in ("", "NA")
    )
    pct("sweep.avll.bias.max", avll_max, 2)
    hbs_4_16 = [
        float(r["mvp_ram"]) for r in rows
        if r["method"] == "HBS" and r["budget_kb"] in ("4KB", "8KB", "16KB") and float(r["cardinality"]) == 1e6
    ]
    check(len(hbs_4_16) == 3, "HBS 4/8/16KB rows present at n=1e6")
    num("hbs.ram.mvp.min", min(hbs_4_16), 2)
    num("hbs.ram.mvp.max", max(hbs_4_16), 2)


def ablation(out: Path) -> None:
    """Design choices: anchor rank k and slot width (Figure fig:ablation)."""
    rows = read_tsv(out / "m2_geometry_ablation.tsv")
    kk = {int(r["k"]): float(r["theory_mvp_payload"]) for r in rows
          if r["part"] == "B_Threshold_K" and r["config"].startswith("5b_g512_")}
    for k in (1, 8, 20, 24, 32):
        num(f"abl.k{k}", kk[k], 2)
    times("abl.k1.ratio", kk[1] / kk[24], 1)
    best = min(kk, key=kk.get)
    integer("abl.kbest", best)
    pct("abl.k24.vs.best", 100 * (kk[24] / kk[best] - 1), 1)
    check(best == 20 and kk[24] / kk[best] < 1.01, "k=20 minimizes the theoretical MVP for g=512 and k=24 is within 1%")
    check(kk[1] > 5 * kk[24], "the maximum anchor (k=1) has more than 5x the MVP of k=24")
    g256 = {int(r["k"]): float(r["theory_mvp_payload"]) for r in rows
            if r["part"] == "B_Threshold_K" and r["config"].startswith("5b_g256_")}
    integer("abl.g256.kbest", min(g256, key=g256.get))
    rows4k = read_tsv(out / "m2_geometry_ablation_4k.tsv")
    kk_emp = {int(r["k"]): float(r["emp_mvp_payload"]) for r in rows4k
              if r["part"] == "B_Threshold_K" and r["config"].startswith("5b_g512_")}
    for k in (16, 20, 24):
        num(f"abl.emp.k{k}", kk_emp[k], 2)
    check(kk_emp[16] < kk_emp[20] < kk_emp[24], "2000-trial MC reaches minimum at k=16 before k=20,24")
    w = [r for r in rows if r["part"] == "A_Geom_Width" and r["budget"] == "4KB" and r["cardinality"] == "1.000e6"]
    drops = []
    mc_reversals = 0
    for g in ("128", "256", "384", "512"):
        a = [r for r in w if r["config"].startswith("5b_g" + g + "_")]
        b = [r for r in w if r["config"].startswith("4b_g" + g + "_")]
        if len(a) != 1 or len(b) != 1:
            die(f"slot-width ablation: missing rows for g={g}")
        drops.append(100 * (1 - float(a[0]["theory_mvp_payload"]) / float(b[0]["theory_mvp_payload"])))
        if float(a[0]["emp_mvp_payload"]) > float(b[0]["emp_mvp_payload"]):
            mc_reversals += 1
    pct("abl.5b.vs.4b.min", min(drops), 1)
    pct("abl.5b.vs.4b.max", max(drops), 1)
    check(min(drops) > 0, "5-bit slots have a lower theoretical MVP than 4-bit slots for every block size")
    check(mc_reversals == 2, "two of four 200-trial MC cells reverse the theoretical 5b vs 4b order")


def real_streams(out: Path) -> None:
    """Real public graphs and unified Rust streams (m2_real_streams.tsv)."""
    rows = read_tsv(out / "m2_real_streams.tsv")
    wls = sorted({r["workload"] for r in rows})
    check(len(wls) == 8, "real-stream suite covers 8 workloads")
    integer("rs.nstreams", len(wls))

    def mean_mvp(budget: str, method: str) -> float:
        vals = [float(r["eff_mvp_ram"]) for r in rows if r["budget"] == budget and r["method"] == method]
        check(len(vals) == len(wls), f"real streams {budget} {method} has 8 rows")
        return sum(vals) / len(vals)

    for tag, m in (("5b", "OS-BMW-5b"), ("256", "OS-BMW-5b-256B"), ("ell24", "ELL(2,24)"),
                   ("ell20", "ELL(2,20)"), ("ull", "ULL"), ("avll", "AVLL"), ("hbs", "HBS")):
        num(f"rs.4k.mvp.{tag}", mean_mvp("4KB", m), 2)
        num(f"rs.1k.mvp.{tag}", mean_mvp("1KB", m), 2)

    wins_4k = sum(
        1 for w in wls
        if float(one(rows, budget="4KB", workload=w, method="OS-BMW-5b-256B")["rmse_pct"])
        < float(one(rows, budget="4KB", workload=w, method="ELL(2,24)")["rmse_pct"])
    )
    wins_1k = sum(
        1 for w in wls
        if float(one(rows, budget="1KB", workload=w, method="OS-BMW-5b-256B")["rmse_pct"])
        < float(one(rows, budget="1KB", workload=w, method="ELL(2,24)")["rmse_pct"])
    )
    check(wins_4k == 6 and wins_1k == 4, "256B beats ELL(2,24) RMSE on 6/8 streams at 4KB and 4/8 at 1KB")
    integer("rs.4k.wins", wins_4k)
    integer("rs.1k.wins", wins_1k)


def official_libs(out: Path) -> None:
    """Official library baselines, reconciliation counts, rank wire extension, and HIP."""
    import json
    ds_p = read_tsv(out / "official_datasketches_pareto.tsv")
    cpc12 = one(ds_p, method="Apache_CPC", lg_k="12", n="1.0000e+06")
    hll4_13 = one(ds_p, method="Apache_HLL4_Dyn", lg_k="13", n="1.0000e+06")
    integer("cpc.lg12.live", int(float(cpc12["ram_live_bytes"])))
    integer("cpc.lg12.peak.b", int(round(float(cpc12["ram_peak_bytes"]))))
    num("cpc.lg12.peak.kb", float(cpc12["ram_peak_bytes"]) / 1024, 1)
    integer("cpc.lg12.wire.b", int(round(float(cpc12["wire_compact_bytes"]))))
    num("cpc.lg12.wire.kb", float(cpc12["wire_compact_bytes"]) / 1024, 1)
    num("cpc.lg12.wire.mvp", float(cpc12["mvp_wire_compact"]), 2)
    num("hll4.lg13.mvp.ram", float(hll4_13["mvp_ram_live"]), 2)
    num("hll4.lg13.peak.kb", float(hll4_13["ram_peak_bytes"]) / 1024, 1)

    ds_s = read_tsv(out / "official_datasketches_sys.tsv")
    cpc11_s = one(ds_s, method="Apache_CPC", lg_k="11")
    cpc12_s = one(ds_s, method="Apache_CPC", lg_k="12")
    hll4_13_s = one(ds_s, method="Apache_HLL4_Dyn", lg_k="13")
    num("cpc.lg11.merge.us", float(cpc11_s["merge_equal_us"]), 1)
    num("cpc.lg12.merge.us", float(cpc12_s["merge_equal_us"]), 1)
    num("cpc.lg12.ser.us", float(cpc12_s["serialize_ns"]) / 1000, 1)
    num("hll4.lg13.merge.us", float(hll4_13_s["merge_equal_us"]), 1)

    h4j = read_tsv(out / "official_hash4j_pareto.tsv")
    h4j_ull = one(h4j, method="Hash4j_UltraLogLog", estimator="MLE", p="12", n="1000000")
    num("hash4j.ull.mvp", float(h4j_ull["mvp_state"]), 2)

    ms = read_tsv(out / "m3_256b_mulshift.tsv")
    orig_l1 = float(one(ms, method="OS-BMW-5b-256B_Div3_Orig")["l1_steady_1e7_ns"])
    exact_l1 = float(one(ms, method="OS-BMW-5b-256B_MulShift128_Exact")["l1_steady_1e7_ns"])
    zero_l1 = float(one(ms, method="OS-BMW-5b-256B_MulShift384_ZeroCP")["l1_steady_1e7_ns"])
    num("mulshift.l1.orig", orig_l1, 2)
    num("mulshift.l1.exact", exact_l1, 2)
    num("mulshift.l1.zero", zero_l1, 2)
    pct("mulshift.l1.drop", 100 * (1 - exact_l1 / orig_l1), 1)
    pct("mulshift.l1.zero.drop", 100 * (1 - zero_l1 / orig_l1), 1)
    check(exact_l1 < orig_l1 and zero_l1 < exact_l1, "MulShift128 and MulShift384 speed up 256B L1 steady updates")

    rec = json.loads((out / "m0_m1_reconcile.json").read_text())
    alg = rec["m1_algebraic_audit"]
    integer("reconcile.pairs", alg["exhaustive_512x512_pairs"])
    integer("reconcile.triples", alg["inplace_triple_merge_trials"])
    rr = rec["m0_reconciliation_3_0061_vs_2_9977"]
    b_rank = int(rr["rank_wire_k60_fixed_bytes"])
    mvp_rank = float(rr["rank_wire_k60_byte_mvp"])
    integer("rank.bytes", b_rank)
    num("rank.mvp.theory", mvp_rank, 3)
    num("rank.mvp.hdr", mvp_rank * (2 * b_rank + 8) / (2 * b_rank), 3)
    num("rank.enc.us", float(rr["rank_codec_encode_us_per_block"]), 1)
    num("rank.dec.us", float(rr["rank_codec_decode_us_per_block"]), 1)

    mc_rows = read_tsv(out / "m2_poisson_16k_mc.tsv")
    rank_mc = one(mc_rows, method="OS-BMW-Rank(Wire1277B)")
    num("rank.mvp.emp", float(rank_mc["mc_mvp_payload"]), 3)
    num("rank.mvp.hdr.emp", float(rank_mc["mc_mvp_with_hdr"]), 3)
    num("rank.mvp.se", float(rank_mc["mc_mvp_payload_se"]), 3)

    hip_rows = read_tsv(out / "m2_equal_error_hip.tsv")
    eq_256 = one(hip_rows, part="A_EqualError", tier="1.0pct", method="OS-BMW-5b-256B(16blk)", n_shards="1")
    eq_ell = one(hip_rows, part="A_EqualError", tier="1.0pct", method="ELL(2,24)(p=10)", n_shards="1")
    integer("eqerr.1pct.256.b", int(round(float(eq_256["eq_err_ram_bytes"]))))
    integer("eqerr.1pct.ell24.b", int(round(float(eq_ell["eq_err_ram_bytes"]))))
    pct("eqerr.1pct.save", 100 * (1 - float(eq_256["eq_err_ram_bytes"]) / float(eq_ell["eq_err_ram_bytes"])), 1)
    hip_256 = one(hip_rows, part="B_Martingale_vs_Merge", method="OS-BMW-5b-256B", estimator="Martingale_HIP", n_shards="1")
    hip_5b = one(hip_rows, part="B_Martingale_vs_Merge", method="OS-BMW-5b(k24)", estimator="Martingale_HIP", n_shards="1")
    hip_256_m = one(hip_rows, part="B_Martingale_vs_Merge", method="OS-BMW-5b-256B", estimator="Mergeable_MLE", n_shards="2")
    hip_5b_m = one(hip_rows, part="B_Martingale_vs_Merge", method="OS-BMW-5b(k24)", estimator="Mergeable_MLE", n_shards="2")
    hip_ell16 = one(hip_rows, part="B_Martingale_vs_Merge", method="ELL(2,16)", estimator="Martingale_HIP", n_shards="1")
    hip_ell16_m = one(hip_rows, part="B_Martingale_vs_Merge", method="ELL(2,16)", estimator="Mergeable_MLE", n_shards="2")
    num("hip.256.mvp", float(hip_256["mvp_ram"]), 2)
    num("hip.5b.mvp", float(hip_5b["mvp_ram"]), 2)
    num("hip.256.mrg.mvp", float(hip_256_m["mvp_ram"]), 2)
    num("hip.5b.mrg.mvp", float(hip_5b_m["mvp_ram"]), 2)
    num("hip.ell16.mvp", float(hip_ell16["mvp_ram"]), 2)
    num("hip.ell16.mrg.mvp", float(hip_ell16_m["mvp_ram"]), 2)
    pct("hip.ell16.jump", 100 * (float(hip_ell16_m["mvp_ram"]) / float(hip_ell16["mvp_ram"]) - 1), 1)

    h64 = read_tsv(out / "m2_hash64_vs_128_ablation.tsv")
    integer("h64v128.cells", len(h64))
    pct("h64v128.bias.max", math.ceil(max(abs(float(r["rel_bias_pct"])) for r in h64) * 100) / 100, 2)


def supplementary_exps(out: Path) -> None:
    """Supplementary empirical verification & diagnostics (E-01..E-11)."""
    import json

    # E-04: Direct AVX-512 vs forced-scalar parity
    par = json.loads((out / "m1_simd_scalar_parity.json").read_text())
    check(par.get("avx512_compiled") is True, "E-04 parity run compiled AVX-512")
    check(int(par.get("mismatches", -1)) == 0 and par.get("status") == "PASS",
          "E-04 AVX-512 vs forced-scalar merge parity has 0 byte mismatches")
    integer("parity.merges", int(par["total_merges_checked"]))

    # E-11: Closest equal-byte cells & official C++/Java wrapper verification
    c11 = json.loads((out / "m2_closest_cells_and_wrappers.json").read_text())
    sw11 = c11["equal_byte_sweep_audit"]["osbmw256b_vs_ell24_all_budgets"]
    check(sw11["all_cells"]["count"] == sw11["all_cells"]["point_wins_a"] == 72,
          "E-11 closest-cell audit confirms 72/72 point wins")
    check(all(c11["wrapper_audit"]["checks"].values()),
          "E-11 official C++/Java wrapper checks pass")
    num("closest.z.all", float(sw11["all_cells"]["pooled_z_conservative"]), 1)
    num("closest.z.dense24", float(sw11["dense_1e6_to_1e9"]["pooled_z_conservative"]), 1)
    num("closest.z.dense48", float(sw11["dense_all_ge_1e6"]["pooled_z_conservative"]), 1)
    integer("closest.trials.dense48", int(sw11["dense_all_ge_1e6"]["total_trials_a"]))
    integer("closest.n.lt196", int(sw11["all_cells"]["abs_z_lt_1_96_count"]))
    integer("closest.n.overlap", int(sw11["all_cells"]["ci95_overlap_count"]))
    num("closest.min.z", float(sw11["min_margin_cell"]["z_score_mvp"]), 2)
    pct("closest.min.a", float(sw11["min_margin_cell"]["rmse_a_pct"]), 3)
    pct("closest.min.b", float(sw11["min_margin_cell"]["rmse_b_pct"]), 3)

    # E-10: 200-seed real-stream bootstrap CIs
    rsci = read_tsv(out / "m2_real_streams_ci.tsv")
    for btag, bval in (("4k", "4KB"), ("1k", "1KB")):
        s256 = [r for r in rsci if r["budget"] == bval and r["method"] == "OS-BMW-5b-256B"]
        s5b = [r for r in rsci if r["budget"] == bval and r["method"] == "OS-BMW-5b"]
        sell = [r for r in rsci if r["budget"] == bval and r["method"] == "ELL(2,24)"]
        check(len(s256) == 8 and len(s5b) == 8 and len(sell) == 8, f"E-10 {bval} has 8 streams")
        wins = sum(1 for r in s256 if float(r["diff_vs_ell24_rmse_pct"]) < 0.0)
        sig = sum(1 for r in s256 if float(r["diff_ci95_hi"]) < 0.0)
        check(wins == 8, f"E-10 256B beats ELL(2,24) on all 8 streams at {bval} with 200 seeds")
        integer(f"rsci.{btag}.wins", wins)
        integer(f"rsci.{btag}.sig", sig)
        num(f"rsci.{btag}.mvp.256", sum(float(r["mvp_ram"]) for r in s256) / 8.0, 2)
        num(f"rsci.{btag}.mvp.5b", sum(float(r["mvp_ram"]) for r in s5b) / 8.0, 2)
        num(f"rsci.{btag}.mvp.ell24", sum(float(r["mvp_ram"]) for r in sell) / 8.0, 2)
    integer("rsci.seeds", int(rsci[0]["seeds"]))

    # E-03: 4,000-trial 5-bit vs 4-bit & 2,000-trial top-k ablation
    g4k = read_tsv(out / "m2_geometry_ablation_4k.tsv")
    pa4 = [r for r in g4k if r["part"] == "A_Geom_Width" and r["budget"] == "4KB" and abs(float(r["cardinality"]) - 1e6) < 1.0]
    pb4 = [r for r in g4k if r["part"] == "B_Threshold_K"]
    wins_4k_abl = 0
    for g in ("128", "256", "384", "512"):
        r5 = [r for r in pa4 if r["g"] == g and r["slot_bits"] == "5"][0]
        r4 = [r for r in pa4 if r["g"] == g and r["slot_bits"] == "4"][0]
        if float(r5["emp_mvp_payload"]) < float(r4["emp_mvp_payload"]):
            wins_4k_abl += 1
    check(wins_4k_abl == 4, "E-03 5-bit beats 4-bit empirically in all 4/4 block sizes at 4,000 trials")
    integer("abl4k.wins", wins_4k_abl)
    integer("abl4k.trials.a", int(pa4[0]["trials"]))
    integer("abl4k.trials.b", int(pb4[0]["trials"]))

    # E-01: ELL(2,24) + Filter-8B/16B & ULL + Filter-8B microbenchmark ablation
    ef = read_tsv(out / "m3_ell_filter_ablation.tsv")
    ef5b = one(ef, method="OS-BMW-5b(k24)")
    efell = one(ef, method="ELL(2,24)")
    eff8 = one(ef, method="ELL(2,24)+Filter-8B")
    eff16 = one(ef, method="ELL(2,24)+Filter-16B")
    num("efilt.64k.1e7.ell", float(efell["g64k_n1e7_ns"]), 1)
    num("efilt.64k.1e7.f8", float(eff8["g64k_n1e7_ns"]), 1)
    num("efilt.64k.1e7.5b", float(ef5b["g64k_n1e7_ns"]), 1)
    num("efilt.64k.1e6.ell", float(efell["g64k_n1e6_ns"]), 1)
    num("efilt.64k.1e6.f8", float(eff8["g64k_n1e6_ns"]), 1)
    num("efilt.4k.1e6.ell", float(efell["g4k_n1e6_ns"]), 1)
    num("efilt.4k.1e6.f8", float(eff8["g4k_n1e6_ns"]), 1)
    pct("efilt.ee.1e6.f8", float(eff8["ee_1e6_pct"]), 1)
    pct("efilt.ee.1e6.5b", float(ef5b["ee_1e6_pct"]), 1)
    times("efilt.f16.mrg.x", float(eff16["merge_eq_ns"]) / float(efell["merge_eq_ns"]), 1)
    check(float(eff8["g64k_n1e7_ns"]) < float(efell["g64k_n1e7_ns"]) and float(eff8["g64k_n1e6_ns"]) > float(efell["g64k_n1e6_ns"]),
          "E-01 ELL+Filter-8B speeds up at 1e7 but regresses at 1e6")

    # E-09: Moderate cardinality (1e3..1e6) 4K & 64K transition
    mc = read_tsv(out / "m3_moderate_card_micro.tsv")
    mc_1k_5b = one(mc, prefill_n="1000", method="OS-BMW-5b(k24)")
    mc_1k_ell = one(mc, prefill_n="1000", method="ELL(2,24)")
    mc_10k_5b = one(mc, prefill_n="10000", method="OS-BMW-5b(k24)")
    mc_10k_ell = one(mc, prefill_n="10000", method="ELL(2,24)")
    mc_100k_5b = one(mc, prefill_n="100000", method="OS-BMW-5b(k24)")
    mc_100k_ell = one(mc, prefill_n="100000", method="ELL(2,24)")
    mc_100k_f8 = one(mc, prefill_n="100000", method="ELL(2,24)+Filter-8B")
    mc_1m_5b = one(mc, prefill_n="1000000", method="OS-BMW-5b(k24)")
    mc_1m_ell = one(mc, prefill_n="1000000", method="ELL(2,24)")
    check(float(mc_1k_5b["ee_rate_pct"]) == 0.0 and float(mc_10k_5b["ee_rate_pct"]) == 0.0
          and float(mc_100k_5b["ee_rate_pct"]) > 50.0 and float(mc_1m_5b["ee_rate_pct"]) > 90.0,
          "E-09 moderate-card early exit turns on cleanly between 1e4 and 1e5")
    num("modc.1k.5b.4k", float(mc_1k_5b["g4k_ns"]), 1)
    num("modc.1k.ell.4k", float(mc_1k_ell["g4k_ns"]), 1)
    num("modc.1k.5b.64k", float(mc_1k_5b["g64k_ns"]), 1)
    num("modc.1k.ell.64k", float(mc_1k_ell["g64k_ns"]), 1)
    pct("modc.100k.5b.ee", float(mc_100k_5b["ee_rate_pct"]), 1)
    num("modc.100k.5b.64k", float(mc_100k_5b["g64k_ns"]), 1)
    num("modc.100k.ell.64k", float(mc_100k_ell["g64k_ns"]), 1)
    num("modc.100k.f8.64k", float(mc_100k_f8["g64k_ns"]), 1)
    pct("modc.1m.5b.ee", float(mc_1m_5b["ee_rate_pct"]), 1)
    num("modc.1m.5b.64k", float(mc_1m_5b["g64k_ns"]), 1)
    num("modc.1m.ell.64k", float(mc_1m_ell["g64k_ns"]), 1)

    # E-08: Non-AVX-512 merge timings (x86 no-AVX-512 + macOS ARM64)
    na = read_tsv(out / "m3_non_avx512_merge.tsv")
    avx_5b = float(one(na, target_tag="yao_avx512", merge_path="compiled_avx512", method="OS-BMW-5b(k24)")["merge_eq_ns"])
    noavx_5b = float(one(na, target_tag="yao_x86_noavx512", merge_path="compiled_portable", method="OS-BMW-5b(k24)")["merge_eq_ns"])
    noavx_ell = float(one(na, target_tag="yao_x86_noavx512", merge_path="compiled_portable", method="ELL(2,24)")["merge_eq_ns"])
    arm_5b = float(one(na, target_tag="macos_arm64_scalar", merge_path="compiled_portable", method="OS-BMW-5b(k24)")["merge_eq_ns"])
    arm_ell = float(one(na, target_tag="macos_arm64_scalar", merge_path="compiled_portable", method="ELL(2,24)")["merge_eq_ns"])
    num("noavx.avx512.eq.us", avx_5b / 1000.0, 2)
    num("noavx.5b.eq.us", noavx_5b / 1000.0, 1)
    times("noavx.5b.vs.avx.x", noavx_5b / avx_5b, 1)
    num("noavx.ell.eq.us", noavx_ell / 1000.0, 2)
    num("arm64.5b.eq.us", arm_5b / 1000.0, 1)
    num("arm64.ell.eq.us", arm_ell / 1000.0, 2)

    # E-02: W7 root cause diagnostics
    w7d = json.loads((out / "m4_w7_diag.json").read_text())["profiles"]
    p5 = [p for p in w7d if p["method"] == "OS-BMW-5b(k24)"][0]
    p16 = [p for p in w7d if p["method"] == "OS-BMW-5b(k24,16blk)"][0]
    w7r = read_tsv(out / "m4_w7_rootcause.tsv")
    w7_a_5b = one(w7r, build_tag="avx512", method="OS-BMW-5b(k24,Slab+Sp)")
    w7_s_16b = one(w7r, build_tag="scalar", method="OS-BMW-5b(k24,16blk)")
    w7_s_ell = one(w7r, build_tag="scalar", method="ELL(2,24)")
    w7_s_5b = one(w7r, build_tag="scalar", method="OS-BMW-5b(k24)")
    check(float(w7_s_16b["wall_median_s"]) < float(w7_s_ell["wall_median_s"]) < float(w7_s_5b["wall_median_s"]),
          "E-02 scalar W7 preserves exact 16blk < ELL < 5b ordering")
    freqs = [float(r["freq_med_mhz"]) for r in w7r]
    num("w7d.mrg.ms", float(w7_a_5b["mrg_cpu_ms"]), 1)
    num("w7d.freq.min", min(freqs) / 1000.0, 2)
    num("w7d.freq.max", max(freqs) / 1000.0, 2)
    num("w7d.sc.16b.s", float(w7_s_16b["wall_median_s"]), 2)
    num("w7d.sc.ell.s", float(w7_s_ell["wall_median_s"]), 2)
    num("w7d.sc.5b.s", float(w7_s_5b["wall_median_s"]), 2)
    pct("w7d.ee.5b", float(p5["ee_pct"]), 1)
    pct("w7d.lut.5b", float(p5["lut_pct"]), 1)
    num("w7d.anc.5b", float(p5["anchor_mean"]), 2)
    pct("w7d.ee.16b", float(p16["ee_pct"]), 1)
    pct("w7d.lut.16b", float(p16["lut_pct"]), 1)
    num("w7d.anc.16b", float(p16["anchor_mean"]), 2)

    # E-05: W10 out-of-L3 dense SQL
    w10 = read_tsv(out / "m4_w10_outofl3.tsv")
    w10_5bd = one(w10, method="OS-BMW-5b(k24)")
    w10_5bs = one(w10, method="OS-BMW-5b(k24,Slab+Sp)")
    w10_16bd = one(w10, method="OS-BMW-5b(k24,16blk)")
    w10_elld = one(w10, method="ELL(2,24)")
    w10_ells = one(w10, method="ELL(2,24,Slab+Sp)")
    w10_ulld = one(w10, method="ULL")
    w10_ulls = one(w10, method="ULL(Slab+Sp)")
    w10_avll = one(w10, method="AVLL(base56)")
    w10_hll = one(w10, method="DF-BuiltIn-HLL(16KB)")
    w10_hllpp = one(w10, method="HLL++(6b)")
    integer("w10.5b.d.mem", int(round(float(w10_5bd["rss_peak_over_table_mb"]))))
    integer("w10.5b.sp.mem", int(round(float(w10_5bs["rss_peak_over_table_mb"]))))
    integer("w10.ell.d.mem", int(round(float(w10_elld["rss_peak_over_table_mb"]))))
    integer("w10.16b.d.mem", int(round(float(w10_16bd["rss_peak_over_table_mb"]))))
    integer("w10.hll.mem", int(round(float(w10_hll["rss_peak_over_table_mb"]))))
    pct("w10.5b.vs.ell.mem", 100.0 * (1.0 - float(w10_5bd["rss_peak_over_table_mb"]) / float(w10_elld["rss_peak_over_table_mb"])), 0)
    pct("w10.16b.vs.hll.mem", 100.0 * (1.0 - float(w10_16bd["rss_peak_over_table_mb"]) / float(w10_hll["rss_peak_over_table_mb"])), 0)
    times("w10.5b.sp.vs.ell", float(w10_ells["wall_median_s"]) / float(w10_5bs["wall_median_s"]), 2)
    times("w10.5b.sp.vs.hll", float(w10_hll["wall_median_s"]) / float(w10_5bs["wall_median_s"]), 2)
    times("w10.5b.sp.vs.avll", float(w10_avll["wall_median_s"]) / float(w10_5bs["wall_median_s"]), 2)
    num("w10.avll.mrg.s", float(w10_avll["mrg_cpu_ms"]) / 1000.0, 1)
    integer("w10.5b.mrg.ms", int(round(float(w10_5bs["mrg_cpu_ms"]))))
    num("w10.ull.sp.s", float(w10_ulls["wall_median_s"]), 2)
    num("w10.ull.d.s", float(w10_ulld["wall_median_s"]), 2)
    num("w10.ell.d.s", float(w10_elld["wall_median_s"]), 2)
    num("w10.5b.d.s", float(w10_5bd["wall_median_s"]), 2)
    num("w10.hllpp.s", float(w10_hllpp["wall_median_s"]), 2)
    check(all(int(float(r["stage_mismatch_groups"])) == 0 for r in w10),
          "E-05 W10 shows 0 mismatch groups across exact engines (single-run check)")

    # E-06: DataFusion stage breakdown & 64-partition skewed W5
    sb = read_tsv(out / "m4_stage_breakdown.tsv")
    sb_64p_5b = one(sb, workload="W5_64P_16Kg_64M_Skew", method="OS-BMW-5b(k24)")
    sb_64p_ell = one(sb, workload="W5_64P_16Kg_64M_Skew", method="ELL(2,24)")
    sb_w6_5b = one(sb, workload="W6_8T_65Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    sb_w6_ell = one(sb, workload="W6_8T_65Kg_64M", method="ELL(2,24,Slab+Sp)")
    pct("stg.w64p.mem.drop", 100.0 * (1.0 - float(sb_64p_5b["rss_peak_over_table_mb"]) / float(sb_64p_ell["rss_peak_over_table_mb"])), 0)
    num("stg.w64p.5b.gb", float(sb_64p_5b["rss_peak_over_table_mb"]) / 1024.0, 1)
    num("stg.w64p.ell.gb", float(sb_64p_ell["rss_peak_over_table_mb"]) / 1024.0, 1)
    tot_64p_5b = sum(float(sb_64p_5b[c]) for c in ("upd_cpu_ms", "mrg_cpu_ms", "eval_cpu_ms", "state_cpu_ms"))
    pct("stg.w64p.mrg.pct", 100.0 * float(sb_64p_5b["mrg_cpu_ms"]) / tot_64p_5b, 1)
    tot_w6_5b = sum(float(sb_w6_5b[c]) for c in ("upd_cpu_ms", "mrg_cpu_ms", "eval_cpu_ms", "state_cpu_ms"))
    pct("stg.w6.mrg.pct", 100.0 * float(sb_w6_5b["mrg_cpu_ms"]) / tot_w6_5b, 1)

    # E-07: W6 fixed sparse cap + MulShift384 DataFusion SQL ablation
    fc = read_tsv(out / "m4_w6_fixedcap.tsv")
    fc_1k_5b = one(fc, build_tag="cap1024", method="OS-BMW-5b(k24,Slab+Sp)")
    fc_1k_ell = one(fc, build_tag="cap1024", method="ELL(2,24,Slab+Sp)")
    fc_4k_5b = one(fc, build_tag="cap4096", method="OS-BMW-5b(k24,Slab+Sp)")
    fc_4k_ell = one(fc, build_tag="cap4096", method="ELL(2,24,Slab+Sp)")
    times("w6cap.1k.spd", float(fc_1k_ell["wall_median_s"]) / float(fc_1k_5b["wall_median_s"]), 2)
    pct("w6cap.1k.mem", 100.0 * (1.0 - float(fc_1k_5b["rss_peak_over_table_mb"]) / float(fc_1k_ell["rss_peak_over_table_mb"])), 1)
    times("w6cap.4k.spd", float(fc_4k_ell["wall_median_s"]) / float(fc_4k_5b["wall_median_s"]), 2)

    ms_sql = read_tsv(out / "m4_mulshift_sql.tsv")
    ms_w7_orig = one(ms_sql, workload="W7_8T_128g_768M_Dense", method="OS-BMW-5b-256B")
    ms_w7_384 = one(ms_sql, workload="W7_8T_128g_768M_Dense", method="OS-BMW-5b-256B(MS384)")
    ms_w9_orig = one(ms_sql, workload="W9_8T_ClickBench_32M", method="OS-BMW-5b-256B(Slab+Sp)")
    ms_w9_384 = one(ms_sql, workload="W9_8T_ClickBench_32M", method="OS-BMW-5b-256B(MS384,Slab+Sp)")
    pct("ms384.w7.drop", 100.0 * (1.0 - float(ms_w7_384["wall_median_s"]) / float(ms_w7_orig["wall_median_s"])), 1)
    pct("ms384.w9.drop", 100.0 * (1.0 - float(ms_w9_384["wall_median_s"]) / float(ms_w9_orig["wall_median_s"])), 1)

    # EXP-OPT-1a: AVX2 (x86-64-v3) auto-vectorized merge check
    avx2 = read_tsv(out / "m3_avx2_merge.tsv")
    avx2_5b = float(one(avx2, target_tag="yao_avx2", merge_path="compiled_portable", method="OS-BMW-5b(k24)")["merge_eq_ns"])
    avx2_ell = float(one(avx2, target_tag="yao_avx2", merge_path="compiled_portable", method="ELL(2,24)")["merge_eq_ns"])
    avx2_ull = float(one(avx2, target_tag="yao_avx2", merge_path="compiled_portable", method="ULL")["merge_eq_ns"])
    num("avx2.5b.eq.us", avx2_5b / 1000.0, 1)
    num("avx2.ell.eq.us", avx2_ell / 1000.0, 2)
    num("avx2.ull.eq.us", avx2_ull / 1000.0, 2)

    # EXP-OPT-2a & 2b: ULL-FGRA single-sketch microbench + 11-run SQL ablation (W1, W3, W5, W6, W9)
    fgm = read_tsv(out / "m3_ull_fgra_micro.tsv")
    fgm_5b = float(one(fgm, method="OS-BMW-5b(k24)", estimator="MLE")["per_sketch_est_us"])
    fgm_ell = float(one(fgm, method="ELL(2,24)", estimator="MLE")["per_sketch_est_us"])
    fgm_ull_mle = float(one(fgm, method="ULL", estimator="MLE")["per_sketch_est_us"])
    fgm_ull_fgra = float(one(fgm, method="ULL", estimator="FGRA")["per_sketch_est_us"])
    num("fgra.micro.5b.us", fgm_5b, 1)
    num("fgra.micro.ell.us", fgm_ell, 1)
    num("fgra.micro.ull.mle.us", fgm_ull_mle, 1)
    num("fgra.micro.ull.fgra.us", fgm_ull_fgra, 1)
    times("fgra.micro.speedup", fgm_ull_mle / fgm_ull_fgra, 2)

    fga = read_tsv(out / "m4_ull_fgra_ablation.tsv")
    fga_w6_5b = one(fga, workload="W6_8T_65Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    fga_w6_ell = one(fga, workload="W6_8T_65Kg_64M", method="ELL(2,24,Slab+Sp)")
    times("fgra.w6.eval.x", float(fga_w6_ell["eval_cpu_ms"]) / float(fga_w6_5b["eval_cpu_ms"]), 2)
    fgra_rmse_pen = []
    fgra_wall_vs_ull = []
    for w_full in ("W1_1T_16Kg_32M", "W3_8T_Zipf_4Kg_40M", "W5_32T_16Kg_64M", "W6_8T_65Kg_64M", "W9_8T_ClickBench_32M"):
        r5 = one(fga, workload=w_full, method="OS-BMW-5b(k24,Slab+Sp)")
        ru = one(fga, workload=w_full, method="ULL(Slab+Sp)")
        rf = one(fga, workload=w_full, method="ULL-FGRA(Slab+Sp)")
        fgra_rmse_pen.append(100.0 * (float(rf["rmse_pct"]) / float(r5["rmse_pct"]) - 1.0))
        fgra_wall_vs_ull.append(100.0 * (float(ru["wall_median_s"]) / float(rf["wall_median_s"]) - 1.0))
    pct("fgra.rmse.pen.min", min(fgra_rmse_pen), 1)
    pct("fgra.rmse.pen.max", max(fgra_rmse_pen), 1)
    pct("fgra.sql.gain.min", min(fgra_wall_vs_ull), 1)
    pct("fgra.sql.gain.max", max(fgra_wall_vs_ull), 1)

    # EXP-OPT-1b: Scalar x86-64-v2 SQL generalization (W5, W6, W9, W10)
    sc = read_tsv(out / "m4_scalar_sql.tsv")
    check(all(int(r["stage_mismatch_groups"]) == 0 for r in sc), "all scalar SQL rows have 0 mismatch groups")
    sc_w5_5b = one(sc, workload="W5_32T_16Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w5_ell = one(sc, workload="W5_32T_16Kg_64M", method="ELL(2,24,Slab+Sp)")
    sc_w6_5b = one(sc, workload="W6_8T_65Kg_64M", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w6_ell = one(sc, workload="W6_8T_65Kg_64M", method="ELL(2,24,Slab+Sp)")
    sc_w6_5bd = one(sc, workload="W6_8T_65Kg_64M", method="OS-BMW-5b(k24)")
    sc_w6_elld = one(sc, workload="W6_8T_65Kg_64M", method="ELL(2,24)")
    sc_w9_5b = one(sc, workload="W9_8T_ClickBench_32M", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w9_ell = one(sc, workload="W9_8T_ClickBench_32M", method="ELL(2,24,Slab+Sp)")
    sc_w10_5bsp = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="OS-BMW-5b(k24,Slab+Sp)")
    sc_w10_5bd = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="OS-BMW-5b(k24)")
    sc_w10_ellsp = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="ELL(2,24,Slab+Sp)")
    sc_w10_elld = one(sc, workload="W10_32T_2Kg_768M_DenseOOC", method="ELL(2,24)")
    num("sc.w5.5b.s", float(sc_w5_5b["wall_median_s"]), 2)
    num("sc.w5.ell.s", float(sc_w5_ell["wall_median_s"]), 2)
    num("sc.w6.5b.s", float(sc_w6_5b["wall_median_s"]), 2)
    num("sc.w6.ell.s", float(sc_w6_ell["wall_median_s"]), 2)
    num("sc.w6.5b.d.s", float(sc_w6_5bd["wall_median_s"]), 2)
    num("sc.w6.ell.d.s", float(sc_w6_elld["wall_median_s"]), 2)
    num("sc.w9.5b.s", float(sc_w9_5b["wall_median_s"]), 3)
    num("sc.w9.ell.s", float(sc_w9_ell["wall_median_s"]), 3)
    num("sc.w10.5b.sp.s", float(sc_w10_5bsp["wall_median_s"]), 2)
    num("sc.w10.5b.d.s", float(sc_w10_5bd["wall_median_s"]), 2)
    num("sc.w10.ell.sp.s", float(sc_w10_ellsp["wall_median_s"]), 2)
    num("sc.w10.ell.d.s", float(sc_w10_elld["wall_median_s"]), 2)
    integer("sc.w10.5b.d.mem", int(round(float(sc_w10_5bd["rss_peak_over_table_mb"]))))
    integer("sc.w10.ell.d.mem", int(round(float(sc_w10_elld["rss_peak_over_table_mb"]))))
    pct("sc.w9.mem.drop", 100.0 * (1.0 - float(sc_w9_5b["rss_peak_over_table_mb"]) / float(sc_w9_ell["rss_peak_over_table_mb"])), 1)
    pct("sc.w10.d.mem.drop", 100.0 * (1.0 - float(sc_w10_5bd["rss_peak_over_table_mb"]) / float(sc_w10_elld["rss_peak_over_table_mb"])), 1)
    times("sc.w6.d.ell.ratio", float(sc_w6_5bd["wall_median_s"]) / float(sc_w6_elld["wall_median_s"]), 2)
    times("sc.w9.ell.ratio", float(sc_w9_5b["wall_median_s"]) / float(sc_w9_ell["wall_median_s"]), 2)
    times("sc.w10.d.ell.ratio", float(sc_w10_5bd["wall_median_s"]) / float(sc_w10_elld["wall_median_s"]), 2)
    times("sc.w10.sp.ell.ratio", float(sc_w10_5bsp["wall_median_s"]) / float(sc_w10_ellsp["wall_median_s"]), 2)


def build(out: Path) -> None:
    monte_carlo(out)
    sizes(out)
    multishard(out)
    datafusion(out)
    micro(out)
    onset(out)
    sweep(out)
    ablation(out)
    real_streams(out)
    official_libs(out)
    supplementary_exps(out)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(MS.parent / "artifact" / "out"))
    ap.add_argument("--dest", default=str(MS / "tables" / "story_macros.tex"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    if not out.is_dir():
        die(f"result directory {out} not found")
    build(out)
    if FAILED:
        print("Statements made in the text that the result files do not support:", file=sys.stderr)
        for s in FAILED:
            print(f"  - {s}", file=sys.stderr)
        sys.exit(1)
    lines = [
        "% Auto-generated by gen/build_story_macros.py -- DO NOT EDIT BY HAND",
        r"\makeatletter",
        r"\newcommand{\Nm}[1]{\ifcsname osn@#1\endcsname\csname osn@#1\endcsname\else"
        r"\PackageError{osbmw}{Unknown number key: #1}{Add it to gen/build_story_macros.py}\fi}",
    ]
    for k in sorted(M):
        lines.append(rf"\expandafter\def\csname osn@{k}\endcsname{{{M[k]}}}")
    lines.append(r"\makeatother")
    dest = Path(args.dest)
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(M)} keys to {dest}")


if __name__ == "__main__":
    main()
