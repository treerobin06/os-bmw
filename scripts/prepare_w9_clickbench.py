#!/usr/bin/env python3
"""
Deterministic ClickBench W9 Dataset Downloader, Converter, and Exact Verifier.

Produces the binary file `w9_clickbench.bin` consumed by `s1_v2_datafusion`
(`--mode v5_hardened --w9-file ./data/w9_clickbench.bin`) from the official
partitioned ClickBench Parquet release (`hits_0.parquet` .. `hits_31.parquet`):

  Official URL pattern:
    https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{i}.parquet

Binary layout (little-endian, exact match to `s1_v2_datafusion/src/main.rs::load_external_table`):
  [u64 n_rows]
  [u64 n_groups]
  [u64 * n_groups true_distinct]
  [u32 * n_rows   gids]
  [u64 * n_rows   vals]

Exactness guarantees:
  1. Reads shards `hits_0.parquet .. hits_{S-1}.parquet` in strict numeric order
     and takes the exact deterministic prefix of `max_rows` (default 32,000,000)
     rows of `(RegionID, URLHash)`.
  2. Maps unique `RegionID` values deterministically (sorted ascending) to
     contiguous `u32` group IDs `0..n_groups-1` and preserves all 64 bits of
     `URLHash` as `u64` (two's-complement bitcast from Parquet `INT64`).
  3. Computes exact per-group `COUNT(DISTINCT URLHash)` via TWO independent
     algorithms (NumPy sorted-pair deduplication + independent exact hash-set
     pass across bounded group buckets) and asserts 0/n_groups mismatches.
  4. Emits SHA-256 checksums for every input Parquet shard and for the output
     `w9_clickbench.bin`, plus a JSON manifest for independent auditing.

Usage examples:
  # Download official shards hits_0..31.parquet, convert, and verify exact COUNT(DISTINCT):
  python3 scripts/prepare_w9_clickbench.py --download \\
      --parquet-dir ./data/clickbench_parquet \\
      --out ./data/w9_clickbench.bin

  # Convert from already-downloaded shards without network access:
  python3 scripts/prepare_w9_clickbench.py \\
      --parquet-dir ./data/clickbench_parquet \\
      --out ./data/w9_clickbench.bin

  # Independently verify an existing w9_clickbench.bin header vs raw (gid, val) arrays:
  python3 scripts/prepare_w9_clickbench.py --verify ./data/w9_clickbench.bin

  # Run built-in deterministic self-test (no network required):
  python3 scripts/prepare_w9_clickbench.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CLICKBENCH_BASE_URL = (
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned"
)
DEFAULT_SHARDS = 32
DEFAULT_MAX_ROWS = 32_000_000


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_shards(parquet_dir: Path, n_shards: int) -> List[Path]:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx in range(n_shards):
        fname = f"hits_{idx}.parquet"
        dst = parquet_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[download] exists: {dst} ({dst.stat().st_size:,} B)")
            paths.append(dst)
            continue
        url = f"{CLICKBENCH_BASE_URL}/{fname}"
        tmp = parquet_dir / f".{fname}.part"
        print(f"[download] fetching {url} -> {dst}")
        req = urllib.request.Request(url, headers={"User-Agent": "os-bmw-artifact/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dst)
        print(f"[download] saved {dst} ({dst.stat().st_size:,} B)")
        paths.append(dst)
    return paths


def exact_count_distinct_numpy(
    gids: np.ndarray, vals: np.ndarray, n_groups: int
) -> np.ndarray:
    """Method A: Exact vectorized structured-array sort & deduplicate."""
    assert gids.dtype == np.uint32 and vals.dtype == np.uint64
    assert len(gids) == len(vals)
    if len(gids) == 0:
        return np.zeros(n_groups, dtype=np.uint64)
    pairs = np.empty(len(gids), dtype=[("g", "<u4"), ("v", "<u8")])
    pairs["g"] = gids
    pairs["v"] = vals
    pairs.sort(order=("g", "v"))
    diff = np.empty(len(pairs), dtype=bool)
    diff[0] = True
    diff[1:] = (pairs["g"][1:] != pairs["g"][:-1]) | (pairs["v"][1:] != pairs["v"][:-1])
    uniq_g = pairs["g"][diff]
    counts = np.bincount(uniq_g, minlength=n_groups)
    return counts.astype(np.uint64)


def exact_count_distinct_independent(
    gids: np.ndarray, vals: np.ndarray, n_groups: int, bucket_stride: int = 1024
) -> np.ndarray:
    """Method B: Independent exact Python hash-set verification over bounded group windows."""
    assert gids.dtype == np.uint32 and vals.dtype == np.uint64
    out = np.zeros(n_groups, dtype=np.uint64)
    for g_lo in range(0, n_groups, bucket_stride):
        g_hi = min(n_groups, g_lo + bucket_stride)
        mask = (gids >= g_lo) & (gids < g_hi)
        sub_g = gids[mask].tolist()
        sub_v = vals[mask].tolist()
        sets: Dict[int, set] = {}
        for g, v in zip(sub_g, sub_v):
            s = sets.get(g)
            if s is None:
                sets[g] = {v}
            else:
                s.add(v)
        for g, s in sets.items():
            out[g] = len(s)
    return out


def write_w9_binary(
    out_path: Path, gids: np.ndarray, vals: np.ndarray, true_distinct: np.ndarray
) -> str:
    n_rows = int(len(gids))
    n_groups = int(len(true_distinct))
    assert gids.dtype == np.uint32 and vals.dtype == np.uint64
    assert true_distinct.dtype == np.uint64
    assert len(vals) == n_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(struct.pack("<QQ", n_rows, n_groups))
        f.write(np.ascontiguousarray(true_distinct, dtype="<u8").tobytes())
        f.write(np.ascontiguousarray(gids, dtype="<u4").tobytes())
        f.write(np.ascontiguousarray(vals, dtype="<u8").tobytes())
    os.replace(tmp_path, out_path)
    expected_len = 16 + n_groups * 8 + n_rows * 4 + n_rows * 8
    actual_len = out_path.stat().st_size
    if actual_len != expected_len:
        raise RuntimeError(f"Output byte length mismatch: {actual_len} != {expected_len}")
    return sha256_file(out_path)


def verify_w9_binary(bin_path: Path) -> Dict[str, object]:
    """Read an existing w9_clickbench.bin and independently verify every group's COUNT(DISTINCT)."""
    if not bin_path.exists():
        raise FileNotFoundError(f"W9 binary not found: {bin_path}")
    file_size = bin_path.stat().st_size
    if file_size < 16:
        raise ValueError(f"W9 binary too short ({file_size} B)")
    with bin_path.open("rb") as f:
        n_rows, n_groups = struct.unpack("<QQ", f.read(16))
        expected_size = 16 + n_groups * 8 + n_rows * 4 + n_rows * 8
        if file_size != expected_size:
            raise ValueError(
                f"W9 binary size mismatch: got {file_size} B, expected {expected_size} B "
                f"(n_rows={n_rows}, n_groups={n_groups})"
            )
        hdr_td = np.frombuffer(f.read(n_groups * 8), dtype="<u8").copy()
        gids = np.frombuffer(f.read(n_rows * 4), dtype="<u4").copy()
        vals = np.frombuffer(f.read(n_rows * 8), dtype="<u8").copy()

    if n_rows > 0 and int(gids.max()) >= n_groups:
        raise ValueError(f"Invalid group_id {int(gids.max())} >= n_groups {n_groups}")

    recomputed_np = exact_count_distinct_numpy(gids, vals, n_groups)
    mismatches_np = int(np.count_nonzero(recomputed_np != hdr_td))
    if mismatches_np != 0:
        raise AssertionError(
            f"Header true_distinct mismatch vs NumPy exact COUNT(DISTINCT): {mismatches_np}/{n_groups} groups"
        )

    recomputed_py = exact_count_distinct_independent(gids, vals, n_groups)
    mismatches_py = int(np.count_nonzero(recomputed_py != hdr_td))
    if mismatches_py != 0:
        raise AssertionError(
            f"Header true_distinct mismatch vs Python exact set COUNT(DISTINCT): {mismatches_py}/{n_groups} groups"
        )

    digest = sha256_file(bin_path)
    nz = hdr_td[hdr_td > 0]
    summary: Dict[str, object] = {
        "file": str(bin_path),
        "size_bytes": file_size,
        "sha256": digest,
        "n_rows": int(n_rows),
        "n_groups": int(n_groups),
        "distinct_min": int(nz.min()) if len(nz) else 0,
        "distinct_p50": float(np.percentile(nz, 50)) if len(nz) else 0.0,
        "distinct_p90": float(np.percentile(nz, 90)) if len(nz) else 0.0,
        "distinct_max": int(nz.max()) if len(nz) else 0,
        "verified_exact_groups": int(n_groups),
        "mismatches": 0,
    }
    print(
        f"[verify] PASS: {bin_path} | rows={n_rows:,} | groups={n_groups:,} | "
        f"distinct=[{summary['distinct_min']}..{summary['distinct_max']}], "
        f"p50={summary['distinct_p50']:.1f}, p90={summary['distinct_p90']:.1f} | "
        f"mismatches=0/{n_groups} | sha256={digest}"
    )
    return summary


def load_clickbench_parquet_shards(
    parquet_dir: Path, n_shards: int, max_rows: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: `pyarrow` is required to read ClickBench Parquet shards.\n"
            "Install it with:  pip install pyarrow\n"
            "(Only needed when converting hits_0..31.parquet -> w9_clickbench.bin.)"
        ) from exc

    region_chunks: List[np.ndarray] = []
    url_chunks: List[np.ndarray] = []
    shard_records: List[Dict[str, object]] = []
    collected = 0

    for idx in range(n_shards):
        if collected >= max_rows:
            break
        shard = parquet_dir / f"hits_{idx}.parquet"
        if not shard.exists():
            raise FileNotFoundError(
                f"Missing shard {shard}. Pass --download to fetch official ClickBench "
                f"shards from {CLICKBENCH_BASE_URL}/hits_{{0..{n_shards - 1}}}.parquet"
            )
        shard_sha = sha256_file(shard)
        table = pq.read_table(shard, columns=["RegionID", "URLHash"])
        reg = table.column("RegionID").to_numpy(zero_copy_only=False)
        url = table.column("URLHash").to_numpy(zero_copy_only=False)
        need = max_rows - collected
        if len(reg) > need:
            reg = reg[:need]
            url = url[:need]
        reg_i64 = np.asarray(reg, dtype=np.int64)
        url_u64 = np.asarray(url, dtype=np.int64).view(np.uint64)
        region_chunks.append(reg_i64)
        url_chunks.append(url_u64)
        collected += len(reg_i64)
        shard_records.append(
            {
                "shard": shard.name,
                "url": f"{CLICKBENCH_BASE_URL}/{shard.name}",
                "rows_used": int(len(reg_i64)),
                "size_bytes": shard.stat().st_size,
                "sha256": shard_sha,
            }
        )
        print(
            f"[read] {shard.name}: +{len(reg_i64):,} rows (total {collected:,}/{max_rows:,}) "
            f"sha256={shard_sha[:16]}..."
        )

    if collected != max_rows:
        raise RuntimeError(
            f"Collected only {collected:,} rows across {len(shard_records)} shards; "
            f"expected {max_rows:,} rows."
        )
    all_regions = np.concatenate(region_chunks)
    all_urls = np.concatenate(url_chunks)
    return all_regions, all_urls, shard_records


def convert_arrays_to_w9(
    raw_regions: np.ndarray,
    raw_urls_u64: np.ndarray,
    out_path: Path,
    manifest_path: Optional[Path] = None,
    shard_records: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    # Deterministic ascending mapping of unique RegionID -> contiguous u32 group_id
    uniq_regions, inv = np.unique(raw_regions, return_inverse=True)
    n_groups = int(len(uniq_regions))
    gids = inv.astype(np.uint32, copy=False)
    vals = np.ascontiguousarray(raw_urls_u64, dtype=np.uint64)

    print(
        f"[exact] computing exact per-group COUNT(DISTINCT URLHash) for "
        f"{len(gids):,} rows across {n_groups:,} RegionID groups..."
    )
    td_np = exact_count_distinct_numpy(gids, vals, n_groups)
    td_py = exact_count_distinct_independent(gids, vals, n_groups)
    if not np.array_equal(td_np, td_py):
        bad = int(np.count_nonzero(td_np != td_py))
        raise AssertionError(f"Independent COUNT(DISTINCT) disagreement on {bad} groups!")
    print(f"[exact] two independent exact COUNT(DISTINCT) passes matched on {n_groups:,}/{n_groups:,} groups.")

    out_sha = write_w9_binary(out_path, gids, vals, td_np)
    ver = verify_w9_binary(out_path)
    assert ver["sha256"] == out_sha

    manifest: Dict[str, object] = {
        "dataset": "ClickBench hits_compatible/athena_partitioned (RegionID, URLHash)",
        "source_base_url": CLICKBENCH_BASE_URL,
        "output_binary": str(out_path),
        "output_size_bytes": out_path.stat().st_size,
        "output_sha256": out_sha,
        "n_rows": int(len(gids)),
        "n_groups": n_groups,
        "distinct_min": ver["distinct_min"],
        "distinct_p50": ver["distinct_p50"],
        "distinct_p90": ver["distinct_p90"],
        "distinct_max": ver["distinct_max"],
        "independent_verification": "PASS (numpy_lexsort == python_hashset == binary_readback)",
        "input_shards": shard_records or [],
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[manifest] wrote {manifest_path}")
    return manifest


def run_self_test() -> None:
    """Deterministic round-trip self-test (no network required)."""
    rng = np.random.default_rng(20270202)
    n_rows = 50_000
    # Skewed RegionIDs (including negative/large values) and signed 64-bit URLHashes
    raw_regions = rng.choice(
        np.array([-7, 0, 42, 225, 10001, 2147483647], dtype=np.int64),
        size=n_rows,
        p=[0.02, 0.15, 0.35, 0.30, 0.13, 0.05],
    )
    raw_urls_i64 = rng.integers(-5000, 5000, size=n_rows, dtype=np.int64)
    raw_urls_u64 = raw_urls_i64.view(np.uint64)

    with tempfile.TemporaryDirectory(prefix="s1_w9_selftest_") as td:
        tmp = Path(td)
        out_bin = tmp / "w9_selftest.bin"
        manifest_path = tmp / "w9_selftest.manifest.json"

        # If pyarrow is available, exercise the full Parquet -> binary -> verify pipeline.
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            pq_dir = tmp / "parquet"
            pq_dir.mkdir()
            half = n_rows // 2
            for idx, (lo, hi) in enumerate([(0, half), (half, n_rows)]):
                tbl = pa.table(
                    {
                        "RegionID": pa.array(raw_regions[lo:hi], type=pa.int64()),
                        "URLHash": pa.array(raw_urls_i64[lo:hi], type=pa.int64()),
                    }
                )
                pq.write_table(tbl, pq_dir / f"hits_{idx}.parquet")
            regs, urls, shards = load_clickbench_parquet_shards(pq_dir, n_shards=2, max_rows=n_rows)
            manifest = convert_arrays_to_w9(regs, urls, out_bin, manifest_path, shards)
            assert len(manifest["input_shards"]) == 2
            mode = "full Parquet -> binary -> independent COUNT(DISTINCT) readback"
        except ImportError:
            manifest = convert_arrays_to_w9(raw_regions, raw_urls_u64, out_bin, manifest_path)
            mode = "in-memory arrays -> binary -> independent COUNT(DISTINCT) readback (pyarrow not installed)"

        assert manifest["n_rows"] == n_rows
        assert manifest["n_groups"] == 6
        print(f"[self-test] PASS ({mode}): sha256={manifest['output_sha256'][:16]}...")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download, deterministically convert, and verify ClickBench W9 binary for s1_v2_datafusion."
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download official ClickBench hits_0..{shards-1}.parquet shards before converting.",
    )
    ap.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("./data/clickbench_parquet"),
        help="Directory holding official hits_0..31.parquet shards (default: ./data/clickbench_parquet).",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=DEFAULT_SHARDS,
        help=f"Number of sequential shards hits_0..{{shards-1}}.parquet to scan (default: {DEFAULT_SHARDS}).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Exact deterministic prefix row count (default: {DEFAULT_MAX_ROWS:,}).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("./data/w9_clickbench.bin"),
        help="Output binary path consumed by s1_v2_datafusion --w9-file (default: ./data/w9_clickbench.bin).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON checksum manifest path (default: <out>.manifest.json).",
    )
    ap.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Independently verify an existing w9_clickbench.bin file without reconverting.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run a fast deterministic round-trip self-test without network access.",
    )
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.verify is not None:
        verify_w9_binary(args.verify)
        return

    if args.download:
        download_shards(args.parquet_dir, args.shards)

    regs, urls, shards = load_clickbench_parquet_shards(
        args.parquet_dir, n_shards=args.shards, max_rows=args.max_rows
    )
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    convert_arrays_to_w9(regs, urls, args.out, manifest_path=manifest_path, shard_records=shards)


if __name__ == "__main__":
    main()
