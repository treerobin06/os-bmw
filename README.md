# OS-BMW: Top-k Order-Statistic Anchoring for Space-Efficient, Semilattice-Exact Cardinality Sketches

Reproducibility artifact for **OS-BMW** (*PVLDB Vol. 20 / VLDB 2027 Research Track*).

## Repository Layout

```text
.
├── s1_v2_rust/            # Core OS-BMW library (zero external crate deps) & benchmark binaries
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── scripts/env_info.sh
│   └── src/
│       ├── lib.rs         # 29 core unit tests (semilattice laws, exact-semilattice CPC two-shard merge,
│       │                  #   AVLL exact-mod preimage extreme-state check at 2^63-2^64 [1,500 tr.,
│       │                  #   prefilled to avoid historical clamping], lossless HBS, 128b ref-vs-inplace,
│       │                  #   PMF, MLE, Rank wire codec, 2x-reducibility, release k>=2 & flat-slice bounds,
│       │                  #   AVX-512 vs. forced-scalar byte parity, filtered ELL/ULL equivalence,
│       │                  #   ULL-FGRA unbiasedness & large-range bounds via test_ull_fgra_estimator_unbiased_and_bounds)
│       ├── osbmw_inplace.rs
│       ├── osbmw_rank.rs
│       ├── osbmw_ref.rs
│       ├── baselines.rs   # ELL(2,24), ELL(2,20), ULL (MLE + hash4j FGRA), HLL++, AVLL(exact-mod preimage ML), HLL4, HLLL, CPC, HBS
│       ├── hash.rs
│       ├── pmf.rs
│       └── bin/           # Accuracy, equal-error/HIP, 256B mul-shift, ablation (1K/4K), real-stream CI,
│                          #   SIMD parity, systems, and supplementary microbenchmarks (s1_v2_bench_supp.rs:
│                          #   --mode ell-filter | modcard | non-avx512-merge | ull-fgra-micro)
├── s1_v2_datafusion/      # Apache DataFusion 55.3 GROUP BY integration (symmetric exact-cell sparse+flat-slab)
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/main.rs        # 4 tests (0x00..=0xFF symmetric Sl+Sp wire framing, malformed frames,
│                          #   W7/W8 M_(24)>8 truncation, sparse->dense promotion & 95% CI summary)
├── out/                   # Complete raw TSV/JSON measurements backing all tables & figures (42 files total)
├── scripts/
│   ├── verify_proofs.py                    # Exact exhaustive/numerical verification (Theorems 1-3 + §8 wire-ranking bound)
│   ├── reconcile_pmf_and_rank.py           # Independent Python PMF, Fisher/FiSh, and 1,277 B rank codec audit
│   ├── prepare_w9_clickbench.py            # Deterministic ClickBench hits_0..31.parquet -> w9_clickbench.bin + SHA256
│   ├── audit_closest_cells_and_wrappers.py # Pooled sweep z-scores & C++/Java wrapper parity audit
│   ├── dump_protocol_and_constants.py      # Extracts & verifies (--verify) out/m5_protocol_and_constants.json
│   └── build_tables_and_figs.py            # Rebuilds 9 LaTeX files (8 artifact tables + metrics_macros.tex) & 4 PDF figures
├── reproduce.sh           # One-command offline clean-room verification (34/34 Rust tests [29/29 lib + 1/1 martingale + 4/4 DataFusion] + 4 proof checks + m5 verify + tables/figs)
└── LICENSE                # MIT License
```

## Requirements

- **Rust toolchain**: `rustc` / `cargo` **1.94+** (enforced via `rust-version = "1.94"` in both `Cargo.toml` files; required by `Cargo.lock` lockfile v4 and DataFusion 55.3 dependencies; tested on `rustc 1.95.0` macOS arm64 and `rustc 1.100.0` Linux x86-64).
- **Python**: Python 3.9+ with **`numpy`** and **`matplotlib`** (`scripts/verify_proofs.py`, `scripts/reconcile_pmf_and_rank.py`, and `scripts/prepare_w9_clickbench.py --verify/--self-test` require `numpy`; `scripts/build_tables_and_figs.py` requires `numpy` and `matplotlib`; `pyarrow` is only needed when converting raw ClickBench `.parquet` shards via `scripts/prepare_w9_clickbench.py`).
- **Hardware**: All 34/34 Rust tests (29/29 lib + 1/1 martingale + 4/4 DataFusion), theorem proofs, protocol-constant verification, and table/figure generation run portably on any x86-64 or ARM64 CPU. Full SIMD microbenchmarks (`s1_v2_bench_sys`, `s1_v2_256b_mulshift`, `s1_v2_bench_supp`, `s1_v2_datafusion`) use AVX-512BW/VBMI/VBMI2 when compiled with `RUSTFLAGS="-C target-cpu=native"` on AVX-512 hosts and fall back cleanly to portable scalar kernels elsewhere.

## Quick Verification (`reproduce.sh`)

`./reproduce.sh` runs **100% offline** with zero external datasets (`w9_clickbench.bin` is not required; `prepare_w9_clickbench.py --self-test` tests the full Parquet-to-binary readback path when `pyarrow` is installed and otherwise runs an in-memory array-to-binary readback self-test):

```bash
chmod +x reproduce.sh
./reproduce.sh
```

On x86-64 hosts with AVX-512, enable native SIMD instructions:

```bash
RUSTFLAGS="-C target-cpu=native" ./reproduce.sh
```

Expected verification summary (**34/34 Rust tests** = `29/29` lib + `1/1` martingale + `4/4` DataFusion):
1. `s1_v2_rust --lib`: **29/29 passed** (exhaustive 512×512 ordered-pair commutativity/idempotence/merge checks + independent random-triple associativity checks, floor-clamped isomorphism, exact-semilattice local `CPC` two-shard `lgK=8` and 8-shard merge regression, `AVLL` `h % m` joint preimages, uniform `[2^20, 2^21)` 5,632-bucket coverage, extreme-state `AVLL` estimator check using exact modulo preimage counts $C(r,k)$ under an unclamped Poisson likelihood [approximate after historical old-anchor clamping; 1,500 trials prefilled at saturated lower levels to avoid historical clamping, bias within SE, $|\mathrm{bias}| \le 0.10\%$ at $2^{63}$ and $2^{64}$; main direct-stream sweeps stop at $n \le 10^6$], lossless Karppa 2026 `HBS` register/merge/wire equivalence incl. empty `p=6` round-trip, 128-bit `osbmw_ref` vs. `osbmw_inplace` bit-identical equivalence, flat-slice `ELL`/`ULL` equivalence, exact 64-bit preimage PMF closure, dyadic 2×-reducibility, safeguarded Newton-Raphson MLE, 1,277 B combinatorial rank codec round-trip, single-64-bit `max_cap` construction/downsize/MLE consistency, `256B` `MulShift384`/`Div3`/`Pow2` wire-mode isolation + deferred interop, `Chunk` 64-bit/128-bit wire-mode isolation, `test_r22_k1_rejected_and_flat_slice_bounds_checked`, `test_direct_simd_vs_scalar_byte_parity`, `test_ell24_and_ull_filtered_exact_equivalence`, and `test_ull_fgra_estimator_unbiased_and_bounds` [verifying our Rust port of Dynatrace hash4j 0.22.0 `OptimalFGRAEstimator` on empty, moderate, and large-range register states]).
2. `s1_v2_rust --bin s1_v2_equal_error_hip`: **1/1 passed** (`OsBmwMartingale` exact open-cell tracking across anchor promotions up to $n=10^6$).
3. `s1_v2_datafusion`: **4/4 passed** (exhaustive `0x00..=0xFF` first-byte wire-framing test across all 5 symmetric `Slab+Sp` engines including `OS-BMW`, `ELL(2,24)`, and `ULL`, verifying `5B->9B` sparse append vs. bit-identical canonical dense projection and malformed-frame rejection, high-cardinality `W7`/`W8` $M_{(24)}>8$ truncation and estimate divergence vs. `ULL`, plus sparse→dense promotion counter and bootstrap 95% CI verification).
4. `scripts/verify_proofs.py`, `scripts/prepare_w9_clickbench.py --self-test`, & `scripts/dump_protocol_and_constants.py --verify`: **4/4 proof checks passed** (`theorem_1`, `generalized_theorem_2`, `theorem_3`, `theorem_4`, writing `reproduced/proofs_verification.json`), deterministic ClickBench W9 binary converter/verifier self-test, and source-to-JSON consistency verification of `out/m5_protocol_and_constants.json`.
5. `scripts/build_tables_and_figs.py`: Regenerates **9 LaTeX files** (`8` artifact tables + `metrics_macros.tex` in `reproduced/tables/*.tex`) and **4 PDF figures** (`reproduced/figures/*.pdf`) directly from `out/` (`42` raw TSV/JSON files).

### Standalone `verify_proofs.py` & `dump_protocol_and_constants.py` Usage

- `reproduce.sh` invokes `python3 scripts/verify_proofs.py reproduced/proofs_verification.json` so the frozen `out/proofs_verification.json` stays untouched.
- **Standalone execution**: Running `python3 scripts/verify_proofs.py` with **no arguments** overwrites `out/proofs_verification.json`; pass a destination path (or set `S1_PROOFS_OUT=/path/to/out.json`) for clean-room standalone runs. It requires only Python 3.9+, `numpy`, and the sibling `scripts/reconcile_pmf_and_rank.py` (no external data, no network, and no separate `PROOFS.md` file).
- **Mapping to manuscript sections**: `theorem_1` → Theorem 1 (§3.2), `generalized_theorem_2` → Theorem 2 (§3.3), `theorem_3` → Theorem 3 (§4.1 & §4.3 discrete preimages), `theorem_4` → `OS-BMW-Rank` archival 1,277 B wire-ranking bound (§8).
- **Protocol & Design Constants Audit (`scripts/dump_protocol_and_constants.py`)**: Deterministically parses live source files (`pmf.rs`, `baselines.rs`, `lib.rs`, `s1_v2_equal_error_hip.rs`, `s1_v2_accuracy.rs`, `s1_v2_reconcile.rs`, `s1_v2_datafusion/src/main.rs`) and baseline metadata (`out/env_info.json`, `out/official_hash4j_sys.tsv`, `out/official_datasketches_sys.tsv`) to emit or verify `out/m5_protocol_and_constants.json`:

```bash
# Verify frozen out/m5_protocol_and_constants.json matches live artifact sources (also run by reproduce.sh)
python3 scripts/dump_protocol_and_constants.py --verify

# Independently re-compute pooled closest-cell z-scores to a scratch JSON (pass --official-bench-dir / --official-hash4j-dir when external C++/Java wrappers are present)
python3 scripts/audit_closest_cells_and_wrappers.py --dest reproduced/m2_closest_cells_and_wrappers.json
```

### Raw Measurements in `out/` (42 Files Total)

All 42 archived raw TSV/JSON files in `out/` back the manuscript's tables, figures, and supplementary ablations. In addition to the core M0–M4 sweeps, proof/reconciliation logs, official DataSketches/hash4j baselines, and earlier microbenchmarks/SQL diagnostics (37 files), `out/` includes 5 supplementary measurement and metadata files:
- `out/m3_avx2_merge.tsv`: Non-AVX-512 x86-64 AVX2/portable-scalar merge latency (`s1_v2_bench_supp --mode non-avx512-merge`).
- `out/m3_ull_fgra_micro.tsv`: Single-sketch $n=10^7$ estimation latency comparing `OS-BMW-5b(k24)` MLE, `ELL(2,24)` MLE, `ULL` MLE, and `ULL` FGRA (`s1_v2_bench_supp --mode ull-fgra-micro`).
- `out/m4_scalar_sql.tsv`: Portable non-AVX-512 scalar-build Apache DataFusion 55.3 GROUP BY ablation on `W5`, `W6`, `W9`, and `W10` (`s1_v2_datafusion` built without AVX-512 flags using `--build-tag scalar`).
- `out/m4_ull_fgra_ablation.tsv`: End-to-end Apache DataFusion 55.3 `ULL` MLE vs. `ULL-FGRA` (`ull_fgra_v3_cd`) vs. `OS-BMW-5b(k24,Slab+Sp)` and `ELL(2,24,Slab+Sp)` ablation across `W1`, `W3`, `W5`, `W6`, and `W9`.
- `out/m5_protocol_and_constants.json`: Deterministic snapshot of solver tolerances, wire tags, sparse chunk geometry, bootstrap settings, hardware/toolchain versions, and test counts (`34/34`), verified by `scripts/dump_protocol_and_constants.py --verify`.

## Preparing Real ClickBench Workload `W9` (`w9_clickbench.bin`)

`W9` evaluates `GROUP BY RegionID, COUNT(DISTINCT URLHash)` over the exact first 32,000,000 rows of the official partitioned ClickBench Parquet shards (`https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet` .. `hits_31.parquet`, 6,916 `RegionID` groups). Verified statistical summary of our `w9_clickbench.bin` input (not archived in this repository due to ~384 MB binary size; regenerate deterministically from the official Parquet shards below to emit local SHA-256 checksums):
- **Rows / Groups**: `32,000,000` rows across `6,916` unique `RegionID` groups (`~371.2 MiB` Arrow MemTable footprint over baseline RSS; `~380.7 MiB` `VmRSS_post_table`).
- **Per-group exact `COUNT(DISTINCT URLHash)`**: min `1`, median (`p50`) `10`, `p90` `626`, max `1,336,731`. (The 2-shard `--self-test` Parquet-to-binary regression produces deterministic output SHA-256 `2dd9a19341410a20c6e38e490a2c09fb24b03cc0bf2eb93fe75b6d72fb7cf225` with `0/6` mismatches; the full `~384 MB` `w9_clickbench.bin` is generated on demand and was not retained on the remote benchmark server after `W9` completed.)

One command downloads the official shards, deterministically extracts `(RegionID, URLHash)`, independently cross-checks exact per-group `COUNT(DISTINCT)` via two algorithms, writes `./data/w9_clickbench.bin`, and emits SHA-256 checksums (`./data/w9_clickbench.manifest.json`):

```bash
# Requires pyarrow (`pip install pyarrow`) + numpy
python3 scripts/prepare_w9_clickbench.py --download \
  --parquet-dir ./data/clickbench_parquet \
  --out ./data/w9_clickbench.bin

# Independently re-verify an existing w9_clickbench.bin header against its raw (gid, val) arrays:
python3 scripts/prepare_w9_clickbench.py --verify ./data/w9_clickbench.bin
```

**Missing-data behavior**:
- `prepare_w9_clickbench.py` without `--download` raises `FileNotFoundError` naming the missing `hits_{i}.parquet` shard if `--parquet-dir` is empty/incomplete, and exits with a `pip install pyarrow` hint if `pyarrow` is absent.
- `s1_v2_datafusion --mode v5_hardened` (and `--mode w9`) requires `w9_clickbench.bin` (via `--w9-file <path>` or `S1_W9_BIN`, default `./data/w9_clickbench.bin`) and **panics immediately** (`failed to read external W9 ...`) if missing rather than skipping `W9`. Without external ClickBench data, run `--mode v4_symmetric` for all synthetic workloads `W1`–`W8` (plus `--mode w10` / `--mode w5_64p` for supplementary out-of-L3 / 64-partition runs).
- `s1_v2_accuracy` checks the data directory for three optional real-stream files (`web-Google.txt.gz`, `cit-Patents.txt.gz`, `ecoli_k12.fna.gz`). **If any file is absent, `s1_v2_accuracy` silently skips those streams** and writes only the remaining synthetic streams to `m2_real_streams.tsv`; point the first CLI argument at a scratch directory (not `../out`) to avoid overwriting the archived 8-stream `out/m2_real_streams.tsv`.

## Running Full Benchmark Suites (Optional)

```bash
# M0/M1 reconciliation & algebraic counterexample audit
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_reconcile -- ../out)

# M2 equal-error + paired-seed Martingale/HIP sweep
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_equal_error_hip -- ../out)

# M3 systems microbenchmark (4K-LLC + 64K-DRAM + HBS; pin to a single core on Linux)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_bench_sys -- ../out)

# M3 256B multiply-shift indexing microbenchmark
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_256b_mulshift -- ../out)

# M3 supplementary microbenchmarks (s1_v2_bench_supp.rs: ell-filter | modcard | non-avx512-merge | ull-fgra-micro)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_bench_supp -- --mode ell-filter --out ../out)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_bench_supp -- --mode modcard --out ../out)
(cd s1_v2_rust && cargo run --release --bin s1_v2_bench_supp -- --mode non-avx512-merge --tag portable_x86 --out ../out)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_bench_supp -- --mode ull-fgra-micro --out ../out)

# M2 full accuracy sweep (optional SNAP graph directory via S1_DATA_DIR or 2nd CLI arg; defaults to ./data)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_accuracy -- ../out ./data 8)

# M4 Apache DataFusion 55.3 GROUP BY hardened 9-workload benchmark (W1-W9 x 12 engines, 11 repeats, isolated)
(cd s1_v2_datafusion && RUSTFLAGS="-C target-cpu=native" cargo run --release -- --mode v5_hardened --isolate --runs 11 --w9-file ../data/w9_clickbench.bin --out ../out)

# M4 supplementary SQL ablations (portable scalar build & ULL-FGRA end-to-end estimator comparison)
(cd s1_v2_datafusion && cargo run --release -- --mode w5 --isolate --runs 11 --build-tag scalar --methods osbmw5b_v3_cd,osbmw5b_cd,osbmw256_v3_cd,ell24_v3_cd --supp-tsv m4_scalar_sql.tsv --out ../out)
(cd s1_v2_datafusion && RUSTFLAGS="-C target-cpu=native" cargo run --release -- --mode w1 --isolate --runs 11 --methods osbmw5b_v3_cd,ull_v3_cd,ull_fgra_v3_cd,ell24_v3_cd --supp-tsv m4_ull_fgra_ablation.tsv --out ../out)
```

## License

Released under the [MIT License](LICENSE).
