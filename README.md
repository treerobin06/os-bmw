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
│       ├── lib.rs         # 18 core unit tests (semilattice laws, PMF closure, MLE, Rank wire, 2x-reducibility)
│       ├── osbmw_inplace.rs
│       ├── osbmw_rank.rs
│       ├── osbmw_ref.rs
│       ├── baselines.rs
│       ├── hash.rs
│       ├── pmf.rs
│       └── bin/           # Accuracy, equal-error/HIP, 256B mul-shift, ablation, and systems harnesses
├── s1_v2_datafusion/      # Apache DataFusion 55.3 GROUP BY integration (exact-cell sparse+flat-slab)
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/main.rs        # Includes 2 exhaustive wire-framing & canonical-state equivalence tests
├── scripts/
│   ├── verify_proofs.py           # Exact exhaustive/numerical verification of Theorems 1-4
│   └── reconcile_pmf_and_rank.py  # Independent Python PMF, Fisher/FiSh, and 1,277 B rank codec audit
├── reproduce.sh           # One-command clean-room verification (21/21 Rust tests + Theorems 1-4)
└── LICENSE                # MIT License
```

## Requirements

- **Rust toolchain**: `rustc` / `cargo` 1.85+ (tested on `rustc 1.95.0` macOS arm64 and `rustc 1.100.0` Linux x86-64).
- **Python**: Python 3.9+ with `numpy` (`scripts/verify_proofs.py` requires only Python standard library; `scripts/reconcile_pmf_and_rank.py` uses `numpy`).
- **Hardware**: All 21/21 correctness tests and theorem proofs run portably on any x86-64 or ARM64 CPU. Full SIMD microbenchmarks (`s1_v2_bench_sys`, `s1_v2_256b_mulshift`, `s1_v2_datafusion`) use AVX-512BW/VBMI/VBMI2 when compiled with `RUSTFLAGS="-C target-cpu=native"` on AVX-512 hosts and fall back cleanly to portable scalar kernels elsewhere.

## Quick Verification (`reproduce.sh`)

```bash
chmod +x reproduce.sh
./reproduce.sh
```

On x86-64 hosts with AVX-512, enable native SIMD instructions:

```bash
RUSTFLAGS="-C target-cpu=native" ./reproduce.sh
```

Expected verification summary:
1. `s1_v2_rust --lib`: **18/18 passed** (exhaustive 512×512 merge associativity/commutativity/idempotence, floor-clamped isomorphism, exact 64-bit preimage PMF closure, dyadic 2×-reducibility, safeguarded Newton-Raphson MLE, 1,277 B combinatorial rank codec round-trip).
2. `s1_v2_rust --bin s1_v2_equal_error_hip`: **1/1 passed** (`OsBmwMartingale` exact open-cell tracking across anchor promotions up to $n=10^6$).
3. `s1_v2_datafusion`: **2/2 passed** (exhaustive `0x00..=0xFF` first-byte wire-framing test verifying mutually exclusive `0xE1`/`0xE5`/`0x53` tags, canonical dense byte equivalence across direct update vs. `convert_to_state -> merge_batch`, and malformed-frame rejection).
4. `scripts/verify_proofs.py`: **4/4 theorems passed** (`theorem_1`, `generalized_theorem_2`, `theorem_3`, `theorem_4`), writing `out/proofs_verification.json`.

## Running Full Benchmark Suites (Optional)

```bash
# M0/M1 reconciliation & algebraic counterexample audit
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_reconcile -- ../out)

# M2 equal-error + paired-seed Martingale/HIP sweep
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_equal_error_hip -- ../out)

# M3 systems microbenchmark (pin to a single core on Linux for reproducible cache latency)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_bench_sys -- ../out)

# M3 256B multiply-shift indexing microbenchmark
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_256b_mulshift -- ../out)

# M2 full accuracy sweep (optional SNAP graph directory via S1_DATA_DIR or 2nd CLI arg)
(cd s1_v2_rust && RUSTFLAGS="-C target-cpu=native" cargo run --release --bin s1_v2_accuracy -- ../out ./data 8)

# M4 Apache DataFusion 55.3 GROUP BY benchmark
(cd s1_v2_datafusion && RUSTFLAGS="-C target-cpu=native" cargo run --release -- --help)
```

## License

Released under the [MIT License](LICENSE).
