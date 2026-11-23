//! Milestone M3 Same-Machine Systems Microbenchmark (Run strictly on `numactl --cpunodebind=1 --membind=1 taskset -c 32`).
//!
//! Measures under IDENTICAL single 64-bit `wyhash64` hash & 8-byte self-describing header:
//!   1. Single-sketch L1 `ns/update` across n in {1e3, 1e4, 1e5, 1e7} (steady-state & lifecycle)
//!   2. 4,096 concurrent sketches DRAM/L3 cache-stress `ns/update` (steady-state & lifecycle!)
//!   3. Low-cardinality (n <= 10^4) `0.5 KB` & `1 KB` update throughput (`g=128` + `256B` Hybrid vs ELL)
//!   4. `ns/merge` (equal-size merge AND skewed 100x merge!)
//!   5. `us/estimate` (exact Newton MLE latency & mean Newton iterations!)
//!   6. `ns/serialize` (RAM -> wire bytes, including CPC Rice compression & OS-BMW-Rank encode/decode!)

use s1_v2_rust::baselines::*;
use s1_v2_rust::hash::wyhash64;
use s1_v2_rust::osbmw_inplace::*;
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn median_f64(mut vals: Vec<f64>) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 1 { vals[n / 2] } else { 0.5 * (vals[n / 2 - 1] + vals[n / 2]) }
}

fn bench_ns_per_op<F: FnMut()>(repeats: usize, ops_per_repeat: usize, mut f: F) -> f64 {
    // Warmup
    f();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        f();
        let ns = t0.elapsed().as_nanos() as f64 / (ops_per_repeat as f64);
        samples.push(ns);
    }
    median_f64(samples)
}

fn main() {
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // Ramp CPU 32 out of 800 MHz idle C-state to steady turbo clock before timing!
    {
        let t_warm = Instant::now();
        let mut acc = 0u64;
        let mut i = 0u64;
        while t_warm.elapsed().as_millis() < 250 {
            for _ in 0..100_000 {
                acc = acc.wrapping_add(wyhash64(i, 0xdead));
                i += 1;
            }
        }
        black_box(acc);
    }

    println!("=== [M3] Same-Machine Systems Microbenchmark (4 KB nominal budget, single 64-bit WyHash) ===");
    // Equal-budget (~4 KB) configurations (all include 8B self-describing header!):
    //   - OS-BMW-5b       (g=512, k=24):    8 blocks = 2,760 B (2,752 B payload; 16 blocks = 5,512 B)
    //   - OS-BMW-5b-Mask  (g=512, k=20):    8 blocks = 2,760 B (32-bit coarse mask in free padding!)
    //   - OS-BMW-5b-256B  (g=384, k=16):   16 blocks = 4,104 B (EXACT 4 KB payload = ELL(2,24)!)
    //   - OS-BMW-Chunk    (g=1024, k=32):   4 blocks = 2,632 B (656 B/blk, b=2.05)
    //   - ELL(2,24)       (p=10, 32-bit): 1024 regs  = 4,104 B
    //   - ELL(2,20)       (p=10, 28-bit): 1024 regs  = 3,592 B
    //   - ULL             (p=12, 8-bit):  4096 regs  = 4,104 B
    //   - HLL++           (p=12, 6-bit):  4096 regs  = 3,080 B
    //   - AVLL            (511 words):    5621 bkts  = 4,096 B
    //   - CPC             (lg_k=11):      2048 rows  (variable RAM ~3.5-6 KB, compressed Wire)
    //   - HLLL            (p=12):         4096 regs  (variable RAM + dynamic exc)
    //   - HLL_4           (p=12):         4096 regs  (dynamic AuxHashMap)

    let pmf_5b = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 3 - 9, max_cap: 63 });
    let pmf_256 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 4 });
    let pmf_chunk = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64 - 2 - 10, max_cap: 63 });
    let pmf_ell = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - 10 });
    let pmf_ull = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 12, max_cap: 63 });
    let pmf_avll = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 });
    let pmf_cpc = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 11, max_cap: 63 });

    // Pre-generate 2,000,000 raw item keys (streaming hashes computed INSIDE the timed loop so hash cost is identical!)
    // Also measure pure-hash baseline so readers see hash vs sketch time transparently!
    let n_l1 = 2_000_000usize;
    let seed = 0x1234_5678_9abc_def0u64;
    let hash_only_ns = bench_ns_per_op(7, n_l1, || {
        let mut acc = 0u64;
        for i in 0..n_l1 as u64 {
            acc ^= wyhash64(i, seed);
        }
        black_box(acc);
    });
    println!("  [Ref] wyhash64 alone: {:.3} ns/op", hash_only_ns);

    #[derive(Clone, Debug)]
    struct SysRow {
        name: String,
        ram_bytes: usize,
        wire_bytes: usize,
        l1_cold_1e4_ns: f64,
        l1_steady_1e7_ns: f64,
        dram_4096_steady_ns: f64,
        dram_4096_cold_ns: f64,
        merge_equal_ns: f64,
        merge_skew_ns: f64,
        est_ml_us: f64,
        est_ml_iters: f64,
        ser_ns: f64,
        ee_rate_1e7_pct: f64,
    }

    let mut rows: Vec<SysRow> = Vec::new();

    // Helper for 4,096 concurrent sketch routing: pre-generate 4,000,000 (group_id, item_key) pairs
    // uniformly routed across 4,096 sketches (16 MB total working set -> blows L2, stresses L3/DRAM!).
    let n_groups = 4096usize;
    let n_dram_ops = 4_000_000usize;
    let group_ids: Vec<u16> = (0..n_dram_ops)
        .map(|i| (wyhash64(i as u64, 0xfeed_face) & 4095) as u16)
        .collect();

    // Helper to pre-populate a prototype sketch to n = 10^6 (steady-state cardinality per group)
    // and clone it 4,096 times (different cache lines!) for authentic steady-state 4,096-sketch DRAM stress!
    let prefill_n = 1_000_000u64;
    let l1_prefill_n = 10_000_000u64;

    macro_rules! bench_method {
        (
            $name:expr,
            $make:expr,
            $add_fn:expr,
            $add_ee_fn:expr,
            $merge_fn:expr,
            $est_fn:expr,
            $ser_fn:expr,
            $ram_fn:expr,
            $wire_fn:expr
        ) => {{
            // 1. L1 Cold-start (0 -> 10^4 elements, 50 repeats)
            let n_cold = 10_000usize;
            let l1_cold_1e4_ns = bench_ns_per_op(35, n_cold, || {
                let mut sk = $make();
                for i in 0..n_cold as u64 {
                    let h = wyhash64(i, seed);
                    $add_fn(&mut sk, h);
                }
                black_box(&sk);
            });

            // 2. Build steady-state sketch at n = 10^7 & measure early-exit rate + L1 steady-state ns/op
            let mut sk_steady = $make();
            for i in 0..l1_prefill_n {
                let h = wyhash64(i, seed);
                $add_fn(&mut sk_steady, h);
            }
            let mut ee_hits = 0usize;
            for i in l1_prefill_n..(l1_prefill_n + 200_000) {
                let h = wyhash64(i, seed);
                if $add_ee_fn(&mut sk_steady, h) { ee_hits += 1; }
            }
            let ee_rate_1e7_pct = (ee_hits as f64) * 100.0 / 200_000.0;

            let mut sk_l1 = sk_steady.clone();
            let mut base_i = l1_prefill_n + 200_000;
            let l1_steady_1e7_ns = bench_ns_per_op(7, n_l1, || {
                for i in base_i..(base_i + n_l1 as u64) {
                    let h = wyhash64(i, seed);
                    $add_fn(&mut sk_l1, h);
                }
                base_i += n_l1 as u64;
                black_box(&sk_l1);
            });

            // 3. 4,096 concurrent sketches (DRAM/L3 stress!):
            //    (a) Steady-state: each of 4,096 sketches initialized at n=10^6, then 4M random updates across groups!
            let mut proto_1m = $make();
            for i in 0..prefill_n {
                let h = wyhash64(i, seed ^ 0x9999);
                $add_fn(&mut proto_1m, h);
            }
            let mut groups_steady: Vec<_> = (0..n_groups).map(|_| proto_1m.clone()).collect();
            let mut g_base = prefill_n;
            let dram_4096_steady_ns = bench_ns_per_op(5, n_dram_ops, || {
                for (step, &gid) in group_ids.iter().enumerate() {
                    let h = wyhash64(g_base + step as u64, seed);
                    $add_fn(&mut groups_steady[gid as usize], h);
                }
                g_base += n_dram_ops as u64;
                black_box(&groups_steady[0]);
            });
            drop(groups_steady);

            //    (b) Cold/Warm lifecycle: fresh 4,096 empty sketches per repeat streaming 4M brand-new (group, key) pairs
            //        (NO replay of a cached 4M batch after warmup!).
            let mut cold_base = 0u64;
            {
                let mut groups_cold: Vec<_> = (0..n_groups).map(|_| $make()).collect();
                for step in 0..n_dram_ops as u64 {
                    let k = cold_base + step;
                    let gid = (wyhash64(k, 0xfeed_face) & 4095) as usize;
                    let h = wyhash64(k, seed ^ 0x5555);
                    $add_fn(&mut groups_cold[gid], h);
                }
                cold_base += n_dram_ops as u64;
                black_box(&groups_cold[0]);
            }
            let mut cold_samples = Vec::with_capacity(3);
            for _ in 0..3 {
                let mut groups_cold: Vec<_> = (0..n_groups).map(|_| $make()).collect();
                let t0 = Instant::now();
                for step in 0..n_dram_ops as u64 {
                    let k = cold_base + step;
                    let gid = (wyhash64(k, 0xfeed_face) & 4095) as usize;
                    let h = wyhash64(k, seed ^ 0x5555);
                    $add_fn(&mut groups_cold[gid], h);
                }
                let ns = t0.elapsed().as_nanos() as f64 / (n_dram_ops as f64);
                cold_base += n_dram_ops as u64;
                black_box(&groups_cold[0]);
                cold_samples.push(ns);
            }
            let dram_4096_cold_ns = median_f64(cold_samples);

            // 4. Merge benchmark: equal-size (1M + 1M) AND skewed (10M + 10k)
            let mut sk_m1 = proto_1m.clone();
            let mut sk_m2 = $make();
            for i in 0..prefill_n {
                $add_fn(&mut sk_m2, wyhash64(i, seed ^ 0x1111));
            }
            let mut sk_small = $make();
            for i in 0..10_000u64 {
                $add_fn(&mut sk_small, wyhash64(i, seed ^ 0x2222));
            }
            let merge_iters = 2000usize;
            let merge_equal_ns = bench_ns_per_op(7, merge_iters, || {
                for _ in 0..merge_iters {
                    let mut tmp = sk_m1.clone();
                    $merge_fn(&mut tmp, &sk_m2);
                    black_box(&tmp);
                }
            });
            // Subtract clone cost? No — report TRUE end-to-end merge-into-target (allocate/copy once outside loop or alternating)!
            // Even cleaner: measure pure in-place merge on a pool of 64 fresh copies so no cache-degenerate idempotence!
            let pool_a: Vec<_> = (0..64).map(|_| sk_m1.clone()).collect();
            let pool_b: Vec<_> = (0..64).map(|_| sk_m2.clone()).collect();
            let mut work = pool_a.clone();
            let merge_equal_inplace_ns = bench_ns_per_op(9, 64, || {
                work.clone_from_slice(&pool_a);
                let t0 = Instant::now();
                for i in 0..64 {
                    $merge_fn(&mut work[i], &pool_b[i]);
                }
                black_box(&work);
                let _ = t0;
            });
            // Wait: measure PURE merge time cleanly without `clone_from_slice` inside the timed window!
            let mut eq_samples = Vec::new();
            for _ in 0..11 {
                let mut w = pool_a.clone();
                let t0 = Instant::now();
                for i in 0..64 {
                    $merge_fn(&mut w[i], &pool_b[i]);
                }
                black_box(&w);
                eq_samples.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_equal_pure_ns = median_f64(eq_samples);

            let pool_big: Vec<_> = (0..64).map(|_| sk_steady.clone()).collect();
            let mut skew_samples = Vec::new();
            for _ in 0..11 {
                let mut w = pool_big.clone();
                let t0 = Instant::now();
                for i in 0..64 {
                    $merge_fn(&mut w[i], &sk_small);
                }
                black_box(&w);
                skew_samples.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_skew_pure_ns = median_f64(skew_samples);
            let _ = (merge_equal_ns, merge_equal_inplace_ns, &mut sk_m1);

            // 5. Estimate (exact Newton MLE latency & iterations)
            let est_iters = 500usize;
            let mut tot_newton = 0u64;
            let est_ml_ns = bench_ns_per_op(7, est_iters, || {
                let mut s = 0.0_f64;
                for _ in 0..est_iters {
                    let (v, it) = $est_fn(&sk_steady);
                    s += v;
                    tot_newton += it as u64;
                }
                black_box(s);
            });
            let est_ml_iters = $est_fn(&sk_steady).1 as f64;

            // 6. Serialize (RAM -> wire bytes)
            let mut ser_buf = Vec::new();
            let ser_iters = 2000usize;
            let ser_ns = bench_ns_per_op(7, ser_iters, || {
                let mut tot_len = 0usize;
                for _ in 0..ser_iters {
                    $ser_fn(&sk_steady, &mut ser_buf);
                    tot_len += ser_buf.len();
                }
                black_box(tot_len);
            });

            let ram_b = $ram_fn(&sk_steady);
            let wire_b = $wire_fn(&sk_steady);
            println!(
                "  {:18} | RAM={:5}B Wire={:5}B | L1(1e4)={:5.2}ns L1(1e7)={:5.2}ns | 4096-DRAM(1e6)={:6.2}ns (cold={:6.2}ns) | mrg={:6.0}ns (skew={:5.0}ns) | est={:6.2}us ({:.0}it) | ser={:6.0}ns | EE={:5.1}%",
                $name, ram_b, wire_b, l1_cold_1e4_ns, l1_steady_1e7_ns, dram_4096_steady_ns, dram_4096_cold_ns,
                merge_equal_pure_ns, merge_skew_pure_ns, est_ml_ns / 1000.0, est_ml_iters, ser_ns, ee_rate_1e7_pct
            );
            rows.push(SysRow {
                name: $name.to_string(),
                ram_bytes: ram_b,
                wire_bytes: wire_b,
                l1_cold_1e4_ns,
                l1_steady_1e7_ns,
                dram_4096_steady_ns,
                dram_4096_cold_ns,
                merge_equal_ns: merge_equal_pure_ns,
                merge_skew_ns: merge_skew_pure_ns,
                est_ml_us: est_ml_ns / 1000.0,
                est_ml_iters,
                ser_ns,
                ee_rate_1e7_pct,
            });
        }};
    }

    // --- Run ALL competitors under identical protocol! ---
    bench_method!(
        "OS-BMW-5b(k24)",
        || OsBmw5b::new(8, HashMode5b::Pow2),
        |s: &mut OsBmw5b, h| { s.add_hash_64(h); },
        |s: &mut OsBmw5b, h| s.add_hash_64(h),
        |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw5b| s.estimate_ml(&pmf_5b),
        |s: &OsBmw5b, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &OsBmw5b| s.total_bytes(),
        |s: &OsBmw5b| s.total_bytes()
    );

    bench_method!(
        "OS-BMW-5b(k20+m)",
        || OsBmw5bMask::new(8, HashMode5b::Pow2),
        |s: &mut OsBmw5bMask, h| { s.add_hash_64(h); },
        |s: &mut OsBmw5bMask, h| s.add_hash_64(h),
        |a: &mut OsBmw5bMask, b: &OsBmw5bMask| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw5bMask| s.estimate_ml(&pmf_5b),
        |s: &OsBmw5bMask, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &OsBmw5bMask| s.total_bytes(),
        |s: &OsBmw5bMask| s.total_bytes()
    );

    bench_method!(
        "OS-BMW-5b-256B",
        || OsBmw256::new(16, HashMode5b::Div3),
        |s: &mut OsBmw256, h| { s.add_hash_64(h); },
        |s: &mut OsBmw256, h| s.add_hash_64(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.estimate_ml(&pmf_256),
        |s: &OsBmw256, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &OsBmw256| s.total_bytes(),
        |s: &OsBmw256| s.total_bytes()
    );

    bench_method!(
        "OS-BMW-Chunk",
        || OsBmwChunkInPlace::new(4),
        |s: &mut OsBmwChunkInPlace, h| { s.add_hash_64(h, &pmf_chunk); },
        |s: &mut OsBmwChunkInPlace, h| s.add_hash_64(h, &pmf_chunk),
        |a: &mut OsBmwChunkInPlace, b: &OsBmwChunkInPlace| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmwChunkInPlace| s.estimate_ml(&pmf_chunk),
        |s: &OsBmwChunkInPlace, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(&s.buf); },
        |s: &OsBmwChunkInPlace| s.buf.len(),
        |s: &OsBmwChunkInPlace| s.buf.len()
    );

    bench_method!(
        "ELL(2,24)",
        || Ell24Sketch::new(10),
        |s: &mut Ell24Sketch, h| { s.add_hash_64(h); },
        |s: &mut Ell24Sketch, h| { s.add_hash_64(h); false },
        |a: &mut Ell24Sketch, b: &Ell24Sketch| { a.merge_in_place(b); },
        |s: &Ell24Sketch| s.estimate_ml(&pmf_ell),
        |s: &Ell24Sketch, out: &mut Vec<u8>| { s.serialize_bytes(out); },
        |s: &Ell24Sketch| s.total_bytes(),
        |s: &Ell24Sketch| s.total_bytes()
    );

    bench_method!(
        "ELL(2,20)",
        || Ell20Sketch::new(10),
        |s: &mut Ell20Sketch, h| { s.add_hash_64(h); },
        |s: &mut Ell20Sketch, h| { s.add_hash_64(h); false },
        |a: &mut Ell20Sketch, b: &Ell20Sketch| { a.merge_in_place(b); },
        |s: &Ell20Sketch| s.estimate_ml(&pmf_ell),
        |s: &Ell20Sketch, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &Ell20Sketch| s.total_bytes(),
        |s: &Ell20Sketch| s.total_bytes()
    );

    bench_method!(
        "ULL",
        || UllSketch::new(12),
        |s: &mut UllSketch, h| { s.add_hash_64(h); },
        |s: &mut UllSketch, h| { s.add_hash_64(h); false },
        |a: &mut UllSketch, b: &UllSketch| { a.merge_in_place(b); },
        |s: &UllSketch| s.estimate_ml(&pmf_ull),
        |s: &UllSketch, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &UllSketch| s.total_bytes(),
        |s: &UllSketch| s.total_bytes()
    );

    bench_method!(
        "HLL++(6b)",
        || Hll6Sketch::new(12),
        |s: &mut Hll6Sketch, h| { s.add_hash_64(h); },
        |s: &mut Hll6Sketch, h| { s.add_hash_64(h); false },
        |a: &mut Hll6Sketch, b: &Hll6Sketch| { a.merge_in_place(b); },
        |s: &Hll6Sketch| s.estimate_ml(&pmf_ull),
        |s: &Hll6Sketch, out: &mut Vec<u8>| { out.clear(); out.extend_from_slice(s.as_wire_bytes()); },
        |s: &Hll6Sketch| s.total_bytes(),
        |s: &Hll6Sketch| s.total_bytes()
    );

    bench_method!(
        "AVLL(base56)",
        || AvllSketch::new_words(511),
        |s: &mut AvllSketch, h| { s.add_hash_64(h); },
        |s: &mut AvllSketch, h| s.add_hash_64(h),
        |a: &mut AvllSketch, b: &AvllSketch| { *a = a.merge(b); },
        |s: &AvllSketch| s.estimate_ml(&pmf_avll),
        |s: &AvllSketch, out: &mut Vec<u8>| {
            out.clear();
            out.extend_from_slice(&(s.global_nlz as u64).to_le_bytes());
            let ptr = s.words.as_ptr() as *const u8;
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, s.words.len() * 8) });
        },
        |s: &AvllSketch| s.total_bytes(),
        |s: &AvllSketch| s.total_bytes()
    );

    bench_method!(
        "CPC(lgK=11)",
        || CpcSketch::new(11),
        |s: &mut CpcSketch, h| { s.add_hash_64(h); },
        |s: &mut CpcSketch, h| { s.add_hash_64(h); false },
        |a: &mut CpcSketch, b: &CpcSketch| { a.merge_in_place(b); },
        |s: &CpcSketch| s.estimate_ml(&pmf_cpc),
        |s: &CpcSketch, out: &mut Vec<u8>| { *out = s.serialize_compressed(); },
        |s: &CpcSketch| s.ram_bytes(),
        |s: &CpcSketch| s.serialize_compressed().len()
    );

    bench_method!(
        "HLLL(p=12)",
        || HlllSketch::new(12),
        |s: &mut HlllSketch, h| { s.add_hash_64(h); },
        |s: &mut HlllSketch, h| { s.add_hash_64(h); false },
        |a: &mut HlllSketch, b: &HlllSketch| { a.merge_in_place(b); },
        |s: &HlllSketch| s.estimate_ml(&pmf_ull),
        |s: &HlllSketch, out: &mut Vec<u8>| { s.serialize_into(out); },
        |s: &HlllSketch| s.ram_bytes(),
        |s: &HlllSketch| s.wire_bytes()
    );

    bench_method!(
        "HLL_4(dyn)",
        || Hll4Sketch::new_dynamic(12),
        |s: &mut Hll4Sketch, h| { s.add_hash_64(h); },
        |s: &mut Hll4Sketch, h| { s.add_hash_64(h); false },
        |a: &mut Hll4Sketch, b: &Hll4Sketch| { a.merge_in_place(b); },
        |s: &Hll4Sketch| s.estimate_ml(&pmf_ull),
        |s: &Hll4Sketch, out: &mut Vec<u8>| { s.serialize_into(out); },
        |s: &Hll4Sketch| s.ram_bytes(),
        |s: &Hll4Sketch| s.wire_bytes()
    );

    // --- A4 Small-Sketch Low-Cardinality Throughput Audit (0.5 KB & 1 KB, n in {100, 1000, 10000}) ---
    println!("\n=== [M3/A4] Small-Sketch Low-Cardinality Throughput (0.5 KB & 1 KB, n <= 10^4) ===");
    let mut small_rows = Vec::new();
    for &(budget_label, p_ell, nblk_g128, nblk_256) in &[("0.5KB", 7u32, 4usize, 2usize), ("1KB", 8u32, 8usize, 4usize)] {
        for &n_low in &[100usize, 1_000, 10_000] {
            let reps = if n_low <= 1_000 { 200 } else { 40 };
            let ns_ell = bench_ns_per_op(reps, n_low, || {
                let mut sk = Ell24Sketch::new(p_ell);
                for i in 0..n_low as u64 { sk.add_hash_64(wyhash64(i, seed)); }
                black_box(&sk);
            });
            let ns_g128_hy = bench_ns_per_op(reps, n_low, || {
                let mut sk = OsBmwHybrid::<128, 8>::new(nblk_g128, HashMode5b::Pow2);
                for i in 0..n_low as u64 { sk.add_hash_64(wyhash64(i, seed)); }
                black_box(&sk);
            });
            let ns_g128_dn = bench_ns_per_op(reps, n_low, || {
                let mut sk = OsBmwG128::new(nblk_g128, HashMode5b::Pow2);
                for i in 0..n_low as u64 { sk.add_hash_64(wyhash64(i, seed)); }
                black_box(&sk);
            });
            let ns_256_hy = bench_ns_per_op(reps, n_low, || {
                let mut sk = OsBmwHybrid::<384, 16>::new(nblk_256, HashMode5b::Div3);
                for i in 0..n_low as u64 { sk.add_hash_64(wyhash64(i, seed)); }
                black_box(&sk);
            });
            let ns_256_dn = bench_ns_per_op(reps, n_low, || {
                let mut sk = OsBmw256::new(nblk_256, HashMode5b::Div3);
                for i in 0..n_low as u64 { sk.add_hash_64(wyhash64(i, seed)); }
                black_box(&sk);
            });
            println!(
                "  {} | n={:5} | ELL(2,24)={:5.2}ns | g128-Hybrid={:5.2}ns ({:.2}x) | g128-Dense={:5.2}ns ({:.2}x) | 256B-Hybrid={:5.2}ns ({:.2}x) | 256B-Dense={:5.2}ns ({:.2}x)",
                budget_label, n_low, ns_ell, ns_g128_hy, ns_g128_hy / ns_ell, ns_g128_dn, ns_g128_dn / ns_ell, ns_256_hy, ns_256_hy / ns_ell, ns_256_dn, ns_256_dn / ns_ell
            );
            small_rows.push((budget_label, n_low, ns_ell, ns_g128_hy, ns_g128_dn, ns_256_hy));
        }
    }

    // Write TSV and JSON
    let mut tsv = File::create(out_dir.join("m3_sys_bench.tsv")).unwrap();
    writeln!(
        tsv,
        "method\tram_bytes\twire_bytes\tl1_cold_1e4_ns\tl1_steady_1e7_ns\tdram_4096_steady_ns\tdram_4096_cold_ns\tmerge_equal_ns\tmerge_skew_ns\test_ml_us\test_ml_iters\tser_ns\tee_rate_1e7_pct"
    ).unwrap();
    for r in &rows {
        writeln!(
            tsv,
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.1}\t{:.3}\t{:.1}\t{:.1}\t{:.2}",
            r.name, r.ram_bytes, r.wire_bytes, r.l1_cold_1e4_ns, r.l1_steady_1e7_ns,
            r.dram_4096_steady_ns, r.dram_4096_cold_ns, r.merge_equal_ns, r.merge_skew_ns,
            r.est_ml_us, r.est_ml_iters, r.ser_ns, r.ee_rate_1e7_pct
        ).unwrap();
    }

    // Evaluate A2 System Gate ratios directly against ELL(2,24)!
    let ell24 = rows.iter().find(|r| r.name == "ELL(2,24)").unwrap();
    let mut gate_lines = Vec::new();
    for r in rows.iter().filter(|r| r.name.starts_with("OS-BMW")) {
        let l1_ratio = r.l1_steady_1e7_ns / ell24.l1_steady_1e7_ns;
        let l1_cold_ratio = r.l1_cold_1e4_ns / ell24.l1_cold_1e4_ns;
        let dram_speedup = ell24.dram_4096_steady_ns / r.dram_4096_steady_ns;
        let merge_ratio = r.merge_equal_ns / ell24.merge_equal_ns;
        let pass_a2 = l1_ratio <= 1.5 && dram_speedup >= 1.0 && merge_ratio <= 2.0;
        println!(
            "\n  [A2 Gate Check] {:16} vs ELL(2,24): L1_steady={:.2}x (<=1.5), L1_cold={:.2}x, 4096-DRAM_speedup={:.2}x (>=1.0), merge={:.2}x (<=2.0) => {}",
            r.name, l1_ratio, l1_cold_ratio, dram_speedup, merge_ratio, if pass_a2 { "PASS" } else { "FAIL" }
        );
        gate_lines.push(format!(
            "    {{\"method\": \"{}\", \"l1_steady_vs_ell24\": {:.4}, \"l1_cold_vs_ell24\": {:.4}, \"dram_4096_speedup_vs_ell24\": {:.4}, \"merge_vs_ell24\": {:.4}, \"pass_a2_sys_gate\": {}}}",
            r.name, l1_ratio, l1_cold_ratio, dram_speedup, merge_ratio, pass_a2
        ));
    }

    let rows_json: Vec<String> = rows.iter().map(|r| format!(
        "    {{\"method\": \"{}\", \"ram_bytes\": {}, \"wire_bytes\": {}, \"l1_cold_1e4_ns\": {:.3}, \"l1_steady_1e7_ns\": {:.3}, \"dram_4096_steady_ns\": {:.3}, \"dram_4096_cold_ns\": {:.3}, \"merge_equal_ns\": {:.1}, \"merge_skew_ns\": {:.1}, \"est_ml_us\": {:.3}, \"est_ml_iters\": {:.1}, \"ser_ns\": {:.1}, \"ee_rate_1e7_pct\": {:.2}}}",
        r.name, r.ram_bytes, r.wire_bytes, r.l1_cold_1e4_ns, r.l1_steady_1e7_ns,
        r.dram_4096_steady_ns, r.dram_4096_cold_ns, r.merge_equal_ns, r.merge_skew_ns,
        r.est_ml_us, r.est_ml_iters, r.ser_ns, r.ee_rate_1e7_pct
    )).collect();
    let small_json: Vec<String> = small_rows.iter().map(|(b, n, e, gh, gd, h2)| format!(
        "    {{\"budget\": \"{}\", \"n\": {}, \"ell24_ns\": {:.3}, \"g128_hybrid_ns\": {:.3}, \"g128_dense_ns\": {:.3}, \"osbmw256_hybrid_ns\": {:.3}, \"g128_hybrid_ratio\": {:.3}, \"osbmw256_hybrid_ratio\": {:.3}}}",
        b, n, e, gh, gd, h2, gh / e, h2 / e
    )).collect();
    let rustc_ver = Command::new("rustc")
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| if o.status.success() { String::from_utf8(o.stdout).ok() } else { None })
        .map(|s| s.trim().replace('\\', "\\\\").replace('"', "\\\""))
        .unwrap_or_else(|| "unknown".to_string());
    let avx512_5b_enabled = cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2"
    ));
    let mut feat_list: Vec<&str> = Vec::new();
    if cfg!(target_feature = "avx2") { feat_list.push("\"avx2\""); }
    if cfg!(target_feature = "avx512f") { feat_list.push("\"avx512f\""); }
    if cfg!(target_feature = "avx512bw") { feat_list.push("\"avx512bw\""); }
    if cfg!(target_feature = "avx512vbmi") { feat_list.push("\"avx512vbmi\""); }
    if cfg!(target_feature = "avx512vbmi2") { feat_list.push("\"avx512vbmi2\""); }

    let json_text = format!(
        "{{\n  \"build_env\": {{\n    \"rustc_version\": \"{}\",\n    \"target_arch\": \"{}\",\n    \"avx512_5b_kernel_enabled\": {},\n    \"compiled_simd_features\": [{}]\n  }},\n  \"wyhash64_baseline_ns\": {:.3},\n  \"rows\": [\n{}\n  ],\n  \"a2_gate_checks\": [\n{}\n  ],\n  \"a4_small_sketch_low_n\": [\n{}\n  ]\n}}\n",
        rustc_ver,
        std::env::consts::ARCH,
        avx512_5b_enabled,
        feat_list.join(", "),
        hash_only_ns,
        rows_json.join(",\n"),
        gate_lines.join(",\n"),
        small_json.join(",\n")
    );
    fs::write(out_dir.join("m3_sys_bench.json"), json_text).unwrap();
    println!("\nWrote m3_sys_bench.tsv & m3_sys_bench.json to {}", out_dir.display());
}
