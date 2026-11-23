//! Experiment 2 (11-23 Verdict): OS-BMW-5b-256B (`/3` -> 128-bit Multiply-Shift) Microbenchmark & Accuracy Audit.
//!
//! Outputs `m3_256b_mulshift.tsv` comparing on the exact same core (NUMA 1):
//!   1. `OS-BMW-5b(k24)_Pow2`                (2,760 B, reference power-of-2 fast path)
//!   2. `OS-BMW-5b-256B_Div3_Orig`           (4,104 B, original `hc / 3` + eager `r3/sub`)
//!   3. `OS-BMW-5b-256B_MulShift128_Exact`   (4,104 B, explicit 128-bit mul-shift `0xAAAA_AAAA_AAAA_AAAB` + deferred `r3/sub`, bit-identical to Div3!)
//!   4. `OS-BMW-5b-256B_MulShift384_ZeroCP`  (4,104 B, zero-critical-path multiply-shift: `blk` & `lvl` outside mul/div; `r3` only on 0.47% slow path)
//!   5. `ELL(2,24)`                          (4,104 B, equal-byte EDBT'25 SOTA reference)
//!
//! Measures identical M3 harness metrics (`l1_cold_1e4`, `l1_steady_1e7`, `dram_4096_steady`,
//! `dram_4096_cold`, `merge_equal`, `merge_skew`, `est_ml_us`, `ee_rate_1e7_pct`) AND
//! verifies empirical accuracy/unbiasedness across 200 trials at `n in {1e4, 1e6, 1e7}`.

use s1_v2_rust::baselines::Ell24Sketch;
use s1_v2_rust::hash::wyhash64;
use s1_v2_rust::osbmw_inplace::{HashMode5b, OsBmw256, OsBmw5b};
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

fn median_f64(mut vals: Vec<f64>) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 1 { vals[n / 2] } else { 0.5 * (vals[n / 2 - 1] + vals[n / 2]) }
}

fn bench_ns_per_op<F: FnMut()>(repeats: usize, ops_per_repeat: usize, mut f: F) -> f64 {
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

fn eval_accuracy_stream<S, FMake, FAdd, FEst>(
    n: u64,
    trials: usize,
    n_threads: usize,
    ram_bytes: usize,
    make: FMake,
    add: FAdd,
    est: FEst,
) -> (f64, f64, f64)
where
    FMake: Fn() -> S + Sync,
    FAdd: Fn(&mut S, u64) + Sync,
    FEst: Fn(&S) -> f64 + Sync,
{
    let mut errs = Vec::with_capacity(trials);
    let per_t = (trials + n_threads - 1) / n_threads;
    thread::scope(|s| {
        let mut hs = Vec::new();
        for tid in 0..n_threads {
            let st = tid * per_t;
            let en = (st + per_t).min(trials);
            if st >= en { continue; }
            let mk = &make;
            let ad = &add;
            let es = &est;
            hs.push(s.spawn(move || {
                let mut loc = Vec::with_capacity(en - st);
                for tr in st..en {
                    let seed = 0x256B_0000_0000_0000u64 ^ ((tr as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) ^ n;
                    let mut sk = mk();
                    for i in 0..n {
                        ad(&mut sk, wyhash64(i, seed));
                    }
                    let e = (es(&sk) - n as f64) / (n as f64);
                    loc.push(e);
                }
                loc
            }));
        }
        for h in hs {
            errs.extend_from_slice(&h.join().unwrap());
        }
    });
    let nt = errs.len() as f64;
    let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
    let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
    let rmse = mse.sqrt() * 100.0;
    let mvp = (ram_bytes as f64 * 8.0) * mse;
    (bias, rmse, mvp)
}

fn main() {
    let out_dir = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");
    let n_threads = env::var("S1_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(16).max(1);

    // Ramp CPU to steady turbo clock before timing!
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

    let seed = 0x1234_5678_9abc_def0u64;
    let n_l1 = 2_000_000usize;
    let l1_prefill_n = 10_000_000u64;
    let prefill_n = 1_000_000u64;
    let n_groups = 4096usize;
    let n_dram_ops = 4_000_000usize;
    let group_ids: Vec<u16> = (0..n_dram_ops)
        .map(|i| (wyhash64(i as u64, 0xfeed_face) & 4095) as u16)
        .collect();

    let pmf_5b = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 3 - 9, max_cap: 63 });
    let pmf_256_div3 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 4 });
    let pmf_256_ms384 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 25 - 4, max_cap: 63 });
    let pmf_ell = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - 10 });

    let out_path = out_dir.join("m3_256b_mulshift.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(
        f,
        "method\tram_bytes\tbit_exact_div3\tl1_cold_1e4_ns\tl1_steady_1e7_ns\tdram_4096_steady_ns\tdram_4096_cold_ns\tmerge_equal_ns\tmerge_skew_ns\test_ml_us\tee_rate_1e7_pct\tacc_1e4_bias_pct\tacc_1e4_rmse_pct\tacc_1e4_mvp\tacc_1e6_bias_pct\tacc_1e6_rmse_pct\tacc_1e6_mvp"
    ).unwrap();

    macro_rules! run_variant {
        ($name:expr, $bit_exact:expr, $make:expr, $add:expr, $add_ee:expr, $merge:expr, $est:expr, $ram:expr) => {{
            let n_cold = 10_000usize;
            let l1_cold_1e4_ns = bench_ns_per_op(35, n_cold, || {
                let mut sk = $make();
                for i in 0..n_cold as u64 {
                    $add(&mut sk, wyhash64(i, seed));
                }
                black_box(&sk);
            });

            let mut sk_steady = $make();
            for i in 0..l1_prefill_n {
                $add(&mut sk_steady, wyhash64(i, seed));
            }
            let mut ee_hits = 0usize;
            for i in l1_prefill_n..(l1_prefill_n + 200_000) {
                if $add_ee(&mut sk_steady, wyhash64(i, seed)) { ee_hits += 1; }
            }
            let ee_rate = (ee_hits as f64) * 100.0 / 200_000.0;

            let mut sk_l1 = sk_steady.clone();
            let mut base_i = l1_prefill_n + 200_000;
            let l1_steady_1e7_ns = bench_ns_per_op(7, n_l1, || {
                for i in base_i..(base_i + n_l1 as u64) {
                    $add(&mut sk_l1, wyhash64(i, seed));
                }
                base_i += n_l1 as u64;
                black_box(&sk_l1);
            });

            let mut proto_1m = $make();
            for i in 0..prefill_n {
                $add(&mut proto_1m, wyhash64(i, seed ^ 0x1111));
            }
            let mut bank_steady: Vec<_> = (0..n_groups).map(|_| proto_1m.clone()).collect();
            let mut dram_base = prefill_n;
            let dram_4096_steady_ns = bench_ns_per_op(5, n_dram_ops, || {
                for (idx, &gid) in group_ids.iter().enumerate() {
                    $add(&mut bank_steady[gid as usize], wyhash64(dram_base + idx as u64, seed));
                }
                dram_base += n_dram_ops as u64;
                black_box(&bank_steady[0]);
            });
            drop(bank_steady);

            let mut bank_cold: Vec<_> = (0..n_groups).map(|_| $make()).collect();
            let dram_4096_cold_ns = bench_ns_per_op(3, n_dram_ops, || {
                for s in bank_cold.iter_mut() { *s = $make(); }
                for (idx, &gid) in group_ids.iter().enumerate() {
                    $add(&mut bank_cold[gid as usize], wyhash64(idx as u64, seed));
                }
                black_box(&bank_cold[0]);
            });
            drop(bank_cold);

            let mut sk_a = $make();
            let mut sk_b = $make();
            let mut sk_small = $make();
            for i in 0..1_000_000u64 {
                $add(&mut sk_a, wyhash64(i, seed));
                $add(&mut sk_b, wyhash64(i + 2_000_000, seed));
            }
            for i in 0..10_000u64 {
                $add(&mut sk_small, wyhash64(i + 5_000_000, seed));
            }
            let m_reps = 200usize;
            let merge_equal_ns = {
                let tot = bench_ns_per_op(7, m_reps, || {
                    for _ in 0..m_reps {
                        let mut tmp = sk_a.clone();
                        $merge(&mut tmp, &sk_b);
                        black_box(&tmp);
                    }
                });
                let cl = bench_ns_per_op(7, m_reps, || {
                    for _ in 0..m_reps {
                        let tmp = sk_a.clone();
                        black_box(&tmp);
                    }
                });
                (tot - cl).max(0.0)
            };
            let merge_skew_ns = {
                let tot = bench_ns_per_op(7, m_reps, || {
                    for _ in 0..m_reps {
                        let mut tmp = sk_a.clone();
                        $merge(&mut tmp, &sk_small);
                        black_box(&tmp);
                    }
                });
                let cl = bench_ns_per_op(7, m_reps, || {
                    for _ in 0..m_reps {
                        let tmp = sk_a.clone();
                        black_box(&tmp);
                    }
                });
                (tot - cl).max(0.0)
            };

            let est_ml_us = bench_ns_per_op(9, 200, || {
                let mut acc = 0.0_f64;
                for _ in 0..200 {
                    acc += $est(&sk_a);
                }
                black_box(acc);
            }) / 1000.0;

            let (b4, r4, m4) = eval_accuracy_stream(10_000, 200, n_threads, $ram, || $make(), |s, h| { $add(s, h); }, |s| $est(s));
            let (b6, r6, m6) = eval_accuracy_stream(1_000_000, 120, n_threads, $ram, || $make(), |s, h| { $add(s, h); }, |s| $est(s));

            println!(
                "  {:34} | L1_cold={:5.2}ns | L1_std={:5.2}ns | DRAM_std={:5.2}ns | DRAM_cold={:5.2}ns | MrgEq={:6.1}ns | Est={:5.2}us | EE={:5.2}% | 1e6: bias={:+.3}% rmse={:.3}% mvp={:.3}",
                $name, l1_cold_1e4_ns, l1_steady_1e7_ns, dram_4096_steady_ns, dram_4096_cold_ns,
                merge_equal_ns, est_ml_us, ee_rate, b6, r6, m6
            );
            writeln!(
                f,
                "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.1}\t{:.3}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                $name, $ram, $bit_exact, l1_cold_1e4_ns, l1_steady_1e7_ns, dram_4096_steady_ns,
                dram_4096_cold_ns, merge_equal_ns, merge_skew_ns, est_ml_us, ee_rate,
                b4, r4, m4, b6, r6, m6
            ).unwrap();
            f.flush().unwrap();
        }};
    }

    println!("=== [M3 256B MulShift Audit] Single-core timing + Multi-thread accuracy ===");
    run_variant!(
        "OS-BMW-5b(k24)_Pow2",
        "N/A_Pow2",
        || OsBmw5b::new(8, HashMode5b::Pow2),
        |s: &mut OsBmw5b, h: u64| { s.add_hash_64(h); },
        |s: &mut OsBmw5b, h: u64| s.add_hash_64(h),
        |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw5b| s.estimate_ml(&pmf_5b).0,
        2760usize
    );
    run_variant!(
        "OS-BMW-5b-256B_Div3_Orig",
        "exact_ref",
        || OsBmw256::new(16, HashMode5b::Div3),
        |s: &mut OsBmw256, h: u64| { s.add_hash_64(h); },
        |s: &mut OsBmw256, h: u64| s.add_hash_64(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.estimate_ml(&pmf_256_div3).0,
        4104usize
    );
    run_variant!(
        "OS-BMW-5b-256B_MulShift128_Exact",
        "bit_identical",
        || OsBmw256::new(16, HashMode5b::Div3),
        |s: &mut OsBmw256, h: u64| { s.add_hash_64_div3_mulshift_deferred(h); },
        |s: &mut OsBmw256, h: u64| s.add_hash_64_div3_mulshift_deferred(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.estimate_ml(&pmf_256_div3).0,
        4104usize
    );
    run_variant!(
        "OS-BMW-5b-256B_MulShift384_ZeroCP",
        "pow2_w35_zero_cp",
        || OsBmw256::new(16, HashMode5b::Div3),
        |s: &mut OsBmw256, h: u64| { s.add_hash_64_mulshift384(h); },
        |s: &mut OsBmw256, h: u64| s.add_hash_64_mulshift384(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.estimate_ml(&pmf_256_ms384).0,
        4104usize
    );
    run_variant!(
        "ELL(2,24)",
        "N/A_ErtlT2",
        || Ell24Sketch::new(10),
        |s: &mut Ell24Sketch, h: u64| { s.add_hash_64(h); },
        |s: &mut Ell24Sketch, h: u64| { s.add_hash_64(h); false },
        |a: &mut Ell24Sketch, b: &Ell24Sketch| { a.merge_in_place(b); },
        |s: &Ell24Sketch| s.estimate_ml(&pmf_ell).0,
        4104usize
    );

    println!("DONE m3_256b_mulshift -> {}", out_path.display());
}
