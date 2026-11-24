//! Supplementary Microbenchmark Suite for E-01, E-08, E-09 (`s1_v2_bench_supp`).
//!
//! Modes:
//!   --mode ell-filter        (E-01: ELL+Filter 8B/16B & ULL+Filter vs ELL, ULL, OS-BMW 2.7/4.0 KiB)
//!   --mode modcard           (E-09: Moderate cardinality 1e3..1e6 across 4K & 64K groups)
//!   --mode non-avx512-merge  (E-08: Merge latency on native AVX-512 vs AVX2/scalar x86 vs arm64 NEON/scalar)

use s1_v2_rust::baselines::*;
use s1_v2_rust::hash::wyhash64;
use s1_v2_rust::osbmw_inplace::*;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn median_f64(mut vals: Vec<f64>) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 1 { vals[n / 2] } else { 0.5 * (vals[n / 2 - 1] + vals[n / 2]) }
}

fn warmup_cpu() {
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

fn run_ell_filter(out_dir: &PathBuf) {
    println!("=== [E-01] ELL(2,24)+Filter & ULL+Filter Footprint vs Early-Exit Ablation ===");
    let seed = 0x1234_5678_9abc_def0u64;
    let n_l1 = 2_000_000usize;
    let n_groups_4k = 4096usize;
    let n_groups_64k = 65_536usize;
    let n_dram_ops = 4_000_000usize;
    let group_ids_4k: Vec<u16> = (0..n_dram_ops).map(|i| (wyhash64(i as u64, 0xfeed_face) & 4095) as u16).collect();
    let group_ids_64k: Vec<u16> = (0..n_dram_ops).map(|i| (wyhash64(i as u64, 0xbeef_cafe) & 65_535) as u16).collect();

    let out_path = out_dir.join("m3_ell_filter_ablation.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(
        f,
        "method\tram_bytes\tl1_cold_1e4_ns\tl1_steady_1e7_ns\tee_1e6_pct\tee_1e7_pct\tg4k_n1e6_ns\tg4k_n1e7_ns\tg64k_n1e6_ns\tg64k_n1e7_ns\tmerge_eq_ns\tmerge_skew_ns"
    ).unwrap();

    macro_rules! eval_e01 {
        ($name:expr, $make:expr, $add_ee:expr, $merge_fn:expr, $bytes_fn:expr) => {{
            let n_cold = 10_000usize;
            let l1_cold = bench_ns_per_op(35, n_cold, || {
                let mut sk = $make();
                for i in 0..n_cold as u64 {
                    black_box($add_ee(&mut sk, wyhash64(i, seed)));
                }
                black_box(&sk);
            });

            let mut sk_1e6 = $make();
            for i in 0..1_000_000u64 {
                $add_ee(&mut sk_1e6, wyhash64(i, seed ^ 0x9999));
            }
            let mut ee_1e6_hits = 0usize;
            for i in 1_000_000..1_200_000u64 {
                if $add_ee(&mut sk_1e6, wyhash64(i, seed ^ 0x7777)) {
                    ee_1e6_hits += 1;
                }
            }
            let ee_1e6_pct = (ee_1e6_hits as f64) * 100.0 / 200_000.0;

            let mut sk_1e7 = $make();
            for i in 0..10_000_000u64 {
                $add_ee(&mut sk_1e7, wyhash64(i, seed));
            }
            let mut ee_1e7_hits = 0usize;
            for i in 10_000_000..10_200_000u64 {
                if $add_ee(&mut sk_1e7, wyhash64(i, seed)) {
                    ee_1e7_hits += 1;
                }
            }
            let ee_1e7_pct = (ee_1e7_hits as f64) * 100.0 / 200_000.0;

            let mut sk_l1 = sk_1e7.clone();
            let mut base_l1 = 10_200_000u64;
            let l1_steady = bench_ns_per_op(7, n_l1, || {
                for i in base_l1..(base_l1 + n_l1 as u64) {
                    $add_ee(&mut sk_l1, wyhash64(i, seed));
                }
                base_l1 += n_l1 as u64;
                black_box(&sk_l1);
            });

            let mut proto_1e6 = $make();
            for i in 0..1_000_000u64 {
                $add_ee(&mut proto_1e6, wyhash64(i, seed ^ 0x9999));
            }

            let mut g4k_1e6_vec: Vec<_> = (0..n_groups_4k).map(|_| proto_1e6.clone()).collect();
            let mut b1 = 1_000_000u64;
            let g4k_n1e6 = bench_ns_per_op(5, n_dram_ops, || {
                for (step, &gid) in group_ids_4k.iter().enumerate() {
                    $add_ee(&mut g4k_1e6_vec[gid as usize], wyhash64(b1 + step as u64, seed));
                }
                b1 += n_dram_ops as u64;
                black_box(&g4k_1e6_vec[0]);
            });
            drop(g4k_1e6_vec);

            let mut g4k_1e7_vec: Vec<_> = (0..n_groups_4k).map(|_| sk_1e7.clone()).collect();
            let mut b2 = 10_200_000u64;
            let g4k_n1e7 = bench_ns_per_op(5, n_dram_ops, || {
                for (step, &gid) in group_ids_4k.iter().enumerate() {
                    $add_ee(&mut g4k_1e7_vec[gid as usize], wyhash64(b2 + step as u64, seed));
                }
                b2 += n_dram_ops as u64;
                black_box(&g4k_1e7_vec[0]);
            });
            drop(g4k_1e7_vec);

            let mut g64k_1e6_vec: Vec<_> = (0..n_groups_64k).map(|_| proto_1e6.clone()).collect();
            let mut b3 = 100_000_000u64;
            let g64k_n1e6 = bench_ns_per_op(3, n_dram_ops, || {
                for (step, &gid) in group_ids_64k.iter().enumerate() {
                    $add_ee(&mut g64k_1e6_vec[gid as usize], wyhash64(b3 + step as u64, seed));
                }
                b3 += n_dram_ops as u64;
                black_box(&g64k_1e6_vec[0]);
            });
            drop(g64k_1e6_vec);

            let mut g64k_1e7_vec: Vec<_> = (0..n_groups_64k).map(|_| sk_1e7.clone()).collect();
            let mut b4 = 200_000_000u64;
            let g64k_n1e7 = bench_ns_per_op(3, n_dram_ops, || {
                for (step, &gid) in group_ids_64k.iter().enumerate() {
                    $add_ee(&mut g64k_1e7_vec[gid as usize], wyhash64(b4 + step as u64, seed));
                }
                b4 += n_dram_ops as u64;
                black_box(&g64k_1e7_vec[0]);
            });
            drop(g64k_1e7_vec);

            let mut sk_m2 = $make();
            for i in 0..1_000_000u64 {
                $add_ee(&mut sk_m2, wyhash64(i, seed ^ 0x1111));
            }
            let mut sk_small = $make();
            for i in 0..10_000u64 {
                $add_ee(&mut sk_small, wyhash64(i, seed ^ 0x2222));
            }
            let pool_a: Vec<_> = (0..64).map(|_| proto_1e6.clone()).collect();
            let pool_b: Vec<_> = (0..64).map(|_| sk_m2.clone()).collect();
            let mut eq_samples = Vec::new();
            for _ in 0..11 {
                let mut w = pool_a.clone();
                let t0 = Instant::now();
                for i in 0..64 { $merge_fn(&mut w[i], &pool_b[i]); }
                black_box(&w);
                eq_samples.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_eq = median_f64(eq_samples);

            let pool_big: Vec<_> = (0..64).map(|_| sk_1e7.clone()).collect();
            let mut skew_samples = Vec::new();
            for _ in 0..11 {
                let mut w = pool_big.clone();
                let t0 = Instant::now();
                for i in 0..64 { $merge_fn(&mut w[i], &sk_small); }
                black_box(&w);
                skew_samples.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_skew = median_f64(skew_samples);
            let ram_b = $bytes_fn(&sk_1e7);

            println!(
                "  {:20} | {:5}B | L1(1e7)={:5.2}ns | EE(1e6/1e7)={:4.1}%/{:4.1}% | 4K(1e6/1e7)={:5.2}/{:5.2}ns | 64K(1e6/1e7)={:5.2}/{:5.2}ns | mrg={:5.0}/{:5.0}ns",
                $name, ram_b, l1_steady, ee_1e6_pct, ee_1e7_pct, g4k_n1e6, g4k_n1e7, g64k_n1e6, g64k_n1e7, merge_eq, merge_skew
            );
            writeln!(
                f,
                "{}\t{}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.1}",
                $name, ram_b, l1_cold, l1_steady, ee_1e6_pct, ee_1e7_pct, g4k_n1e6, g4k_n1e7, g64k_n1e6, g64k_n1e7, merge_eq, merge_skew
            ).unwrap();
            f.flush().unwrap();
        }};
    }

    eval_e01!(
        "OS-BMW-5b(k24)",
        || OsBmw5b::new(8, HashMode5b::Pow2),
        |s: &mut OsBmw5b, h| s.add_hash_64(h),
        |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw5b| s.total_bytes()
    );
    eval_e01!(
        "OS-BMW-5b-256B(Div3)",
        || OsBmw256::new(16, HashMode5b::Div3),
        |s: &mut OsBmw256, h| s.add_hash_64(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.total_bytes()
    );
    eval_e01!(
        "OS-BMW-5b-256B(MS384)",
        || OsBmw256::new(16, HashMode5b::MulShift384),
        |s: &mut OsBmw256, h| s.add_hash_64_mulshift384(h),
        |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); },
        |s: &OsBmw256| s.total_bytes()
    );
    eval_e01!(
        "ELL(2,24)",
        || Ell24Sketch::new(10),
        |s: &mut Ell24Sketch, h| { s.add_hash_64(h); false },
        |a: &mut Ell24Sketch, b: &Ell24Sketch| { a.merge_in_place(b); },
        |s: &Ell24Sketch| s.total_bytes()
    );
    eval_e01!(
        "ELL(2,24)+Filter-8B",
        || Ell24Filtered8B::new(10),
        |s: &mut Ell24Filtered8B, h| s.add_hash_64(h),
        |a: &mut Ell24Filtered8B, b: &Ell24Filtered8B| { a.merge_in_place(b); },
        |s: &Ell24Filtered8B| s.total_bytes()
    );
    eval_e01!(
        "ELL(2,24)+Filter-16B",
        || Ell24Filtered16B::new(10),
        |s: &mut Ell24Filtered16B, h| s.add_hash_64(h),
        |a: &mut Ell24Filtered16B, b: &Ell24Filtered16B| { a.merge_in_place(b); },
        |s: &Ell24Filtered16B| s.total_bytes()
    );
    eval_e01!(
        "ULL",
        || UllSketch::new(12),
        |s: &mut UllSketch, h| { s.add_hash_64(h); false },
        |a: &mut UllSketch, b: &UllSketch| { a.merge_in_place(b); },
        |s: &UllSketch| s.total_bytes()
    );
    eval_e01!(
        "ULL+Filter-8B",
        || UllFiltered8B::new(12),
        |s: &mut UllFiltered8B, h| s.add_hash_64(h),
        |a: &mut UllFiltered8B, b: &UllFiltered8B| { a.merge_in_place(b); },
        |s: &UllFiltered8B| s.total_bytes()
    );
    println!("Wrote E-01 -> {}", out_path.display());
}

fn run_modcard(out_dir: &PathBuf) {
    println!("=== [E-09] Moderate Cardinality (1e3..1e6) 4K & 64K Multi-Group Update Transition ===");
    let seed = 0x2233_4455_6677_8899u64;
    let n_groups_4k = 4096usize;
    let n_groups_64k = 65_536usize;
    let n_ops_4k = 1_000_000usize;
    let n_ops_64k = 4_000_000usize;
    let g4k_ids: Vec<u16> = (0..n_ops_4k).map(|i| (wyhash64(i as u64, 0xfeed_face) & 4095) as u16).collect();
    let g64k_ids: Vec<u16> = (0..n_ops_64k).map(|i| (wyhash64(i as u64, 0xbeef_cafe) & 65_535) as u16).collect();

    let out_path = out_dir.join("m3_moderate_card_micro.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(f, "prefill_n\tmethod\tram_bytes\tee_rate_pct\tg4k_ns\tg64k_ns").unwrap();

    for &prefill_n in &[1_000u64, 10_000, 100_000, 1_000_000] {
        macro_rules! eval_mc {
            ($name:expr, $make:expr, $add_ee:expr, $bytes_fn:expr) => {{
                let mut proto = $make();
                for i in 0..prefill_n {
                    $add_ee(&mut proto, wyhash64(i, seed ^ 0x1111));
                }
                let mut ee_pool: Vec<_> = (0..1024).map(|_| proto.clone()).collect();
                let mut ee_hits = 0usize;
                let mut probe_idx = 0u64;
                for sk in &mut ee_pool {
                    for _ in 0..50 {
                        if $add_ee(sk, wyhash64(prefill_n + 2_000_000 + probe_idx, seed ^ 0x2222)) {
                            ee_hits += 1;
                        }
                        probe_idx += 1;
                    }
                }
                drop(ee_pool);
                let ee_pct = (ee_hits as f64) * 100.0 / 51_200.0;
                let mut g4k: Vec<_> = (0..n_groups_4k).map(|_| proto.clone()).collect();
                let mut b1 = prefill_n + 1_000_000;
                let g4k_ns = bench_ns_per_op(5, n_ops_4k, || {
                    for (step, &gid) in g4k_ids.iter().enumerate() {
                        $add_ee(&mut g4k[gid as usize], wyhash64(b1 + step as u64, seed));
                    }
                    b1 += n_ops_4k as u64;
                    black_box(&g4k[0]);
                });
                drop(g4k);

                let mut g64k: Vec<_> = (0..n_groups_64k).map(|_| proto.clone()).collect();
                let mut b2 = prefill_n + 50_000_000;
                let g64k_ns = bench_ns_per_op(3, n_ops_64k, || {
                    for (step, &gid) in g64k_ids.iter().enumerate() {
                        $add_ee(&mut g64k[gid as usize], wyhash64(b2 + step as u64, seed));
                    }
                    b2 += n_ops_64k as u64;
                    black_box(&g64k[0]);
                });
                drop(g64k);
                let ram_b = $bytes_fn(&proto);
                println!("  n={:7} | {:18} | EE={:5.1}% | 4K={:6.2}ns | 64K={:6.2}ns", prefill_n, $name, ee_pct, g4k_ns, g64k_ns);
                writeln!(f, "{}\t{}\t{}\t{:.2}\t{:.3}\t{:.3}", prefill_n, $name, ram_b, ee_pct, g4k_ns, g64k_ns).unwrap();
                f.flush().unwrap();
            }};
        }
        eval_mc!("OS-BMW-5b(k24)", || OsBmw5b::new(8, HashMode5b::Pow2), |s: &mut OsBmw5b, h| s.add_hash_64(h), |s: &OsBmw5b| s.total_bytes());
        eval_mc!("OS-BMW-5b-256B", || OsBmw256::new(16, HashMode5b::Div3), |s: &mut OsBmw256, h| s.add_hash_64(h), |s: &OsBmw256| s.total_bytes());
        eval_mc!("ELL(2,24)", || Ell24Sketch::new(10), |s: &mut Ell24Sketch, h| { s.add_hash_64(h); false }, |s: &Ell24Sketch| s.total_bytes());
        eval_mc!("ELL(2,24)+Filter-8B", || Ell24Filtered8B::new(10), |s: &mut Ell24Filtered8B, h| s.add_hash_64(h), |s: &Ell24Filtered8B| s.total_bytes());
        eval_mc!("ULL", || UllSketch::new(12), |s: &mut UllSketch, h| { s.add_hash_64(h); false }, |s: &UllSketch| s.total_bytes());
    }
    println!("Wrote E-09 -> {}", out_path.display());
}

fn run_non_avx512_merge(out_dir: &PathBuf, tag: &str, append: bool) {
    let avx512_compiled = cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2"
    ));
    let arch = std::env::consts::ARCH;
    let out_path = out_dir.join("m3_non_avx512_merge.tsv");
    let mut f = if append && out_path.exists() {
        OpenOptions::new().append(true).open(&out_path).unwrap()
    } else {
        let mut nf = File::create(&out_path).unwrap();
        writeln!(nf, "target_tag\tarch\tavx512_compiled\tmerge_path\tmethod\tram_bytes\tmerge_eq_ns\tmerge_skew_ns").unwrap();
        nf
    };
    let seed = 0x51_52_53_54_u64;
    let n_eq = 1_000_000u64;
    let n_hi = 10_000_000u64;
    let n_lo = 10_000u64;

    macro_rules! eval_merge {
        ($path_kind:expr, $name:expr, $make:expr, $add:expr, $merge_fn:expr, $bytes_fn:expr) => {{
            let mut m1 = $make();
            let mut m2 = $make();
            for i in 0..n_eq {
                $add(&mut m1, wyhash64(i, seed ^ 0x1111));
                $add(&mut m2, wyhash64(i, seed ^ 0x2222));
            }
            let mut big = $make();
            for i in 0..n_hi { $add(&mut big, wyhash64(i, seed ^ 0x3333)); }
            let mut small = $make();
            for i in 0..n_lo { $add(&mut small, wyhash64(i, seed ^ 0x4444)); }

            let pa: Vec<_> = (0..64).map(|_| m1.clone()).collect();
            let pb: Vec<_> = (0..64).map(|_| m2.clone()).collect();
            // warmup
            { let mut w = pa.clone(); for i in 0..64 { $merge_fn(&mut w[i], &pb[i]); } black_box(&w); }
            let mut eqs = Vec::with_capacity(15);
            for _ in 0..15 {
                let mut w = pa.clone();
                let t0 = Instant::now();
                for i in 0..64 { $merge_fn(&mut w[i], &pb[i]); }
                black_box(&w);
                eqs.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_eq = median_f64(eqs);

            let pbig: Vec<_> = (0..64).map(|_| big.clone()).collect();
            let mut sks = Vec::with_capacity(15);
            for _ in 0..15 {
                let mut w = pbig.clone();
                let t0 = Instant::now();
                for i in 0..64 { $merge_fn(&mut w[i], &small); }
                black_box(&w);
                sks.push(t0.elapsed().as_nanos() as f64 / 64.0);
            }
            let merge_skew = median_f64(sks);
            let rb = $bytes_fn(&m1);
            println!(
                "  [{} arch={} avx512={} {}] {:22} | {:5}B | eq={:7.1}ns | skew={:7.1}ns",
                tag, arch, avx512_compiled, $path_kind, $name, rb, merge_eq, merge_skew
            );
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}",
                tag, arch, avx512_compiled, $path_kind, $name, rb, merge_eq, merge_skew
            ).unwrap();
            f.flush().unwrap();
        }};
    }

    let path_tag = if avx512_compiled { "compiled_avx512" } else { "compiled_portable" };
    eval_merge!(path_tag, "OS-BMW-5b(k24)", || OsBmw5b::new(8, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.total_bytes());
    eval_merge!(path_tag, "OS-BMW-5b(k24,16blk)", || OsBmw5b::new(16, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.total_bytes());
    eval_merge!(path_tag, "OS-BMW-5b-256B", || OsBmw256::new(16, HashMode5b::Div3), |s: &mut OsBmw256, h| { s.add_hash_64(h); }, |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw256| s.total_bytes());
    eval_merge!("explicit_scalar", "OS-BMW-5b(k24)_scalar", || OsBmw5b::new(8, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place_scalar(b).unwrap(); }, |s: &OsBmw5b| s.total_bytes());
    eval_merge!("explicit_scalar", "OS-BMW-5b-256B_scalar", || OsBmw256::new(16, HashMode5b::Div3), |s: &mut OsBmw256, h| { s.add_hash_64(h); }, |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place_scalar(b).unwrap(); }, |s: &OsBmw256| s.total_bytes());
    eval_merge!(path_tag, "ELL(2,24)", || Ell24Sketch::new(10), |s: &mut Ell24Sketch, h| { s.add_hash_64(h); }, |a: &mut Ell24Sketch, b: &Ell24Sketch| { a.merge_in_place(b); }, |s: &Ell24Sketch| s.total_bytes());
    eval_merge!(path_tag, "ULL", || UllSketch::new(12), |s: &mut UllSketch, h| { s.add_hash_64(h); }, |a: &mut UllSketch, b: &UllSketch| { a.merge_in_place(b); }, |s: &UllSketch| s.total_bytes());
    eval_merge!(path_tag, "CPC(lgK=11)", || CpcSketch::new(11), |s: &mut CpcSketch, h| { s.add_hash_64(h); }, |a: &mut CpcSketch, b: &CpcSketch| { a.merge_in_place(b); }, |s: &CpcSketch| s.ram_bytes());
    eval_merge!(path_tag, "HLL_4(dyn)", || Hll4Sketch::new_dynamic(12), |s: &mut Hll4Sketch, h| { s.add_hash_64(h); }, |a: &mut Hll4Sketch, b: &Hll4Sketch| { a.merge_in_place(b); }, |s: &Hll4Sketch| s.ram_bytes());
    println!("Wrote E-08 ({}) -> {}", tag, out_path.display());
}

fn run_ull_fgra_micro(out_dir: &PathBuf) {
    use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
    let pmf_5b = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 3 - 9, max_cap: 63 });
    let pmf_ell = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - 10 });
    let pmf_ull = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 12, max_cap: 63 });
    let mut sk_5b = OsBmw5b::new(8, HashMode5b::Pow2);
    let mut sk_ell = Ell24Sketch::new(10);
    let mut sk_ull = UllSketch::new(12);
    let seed = 0x1234_5678_9abc_def0u64;
    for i in 0..10_000_000u64 {
        let h = wyhash64(i, seed);
        sk_5b.add_hash_64(h);
        sk_ell.add_hash_64(h);
        sk_ull.add_hash_64(h);
    }

    let out_path = out_dir.join("m3_ull_fgra_micro.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(f, "method\testimator\tn_items\tper_sketch_est_ns\tper_sketch_est_us\testimate_val").unwrap();

    let mut measure = |method: &str, est_name: &str, est_iters: usize, mut func: Box<dyn FnMut() -> f64>| {
        let ns = bench_ns_per_op(11, est_iters, || {
            let mut s = 0.0_f64;
            for _ in 0..est_iters {
                s += func();
            }
            black_box(s);
        });
        let val = func();
        println!("  [FGRA-Micro] {:16} | {:4} | {:8.1} ns ({:6.3} us) | est={:.1}", method, est_name, ns, ns / 1000.0, val);
        writeln!(f, "{}\t{}\t10000000\t{:.2}\t{:.4}\t{:.2}", method, est_name, ns, ns / 1000.0, val).unwrap();
    };

    measure("OS-BMW-5b(k24)", "MLE", 500, Box::new(|| sk_5b.estimate_ml(&pmf_5b).0));
    measure("ELL(2,24)", "MLE", 500, Box::new(|| sk_ell.estimate_ml(&pmf_ell).0));
    measure("ULL", "MLE", 500, Box::new(|| sk_ull.estimate_ml(&pmf_ull).0));
    measure("ULL", "FGRA", 4000, Box::new(|| sk_ull.estimate_fgra()));
    println!("Wrote FGRA single-sketch micro -> {}", out_path.display());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut mode = "ell-filter".to_string();
    let mut out_dir = PathBuf::from(".");
    let mut tag = format!("{}_default", std::env::consts::ARCH);
    let mut append = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            "--out" => { out_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--tag" => { tag = args[i + 1].clone(); i += 2; }
            "--append" => { append = true; i += 1; }
            _ => { i += 1; }
        }
    }
    fs::create_dir_all(&out_dir).expect("create out_dir");
    warmup_cpu();
    match mode.as_str() {
        "ell-filter" => run_ell_filter(&out_dir),
        "modcard" => run_modcard(&out_dir),
        "non-avx512-merge" => run_non_avx512_merge(&out_dir, &tag, append),
        "ull-fgra-micro" => run_ull_fgra_micro(&out_dir),
        other => panic!("Unknown --mode {}", other),
    }
}
