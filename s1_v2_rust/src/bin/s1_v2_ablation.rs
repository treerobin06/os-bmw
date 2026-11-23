//! Generalized Theorem 2.1 & Theorem 3 Empirical Ablation Suite (`s1_v2_ablation`)
//!
//! Sweeps:
//!   Part A: Equal-byte (1 KB & 4 KB) geometry (g in {128, 256, 384, 512}) and
//!           register width (4-bit R=7,d=1 vs 5-bit R=8,d=2) across n in {1e4, 1e5, 1e6, 1e7}
//!           with t=200 trials, reporting theoretical Fisher MVP, empirical RMSE/MVP,
//!           empirical overflow-slot saturation fraction (0 evictions), and in-place 5b latency.
//!   Part B: Tail-slot threshold k (= C_ex + 1) sweep across k in {1, 2, 4, 8, 12, 16, 20, 24, 32}
//!           for g in {256, 512} (5b R=8,d=2 & 4b R=7,d=1) at 4 KB nominal budget & n=1e6.

use s1_v2_rust::hash::{rng_u01, splitmix64, wyhash64, HEADER_BYTES};
use s1_v2_rust::osbmw_inplace::{HashMode5b, OsBmw5bInPlace};
use s1_v2_rust::osbmw_ref::{OsBmwBlock, OsBmwRefSketch};
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn block_bytes_width(g: usize, k: usize, slot_bits: usize) -> usize {
    let slot_bytes = if slot_bits == 4 { g / 2 } else { (g * 5) / 8 };
    1 + slot_bytes + (k - 1)
}

fn nearest_nblk_pow2(target_bytes: usize, blk_bytes: usize) -> usize {
    let mut best_nblk = 1usize;
    let mut best_diff = usize::MAX;
    for p in 0..=10 {
        let nb = 1usize << p;
        let tot = HEADER_BYTES + nb * blk_bytes;
        let d = tot.abs_diff(target_bytes);
        if d < best_diff {
            best_diff = d;
            best_nblk = nb;
        }
    }
    best_nblk
}

fn sample_ref_sketch_at_n(
    n_blk: usize,
    g: usize,
    k: usize,
    r_win: u8,
    d: u8,
    dist: DistSpec,
    pmf: &PreimagePmf,
    n: f64,
    rng: &mut u64,
) -> (OsBmwRefSketch, f64) {
    let mut sk = OsBmwRefSketch::new(n_blk, g, k, r_win, d, 6, dist, true).unwrap();
    let lam = n / ((n_blk * g) as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    let mut us = vec![0u8; g];
    let mut hs = vec![0u32; g];
    let mut ov_slots_used = 0usize;
    for blk in 0..n_blk {
        for r in 0..g {
            let u01 = rng_u01(rng);
            let mut lo = 0usize;
            let mut hi = km;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if cdf[mid] < u01 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let u = lo;
            us[r] = u as u8;
            let mut h = 0u32;
            if u >= 2 {
                let nb = (u - 1).min(d as usize);
                for delta in 1..=nb {
                    let x = lam * pmf.probs[u - delta];
                    let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                    if rng_u01(rng) < p1 {
                        h |= 1u32 << (delta - 1);
                    }
                }
            }
            hs[r] = h;
        }
        let b = OsBmwBlock::pack_from_uh(g, k, r_win, d, &us, &hs);
        ov_slots_used += b.ov_u.len();
        sk.blocks[blk] = b;
    }
    let ov_cap = n_blk * (k.saturating_sub(1));
    let ov_fill = if ov_cap > 0 { (ov_slots_used as f64) / (ov_cap as f64) } else { 0.0 };
    (sk, ov_fill)
}

fn bench_inplace_5b<const G: usize, const K: usize>(n_blk: usize) -> (f64, f64) {
    let mode = if G == 384 { HashMode5b::Div3 } else { HashMode5b::Pow2 };
    let mut sk = OsBmw5bInPlace::<G, K, false>::new(n_blk, mode);
    for i in 0..100_000u64 {
        sk.add_hash_64(wyhash64(i, 0x1234));
    }
    let repeats = 7usize;
    let ops = 200_000usize;
    let mut upd_samples = Vec::with_capacity(repeats);
    for r in 0..repeats {
        let base = (r as u64 + 1) * 1_000_000;
        let t0 = Instant::now();
        for i in 0..ops as u64 {
            black_box(sk.add_hash_64(wyhash64(base + i, 0xbeef)));
        }
        upd_samples.push(t0.elapsed().as_nanos() as f64 / ops as f64);
    }
    upd_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sk2 = OsBmw5bInPlace::<G, K, false>::new(n_blk, mode);
    for i in 0..100_000u64 {
        sk2.add_hash_64(wyhash64(i, 0x5678));
    }
    let m_reps = 200usize;
    let t0 = Instant::now();
    for _ in 0..m_reps {
        let mut tmp = sk.clone();
        tmp.merge_in_place(&sk2).unwrap();
        black_box(&tmp);
    }
    let total_clone_merge = t0.elapsed().as_nanos() as f64 / m_reps as f64;
    let t1 = Instant::now();
    for _ in 0..m_reps {
        let tmp = sk.clone();
        black_box(&tmp);
    }
    let clone_only = t1.elapsed().as_nanos() as f64 / m_reps as f64;
    let merge_us = (total_clone_merge - clone_only).max(0.0) / 1000.0;
    (upd_samples[repeats / 2], merge_us)
}

fn measure_5b_latency(g: usize, k: usize, n_blk: usize) -> (f64, f64) {
    match (g, k) {
        (128, 4) => bench_inplace_5b::<128, 4>(n_blk),
        (128, 6) => bench_inplace_5b::<128, 6>(n_blk),
        (128, 8) => bench_inplace_5b::<128, 8>(n_blk),
        (256, 2) => bench_inplace_5b::<256, 2>(n_blk),
        (256, 4) => bench_inplace_5b::<256, 4>(n_blk),
        (256, 8) => bench_inplace_5b::<256, 8>(n_blk),
        (256, 12) => bench_inplace_5b::<256, 12>(n_blk),
        (256, 16) => bench_inplace_5b::<256, 16>(n_blk),
        (256, 24) => bench_inplace_5b::<256, 24>(n_blk),
        (384, 16) => bench_inplace_5b::<384, 16>(n_blk),
        (384, 20) => bench_inplace_5b::<384, 20>(n_blk),
        (512, 2) => bench_inplace_5b::<512, 2>(n_blk),
        (512, 4) => bench_inplace_5b::<512, 4>(n_blk),
        (512, 8) => bench_inplace_5b::<512, 8>(n_blk),
        (512, 12) => bench_inplace_5b::<512, 12>(n_blk),
        (512, 16) => bench_inplace_5b::<512, 16>(n_blk),
        (512, 20) => bench_inplace_5b::<512, 20>(n_blk),
        (512, 24) => bench_inplace_5b::<512, 24>(n_blk),
        (512, 32) => bench_inplace_5b::<512, 32>(n_blk),
        _ => (f64::NAN, f64::NAN),
    }
}

fn run_cell(
    part: &str,
    budget_label: &str,
    label: &str,
    slot_bits: usize,
    g: usize,
    k: usize,
    r_win: u8,
    d: u8,
    n_blk: usize,
    card: f64,
    trials: usize,
    n_threads: usize,
    theory_mvp: f64,
    upd_ns: f64,
    merge_us: f64,
) -> String {
    let blk_bytes = block_bytes_width(g, k, slot_bits);
    let tot_bytes = HEADER_BYTES + n_blk * blk_bytes;
    let total_regs = n_blk * g;
    let dist = if g == 384 {
        DistSpec::Pow2Div3 { p_tot: 7 + n_blk.trailing_zeros() }
    } else {
        DistSpec::Pow2 { w_bits: 64 - n_blk.trailing_zeros() - g.trailing_zeros(), max_cap: 63 }
    };
    let pmf = Arc::new(PreimagePmf::build(dist));
    let is_stream = card <= 1e5;
    let card_u64 = card as u64;

    let mut errs = Vec::with_capacity(trials);
    let mut fills = Vec::with_capacity(trials);
    let per_t = (trials + n_threads - 1) / n_threads;
    thread::scope(|s| {
        let mut hs = Vec::new();
        for tid in 0..n_threads {
            let st = tid * per_t;
            let en = (st + per_t).min(trials);
            if st >= en {
                continue;
            }
            let pmf_c = Arc::clone(&pmf);
            hs.push(s.spawn(move || {
                let mut loc_e = Vec::with_capacity(en - st);
                let mut loc_f = Vec::with_capacity(en - st);
                for tr in st..en {
                    let seed = splitmix64(
                        0xABC0_0000_0000_0000u64
                            ^ ((tr as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                            ^ (card.to_bits().wrapping_mul(0xbf58_476d_1ce4_e5b9))
                            ^ ((g as u64) << 16)
                            ^ ((k as u64) << 8)
                            ^ (slot_bits as u64),
                    );
                    let (sk, ov_fill) = if is_stream {
                        let mut sref = OsBmwRefSketch::new(n_blk, g, k, r_win, d, 6, dist, true).unwrap();
                        for i in 0..card_u64 {
                            sref.add_hash_64(wyhash64(i, seed), &pmf_c);
                        }
                        let mut ov_used = 0usize;
                        for b in &sref.blocks {
                            ov_used += b.ov_u.len();
                        }
                        let cap = n_blk * (k.saturating_sub(1));
                        let f = if cap > 0 { (ov_used as f64) / (cap as f64) } else { 0.0 };
                        (sref, f)
                    } else {
                        let mut rng = seed;
                        sample_ref_sketch_at_n(n_blk, g, k, r_win, d, dist, &pmf_c, card, &mut rng)
                    };
                    let est = sk.estimate_ml(&pmf_c).0;
                    loc_e.push((est - card) / card);
                    loc_f.push(ov_fill);
                }
                (loc_e, loc_f)
            }));
        }
        for h in hs {
            let (le, lf) = h.join().unwrap();
            errs.extend_from_slice(&le);
            fills.extend_from_slice(&lf);
        }
    });

    let nt = errs.len() as f64;
    let bias = errs.iter().sum::<f64>() / nt;
    let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
    let rmse = mse.sqrt();
    let emp_mvp = (tot_bytes as f64 * 8.0) * mse;
    let emp_mvp_payload = ((n_blk * blk_bytes) as f64 * 8.0) * mse;
    let mean_ov_fill = fills.iter().sum::<f64>() / nt;

    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3e}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.3}",
        part, budget_label, label, slot_bits, g, k, k - 1, r_win, d, n_blk, blk_bytes, tot_bytes,
        card, total_regs, theory_mvp, emp_mvp_payload, emp_mvp, bias * 100.0, rmse * 100.0,
        mean_ov_fill * 100.0, upd_ns, merge_us
    )
}

fn main() {
    let out_dir = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");
    let n_threads = env::var("S1_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .max(1);

    let out_path = out_dir.join("m2_geometry_ablation.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(
        f,
        "part\tbudget\tconfig\tslot_bits\tg\tk\tc_ex\tr_win\td\tn_blk\tblock_bytes\ttotal_bytes\tcardinality\ttotal_regs\ttheory_mvp_payload\temp_mvp_payload\temp_mvp_total\trel_bias_pct\trmse_pct\tov_buffer_fill_pct\tupd_ns\tmerge_us"
    ).unwrap();

    // Part A: Equal-byte (1 KB & 4 KB) g in {128, 256, 384, 512}, 4-bit vs 5-bit, n in {1e4, 1e5, 1e6, 1e7}
    let configs_a: Vec<(&str, usize, usize, usize, u8, u8)> = vec![
        ("5b_g512_k24", 5, 512, 24, 8, 2),
        ("4b_g512_k24", 4, 512, 24, 7, 1),
        ("5b_g384_k16", 5, 384, 16, 8, 2),
        ("4b_g384_k16", 4, 384, 16, 7, 1),
        ("5b_g256_k12", 5, 256, 12, 8, 2),
        ("4b_g256_k12", 4, 256, 12, 7, 1),
        ("5b_g128_k8",  5, 128, 8,  8, 2),
        ("4b_g128_k8",  4, 128, 8,  7, 1),
    ];

    let cards = [1e4_f64, 1e5, 1e6, 1e7];
    let trials_a = 200usize;

    for &(blab, target_b) in &[("1KB", 1024usize), ("4KB", 4096usize)] {
        println!("[Part A] Budget {} (target={} B)...", blab, target_b);
        for &(cname, sbits, g, k, r_win, d) in &configs_a {
            let blk_b = block_bytes_width(g, k, sbits);
            let n_blk = nearest_nblk_pow2(target_b, blk_b);
            let dist = if g == 384 {
                DistSpec::Pow2Div3 { p_tot: 7 + n_blk.trailing_zeros() }
            } else {
                DistSpec::Pow2 { w_bits: 64 - n_blk.trailing_zeros() - g.trailing_zeros(), max_cap: 63 }
            };
            let pmf = PreimagePmf::build(dist);
            let (j_th, _, _) = pmf.fisher_osbmw_periodic(g, k, r_win as usize, d as usize, 2.0, 1e6, 32);
            let theory_mvp = (blk_b as f64 * 8.0) / (g as f64) / j_th;
            let (upd_ns, merge_us) = if sbits == 5 { measure_5b_latency(g, k, n_blk) } else { (f64::NAN, f64::NAN) };
            for &c in &cards {
                let row = run_cell(
                    "A_Geom_Width", blab, cname, sbits, g, k, r_win, d, n_blk, c, trials_a,
                    n_threads, theory_mvp, upd_ns, merge_us,
                );
                writeln!(f, "{}", row).unwrap();
            }
            f.flush().unwrap();
        }
    }

    // Part B: Threshold k (= C_ex + 1) sweep at ~4 KB & n=1e6 (g=512 5b, g=256 5b vs 4b)
    println!("[Part B] Tail-slot k (= C_ex + 1) threshold sweep at 4 KB, n=1e6...");
    let ks_512 = [1usize, 2, 4, 8, 12, 16, 20, 24, 32];
    for &k in &ks_512 {
        let g = 512usize;
        let n_blk = 8usize;
        let blk_b = block_bytes_width(g, k, 5);
        let dist = DistSpec::Pow2 { w_bits: 64 - 3 - 9, max_cap: 63 };
        let pmf = PreimagePmf::build(dist);
        let (j_th, _, _) = pmf.fisher_osbmw_periodic(g, k, 8, 2, 2.0, 1e6, 32);
        let theory_mvp = (blk_b as f64 * 8.0) / (g as f64) / j_th;
        let (upd_ns, merge_us) = measure_5b_latency(g, k, n_blk);
        let cname = format!("5b_g512_k{}", k);
        let row = run_cell(
            "B_Threshold_K", "4KB_nom", &cname, 5, g, k, 8, 2, n_blk, 1e6, 400,
            n_threads, theory_mvp, upd_ns, merge_us,
        );
        writeln!(f, "{}", row).unwrap();
        f.flush().unwrap();
    }

    let ks_256 = [1usize, 2, 4, 8, 12, 16, 24];
    for &(sbits, r_win, d) in &[(5usize, 8u8, 2u8), (4usize, 7u8, 1u8)] {
        for &k in &ks_256 {
            let g = 256usize;
            let n_blk = 16usize;
            let blk_b = block_bytes_width(g, k, sbits);
            let dist = DistSpec::Pow2 { w_bits: 64 - 4 - 8, max_cap: 63 };
            let pmf = PreimagePmf::build(dist);
            let (j_th, _, _) = pmf.fisher_osbmw_periodic(g, k, r_win as usize, d as usize, 2.0, 1e6, 32);
            let theory_mvp = (blk_b as f64 * 8.0) / (g as f64) / j_th;
            let (upd_ns, merge_us) = if sbits == 5 { measure_5b_latency(g, k, n_blk) } else { (f64::NAN, f64::NAN) };
            let cname = format!("{}b_g256_k{}", sbits, k);
            let row = run_cell(
                "B_Threshold_K", "4KB_nom", &cname, sbits, g, k, r_win, d, n_blk, 1e6, 400,
                n_threads, theory_mvp, upd_ns, merge_us,
            );
            writeln!(f, "{}", row).unwrap();
            f.flush().unwrap();
        }
    }

    println!("DONE m2_geometry_ablation -> {}", out_path.display());
}
