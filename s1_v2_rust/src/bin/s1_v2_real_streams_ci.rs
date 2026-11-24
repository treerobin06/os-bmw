//! E-10: 200-Seed Real & Skewed Stream Evaluation with Bootstrap 95% CIs (`s1_v2_real_streams_ci`).
//!
//! Runs all 8 public & synthetic streams at 1 KiB and 4 KiB across 200 independent WyHash64 seeds,
//! computing bootstrap 95% CIs on RMSE and RAM MVP plus paired OS-BMW minus ELL(2,24) CIs.

use s1_v2_rust::baselines::*;
use s1_v2_rust::hash::{rng_u01, splitmix64, wyhash64};
use s1_v2_rust::osbmw_inplace::*;
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;

fn load_snap_edges(path: &Path, max_lines: usize) -> Option<(Vec<u64>, usize)> {
    if !path.exists() {
        return None;
    }
    let mut child = Command::new("gzip")
        .args(["-dc", path.to_str().unwrap()])
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let reader = BufReader::new(stdout);
    let mut items = Vec::with_capacity(max_lines.min(20_000_000));
    let mut uniq: HashSet<u64> = HashSet::new();
    for line in reader.lines() {
        let l = match line {
            Ok(s) => s,
            Err(_) => break,
        };
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        let mut it = l.split_whitespace();
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(u), Ok(v)) = (a.parse::<u64>(), b.parse::<u64>()) {
                let key = (u << 32) ^ v.wrapping_mul(0x9e37_79b9);
                items.push(key);
                uniq.insert(key);
                if items.len() >= max_lines {
                    break;
                }
            }
        }
    }
    let _ = child.wait();
    Some((items, uniq.len()))
}

fn load_ecoli_kmers(path: &Path) -> Option<(Vec<u64>, usize)> {
    if !path.exists() {
        return None;
    }
    let mut child = Command::new("gzip")
        .args(["-dc", path.to_str().unwrap()])
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let reader = BufReader::new(stdout);
    let mut items = Vec::with_capacity(5_000_000);
    let mut uniq: HashSet<u64> = HashSet::with_capacity(5_000_000);
    let mask62: u64 = (1u64 << 62) - 1;
    let mut fwd = 0u64;
    let mut rev = 0u64;
    let mut len = 0usize;
    for line in reader.lines() {
        let l = match line {
            Ok(s) => s,
            Err(_) => break,
        };
        if l.starts_with('>') {
            fwd = 0; rev = 0; len = 0;
            continue;
        }
        for &ch in l.as_bytes() {
            let base = match ch {
                b'A' | b'a' => 0u64,
                b'C' | b'c' => 1u64,
                b'G' | b'g' => 2u64,
                b'T' | b't' => 3u64,
                _ => { fwd = 0; rev = 0; len = 0; continue; }
            };
            fwd = ((fwd << 2) | base) & mask62;
            rev = (rev >> 2) | ((3 - base) << 60);
            len += 1;
            if len >= 31 {
                let canon = fwd.min(rev);
                items.push(canon);
                uniq.insert(canon);
            }
        }
    }
    let _ = child.wait();
    Some((items, uniq.len()))
}

fn boot_rmse_and_mvp_ci(errs: &[f64], bytes: f64, salt: u64) -> (f64, f64, f64, f64, f64, f64, f64) {
    let n = errs.len();
    let nf = n as f64;
    let bias = errs.iter().sum::<f64>() / nf;
    let mse = errs.iter().map(|e| e * e).sum::<f64>() / nf;
    let rmse = mse.sqrt();
    let mvp = (bytes * 8.0) * mse;

    const B: usize = 2000;
    let mut b_rmse = Vec::with_capacity(B);
    let mut b_mvp = Vec::with_capacity(B);
    let mut rng = splitmix64(0xB007_1000 ^ salt);
    for _ in 0..B {
        let mut s2 = 0.0_f64;
        for _ in 0..n {
            rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let idx = (splitmix64(rng) as usize) % n;
            let e = errs[idx];
            s2 += e * e;
        }
        let bm = s2 / nf;
        b_rmse.push(bm.sqrt());
        b_mvp.push((bytes * 8.0) * bm);
    }
    b_rmse.sort_by(|a, b| a.partial_cmp(b).unwrap());
    b_mvp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_i = (0.025 * (B as f64)).floor() as usize;
    let hi_i = ((0.975 * (B as f64)).ceil() as usize).min(B - 1);
    (bias, rmse, b_rmse[lo_i], b_rmse[hi_i], mvp, b_mvp[lo_i], b_mvp[hi_i])
}

fn paired_diff_rmse_ci(a_errs: &[f64], b_errs: &[f64], salt: u64) -> (f64, f64, f64, usize) {
    let n = a_errs.len();
    assert_eq!(n, b_errs.len());
    let nf = n as f64;
    let r_a = (a_errs.iter().map(|e| e * e).sum::<f64>() / nf).sqrt();
    let r_b = (b_errs.iter().map(|e| e * e).sum::<f64>() / nf).sqrt();
    let diff = r_a - r_b;
    let a_better_seeds = a_errs.iter().zip(b_errs.iter()).filter(|(x, y)| x.abs() < y.abs()).count();

    const B: usize = 2000;
    let mut b_diff = Vec::with_capacity(B);
    let mut rng = splitmix64(0x9A17_2000 ^ salt);
    for _ in 0..B {
        let mut sa = 0.0_f64;
        let mut sb = 0.0_f64;
        for _ in 0..n {
            rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let idx = (splitmix64(rng) as usize) % n;
            sa += a_errs[idx] * a_errs[idx];
            sb += b_errs[idx] * b_errs[idx];
        }
        b_diff.push((sa / nf).sqrt() - (sb / nf).sqrt());
    }
    b_diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_i = (0.025 * (B as f64)).floor() as usize;
    let hi_i = ((0.975 * (B as f64)).ceil() as usize).min(B - 1);
    (diff, b_diff[lo_i], b_diff[hi_i], a_better_seeds)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let data_dir = args
        .get(2)
        .map(PathBuf::from)
        .or_else(|| env::var("S1_DATA_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("./data"));
    let n_threads: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| env::var("S1_THREADS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(32)
        .max(1);
    let n_seeds: usize = env::var("S1_REAL_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    fs::create_dir_all(&out_dir).expect("create out_dir");

    println!("=== [E-10] 200-Seed Real & Skewed Streams CI Audit (seeds={}, threads={}) ===", n_seeds, n_threads);
    let mut workloads: Vec<(String, Vec<u64>, usize)> = Vec::new();
    workloads.push(("Synthetic_Uniform_2M".to_string(), (0..2_000_000u64).collect(), 2_000_000));
    for &(label, alpha) in &[("Zipf_a0.8_5M", 0.8_f64), ("Zipf_a1.2_5M", 1.2_f64), ("Zipf_a1.6_5M", 1.6_f64)] {
        let dom = 2_000_000usize;
        let mut rng = splitmix64(label.len() as u64 ^ alpha.to_bits());
        let mut stream = Vec::with_capacity(5_000_000);
        let mut uniq = HashSet::new();
        let one_m_a = 1.0 - alpha;
        let dom_pow = (dom as f64).powf(one_m_a);
        for _ in 0..5_000_000 {
            let u = rng_u01(&mut rng);
            let k = (u * (dom_pow - 1.0) + 1.0).powf(1.0 / one_m_a).floor() as u64;
            let key = k.clamp(1, dom as u64);
            stream.push(key);
            uniq.insert(key);
        }
        workloads.push((label.to_string(), stream, uniq.len()));
    }
    {
        let mut stream = Vec::with_capacity(5_000_000);
        for i in 0..5_000_000u64 { stream.push(splitmix64(i) % 10_000); }
        workloads.push(("HighDup_5M_10kDistinct".to_string(), stream, 10_000));
    }
    if let Some((items, distinct)) = load_snap_edges(&data_dir.join("web-Google.txt.gz"), 6_000_000) {
        workloads.push(("SNAP_webGoogle_Edges".to_string(), items, distinct));
    }
    if let Some((items, distinct)) = load_snap_edges(&data_dir.join("cit-Patents.txt.gz"), 17_000_000) {
        workloads.push(("SNAP_citPatents_Edges".to_string(), items, distinct));
    }
    if let Some((items, distinct)) = load_ecoli_kmers(&data_dir.join("ecoli_k12.fna.gz")) {
        workloads.push(("NCBI_Ecoli_K12_31mers".to_string(), items, distinct));
    }
    println!("  Loaded {} workloads from {:?}", workloads.len(), data_dir);

    let out_path = out_dir.join("m2_real_streams_ci.tsv");
    let mut out = File::create(&out_path).unwrap();
    writeln!(
        out,
        "workload\ttotal_events\ttrue_distinct\tbudget\tmethod\tram_bytes\tseeds\trel_bias_pct\trmse_pct\trmse_ci95_lo\trmse_ci95_hi\tmvp_ram\tmvp_ci95_lo\tmvp_ci95_hi\tdiff_vs_ell24_rmse_pct\tdiff_ci95_lo\tdiff_ci95_hi\twin_seeds_vs_ell24"
    ).unwrap();

    for (w_idx, (wname, items, distinct)) in workloads.iter().enumerate() {
        let items_arc = Arc::new(items.clone());
        let d_f64 = *distinct as f64;
        for &(blab, nblk_256, nblk_5b, p_ell, p_ull, avll_w) in &[("1KB", 4usize, 2usize, 8u32, 10u32, 128usize), ("4KB", 16usize, 8usize, 10u32, 12u32, 512usize)] {
            let pmf_256 = Arc::new(PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + nblk_256.trailing_zeros() }));
            let pmf_5b = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 9 - nblk_5b.trailing_zeros(), max_cap: 63 }));
            let pmf_ell = Arc::new(PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - p_ell }));
            let pmf_ull = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - p_ull, max_cap: 63 }));

            let run_method = |f: &(dyn Fn(&[u64], u64) -> (f64, usize) + Sync)| -> (Vec<f64>, usize) {
                let mut errs = vec![0.0_f64; n_seeds];
                let mut bytes = 0usize;
                let per_t = (n_seeds + n_threads - 1) / n_threads;
                thread::scope(|s| {
                    let mut hs = Vec::new();
                    for tid in 0..n_threads {
                        let st = tid * per_t;
                        let en = (st + per_t).min(n_seeds);
                        if st >= en { continue; }
                        let data = Arc::clone(&items_arc);
                        hs.push(s.spawn(move || {
                            let mut loc = Vec::with_capacity(en - st);
                            let mut b0 = 0usize;
                            for sd in st..en {
                                let seed = splitmix64(0x9999_1111 + sd as u64);
                                let (e, b) = f(&data, seed);
                                loc.push((sd, e));
                                b0 = b;
                            }
                            (loc, b0)
                        }));
                    }
                    for h in hs {
                        let (loc, b0) = h.join().unwrap();
                        bytes = b0;
                        for (sd, e) in loc { errs[sd] = e; }
                    }
                });
                (errs, bytes)
            };

            let (e_e24, b_e24) = run_method(&|data, seed| {
                let mut sk = Ell24Sketch::new(p_ell);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ell).0 - d_f64) / d_f64, sk.total_bytes())
            });
            let (e_256, b_256) = run_method(&|data, seed| {
                let mut sk = OsBmw256::new(nblk_256, HashMode5b::Div3);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_256).0 - d_f64) / d_f64, sk.total_bytes())
            });
            let (e_5b, b_5b) = run_method(&|data, seed| {
                let mut sk = OsBmw5b::new(nblk_5b, HashMode5b::Pow2);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_5b).0 - d_f64) / d_f64, sk.total_bytes())
            });
            let (e_ull, b_ull) = run_method(&|data, seed| {
                let mut sk = UllSketch::new(p_ull);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ull).0 - d_f64) / d_f64, sk.total_bytes())
            });
            let (e_hbs, b_hbs) = run_method(&|data, seed| {
                let mut sk = HbsSketch::new(p_ull);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ull).0 - d_f64) / d_f64, sk.ram_bytes())
            });
            let (e_avll, b_avll) = run_method(&|data, seed| {
                let mut sk = AvllSketch::new_words(avll_w);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml_exact_mod().0 - d_f64) / d_f64, sk.total_bytes())
            });

            let methods = [
                ("OS-BMW-5b-256B", &e_256, b_256),
                ("OS-BMW-5b", &e_5b, b_5b),
                ("ELL(2,24)", &e_e24, b_e24),
                ("ULL", &e_ull, b_ull),
                ("HBS", &e_hbs, b_hbs),
                ("AVLL", &e_avll, b_avll),
            ];
            for (m_idx, &(mname, errs, bytes)) in methods.iter().enumerate() {
                let salt = ((w_idx as u64) << 20) ^ ((b_256 as u64) << 8) ^ (m_idx as u64);
                let (bias, rmse, r_lo, r_hi, mvp, m_lo, m_hi) = boot_rmse_and_mvp_ci(errs, bytes as f64, salt);
                let (d_rmse, d_lo, d_hi, wins) = paired_diff_rmse_ci(errs, &e_e24, salt ^ 0xFFFF);
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
                    wname, items.len(), distinct, blab, mname, bytes, n_seeds,
                    bias * 100.0, rmse * 100.0, r_lo * 100.0, r_hi * 100.0,
                    mvp, m_lo, m_hi, d_rmse * 100.0, d_lo * 100.0, d_hi * 100.0, wins
                ).unwrap();
            }
            out.flush().unwrap();
            if blab == "4KB" {
                let (_, r256, _, _, m256, _, _) = boot_rmse_and_mvp_ci(&e_256, b_256 as f64, 1);
                let (_, re24, _, _, me24, _, _) = boot_rmse_and_mvp_ci(&e_e24, b_e24 as f64, 2);
                let (dr, dlo, dhi, wins) = paired_diff_rmse_ci(&e_256, &e_e24, 3);
                println!(
                    "  {:24} (4KB) | 256B={:.3}% (MVP {:.3}) vs ELL={:.3}% (MVP {:.3}) | diff={:+.3}% [{:+.3}%, {:+.3}%] | wins={}/{}",
                    wname, r256 * 100.0, m256, re24 * 100.0, me24, dr * 100.0, dlo * 100.0, dhi * 100.0, wins, n_seeds
                );
            }
        }
    }
    println!("DONE E-10 -> {}", out_path.display());
}
