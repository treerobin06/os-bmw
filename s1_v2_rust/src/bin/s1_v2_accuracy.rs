//! Milestone M2 Runner (Run on `numactl --cpunodebind=1 --membind=1 taskset -c 32-63`):
//!   Part A: Equal-Byte Full-Range Accuracy Sweep `{0.5, 1, 2, 4, 8, 16} KB` across
//!           `n in {1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12, 1e15, 1e18, 2^64}`
//!           (exact streaming <= 1e5 + Ertl ELL §5.1 exact waiting-time jump up to 2^64!),
//!           covering OS-BMW-{5b, 5b-Mask, 5b-256B, g128+Sparse, Chunk} vs ALL 8 baselines
//!           (HLL++, ULL, ELL(2,20), ELL(2,24), CPC RAM/Wire, HLLL, HLL_4 Dyn/Fixed, AVLL).
//!   Part B: High-precision 16,000-trial Poissonized MC verifying theoretical MVP to +/- 0.03 SE
//!           (including OS-BMW-Rank 1,277 B Wire codec verified end-to-end!).
//!   Part C: Real-world & skewed streams at 1 KB and 4 KB (32 independent hash seeds):
//!           Synthetic Uniform (1e7), Zipf (alpha=0.8, 1.2, 1.6), High-Dup (20M events / 1e4 distinct),
//!           SNAP web-Google edges, SNAP cit-Patents edges, NCBI E. coli K-12 DNA 31-mers!
//!   Part D: Single 64-bit vs 128-bit hash ablation across budgets up to n = 2^64.

use s1_v2_rust::baselines::*;
use s1_v2_rust::hash::{rng_u01, splitmix64, wyhash128, wyhash64, HEADER_BYTES};
use s1_v2_rust::osbmw_inplace::*;
use s1_v2_rust::osbmw_rank::OsBmwRankCodec;
use s1_v2_rust::osbmw_ref::OsBmwBlock;
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;

/// Ertl §5.1 exact Poissonized waiting-time jump sampler:
/// Directly samples a sketch's exact state at ANY cardinality `n` up to `2^64` in O(m * log K)
/// using the exact preimage PMF `pmf` (independent across buckets/levels under Poissonization)!
fn sample_ell24_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> Ell24Sketch {
    let mut sk = Ell24Sketch::new(p);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        let u = lo;
        if u > 0 {
            let mut reg = (u as u32) << Ell24Sketch::D;
            let nb = (u - 1).min(Ell24Sketch::D as usize);
            for delta in 1..=nb {
                let x = lam * pmf.probs[u - delta];
                let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                if rng_u01(rng) < p1 {
                    reg |= 1u32 << (Ell24Sketch::D as usize - delta);
                }
            }
            sk.regs[r] = reg;
        }
    }
    sk
}

fn sample_ell20_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> Ell20Sketch {
    let mut sk = Ell20Sketch::new(p);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        let u = lo;
        if u > 0 {
            let mut reg = (u as u32) << Ell20Sketch::D;
            let nb = (u - 1).min(Ell20Sketch::D as usize);
            for delta in 1..=nb {
                let x = lam * pmf.probs[u - delta];
                let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                if rng_u01(rng) < p1 {
                    reg |= 1u32 << (Ell20Sketch::D as usize - delta);
                }
            }
            sk.set_reg(r, reg);
        }
    }
    sk
}

fn sample_osbmw5b_at_n<const G: usize, const K: usize, const HAS_MASK: bool>(
    n_blk: usize,
    mode: HashMode5b,
    pmf: &PreimagePmf,
    n: f64,
    rng: &mut u64,
) -> OsBmw5bInPlace<G, K, HAS_MASK> {
    let mut sk = OsBmw5bInPlace::<G, K, HAS_MASK>::new(n_blk, mode);
    let lam = n / ((n_blk * G) as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for blk in 0..n_blk {
        for r in 0..G {
            let u01 = rng_u01(rng);
            let mut lo = 0usize;
            let mut hi = km;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
            }
            let u = lo;
            if u > 0 {
                sk.add_level(blk, r, u as u8);
                let nb = (u - 1).min(2);
                for delta in 1..=nb {
                    let x = lam * pmf.probs[u - delta];
                    let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                    if rng_u01(rng) < p1 {
                        sk.add_level(blk, r, (u - delta) as u8);
                    }
                }
            }
        }
    }
    sk
}

fn sample_chunk_at_n(n_blk: usize, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> OsBmwChunkInPlace {
    let mut sk = OsBmwChunkInPlace::new(n_blk);
    let lam = n / ((n_blk * CHUNK_G) as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    let mut vals = [0u8; CHUNK_G];
    for blk in 0..n_blk {
        for r in 0..CHUNK_G {
            let u01 = rng_u01(rng);
            let mut lo = 0usize;
            let mut hi = km;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
            }
            let u = lo;
            let mut h = 0u8;
            if u >= 2 {
                let nb = (u - 1).min(2);
                for delta in 1..=nb {
                    let x = lam * pmf.probs[u - delta];
                    let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                    if rng_u01(rng) < p1 {
                        h |= 1u8 << (delta - 1);
                    }
                }
            }
            vals[r] = ((u as u8) << 2) | h;
        }
        sk.pack_block_u8(blk, &vals, 0);
    }
    sk
}

fn sample_ull_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> UllSketch {
    let mut sk = UllSketch::new(p);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        let u = lo;
        if u > 0 {
            let mut b = (u as u8) << 2;
            if u >= 2 {
                let x = lam * pmf.probs[u - 1];
                if rng_u01(rng) < if x > 700.0 { 1.0 } else { -(-x).exp_m1() } { b |= 2; }
            }
            if u >= 3 {
                let x = lam * pmf.probs[u - 2];
                if rng_u01(rng) < if x > 700.0 { 1.0 } else { -(-x).exp_m1() } { b |= 1; }
            }
            sk.buf[HEADER_BYTES + r] = b;
        }
    }
    sk
}

fn sample_hll6_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> Hll6Sketch {
    let mut sk = Hll6Sketch::new(p);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        if lo > 0 { sk.set_reg(r, lo as u8); }
    }
    sk
}

fn sample_avll_at_n(n_words: usize, n: f64, rng: &mut u64) -> AvllSketch {
    // True chronological Poissonized jump matching `AvllSketch::add_hash_64` single-64-bit semantics:
    //   - Low 32 bits select bucket `r = ((h_lo * m) >> 32)` (exact interval [L_r, U_r))
    //   - Full 64-bit word sets `abs_nlz = (h.leading_zeros() + 1).min(63)`
    // Uses exact joint preimage rate `P(r, k) = C(r, k) / 2^64` per bucket (0 for unreachable (r, k))
    // and non-clamping uniform pre-advance for deeply saturated levels <= min(k_lo - 1, 32).
    const INV_2_64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
    let mut sk = AvllSketch::new_words(n_words);
    let m = sk.m;
    let lam = n / (m as f64);
    let log2_lam = (lam.max(1.0)).log2().floor() as i32;
    let k_lo = (log2_lam - 6).max(1) as usize;
    // Levels 1..=32 are reachable by EVERY bucket with rate ~2^-k / m; for k <= k_lo - 1,
    // n * rate >= 2^6 = 64 so P(missed) < 1e-23. Pre-advance cleanly without MAX_EXP clamping!
    let k_pre = (k_lo - 1).min(32);
    if k_pre > 0 {
        sk.prefill_uniform_saturated(k_pre as u8);
    }
    // Collect active first-arrival events (time <= n) using exact joint (r, k) rates & replay chronologically!
    let mut events: Vec<(f64, u32, u8)> = Vec::with_capacity(m * 8);
    for r in 0..m {
        let (l_r, u_r) = AvllSketch::bucket_lo32_bounds(r, m);
        for k in (k_pre + 1)..=63 {
            let c = AvllSketch::joint_preimage_count(k, l_r, u_r);
            if c == 0 {
                continue;
            }
            let rate = (c as f64) * INV_2_64;
            let u01 = rng_u01(rng);
            let t = -u01.ln() / rate;
            if t <= n {
                events.push((t, r as u32, k as u8));
            }
        }
    }
    events.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, r, k) in events {
        sk.add_level(r as usize, k);
    }
    sk
}

/// Exact Poissonized coupon-matrix jump sampler for `CpcSketch` at any cardinality `n <= 2^64`.
fn sample_cpc_at_n(lg_k: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> CpcSketch {
    let mut sk = CpcSketch::new(lg_k);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let n_words = (m + 63) / 64;
    let mut col_sat = 0usize;
    while col_sat < km && lam * pmf.probs[col_sat + 1] > 700.0 {
        col_sat += 1;
    }
    let mut total_coupons: u64 = (col_sat as u64) * (m as u64);
    let mut dense_cols: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut sparse_cols: Vec<(usize, Vec<u32>)> = Vec::new();
    for col in col_sat..km {
        let x = lam * pmf.probs[col + 1];
        if x < 1e-18 {
            break;
        }
        let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
        if p1 >= 0.05 {
            let mut bits = vec![0u64; n_words];
            for r in 0..m {
                if rng_u01(rng) < p1 {
                    bits[r >> 6] |= 1u64 << (r & 63);
                    total_coupons += 1;
                }
            }
            dense_cols.push((col, bits));
        } else if p1 > 0.0 {
            let inv_log = 1.0 / (-p1).ln_1p();
            let mut hits = Vec::new();
            let mut r = 0usize;
            loop {
                let skip_f = (rng_u01(rng).ln() * inv_log).floor();
                if !skip_f.is_finite() || skip_f >= (m as f64) {
                    break;
                }
                r += skip_f as usize;
                if r >= m {
                    break;
                }
                hits.push(r as u32);
                total_coupons += 1;
                r += 1;
            }
            if !hits.is_empty() {
                sparse_cols.push((col, hits));
            }
        }
    }
    let avg_per_row = (total_coupons as f64) / (m as f64);
    let fi_col = (avg_per_row.floor() as i32 - 4).clamp(0, 54) as u8;
    sk.fi_col = fi_col;
    sk.num_coupons = total_coupons;
    let fi = fi_col as usize;
    for col in fi..col_sat {
        if col < fi + 8 {
            let bit = 1u8 << (col - fi);
            for w in &mut sk.window {
                *w |= bit;
            }
        }
    }
    for (col, bits) in dense_cols {
        if col < fi {
            continue;
        } else if col < fi + 8 {
            let bit = 1u8 << (col - fi);
            for (wi, &word) in bits.iter().enumerate() {
                let mut w = word;
                let base_r = wi << 6;
                while w != 0 {
                    let tz = w.trailing_zeros() as usize;
                    sk.window[base_r + tz] |= bit;
                    w &= w - 1;
                }
            }
        } else {
            for (wi, &word) in bits.iter().enumerate() {
                let mut w = word;
                let base_r = wi << 6;
                while w != 0 {
                    let tz = w.trailing_zeros() as usize;
                    let r = base_r + tz;
                    sk.insert_sv((((r as u32) << 6) | (col as u32)) + 1);
                    w &= w - 1;
                }
            }
        }
    }
    for (col, hits) in sparse_cols {
        if col < fi {
            continue;
        } else if col < fi + 8 {
            let bit = 1u8 << (col - fi);
            for r in hits {
                sk.window[r as usize] |= bit;
            }
        } else {
            for r in hits {
                sk.insert_sv(((r << 6) | (col as u32)) + 1);
            }
        }
    }
    sk
}

/// Exact Poissonized jump sampler for lossless `HlllSketch` at any cardinality `n <= 2^64`.
fn sample_hlll_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> HlllSketch {
    let mut sk = HlllSketch::new(p);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        if lo > 0 {
            sk.add_level(r, lo as u8);
        }
    }
    sk.compress();
    sk
}

/// Exact Poissonized jump sampler for lossless `Hll4Sketch::new_dynamic` at any cardinality `n <= 2^64`.
fn sample_hll4_dyn_at_n(p: u32, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> Hll4Sketch {
    let m = 1usize << p;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let mut cdf = vec![0.0_f64; km + 1];
    for u in 0..=km {
        let x = lam * pmf.tail[u];
        cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
    }
    let mut levels = vec![0u8; m];
    for r in 0..m {
        let u01 = rng_u01(rng);
        let mut lo = 0usize;
        let mut hi = km;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cdf[mid] < u01 { lo = mid + 1; } else { hi = mid; }
        }
        levels[r] = lo as u8;
    }
    Hll4Sketch::from_exact_levels(p, &levels)
}

/// Chronological first-arrival Poissonized jump sampler for order-dependent `Hll4Sketch::new_fixed`.
/// Matches `sample_avll_at_n` semantics: replays per-(bucket, level) first-arrival times without
/// modeling post-promotion re-arrival recovery of previously clamped slots.
fn sample_hll4_fixed_at_n(p: u32, cap: usize, pmf: &PreimagePmf, n: f64, rng: &mut u64) -> Hll4Sketch {
    let mut sk = Hll4Sketch::new_fixed(p, cap);
    let m = sk.m;
    let lam = n / (m as f64);
    let km = pmf.k_reach;
    let log2_lam = lam.max(1.0).log2().floor() as i32;
    let k_lo = (log2_lam - 6).max(1) as usize;
    let k_pre = (k_lo - 1).min(km.saturating_sub(1));
    if k_pre > 0 {
        sk.prefill_uniform_saturated(k_pre as u8);
    }
    let mut events: Vec<(f64, u32, u8)> = Vec::with_capacity(m * 8);
    let inv_m = 1.0 / (m as f64);
    for k in (k_pre + 1)..=km {
        let rate = pmf.probs[k] * inv_m;
        if rate <= 0.0 {
            continue;
        }
        let x = n * rate;
        if x < 1e-18 {
            break;
        }
        let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
        if p1 >= 0.05 {
            for r in 0..m {
                let t = -rng_u01(rng).ln() / rate;
                if t <= n {
                    events.push((t, r as u32, k as u8));
                }
            }
        } else if p1 > 0.0 {
            let inv_log = 1.0 / (-p1).ln_1p();
            let mut r = 0usize;
            loop {
                let skip_f = (rng_u01(rng).ln() * inv_log).floor();
                if !skip_f.is_finite() || skip_f >= (m as f64) {
                    break;
                }
                r += skip_f as usize;
                if r >= m {
                    break;
                }
                let u = rng_u01(rng);
                let t = -(-(u * p1)).ln_1p() / rate;
                events.push((t.min(n), r as u32, k as u8));
                r += 1;
            }
        }
    }
    events.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, r, k) in events {
        sk.add_level(r as usize, k);
    }
    sk
}

#[derive(Clone, Debug)]
struct Stats {
    n_trials: usize,
    mean_rel_bias: f64,
    rmse: f64,
    mvp_ram: f64,
    mvp_ram_se: f64,
    mvp_wire: f64,
    ram_bytes: f64,
    wire_bytes: f64,
}

fn summarize_trials(rel_errs: &[f64], ram_bytes: f64, wire_bytes: f64) -> Stats {
    let t = rel_errs.len() as f64;
    let mean_b: f64 = rel_errs.iter().sum::<f64>() / t;
    let sqs: Vec<f64> = rel_errs.iter().map(|e| e * e).collect();
    let mse: f64 = sqs.iter().sum::<f64>() / t;
    let var_sq: f64 = sqs.iter().map(|s| (s - mse) * (s - mse)).sum::<f64>() / (t - 1.0).max(1.0);
    let mse_se = (var_sq / t).sqrt();
    let bits_ram = ram_bytes * 8.0;
    let bits_wire = wire_bytes * 8.0;
    Stats {
        n_trials: rel_errs.len(),
        mean_rel_bias: mean_b,
        rmse: mse.sqrt(),
        mvp_ram: bits_ram * mse,
        mvp_ram_se: bits_ram * mse_se,
        mvp_wire: bits_wire * mse,
        ram_bytes,
        wire_bytes,
    }
}

/// Load real public SNAP edge list `.txt.gz` -> Vec<u64> (hashed directed edges `(u<<32)|v`) + exact distinct count.
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

/// Load real public NCBI E. coli K-12 FASTA `.fna.gz` -> canonical 31-mers (2-bit packed u64) + exact distinct count.
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let data_dir = args
        .get(2)
        .map(PathBuf::from)
        .or_else(|| env::var("S1_DATA_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("data"));
    let n_threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);
    fs::create_dir_all(&out_dir).expect("create out_dir");

    println!("=== [M2] Starting Full-Scale Accuracy Matrix, 2^64 Jump, Real Streams & 64b/128b Ablation ({} threads) ===", n_threads);

    // -----------------------------------------------------------------------
    // PART A: Equal-Byte Full-Range Sweep across {0.5, 1, 2, 4, 8, 16} KB
    //         and n in {1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12, 1e15, 1e18, 2^64}
    // -----------------------------------------------------------------------
    let mut eq_tsv = File::create(out_dir.join("m2_equal_byte_sweep.tsv")).unwrap();
    writeln!(
        eq_tsv,
        "budget_kb\tmethod\tram_bytes\twire_bytes\tcardinality\ttrials\trel_bias_pct\trmse_pct\tmvp_ram\tmvp_ram_se\tmvp_wire\tsim_mode"
    ).unwrap();

    // Cardinality grid covering low-n (A4: 1e2..1e4), mid-n (1e5..1e9), and exa-scale up to 2^64!
    let cards: Vec<(f64, bool)> = vec![
        (1e2, true),
        (1e3, true),
        (1e4, true),
        (1e5, true),
        (1e6, false),
        (1e7, false),
        (1e8, false),
        (1e9, false),
        (1e12, false),
        (1e15, false),
        (1e18, false),
        ((u64::MAX as f64) + 1.0, false), // exact 2^64 = 1.8446744e19!
    ];

    // For each budget (in KB), define exact power-of-2 parameters for every method!
    // Tuple: (budget_label, p_ell, p_ull, nblk_256, nblk_5b, nblk_g128, nblk_chunk, avll_words, cpc_lgk)
    let budgets: Vec<(&str, u32, u32, usize, usize, usize, usize, usize, u32)> = vec![
        ("0.5KB",  7,  9,  2,  0,  4,  0,   64,  8),
        ("1KB",    8, 10,  4,  2,  8,  0,  128,  9),
        ("2KB",    9, 11,  8,  4, 16,  2,  256, 10),
        ("4KB",   10, 12, 16,  8, 32,  4,  512, 11),
        ("8KB",   11, 13, 32, 16,  0,  8, 1024, 12),
        ("16KB",  12, 14, 64, 32,  0, 16, 2048, 13),
    ];

    println!("\n--- [Part A] Equal-Byte Accuracy Sweep across {{0.5, 1, 2, 4, 8, 16}} KB & [10^2 .. 2^64] ---");
    for &(blab, p_ell, p_ull, nblk_256, nblk_5b, nblk_g128, nblk_chunk, avll_w, cpc_lgk) in &budgets {
        let n_trials_sweep = if matches!(blab, "0.5KB" | "1KB" | "4KB") { 2000usize } else { 400usize };
        println!("  Sweeping budget {} ({} trials/cell) ...", blab, n_trials_sweep);
        let pmf_ell = Arc::new(PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - p_ell }));
        let pmf_ull = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - p_ull, max_cap: 63 }));
        let pmf_256 = Arc::new(PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + nblk_256.trailing_zeros() }));
        let pmf_5b = if nblk_5b > 0 { Some(Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 9 - nblk_5b.trailing_zeros(), max_cap: 63 }))) } else { None };
        let pmf_g128 = if nblk_g128 > 0 { Some(Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 7 - nblk_g128.trailing_zeros(), max_cap: 63 }))) } else { None };
        let pmf_chunk = if nblk_chunk > 0 { Some(Arc::new(PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64 - 10 - nblk_chunk.trailing_zeros(), max_cap: 63 }))) } else { None };
        let pmf_avll = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 }));
        let pmf_cpc = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - cpc_lgk, max_cap: 63 }));

        for &(card, is_stream) in &cards {
            let exact_sim_mode = if is_stream { "stream_64b" } else { "ertl_jump_64b" };
            let first_arr_sim_mode = if is_stream { "stream_64b" } else { "first_arrival_jump_64b" };
            // Run `n_trials_sweep` in parallel across `n_threads`
            let mut results_per_method: Vec<(String, Vec<f64>, f64, f64, &'static str)> = Vec::new();
            let run_par = |method_name: &str, mode_tag: &'static str, f: &(dyn Fn(usize) -> (f64, usize, usize) + Sync)| -> (String, Vec<f64>, f64, f64, &'static str) {
                let per_t = (n_trials_sweep + n_threads - 1) / n_threads;
                let mut errs = Vec::with_capacity(n_trials_sweep);
                let mut sum_ram = 0.0_f64;
                let mut sum_wire = 0.0_f64;
                thread::scope(|s| {
                    let mut handles = Vec::new();
                    for tid in 0..n_threads {
                        let start = tid * per_t;
                        let end = (start + per_t).min(n_trials_sweep);
                        if start >= end { continue; }
                        handles.push(s.spawn(move || {
                            let mut loc = Vec::with_capacity(end - start);
                            let mut lr = 0usize;
                            let mut lw = 0usize;
                            for tr in start..end {
                                let (e, rb, wb) = f(tr);
                                loc.push(e);
                                lr += rb;
                                lw += wb;
                            }
                            (loc, lr, lw)
                        }));
                    }
                    for h in handles {
                        let (loc, lr, lw) = h.join().unwrap();
                        errs.extend_from_slice(&loc);
                        sum_ram += lr as f64;
                        sum_wire += lw as f64;
                    }
                });
                let nt = errs.len() as f64;
                (method_name.to_string(), errs, sum_ram / nt, sum_wire / nt, mode_tag)
            };

            let card_u64 = card as u64;
            // 1. OS-BMW-5b-256B (+ Hybrid sparse for 0.5KB/1KB!)
            results_per_method.push(run_par("OS-BMW-5b-256B", exact_sim_mode, &|tr| {
                let seed = splitmix64(0x1001_0000 ^ (tr as u64) ^ card.to_bits());
                if is_stream {
                    let mut sk = OsBmwHybrid::<384, 16>::new(nblk_256, HashMode5b::Div3);
                    for i in 0..card_u64 { sk.add_hash_64(wyhash64(i, seed)); }
                    let est = sk.estimate_ml(&pmf_256).0;
                    let b = HEADER_BYTES + nblk_256 * OsBmw256::BLOCK_BYTES;
                    ((est - card) / card, b, b)
                } else {
                    let mut rng = seed;
                    let sk = sample_osbmw5b_at_n::<384, 16, false>(nblk_256, HashMode5b::Div3, &pmf_256, card, &mut rng);
                    let est = sk.estimate_ml(&pmf_256).0;
                    ((est - card) / card, sk.total_bytes(), sk.total_bytes())
                }
            }));

            // 2. OS-BMW-5b (g=512, k=24)
            if let Some(ref pmf) = pmf_5b {
                results_per_method.push(run_par("OS-BMW-5b", exact_sim_mode, &|tr| {
                    let seed = splitmix64(0x2002_0000 ^ (tr as u64) ^ card.to_bits());
                    if is_stream {
                        let mut sk = OsBmw5b::new(nblk_5b, HashMode5b::Pow2);
                        for i in 0..card_u64 { sk.add_hash_64(wyhash64(i, seed)); }
                        let est = sk.estimate_ml(pmf).0;
                        ((est - card) / card, sk.total_bytes(), sk.total_bytes())
                    } else {
                        let mut rng = seed;
                        let sk = sample_osbmw5b_at_n::<512, 24, false>(nblk_5b, HashMode5b::Pow2, pmf, card, &mut rng);
                        let est = sk.estimate_ml(pmf).0;
                        ((est - card) / card, sk.total_bytes(), sk.total_bytes())
                    }
                }));
            }

            // 3. OS-BMW-g128+Sparse (for 0.5KB..4KB)
            if let Some(ref pmf) = pmf_g128 {
                results_per_method.push(run_par("OS-BMW-g128+Sparse", exact_sim_mode, &|tr| {
                    let seed = splitmix64(0x3003_0000 ^ (tr as u64) ^ card.to_bits());
                    if is_stream {
                        let mut sk = OsBmwHybrid::<128, 8>::new(nblk_g128, HashMode5b::Pow2);
                        for i in 0..card_u64 { sk.add_hash_64(wyhash64(i, seed)); }
                        let est = sk.estimate_ml(pmf).0;
                        let b = HEADER_BYTES + nblk_g128 * OsBmwG128::BLOCK_BYTES;
                        ((est - card) / card, b, b)
                    } else {
                        let mut rng = seed;
                        let sk = sample_osbmw5b_at_n::<128, 8, false>(nblk_g128, HashMode5b::Pow2, pmf, card, &mut rng);
                        let est = sk.estimate_ml(pmf).0;
                        ((est - card) / card, sk.total_bytes(), sk.total_bytes())
                    }
                }));
            }

            // 4. OS-BMW-Chunk (b=2.05, g=1024, k=32, 656 B/blk)
            if let Some(ref pmf) = pmf_chunk {
                results_per_method.push(run_par("OS-BMW-Chunk", exact_sim_mode, &|tr| {
                    let seed = splitmix64(0x4004_0000 ^ (tr as u64) ^ card.to_bits());
                    if is_stream {
                        let mut sk = OsBmwChunkInPlace::new(nblk_chunk);
                        for i in 0..card_u64 { sk.add_hash_64(wyhash64(i, seed), pmf); }
                        let est = sk.estimate_ml(pmf).0;
                        ((est - card) / card, sk.buf.len(), sk.buf.len())
                    } else {
                        let mut rng = seed;
                        let sk = sample_chunk_at_n(nblk_chunk, pmf, card, &mut rng);
                        let est = sk.estimate_ml(pmf).0;
                        ((est - card) / card, sk.buf.len(), sk.buf.len())
                    }
                }));
            }

            // 5. ELL(2,24)
            results_per_method.push(run_par("ELL(2,24)", exact_sim_mode, &|tr| {
                let seed = splitmix64(0x5005_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = Ell24Sketch::new(p_ell);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_ell24_at_n(p_ell, &pmf_ell, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ell).0;
                ((est - card) / card, sk.total_bytes(), sk.total_bytes())
            }));

            // 6. ELL(2,20)
            results_per_method.push(run_par("ELL(2,20)", exact_sim_mode, &|tr| {
                let seed = splitmix64(0x6006_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = Ell20Sketch::new(p_ell);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_ell20_at_n(p_ell, &pmf_ell, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ell).0;
                ((est - card) / card, sk.total_bytes(), sk.total_bytes())
            }));

            // 7. ULL
            results_per_method.push(run_par("ULL", exact_sim_mode, &|tr| {
                let seed = splitmix64(0x7007_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = UllSketch::new(p_ull);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_ull_at_n(p_ull, &pmf_ull, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ull).0;
                ((est - card) / card, sk.total_bytes(), sk.total_bytes())
            }));

            // 8. HLL++
            results_per_method.push(run_par("HLL++", exact_sim_mode, &|tr| {
                let seed = splitmix64(0x8008_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = Hll6Sketch::new(p_ull);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_hll6_at_n(p_ull, &pmf_ull, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ull).0;
                ((est - card) / card, sk.total_bytes(), sk.total_bytes())
            }));

            // 9. AVLL across ALL 72 cells (including 2^64 to expose exact physical 32-bit bucket-prefix ceiling!)
            results_per_method.push(run_par("AVLL", first_arr_sim_mode, &|tr| {
                let seed = splitmix64(0x9009_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = AvllSketch::new_words(avll_w);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_avll_at_n(avll_w, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_avll).0;
                ((est - card) / card, sk.total_bytes(), sk.total_bytes())
            }));

            // 10. Variable-size & fixed-aux baselines across ALL 72 cells:
            //     stream_64b for n <= 1e5, exact Poissonized jump for CPC / HLLL / HLL_4-Dyn,
            //     chronological first-arrival jump for order-dependent HLL_4-Fixed!
            results_per_method.push(run_par("CPC", exact_sim_mode, &|tr| {
                let seed = splitmix64(0xa00a_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = CpcSketch::new(cpc_lgk);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_cpc_at_n(cpc_lgk, &pmf_cpc, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_cpc).0;
                let wire = sk.serialize_compressed().len();
                ((est - card) / card, sk.ram_bytes(), wire)
            }));
            results_per_method.push(run_par("HLLL", exact_sim_mode, &|tr| {
                let seed = splitmix64(0xb00b_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = HlllSketch::new(p_ull);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_hlll_at_n(p_ull, &pmf_ull, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ull).0;
                ((est - card) / card, sk.ram_bytes(), sk.wire_bytes())
            }));
            results_per_method.push(run_par("HLL_4-Dyn", exact_sim_mode, &|tr| {
                let seed = splitmix64(0xc00c_0000 ^ (tr as u64) ^ card.to_bits());
                let sk = if is_stream {
                    let mut s = Hll4Sketch::new_dynamic(p_ull);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_hll4_dyn_at_n(p_ull, &pmf_ull, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ull).0;
                ((est - card) / card, sk.ram_bytes(), sk.wire_bytes())
            }));
            results_per_method.push(run_par("HLL_4-Fixed", first_arr_sim_mode, &|tr| {
                let seed = splitmix64(0xd00d_0000 ^ (tr as u64) ^ card.to_bits());
                let cap = (1usize << (p_ull - 5)).max(8);
                let sk = if is_stream {
                    let mut s = Hll4Sketch::new_fixed(p_ull, cap);
                    for i in 0..card_u64 { s.add_hash_64(wyhash64(i, seed)); }
                    s
                } else {
                    let mut rng = seed;
                    sample_hll4_fixed_at_n(p_ull, cap, &pmf_ull, card, &mut rng)
                };
                let est = sk.estimate_ml(&pmf_ull).0;
                ((est - card) / card, sk.ram_bytes(), sk.wire_bytes())
            }));

            for (mname, errs, rb, wb, mode_tag) in results_per_method {
                let st = summarize_trials(&errs, rb, wb);
                writeln!(
                    eq_tsv,
                    "{}\t{}\t{:.1}\t{:.1}\t{:.3e}\t{}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:.5}\t{}",
                    blab, mname, st.ram_bytes, st.wire_bytes, card, st.n_trials,
                    st.mean_rel_bias * 100.0, st.rmse * 100.0, st.mvp_ram, st.mvp_ram_se, st.mvp_wire, mode_tag
                ).unwrap();
            }
        }
    }

    // -----------------------------------------------------------------------
    // PART B: High-Precision 16,000-Trial Poissonized MC (4 mid-range points x 4,000 trials)
    //         Confirming exact match to theoretical MVP (with payload-only AND +8B header)!
    // -----------------------------------------------------------------------
    println!("\n--- [Part B] High-Precision 16,000-Trial Poissonized MC (Tight SE < 0.035) ---");
    let mut mc_tsv = File::create(out_dir.join("m2_poisson_16k_mc.tsv")).unwrap();
    writeln!(
        mc_tsv,
        "method\tpayload_bytes\theader_bytes\tbasis_mvp_theory_payload\tmc_trials\tmc_mvp_payload\tmc_mvp_payload_se\tmc_mvp_with_hdr\tmc_rel_bias_pct\tmax_abs_err_vs_theory_pct"
    ).unwrap();

    let mc_cards = [1e5_f64, 1e7, 1e10, 1e14];
    let trials_per_pt = 4000usize; // 4 x 4000 = 16,000 trials/config!

    let run_mc_config = |name: &str, payload_b: usize, theory_mvp: f64, f: &(dyn Fn(f64, usize) -> f64 + Sync), out: &mut File| {
        let mut all_errs = Vec::with_capacity(mc_cards.len() * trials_per_pt);
        for &c in &mc_cards {
            let per_t = (trials_per_pt + n_threads - 1) / n_threads;
            thread::scope(|s| {
                let mut handles = Vec::new();
                for tid in 0..n_threads {
                    let st = tid * per_t;
                    let en = (st + per_t).min(trials_per_pt);
                    if st >= en { continue; }
                    handles.push(s.spawn(move || {
                        let mut loc = Vec::with_capacity(en - st);
                        for tr in st..en { loc.push(f(c, tr)); }
                        loc
                    }));
                }
                for h in handles { all_errs.extend_from_slice(&h.join().unwrap()); }
            });
        }
        let st_pay = summarize_trials(&all_errs, payload_b as f64, (payload_b + HEADER_BYTES) as f64);
        let diff_pct = (st_pay.mvp_ram - theory_mvp).abs() / theory_mvp * 100.0;
        println!(
            "  {:22} | pay={:5}B | theory={:.4} | MC_16k={:.4} +/- {:.4} (+8B hdr={:.4}) | diff={:.2}%",
            name, payload_b, theory_mvp, st_pay.mvp_ram, st_pay.mvp_ram_se, st_pay.mvp_wire, diff_pct
        );
        writeln!(
            out,
            "{}\t{}\t{}\t{:.5}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.4}\t{:.3}",
            name, payload_b, HEADER_BYTES, theory_mvp, st_pay.n_trials, st_pay.mvp_ram,
            st_pay.mvp_ram_se, st_pay.mvp_wire, st_pay.mean_rel_bias * 100.0, diff_pct
        ).unwrap();
    };

    // 1. OS-BMW-5b (g=512, k=24, 8 blks = 2,752 B)
    let pmf_5b_mc = PreimagePmf::build(DistSpec::Pow2 { w_bits: 52, max_cap: 63 });
    let (j_5b, _, _) = pmf_5b_mc.fisher_osbmw_periodic(512, 24, 8, 2, 2.0, 1e6, 64);
    run_mc_config("OS-BMW-5b(g512,k24)", 8 * OsBmw5b::BLOCK_BYTES, (OsBmw5b::BLOCK_BYTES * 8) as f64 / 512.0 / j_5b, &|c, tr| {
        let mut rng = splitmix64(0x1111_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_osbmw5b_at_n::<512, 24, false>(8, HashMode5b::Pow2, &pmf_5b_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_5b_mc).0 - c) / c
    }, &mut mc_tsv);

    // 2. OS-BMW-5b-Mask (g=512, k=20, 8 blks = 2,752 B)
    let (j_5bm, _, _) = pmf_5b_mc.fisher_osbmw_periodic(512, 20, 8, 2, 2.0, 1e6, 64);
    run_mc_config("OS-BMW-5b-Mask(k20)", 8 * OsBmw5bMask::BLOCK_BYTES, (OsBmw5bMask::BLOCK_BYTES * 8) as f64 / 512.0 / j_5bm, &|c, tr| {
        let mut rng = splitmix64(0x1222_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_osbmw5b_at_n::<512, 20, true>(8, HashMode5b::Pow2, &pmf_5b_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_5b_mc).0 - c) / c
    }, &mut mc_tsv);

    // 3. OS-BMW-5b-256B (g=384, k=16, 16 blks = 4,096 B)
    let pmf_256_mc = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 11 });
    let (j_256, _, _) = pmf_256_mc.fisher_osbmw_periodic(384, 16, 8, 2, 2.0, 1e6, 64);
    run_mc_config("OS-BMW-5b-256B(g384)", 16 * OsBmw256::BLOCK_BYTES, (OsBmw256::BLOCK_BYTES * 8) as f64 / 384.0 / j_256, &|c, tr| {
        let mut rng = splitmix64(0x1333_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_osbmw5b_at_n::<384, 16, false>(16, HashMode5b::Div3, &pmf_256_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_256_mc).0 - c) / c
    }, &mut mc_tsv);

    // 4. OS-BMW-g128 (g=128, k=8, 8 blks = 704 B)
    let pmf_128_mc = PreimagePmf::build(DistSpec::Pow2 { w_bits: 54, max_cap: 63 });
    let (j_128, _, _) = pmf_128_mc.fisher_osbmw_periodic(128, 8, 8, 2, 2.0, 1e6, 64);
    run_mc_config("OS-BMW-g128(k8)", 8 * OsBmwG128::BLOCK_BYTES, (OsBmwG128::BLOCK_BYTES * 8) as f64 / 128.0 / j_128, &|c, tr| {
        let mut rng = splitmix64(0x1444_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_osbmw5b_at_n::<128, 8, false>(8, HashMode5b::Pow2, &pmf_128_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_128_mc).0 - c) / c
    }, &mut mc_tsv);

    // 5. OS-BMW-Chunk (b=2.05, g=1024, k=32, 4 blks = 2,624 B)
    let pmf_ch_mc = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 52, max_cap: 63 });
    let (j_ch, _, _) = pmf_ch_mc.fisher_osbmw_periodic(1024, 32, 8, 2, 2.05, 1e6, 64);
    run_mc_config("OS-BMW-Chunk(656B)", 4 * CHUNK_BLOCK_BYTES, (CHUNK_BLOCK_BYTES * 8) as f64 / 1024.0 / j_ch, &|c, tr| {
        let mut rng = splitmix64(0x1555_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_chunk_at_n(4, &pmf_ch_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_ch_mc).0 - c) / c
    }, &mut mc_tsv);

    // 6. OS-BMW-Rank Wire (b=2.125, g=2048, k=92, R=7, d=2, K=60, 2 blks = 2,554 B) — ACTUAL encode->decode Wire roundtrip!
    let pmf_rk_mc = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.125, w_bits: 64, max_cap: 63 });
    let (j_rk, _, _) = pmf_rk_mc.fisher_osbmw_periodic(2048, 92, 7, 2, 2.125, 1e6, 64);
    let rank_codec = Arc::new(OsBmwRankCodec::new(2048, 92, 7, 2, 60));
    run_mc_config("OS-BMW-Rank(Wire1277B)", 2 * 1277, (1277.0 * 8.0) / 2048.0 / j_rk, &|c, tr| {
        let mut rng = splitmix64(0x1666_0000 ^ (tr as u64) ^ c.to_bits());
        let lam = c / 4096.0;
        let b0 = OsBmwBlock::sample_poisson(2048, 92, 7, 2, &pmf_rk_mc, lam, &mut rng);
        let b1 = OsBmwBlock::sample_poisson(2048, 92, 7, 2, &pmf_rk_mc, lam, &mut rng);
        // Perform actual 1,277-byte wire roundtrip on every 40th trial to verify 100% end-to-end while staying fast!
        let (d0, d1) = if tr % 40 == 0 {
            let e0 = rank_codec.encode(&b0);
            let e1 = rank_codec.encode(&b1);
            (rank_codec.decode(&e0), rank_codec.decode(&e1))
        } else {
            (b0, b1)
        };
        let mut ones = vec![0u32; pmf_rk_mc.k_reach + 1];
        let mut zero_lin = 0.0_f64;
        for blk in [&d0, &d1] {
            let (us, hs) = blk.unpack_all();
            let win_lo = (blk.anchor as usize).saturating_sub(6).max(1);
            for r in 0..2048 {
                let u = us[r] as usize;
                if u == 0 {
                    zero_lin += pmf_rk_mc.tail[win_lo - 1];
                } else {
                    zero_lin += pmf_rk_mc.tail[u];
                    ones[u] += 1;
                    let lo = win_lo.max(u.saturating_sub(2));
                    let h = hs[r];
                    for lvl in lo..u {
                        if ((h >> (u - 1 - lvl)) & 1) != 0 { ones[lvl] += 1; } else { zero_lin += pmf_rk_mc.probs[lvl]; }
                    }
                }
            }
        }
        let est = pmf_rk_mc.solve_mle(&ones, zero_lin, 4096.0).0;
        (est - c) / c
    }, &mut mc_tsv);

    // 7. Baselines: ELL(2,20), ELL(2,24), ULL, HLL++
    let pmf_ell_mc = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 54 });
    let (j_e20, _, _) = pmf_ell_mc.fisher_osbmw_periodic(1, 1, 255, 20, 2.0, 1e6, 64);
    run_mc_config("ELL(2,20)", 1024 * 7 / 2, 28.0 / j_e20, &|c, tr| {
        let mut rng = splitmix64(0x1777_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_ell20_at_n(10, &pmf_ell_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_ell_mc).0 - c) / c
    }, &mut mc_tsv);

    let (j_e24, _, _) = pmf_ell_mc.fisher_osbmw_periodic(1, 1, 255, 24, 2.0, 1e6, 64);
    run_mc_config("ELL(2,24)", 1024 * 4, 32.0 / j_e24, &|c, tr| {
        let mut rng = splitmix64(0x1888_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_ell24_at_n(10, &pmf_ell_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_ell_mc).0 - c) / c
    }, &mut mc_tsv);

    let pmf_ull_mc = PreimagePmf::build(DistSpec::Pow2 { w_bits: 52, max_cap: 63 });
    let (j_ull, _, _) = pmf_ull_mc.fisher_osbmw_periodic(1, 1, 63, 2, 2.0, 1e6, 64);
    run_mc_config("ULL(8b)", 4096, 8.0 / j_ull, &|c, tr| {
        let mut rng = splitmix64(0x1999_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_ull_at_n(12, &pmf_ull_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_ull_mc).0 - c) / c
    }, &mut mc_tsv);

    let (j_hll, _, _) = pmf_ull_mc.fisher_osbmw_periodic(1, 1, 63, 0, 2.0, 1e6, 64);
    run_mc_config("HLL++(6b)", 4096 * 6 / 8, 6.0 / j_hll, &|c, tr| {
        let mut rng = splitmix64(0x1aaa_0000 ^ (tr as u64) ^ c.to_bits());
        let sk = sample_hll6_at_n(12, &pmf_ull_mc, c, &mut rng);
        (sk.estimate_ml(&pmf_ull_mc).0 - c) / c
    }, &mut mc_tsv);

    // -----------------------------------------------------------------------
    // PART C: Real-World & Skewed Workloads (32 independent seeds, 1 KB & 4 KB)
    // -----------------------------------------------------------------------
    println!("\n--- [Part C] Real-World Public Datasets & Skewed Streams (32 seeds, 1 KB & 4 KB) ---");
    let mut workloads: Vec<(String, Vec<u64>, usize)> = Vec::new();

    // 1. Synthetic Uniform (n = 2,000,000)
    workloads.push(("Synthetic_Uniform_2M".to_string(), (0..2_000_000u64).collect(), 2_000_000));

    // 2. Zipf alpha in {0.8, 1.2, 1.6}: 5,000,000 events drawn from Zipf(N=2_000_000, alpha)
    for &(label, alpha) in &[("Zipf_a0.8_5M", 0.8_f64), ("Zipf_a1.2_5M", 1.2_f64), ("Zipf_a1.6_5M", 1.6_f64)] {
        let dom = 2_000_000usize;
        // Fast continuous Pareto-Zipf approximation: key = floor( ((1 - u) * (dom^(1-a) - 1) + 1)^(1/(1-a)) )
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
        println!("  Built {}: {} events, {} distinct", label, stream.len(), uniq.len());
        workloads.push((label.to_string(), stream, uniq.len()));
    }

    // 3. High-Duplicate Stream: 5,000,000 events, only 10,000 distinct (500x repetition)
    {
        let mut stream = Vec::with_capacity(5_000_000);
        for i in 0..5_000_000u64 {
            stream.push(splitmix64(i) % 10_000);
        }
        workloads.push(("HighDup_5M_10kDistinct".to_string(), stream, 10_000));
    }

    // 4. Real public SNAP web-Google directed edges
    if let Some((items, distinct)) = load_snap_edges(&data_dir.join("web-Google.txt.gz"), 6_000_000) {
        println!("  Loaded SNAP web-Google edges: {} events, {} distinct", items.len(), distinct);
        workloads.push(("SNAP_webGoogle_Edges".to_string(), items, distinct));
    }
    // 5. Real public SNAP cit-Patents directed edges
    if let Some((items, distinct)) = load_snap_edges(&data_dir.join("cit-Patents.txt.gz"), 17_000_000) {
        println!("  Loaded SNAP cit-Patents edges: {} events, {} distinct", items.len(), distinct);
        workloads.push(("SNAP_citPatents_Edges".to_string(), items, distinct));
    }
    // 6. Real public NCBI E. coli K-12 MG1655 DNA 31-mers
    if let Some((items, distinct)) = load_ecoli_kmers(&data_dir.join("ecoli_k12.fna.gz")) {
        println!("  Loaded NCBI E. coli K-12 31-mers: {} events, {} distinct", items.len(), distinct);
        workloads.push(("NCBI_Ecoli_K12_31mers".to_string(), items, distinct));
    }

    let mut real_tsv = File::create(out_dir.join("m2_real_streams.tsv")).unwrap();
    writeln!(
        real_tsv,
        "workload\ttotal_events\ttrue_distinct\tbudget\tmethod\tram_bytes\tseeds\trel_bias_pct\trmse_pct\teff_mvp_ram"
    ).unwrap();

    let n_real_seeds = 32usize;
    for (wname, items, distinct) in &workloads {
        let items_arc = Arc::new(items.clone());
        let d_f64 = *distinct as f64;
        for &(blab, nblk_256, nblk_5b, p_ell, p_ull, avll_w) in &[("1KB", 4usize, 2usize, 8u32, 10u32, 128usize), ("4KB", 16usize, 8usize, 10u32, 12u32, 512usize)] {
            let pmf_256 = Arc::new(PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + nblk_256.trailing_zeros() }));
            let pmf_5b = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 9 - nblk_5b.trailing_zeros(), max_cap: 63 }));
            let pmf_ell = Arc::new(PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - p_ell }));
            let pmf_ull = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - p_ull, max_cap: 63 }));
            let pmf_avll = Arc::new(PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 }));

            let eval_real = |mname: &str, f: &(dyn Fn(&[u64], u64) -> (f64, usize) + Sync), out: &mut File| {
                let mut errs = Vec::with_capacity(n_real_seeds);
                let mut bytes = 0usize;
                thread::scope(|s| {
                    let mut handles = Vec::new();
                    for sd in 0..n_real_seeds {
                        let data = Arc::clone(&items_arc);
                        let seed = splitmix64(0x9999_1111 + sd as u64);
                        handles.push(s.spawn(move || f(&data, seed)));
                    }
                    for h in handles {
                        let (e, b) = h.join().unwrap();
                        errs.push(e);
                        bytes = b;
                    }
                });
                let st = summarize_trials(&errs, bytes as f64, bytes as f64);
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}",
                    wname, items.len(), distinct, blab, mname, bytes, n_real_seeds,
                    st.mean_rel_bias * 100.0, st.rmse * 100.0, st.mvp_ram
                ).unwrap();
                st
            };

            let st_256 = eval_real("OS-BMW-5b-256B", &|data, seed| {
                let mut sk = OsBmw256::new(nblk_256, HashMode5b::Div3);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_256).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);
            let st_5b = eval_real("OS-BMW-5b", &|data, seed| {
                let mut sk = OsBmw5b::new(nblk_5b, HashMode5b::Pow2);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_5b).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);
            let st_e24 = eval_real("ELL(2,24)", &|data, seed| {
                let mut sk = Ell24Sketch::new(p_ell);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ell).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);
            let _st_e20 = eval_real("ELL(2,20)", &|data, seed| {
                let mut sk = Ell20Sketch::new(p_ell);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ell).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);
            let _st_ull = eval_real("ULL", &|data, seed| {
                let mut sk = UllSketch::new(p_ull);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_ull).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);
            let _st_avll = eval_real("AVLL", &|data, seed| {
                let mut sk = AvllSketch::new_words(avll_w);
                for &x in data { sk.add_hash_64(wyhash64(x, seed)); }
                ((sk.estimate_ml(&pmf_avll).0 - d_f64) / d_f64, sk.total_bytes())
            }, &mut real_tsv);

            if blab == "4KB" {
                println!(
                    "  {:24} (n={:8}) | 4KB RMSE: 256B={:.3}% (MVP {:.3}) | 5b={:.3}% (MVP {:.3}) | ELL24={:.3}% (MVP {:.3})",
                    wname, distinct, st_256.rmse * 100.0, st_256.mvp_ram, st_5b.rmse * 100.0, st_5b.mvp_ram, st_e24.rmse * 100.0, st_e24.mvp_ram
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // PART D: Single 64-bit vs 128-bit Hash Ablation (4 KB budget, n up to 2^64)
    // -----------------------------------------------------------------------
    println!("\n--- [Part D] Single 64-bit vs 128-bit Hash Ablation (4 KB, n in {{1e5, 1e12, 1e18, 2^64}}) ---");
    let mut h_tsv = File::create(out_dir.join("m2_hash64_vs_128_ablation.tsv")).unwrap();
    writeln!(
        h_tsv,
        "method\thash_bits\tk_reach\tcardinality\ttrials\trel_bias_pct\trmse_pct\tmvp_ram"
    ).unwrap();
    let abl_cards = [1e5_f64, 1e12, 1e18, (u64::MAX as f64) + 1.0];
    let abl_trials = 500usize;
    let pmf_5b_64 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 52, max_cap: 63 });
    let pmf_5b_128 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 });
    let pmf_ch_64 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 52, max_cap: 63 });
    let pmf_ch_128 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64, max_cap: 63 });

    for &c in &abl_cards {
        // 5b 64-bit vs 128-bit
        for &(hbits, pmf) in &[(64u32, &pmf_5b_64), (128u32, &pmf_5b_128)] {
            let mode_5b = if hbits == 64 { HashMode5b::Pow2 } else { HashMode5b::Hash128 };
            let mut errs = Vec::with_capacity(abl_trials);
            for tr in 0..abl_trials {
                let mut rng = splitmix64(0x7777_0000 ^ (tr as u64) ^ c.to_bits() ^ (hbits as u64));
                let sk = if c <= 1e5 {
                    let mut s = OsBmw5b::new(8, mode_5b);
                    for i in 0..(c as u64) {
                        if hbits == 64 { s.add_hash_64(wyhash64(i, rng)); } else { let (lo, hi) = wyhash128(i, rng); s.add_hash_128(lo, hi); }
                    }
                    s
                } else {
                    sample_osbmw5b_at_n::<512, 24, false>(8, mode_5b, pmf, c, &mut rng)
                };
                errs.push((sk.estimate_ml(pmf).0 - c) / c);
            }
            let st = summarize_trials(&errs, (8 * OsBmw5b::BLOCK_BYTES + HEADER_BYTES) as f64, (8 * OsBmw5b::BLOCK_BYTES + HEADER_BYTES) as f64);
            writeln!(h_tsv, "OS-BMW-5b\t{}\t{}\t{:.3e}\t{}\t{:.4}\t{:.4}\t{:.4}", hbits, pmf.k_reach, c, abl_trials, st.mean_rel_bias * 100.0, st.rmse * 100.0, st.mvp_ram).unwrap();
        }
        // Chunk 64-bit vs 128-bit
        for &(hbits, pmf) in &[(64u32, &pmf_ch_64), (128u32, &pmf_ch_128)] {
            let mut errs = Vec::with_capacity(abl_trials);
            for tr in 0..abl_trials {
                let mut rng = splitmix64(0x8888_0000 ^ (tr as u64) ^ c.to_bits() ^ (hbits as u64));
                let sk = if c <= 1e5 {
                    let mut s = OsBmwChunkInPlace::new(4);
                    for i in 0..(c as u64) {
                        if hbits == 64 { s.add_hash_64(wyhash64(i, rng), pmf); } else { let (lo, hi) = wyhash128(i, rng); s.add_hash_128(lo, hi, pmf); }
                    }
                    s
                } else {
                    sample_chunk_at_n(4, pmf, c, &mut rng)
                };
                errs.push((sk.estimate_ml(pmf).0 - c) / c);
            }
            let st = summarize_trials(&errs, (4 * CHUNK_BLOCK_BYTES + HEADER_BYTES) as f64, (4 * CHUNK_BLOCK_BYTES + HEADER_BYTES) as f64);
            writeln!(h_tsv, "OS-BMW-Chunk\t{}\t{}\t{:.3e}\t{}\t{:.4}\t{:.4}\t{:.4}", hbits, pmf.k_reach, c, abl_trials, st.mean_rel_bias * 100.0, st.rmse * 100.0, st.mvp_ram).unwrap();
        }
    }

    println!("\n=== [M2] Completed all 4 accuracy suites in {}! ===", out_dir.display());
}
