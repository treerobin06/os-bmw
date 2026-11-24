//! Experiment 3 (11-23 Verdict): Equal-Error Pareto + Single-Machine Martingale (HIP) vs Distributed Merge.
//!
//! Outputs `m2_equal_error_hip.tsv` with two parts:
//!   - Part A (`A_EqualError`): Matched target RMSE tiers (~2.0%, ~1.0%, ~0.60%) across
//!     strictly mergeable sketches (`OS-BMW-5b`, `OS-BMW-5b-256B`, `ELL(2,24)`, `ULL`, `HLL++(6b)`)
//!     plus non-associative `AVLL(base56)`, reporting exact RAM bytes, error-normalized RAM bytes
//!     `B_eq(target) = RAM * (rmse / target)^2`, `K=1` vs `K=16` RMSE & merge drift, update ns,
//!     merge us, and MLE estimate us.
//!   - Part B (`B_Martingale_vs_Merge`): Exact single-machine Martingale/HIP estimators (`K=1`)
//!     vs strictly mergeable MLE (`K in {1, 2, 4, 16, 64}`) for `OS-BMW-5b(k24)`, `OS-BMW-5b-256B`,
//!     `ELL(2,24)`, `ELL(2,16)` (EDBT'25 optimal `t=2,d=16` 24-bit), `ULL`, and `HLL++(6b)`,
//!     demonstrating both single-stream Martingale behavior AND why distributed merge (`K>=2`)
//!     invalidates Martingale/HIP and locks in OS-BMW's strictly mergeable Pareto dominance.

use s1_v2_rust::baselines::{
    ell_hash_to_idx_k, ell_merge_reg, ell_update_reg, AvllSketch, Ell24Sketch, Hll6Sketch,
    UllSketch,
};
use s1_v2_rust::hash::{wyhash64, HEADER_BYTES};
use s1_v2_rust::osbmw_inplace::{HashMode5b, OsBmw256, OsBmw5b, OsBmw5bInPlace};
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Exact Single-Stream Martingale / HIP Wrappers (Ting 2014 / Cohen 2015 / Ertl 2024-25)
// ---------------------------------------------------------------------------

#[inline]
fn ull_reg_open(r: u8, pmf: &PreimagePmf) -> f64 {
    let u = (r >> 2) as usize;
    let mut open = pmf.tail[u];
    if u >= 2 && (r & 2) == 0 {
        open += pmf.probs[u - 1];
    }
    if u >= 3 && (r & 1) == 0 {
        open += pmf.probs[u - 2];
    }
    open
}

#[inline]
fn ell_reg_open(r: u32, d: u32, pmf: &PreimagePmf) -> f64 {
    let u = (r >> d) as usize;
    let mut open = pmf.tail[u];
    if u > 0 {
        let nb = (u - 1).min(d as usize);
        for delta in 1..=nb {
            if ((r >> (d as usize - delta)) & 1) == 0 {
                open += pmf.probs[u - delta];
            }
        }
    }
    open
}

#[inline]
fn osbmw_reg_open(v: u8, win_lo: usize, pmf: &PreimagePmf) -> f64 {
    let u = (v >> 2) as usize;
    if u == 0 {
        return pmf.tail[win_lo - 1];
    }
    let mut open = pmf.tail[u];
    let h = v & 3;
    if u > win_lo {
        if (h & 1) == 0 {
            open += pmf.probs[u - 1];
        }
        if u >= win_lo + 2 && (h & 2) == 0 {
            open += pmf.probs[u - 2];
        }
    }
    open
}

/// Generic ExaLogLog `ELL(2, D)` (`D = 16` for 24-bit EDBT'25 optimal Martingale, or `D = 24` for 32-bit).
#[derive(Clone)]
struct EllGen {
    p: u32,
    m: usize,
    d: u32,
    regs: Vec<u32>,
}

impl EllGen {
    fn new(p: u32, d: u32) -> Self {
        let m = 1usize << p;
        Self { p, m, d, regs: vec![0u32; m] }
    }
    fn packed_bytes(&self) -> usize {
        // 8-bit base + d-bit history per register + 8B header
        let reg_bits = 8 + self.d as usize;
        HEADER_BYTES + (self.m * reg_bits + 7) / 8
    }
    #[inline(always)]
    fn add_hash_64(&mut self, h: u64) -> Option<(u32, u32)> {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        let cur = self.regs[idx];
        let next = ell_update_reg(cur, k, self.d);
        if next != cur {
            self.regs[idx] = next;
            Some((cur, next))
        } else {
            None
        }
    }
    fn merge_in_place(&mut self, other: &Self) {
        for (a, &b) in self.regs.iter_mut().zip(other.regs.iter()) {
            *a = ell_merge_reg(*a, b, self.d);
        }
    }
    fn estimate_ml(&self, pmf: &PreimagePmf) -> f64 {
        let mut ones = [0u32; 256];
        let mut zero_lin = 0.0_f64;
        for &r in &self.regs {
            let u = (r >> self.d) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 {
                ones[u] += 1;
                let nb = (u - 1).min(self.d as usize);
                for delta in 1..=nb {
                    let lvl = u - delta;
                    if ((r >> (self.d as usize - delta)) & 1) != 0 {
                        ones[lvl] += 1;
                    } else {
                        zero_lin += pmf.probs[lvl];
                    }
                }
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64).0
    }
}

struct OsBmwMartingale<const G: usize, const K: usize> {
    sk: OsBmw5bInPlace<G, K, false>,
    p_change: f64,
    c_mart: f64,
}

impl<const G: usize, const K: usize> OsBmwMartingale<G, K> {
    fn new(n_blk: usize, mode: HashMode5b) -> Self {
        let sk = OsBmw5bInPlace::<G, K, false>::new(n_blk, mode);
        Self {
            sk,
            p_change: 1.0,
            c_mart: 0.0,
        }
    }

    #[inline(always)]
    fn total_regs(&self) -> f64 {
        (self.sk.n_blk * G) as f64
    }

    /// Recomputes exact global open transition probability across all blocks on-demand
    /// (called only on rare anchor promotions: ~160 times per 10^6 updates -> ZERO extra RAM!).
    #[cold]
    #[inline(never)]
    fn recompute_p_change(&self, pmf: &PreimagePmf) -> f64 {
        let mut u_blk = [0u8; G];
        let mut sum = 0.0_f64;
        for blk in 0..self.sk.n_blk {
            let anchor = self.sk.buf[HEADER_BYTES + blk];
            let win_lo = (anchor - 7) as usize;
            self.sk.unpack_block_u8(blk, &mut u_blk);
            for &v in u_blk.iter() {
                sum += osbmw_reg_open(v, win_lo, pmf);
            }
        }
        sum / self.total_regs()
    }

    #[inline(always)]
    fn add_hash_64(&mut self, h: u64, pmf: &PreimagePmf) {
        let (blk, sub, lvl) = OsBmw5bInPlace::<G, K, false>::decompose_hash_64(h, self.sk.n_blk, self.sk.mode);
        let old_anchor = self.sk.buf[HEADER_BYTES + blk];
        if lvl.wrapping_add(8) <= old_anchor {
            return;
        }
        let mut before = [0u8; G];
        self.sk.unpack_block_u8(blk, &mut before);
        let old_v = before[sub];
        self.sk.add_level(blk, sub, lvl);
        let new_anchor = self.sk.buf[HEADER_BYTES + blk];
        if new_anchor != old_anchor {
            self.c_mart += 1.0 / self.p_change;
            self.p_change = self.recompute_p_change(pmf);
        } else {
            let mut after = [0u8; G];
            self.sk.unpack_block_u8(blk, &mut after);
            let new_v = after[sub];
            if new_v != old_v {
                self.c_mart += 1.0 / self.p_change;
                let win_lo = (new_anchor - 7) as usize;
                let d_open = osbmw_reg_open(old_v, win_lo, pmf) - osbmw_reg_open(new_v, win_lo, pmf);
                self.p_change -= d_open / self.total_regs();
            }
        }
    }
}

/// Unified matched-seed schedule shared identically by ALL methods & estimators in Parts A and B!
#[inline(always)]
fn matched_trial_seed(tr: usize, n: u64) -> u64 {
    0xEE00_0000_0000_0000u64
        ^ ((tr as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ (n.wrapping_mul(0xbf58_476d_1ce4_e5b9))
}

fn measure_timing<S, FMake, FAdd, FMerge, FEst>(
    make: FMake,
    add: FAdd,
    merge: FMerge,
    est: FEst,
) -> (f64, f64, f64)
where
    S: Clone,
    FMake: Fn() -> S,
    FAdd: Fn(&mut S, u64),
    FMerge: Fn(&mut S, &S),
    FEst: Fn(&S) -> f64,
{
    let seed = 0x5151_1234u64;
    let mut sk = make();
    for i in 0..200_000u64 {
        add(&mut sk, wyhash64(i, seed));
    }
    let mut samples = Vec::with_capacity(7);
    let ops = 300_000usize;
    for rep in 0..7 {
        let base = (rep as u64 + 1) * 1_000_000;
        let t0 = Instant::now();
        for i in 0..ops as u64 {
            add(&mut sk, wyhash64(base + i, seed));
        }
        samples.push(t0.elapsed().as_nanos() as f64 / ops as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let upd_ns = samples[3];

    let mut sk2 = make();
    for i in 0..200_000u64 {
        add(&mut sk2, wyhash64(i + 5_000_000, seed));
    }
    let m_reps = 100usize;
    let t0 = Instant::now();
    for _ in 0..m_reps {
        let mut tmp = sk.clone();
        merge(&mut tmp, &sk2);
        black_box(&tmp);
    }
    let tot = t0.elapsed().as_nanos() as f64 / m_reps as f64;
    let t1 = Instant::now();
    for _ in 0..m_reps {
        let tmp = sk.clone();
        black_box(&tmp);
    }
    let cl = t1.elapsed().as_nanos() as f64 / m_reps as f64;
    let merge_us = (tot - cl).max(0.0) / 1000.0;

    let e_reps = 100usize;
    let t2 = Instant::now();
    let mut sink = 0.0_f64;
    for _ in 0..e_reps {
        sink += est(&sk);
    }
    black_box(sink);
    let est_us = (t2.elapsed().as_nanos() as f64 / e_reps as f64) / 1000.0;
    (upd_ns, merge_us, est_us)
}

fn eval_k_shards<S, FMake, FAdd, FMerge, FEst>(
    n: u64,
    k_shards: usize,
    trials: usize,
    n_threads: usize,
    make: FMake,
    add: FAdd,
    merge: FMerge,
    est: FEst,
) -> (f64, f64, f64)
where
    FMake: Fn() -> S + Sync,
    FAdd: Fn(&mut S, u64) + Sync,
    FMerge: Fn(&mut S, &S) + Sync,
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
            let mg = &merge;
            let es = &est;
            hs.push(s.spawn(move || {
                let mut loc = Vec::with_capacity(en - st);
                for tr in st..en {
                    let seed = matched_trial_seed(tr, n);
                    let mut acc = mk();
                    if k_shards == 1 {
                        for i in 0..n {
                            ad(&mut acc, wyhash64(i, seed));
                        }
                    } else {
                        for k in 0..k_shards {
                            let mut part = mk();
                            let mut i = k as u64;
                            while i < n {
                                ad(&mut part, wyhash64(i, seed));
                                i += k_shards as u64;
                            }
                            if k == 0 {
                                acc = part;
                            } else {
                                mg(&mut acc, &part);
                            }
                        }
                    }
                    let e = (es(&acc) - n as f64) / (n as f64);
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
    let bias_pct = (errs.iter().sum::<f64>() / nt) * 100.0;
    let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
    let rmse_pct = mse.sqrt() * 100.0;
    (bias_pct, rmse_pct, mse)
}

fn main() {
    let out_dir = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");
    let n_threads = env::var("S1_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(16).max(1);

    let out_path = out_dir.join("m2_equal_error_hip.tsv");
    let mut f = File::create(&out_path).unwrap();
    writeln!(
        f,
        "part\ttier\tmethod\testimator\tmergeable\tn_shards\tn\ttrials\tram_bytes\ttarget_rmse_pct\trel_bias_pct\trmse_pct\tmvp_ram\teq_err_ram_bytes\tupd_ns\tmerge_us\test_us"
    ).unwrap();

    // =========================================================================
    // Part A: Equal-Error Pareto across 3 target RMSE tiers (~2.0%, ~1.0%, ~0.60%)
    // =========================================================================
    let n_a = 1_000_000u64;
    let trials_a = 400usize;

    macro_rules! run_eq_row {
        ($tier:expr, $target_rmse:expr, $name:expr, $mergeable:expr, $ram:expr, $make:expr, $add:expr, $merge:expr, $est:expr) => {{
            let (upd_ns, merge_us, est_us) = measure_timing(|| $make(), |s, h| $add(s, h), |a, b| $merge(a, b), |s| $est(s));
            for &ks in &[1usize, 16usize] {
                let (bias, rmse, mse) = eval_k_shards(n_a, ks, trials_a, n_threads, || $make(), |s, h| $add(s, h), |a, b| $merge(a, b), |s| $est(s));
                let mvp = ($ram as f64 * 8.0) * mse;
                let eq_bytes = ($ram as f64) * (rmse / $target_rmse).powi(2);
                println!(
                    "  [Part A {:6}] {:24} K={:2} | RAM={:5}B | RMSE={:.4}% (eqRAM={:7.1}B) | MVP={:.4} | upd={:.2}ns mrg={:.2}us est={:.1}us",
                    $tier, $name, ks, $ram, rmse, eq_bytes, mvp, upd_ns, merge_us, est_us
                );
                writeln!(
                    f,
                    "A_EqualError\t{}\t{}\tMLE\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.3}\t{:.3}\t{:.3}",
                    $tier, $name, $mergeable, ks, n_a, trials_a, $ram, $target_rmse, bias, rmse, mvp, eq_bytes, upd_ns, merge_us, est_us
                ).unwrap();
                f.flush().unwrap();
            }
        }};
    }

    println!("=== [Part A] Equal-Error Pareto (n=1e6, {} matched-seed trials, K=1 & K=16) ===", trials_a);

    // Tier 1: ~2.0% RMSE
    {
        let pmf_5b_4 = Arc::new(PreimagePmf::build(OsBmw5b::new(4, HashMode5b::Pow2).dist_spec_64()));
        run_eq_row!("2.0pct", 2.0, "OS-BMW-5b(k24,4blk)", "yes", OsBmw5b::state_bytes_for(4),
            || OsBmw5b::new(4, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.estimate_ml(&pmf_5b_4).0);

        let pmf_256_4 = Arc::new(PreimagePmf::build(OsBmw256::new(4, HashMode5b::Div3).dist_spec_64()));
        run_eq_row!("2.0pct", 2.0, "OS-BMW-5b-256B(4blk)", "yes", OsBmw256::state_bytes_for(4),
            || OsBmw256::new(4, HashMode5b::Div3), |s: &mut OsBmw256, h| { s.add_hash_64_div3_mulshift_deferred(h); }, |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw256| s.estimate_ml(&pmf_256_4).0);

        let pmf_ell_8 = Arc::new(PreimagePmf::build(Ell24Sketch::new(8).dist_spec_64()));
        run_eq_row!("2.0pct", 2.0, "ELL(2,24)(p=8)", "yes", Ell24Sketch::new(8).total_bytes(),
            || Ell24Sketch::new(8), |s: &mut Ell24Sketch, h| s.add_hash_64(h), |a: &mut Ell24Sketch, b: &Ell24Sketch| a.merge_in_place(b), |s: &Ell24Sketch| s.estimate_ml(&pmf_ell_8).0);

        let pmf_ull_11 = Arc::new(PreimagePmf::build(UllSketch::new(11).dist_spec_64()));
        run_eq_row!("2.0pct", 2.0, "ULL(p=11)", "yes", UllSketch::new(11).total_bytes(),
            || UllSketch::new(11), |s: &mut UllSketch, h| s.add_hash_64(h), |a: &mut UllSketch, b: &UllSketch| a.merge_in_place(b), |s: &UllSketch| s.estimate_ml(&pmf_ull_11).0);

        let pmf_hll_11 = Arc::new(PreimagePmf::build(Hll6Sketch::new(11).dist_spec_64()));
        run_eq_row!("2.0pct", 2.0, "HLL++(6b,p=11)", "yes", Hll6Sketch::new(11).total_bytes(),
            || Hll6Sketch::new(11), |s: &mut Hll6Sketch, h| s.add_hash_64(h), |a: &mut Hll6Sketch, b: &Hll6Sketch| a.merge_in_place(b), |s: &Hll6Sketch| s.estimate_ml(&pmf_hll_11).0);
    }

    // Tier 2: ~1.0% RMSE
    {
        let pmf_5b_8 = Arc::new(PreimagePmf::build(OsBmw5b::new(8, HashMode5b::Pow2).dist_spec_64()));
        run_eq_row!("1.0pct", 1.0, "OS-BMW-5b(k24,8blk)", "yes", OsBmw5b::state_bytes_for(8),
            || OsBmw5b::new(8, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.estimate_ml(&pmf_5b_8).0);

        let pmf_256_16 = Arc::new(PreimagePmf::build(OsBmw256::new(16, HashMode5b::Div3).dist_spec_64()));
        run_eq_row!("1.0pct", 1.0, "OS-BMW-5b-256B(16blk)", "yes", OsBmw256::state_bytes_for(16),
            || OsBmw256::new(16, HashMode5b::Div3), |s: &mut OsBmw256, h| { s.add_hash_64_div3_mulshift_deferred(h); }, |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw256| s.estimate_ml(&pmf_256_16).0);

        let pmf_ell_10 = Arc::new(PreimagePmf::build(Ell24Sketch::new(10).dist_spec_64()));
        run_eq_row!("1.0pct", 1.0, "ELL(2,24)(p=10)", "yes", Ell24Sketch::new(10).total_bytes(),
            || Ell24Sketch::new(10), |s: &mut Ell24Sketch, h| s.add_hash_64(h), |a: &mut Ell24Sketch, b: &Ell24Sketch| a.merge_in_place(b), |s: &Ell24Sketch| s.estimate_ml(&pmf_ell_10).0);

        let pmf_ull_12 = Arc::new(PreimagePmf::build(UllSketch::new(12).dist_spec_64()));
        run_eq_row!("1.0pct", 1.0, "ULL(p=12)", "yes", UllSketch::new(12).total_bytes(),
            || UllSketch::new(12), |s: &mut UllSketch, h| s.add_hash_64(h), |a: &mut UllSketch, b: &UllSketch| a.merge_in_place(b), |s: &UllSketch| s.estimate_ml(&pmf_ull_12).0);

        let pmf_hll_13 = Arc::new(PreimagePmf::build(Hll6Sketch::new(13).dist_spec_64()));
        run_eq_row!("1.0pct", 1.0, "HLL++(6b,p=13)", "yes", Hll6Sketch::new(13).total_bytes(),
            || Hll6Sketch::new(13), |s: &mut Hll6Sketch, h| s.add_hash_64(h), |a: &mut Hll6Sketch, b: &Hll6Sketch| a.merge_in_place(b), |s: &Hll6Sketch| s.estimate_ml(&pmf_hll_13).0);

        run_eq_row!("1.0pct", 1.0, "AVLL(base56,512w)", "non_assoc", HEADER_BYTES + 512 * 8,
            || AvllSketch::new_words(512), |s: &mut AvllSketch, h| { s.add_hash_64(h); }, |a: &mut AvllSketch, b: &AvllSketch| { *a = a.merge(b); }, |s: &AvllSketch| s.estimate_ml_exact_mod().0);
    }

    // Tier 3: ~0.60% RMSE (matches DataFusion built-in 16KB HLL!)
    {
        let pmf_5b_16 = Arc::new(PreimagePmf::build(OsBmw5b::new(16, HashMode5b::Pow2).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "OS-BMW-5b(k24,16blk)", "yes", OsBmw5b::state_bytes_for(16),
            || OsBmw5b::new(16, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.estimate_ml(&pmf_5b_16).0);

        let pmf_5b_32 = Arc::new(PreimagePmf::build(OsBmw5b::new(32, HashMode5b::Pow2).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "OS-BMW-5b(k24,32blk)", "yes", OsBmw5b::state_bytes_for(32),
            || OsBmw5b::new(32, HashMode5b::Pow2), |s: &mut OsBmw5b, h| { s.add_hash_64(h); }, |a: &mut OsBmw5b, b: &OsBmw5b| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw5b| s.estimate_ml(&pmf_5b_32).0);

        let pmf_256_32 = Arc::new(PreimagePmf::build(OsBmw256::new(32, HashMode5b::Div3).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "OS-BMW-5b-256B(32blk)", "yes", OsBmw256::state_bytes_for(32),
            || OsBmw256::new(32, HashMode5b::Div3), |s: &mut OsBmw256, h| { s.add_hash_64_div3_mulshift_deferred(h); }, |a: &mut OsBmw256, b: &OsBmw256| { a.merge_in_place(b).unwrap(); }, |s: &OsBmw256| s.estimate_ml(&pmf_256_32).0);

        let pmf_ell_11 = Arc::new(PreimagePmf::build(Ell24Sketch::new(11).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "ELL(2,24)(p=11)", "yes", Ell24Sketch::new(11).total_bytes(),
            || Ell24Sketch::new(11), |s: &mut Ell24Sketch, h| s.add_hash_64(h), |a: &mut Ell24Sketch, b: &Ell24Sketch| a.merge_in_place(b), |s: &Ell24Sketch| s.estimate_ml(&pmf_ell_11).0);

        let pmf_ull_14 = Arc::new(PreimagePmf::build(UllSketch::new(14).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "ULL(p=14)", "yes", UllSketch::new(14).total_bytes(),
            || UllSketch::new(14), |s: &mut UllSketch, h| s.add_hash_64(h), |a: &mut UllSketch, b: &UllSketch| a.merge_in_place(b), |s: &UllSketch| s.estimate_ml(&pmf_ull_14).0);

        let pmf_hll_14 = Arc::new(PreimagePmf::build(Hll6Sketch::new(14).dist_spec_64()));
        run_eq_row!("0.60pct", 0.60, "HLL++(6b,p=14)", "yes", Hll6Sketch::new(14).total_bytes(),
            || Hll6Sketch::new(14), |s: &mut Hll6Sketch, h| s.add_hash_64(h), |a: &mut Hll6Sketch, b: &Hll6Sketch| a.merge_in_place(b), |s: &Hll6Sketch| s.estimate_ml(&pmf_hll_14).0);
    }

    // =========================================================================
    // Part B: Single-Machine Martingale/HIP (K=1) vs Distributed Merge (K=1..64)
    // =========================================================================
    let n_b = 1_000_000u64;
    let trials_b = 1000usize;
    println!("\n=== [Part B] Single-Machine Martingale/HIP (K=1) vs Distributed Merge (K in {{1,2,4,16,64}}, n=1e6, {} matched-seed trials) ===", trials_b);

    // 1. OS-BMW-5b(k24, 8blk, 2760 B) — Martingale K=1 (+16B state -> 2776 B) vs Mergeable MLE K=1..64 (2760 B)
    {
        let pmf = Arc::new(PreimagePmf::build(OsBmw5b::new(8, HashMode5b::Pow2).dist_spec_64()));
        let mut errs = Vec::with_capacity(trials_b);
        let per_t = (trials_b + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut hs = Vec::new();
            for tid in 0..n_threads {
                let st = tid * per_t;
                let en = (st + per_t).min(trials_b);
                if st >= en { continue; }
                let pmf_c = Arc::clone(&pmf);
                hs.push(s.spawn(move || {
                    let mut loc = Vec::with_capacity(en - st);
                    for tr in st..en {
                        let seed = matched_trial_seed(tr, n_b);
                        let mut msk = OsBmwMartingale::<512, 24>::new(8, HashMode5b::Pow2);
                        for i in 0..n_b {
                            msk.add_hash_64(wyhash64(i, seed), &pmf_c);
                        }
                        loc.push((msk.c_mart - n_b as f64) / (n_b as f64));
                    }
                    loc
                }));
            }
            for h in hs { errs.extend_from_slice(&h.join().unwrap()); }
        });
        let nt = errs.len() as f64;
        let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
        let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
        let rmse = mse.sqrt() * 100.0;
        let ram = OsBmw5b::state_bytes_for(8) + 16;
        let mvp = (ram as f64 * 8.0) * mse;
        let eq_b = (ram as f64) * (rmse / 1.0).powi(2);
        println!("  [Part B] OS-BMW-5b(k24)     Martingale_HIP K=1  | RAM={}B | bias={:+.4}% rmse={:.4}% mvp={:.4}", ram, bias, rmse, mvp);
        writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tOS-BMW-5b(k24)\tMartingale_HIP\tno_single_only\t1\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
            n_b, trials_b, ram, bias, rmse, mvp, eq_b).unwrap();

        for &ks in &[1usize, 2, 4, 16, 64] {
            let (bias, rmse, mse) = eval_k_shards(n_b, ks, trials_b, n_threads,
                || OsBmw5b::new(8, HashMode5b::Pow2), |s, h| { s.add_hash_64(h); }, |a, b| { a.merge_in_place(b).unwrap(); }, |s| s.estimate_ml(&pmf).0);
            let ram_m = OsBmw5b::state_bytes_for(8);
            let mvp = (ram_m as f64 * 8.0) * mse;
            let eq_b = (ram_m as f64) * (rmse / 1.0).powi(2);
            writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tOS-BMW-5b(k24)\tMergeable_MLE\tyes\t{}\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
                ks, n_b, trials_b, ram_m, bias, rmse, mvp, eq_b).unwrap();
        }
        f.flush().unwrap();
    }

    // 2. OS-BMW-5b-256B (16blk, 4104 B) — Martingale K=1 vs Mergeable MLE K=1..64
    {
        let pmf = Arc::new(PreimagePmf::build(OsBmw256::new(16, HashMode5b::Div3).dist_spec_64()));
        let mut errs = Vec::with_capacity(trials_b);
        let per_t = (trials_b + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut hs = Vec::new();
            for tid in 0..n_threads {
                let st = tid * per_t;
                let en = (st + per_t).min(trials_b);
                if st >= en { continue; }
                let pmf_c = Arc::clone(&pmf);
                hs.push(s.spawn(move || {
                    let mut loc = Vec::with_capacity(en - st);
                    for tr in st..en {
                        let seed = matched_trial_seed(tr, n_b);
                        let mut msk = OsBmwMartingale::<384, 16>::new(16, HashMode5b::Div3);
                        for i in 0..n_b {
                            msk.add_hash_64(wyhash64(i, seed), &pmf_c);
                        }
                        loc.push((msk.c_mart - n_b as f64) / (n_b as f64));
                    }
                    loc
                }));
            }
            for h in hs { errs.extend_from_slice(&h.join().unwrap()); }
        });
        let nt = errs.len() as f64;
        let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
        let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
        let rmse = mse.sqrt() * 100.0;
        let ram = OsBmw256::state_bytes_for(16) + 16;
        let mvp = (ram as f64 * 8.0) * mse;
        let eq_b = (ram as f64) * (rmse / 1.0).powi(2);
        println!("  [Part B] OS-BMW-5b-256B     Martingale_HIP K=1  | RAM={}B | bias={:+.4}% rmse={:.4}% mvp={:.4}", ram, bias, rmse, mvp);
        writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tOS-BMW-5b-256B\tMartingale_HIP\tno_single_only\t1\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
            n_b, trials_b, ram, bias, rmse, mvp, eq_b).unwrap();

        for &ks in &[1usize, 2, 4, 16, 64] {
            let (bias, rmse, mse) = eval_k_shards(n_b, ks, trials_b, n_threads,
                || OsBmw256::new(16, HashMode5b::Div3), |s, h| { s.add_hash_64_div3_mulshift_deferred(h); }, |a, b| { a.merge_in_place(b).unwrap(); }, |s| s.estimate_ml(&pmf).0);
            let ram_m = OsBmw256::state_bytes_for(16);
            let mvp = (ram_m as f64 * 8.0) * mse;
            let eq_b = (ram_m as f64) * (rmse / 1.0).powi(2);
            writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tOS-BMW-5b-256B\tMergeable_MLE\tyes\t{}\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
                ks, n_b, trials_b, ram_m, bias, rmse, mvp, eq_b).unwrap();
        }
        f.flush().unwrap();
    }

    // 3. ELL(2, 24) (p=10, 4104 B) & ELL(2, 16) (p=10, 3080 B, EDBT'25 optimal Martingale!)
    for &(ename, d_bits) in &[("ELL(2,24)", 24u32), ("ELL(2,16)", 16u32)] {
        let p = 10u32;
        let pmf = Arc::new(PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - p }));
        let mut errs = Vec::with_capacity(trials_b);
        let per_t = (trials_b + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut hs = Vec::new();
            for tid in 0..n_threads {
                let st = tid * per_t;
                let en = (st + per_t).min(trials_b);
                if st >= en { continue; }
                let pmf_c = Arc::clone(&pmf);
                hs.push(s.spawn(move || {
                    let mut loc = Vec::with_capacity(en - st);
                    for tr in st..en {
                        let seed = matched_trial_seed(tr, n_b);
                        let mut sk = EllGen::new(p, d_bits);
                        let mut c_mart = 0.0_f64;
                        let mut p_ch = 1.0_f64;
                        let mf = sk.m as f64;
                        for i in 0..n_b {
                            if let Some((old_r, new_r)) = sk.add_hash_64(wyhash64(i, seed)) {
                                c_mart += 1.0 / p_ch;
                                p_ch -= (ell_reg_open(old_r, d_bits, &pmf_c) - ell_reg_open(new_r, d_bits, &pmf_c)) / mf;
                            }
                        }
                        loc.push((c_mart - n_b as f64) / (n_b as f64));
                    }
                    loc
                }));
            }
            for h in hs { errs.extend_from_slice(&h.join().unwrap()); }
        });
        let nt = errs.len() as f64;
        let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
        let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
        let rmse = mse.sqrt() * 100.0;
        let ram_base = EllGen::new(p, d_bits).packed_bytes();
        let ram = ram_base + 16;
        let mvp = (ram as f64 * 8.0) * mse;
        let eq_b = (ram as f64) * (rmse / 1.0).powi(2);
        println!("  [Part B] {:18} Martingale_HIP K=1  | RAM={}B | bias={:+.4}% rmse={:.4}% mvp={:.4}", ename, ram, bias, rmse, mvp);
        writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\t{}\tMartingale_HIP\tno_single_only\t1\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
            ename, n_b, trials_b, ram, bias, rmse, mvp, eq_b).unwrap();

        for &ks in &[1usize, 2, 4, 16, 64] {
            let (bias, rmse, mse) = eval_k_shards(n_b, ks, trials_b, n_threads,
                || EllGen::new(p, d_bits), |s, h| { s.add_hash_64(h); }, |a, b| a.merge_in_place(b), |s| s.estimate_ml(&pmf));
            let mvp = (ram_base as f64 * 8.0) * mse;
            let eq_b = (ram_base as f64) * (rmse / 1.0).powi(2);
            writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\t{}\tMergeable_MLE\tyes\t{}\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
                ename, ks, n_b, trials_b, ram_base, bias, rmse, mvp, eq_b).unwrap();
        }
        f.flush().unwrap();
    }

    // 4. ULL (p=12, 4104 B) — Martingale K=1 vs Mergeable MLE K=1..64
    {
        let p = 12u32;
        let pmf = Arc::new(PreimagePmf::build(UllSketch::new(p).dist_spec_64()));
        let mut errs = Vec::with_capacity(trials_b);
        let per_t = (trials_b + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut hs = Vec::new();
            for tid in 0..n_threads {
                let st = tid * per_t;
                let en = (st + per_t).min(trials_b);
                if st >= en { continue; }
                let pmf_c = Arc::clone(&pmf);
                hs.push(s.spawn(move || {
                    let mut loc = Vec::with_capacity(en - st);
                    for tr in st..en {
                        let seed = matched_trial_seed(tr, n_b);
                        let mut sk = UllSketch::new(p);
                        let mut c_mart = 0.0_f64;
                        let mut p_ch = 1.0_f64;
                        let mf = sk.m as f64;
                        let mask = (sk.m as u64) - 1;
                        let sentinel = 1u64 << (p - 1);
                        for i in 0..n_b {
                            let h = wyhash64(i, seed);
                            let idx = (h & mask) as usize;
                            let old_r = sk.buf[HEADER_BYTES + idx];
                            sk.add_hash_64(h);
                            let new_r = sk.buf[HEADER_BYTES + idx];
                            if new_r != old_r {
                                c_mart += 1.0 / p_ch;
                                p_ch -= (ull_reg_open(old_r, &pmf_c) - ull_reg_open(new_r, &pmf_c)) / mf;
                            }
                            let _ = sentinel;
                        }
                        loc.push((c_mart - n_b as f64) / (n_b as f64));
                    }
                    loc
                }));
            }
            for h in hs { errs.extend_from_slice(&h.join().unwrap()); }
        });
        let nt = errs.len() as f64;
        let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
        let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
        let rmse = mse.sqrt() * 100.0;
        let ram_base = UllSketch::new(p).total_bytes();
        let ram = ram_base + 16;
        let mvp = (ram as f64 * 8.0) * mse;
        let eq_b = (ram as f64) * (rmse / 1.0).powi(2);
        println!("  [Part B] ULL(p=12)          Martingale_HIP K=1  | RAM={}B | bias={:+.4}% rmse={:.4}% mvp={:.4}", ram, bias, rmse, mvp);
        writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tULL(p=12)\tMartingale_HIP\tno_single_only\t1\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
            n_b, trials_b, ram, bias, rmse, mvp, eq_b).unwrap();

        for &ks in &[1usize, 2, 4, 16, 64] {
            let (bias, rmse, mse) = eval_k_shards(n_b, ks, trials_b, n_threads,
                || UllSketch::new(p), |s, h| s.add_hash_64(h), |a, b| a.merge_in_place(b), |s| s.estimate_ml(&pmf).0);
            let mvp = (ram_base as f64 * 8.0) * mse;
            let eq_b = (ram_base as f64) * (rmse / 1.0).powi(2);
            writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tULL(p=12)\tMergeable_MLE\tyes\t{}\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
                ks, n_b, trials_b, ram_base, bias, rmse, mvp, eq_b).unwrap();
        }
        f.flush().unwrap();
    }

    // 5. HLL++(6b, p=12, 3080 B) — HIP K=1 vs Mergeable MLE K=1..64
    {
        let p = 12u32;
        let pmf = Arc::new(PreimagePmf::build(Hll6Sketch::new(p).dist_spec_64()));
        let mut errs = Vec::with_capacity(trials_b);
        let per_t = (trials_b + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut hs = Vec::new();
            for tid in 0..n_threads {
                let st = tid * per_t;
                let en = (st + per_t).min(trials_b);
                if st >= en { continue; }
                let pmf_c = Arc::clone(&pmf);
                hs.push(s.spawn(move || {
                    let mut loc = Vec::with_capacity(en - st);
                    for tr in st..en {
                        let seed = matched_trial_seed(tr, n_b);
                        let mut sk = Hll6Sketch::new(p);
                        let mut c_hip = 0.0_f64;
                        let mut p_ch = 1.0_f64;
                        let mf = sk.m as f64;
                        let mask = (sk.m as u64) - 1;
                        let sentinel = 1u64 << (p - 1);
                        for i in 0..n_b {
                            let h = wyhash64(i, seed);
                            let idx = (h & mask) as usize;
                            let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
                            let old = sk.get_reg(idx);
                            if lvl > old {
                                c_hip += 1.0 / p_ch;
                                p_ch -= (pmf_c.tail[old as usize] - pmf_c.tail[lvl as usize]) / mf;
                                sk.set_reg(idx, lvl);
                            }
                        }
                        loc.push((c_hip - n_b as f64) / (n_b as f64));
                    }
                    loc
                }));
            }
            for h in hs { errs.extend_from_slice(&h.join().unwrap()); }
        });
        let nt = errs.len() as f64;
        let bias = (errs.iter().sum::<f64>() / nt) * 100.0;
        let mse = errs.iter().map(|e| e * e).sum::<f64>() / nt;
        let rmse = mse.sqrt() * 100.0;
        let ram_base = Hll6Sketch::new(p).total_bytes();
        let ram = ram_base + 16;
        let mvp = (ram as f64 * 8.0) * mse;
        let eq_b = (ram as f64) * (rmse / 1.0).powi(2);
        println!("  [Part B] HLL++(6b,p=12)     Martingale_HIP K=1  | RAM={}B | bias={:+.4}% rmse={:.4}% mvp={:.4}", ram, bias, rmse, mvp);
        writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tHLL++(6b,p=12)\tMartingale_HIP\tno_single_only\t1\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
            n_b, trials_b, ram, bias, rmse, mvp, eq_b).unwrap();

        for &ks in &[1usize, 2, 4, 16, 64] {
            let (bias, rmse, mse) = eval_k_shards(n_b, ks, trials_b, n_threads,
                || Hll6Sketch::new(p), |s, h| s.add_hash_64(h), |a, b| a.merge_in_place(b), |s| s.estimate_ml(&pmf).0);
            let mvp = (ram_base as f64 * 8.0) * mse;
            let eq_b = (ram_base as f64) * (rmse / 1.0).powi(2);
            writeln!(f, "B_Martingale_vs_Merge\t4KB_nom\tHLL++(6b,p=12)\tMergeable_MLE\tyes\t{}\t{}\t{}\t{}\t1.00\t{:.4}\t{:.4}\t{:.4}\t{:.1}\tNaN\tNaN\tNaN",
                ks, n_b, trials_b, ram_base, bias, rmse, mvp, eq_b).unwrap();
        }
        f.flush().unwrap();
    }

    println!("DONE m2_equal_error_hip -> {}", out_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit_martingale_invariants<const G: usize, const K: usize>(name: &str, n_blk: usize, mode: HashMode5b) {
        let pmf = PreimagePmf::build(OsBmw5bInPlace::<G, K, false>::new(n_blk, mode).dist_spec_64());
        let mut msk = OsBmwMartingale::<G, K>::new(n_blk, mode);
        let seed = matched_trial_seed(0, 1_000_000);
        let mut max_p_diff = 0.0_f64;
        let mut missed_changes = 0usize;
        let mut false_changes = 0usize;
        let mut prev_buf = msk.sk.buf.clone();
        let mut prev_c = msk.c_mart;

        let mut promotions = 0usize;
        let mut exact_var_sum = 0.0_f64;
        // Verify EXACT buf-change <=> c_mart-change and p_change == recompute_p_change through promotions up to n = 1_000_000!
        for i in 0..1_000_000u64 {
            exact_var_sum += 1.0 / msk.p_change - 1.0;
            let h = wyhash64(i, seed);
            let old_anchors: Vec<u8> = msk.sk.buf[HEADER_BYTES..HEADER_BYTES + n_blk].to_vec();
            msk.add_hash_64(h, &pmf);
            let new_anchors = &msk.sk.buf[HEADER_BYTES..HEADER_BYTES + n_blk];
            let promoted = new_anchors != old_anchors.as_slice();
            if promoted {
                promotions += 1;
            }
            if i < 200_000 || promoted {
                let buf_changed = msk.sk.buf != prev_buf;
                let c_changed = msk.c_mart != prev_c;
                if buf_changed && !c_changed {
                    missed_changes += 1;
                }
                if !buf_changed && c_changed {
                    false_changes += 1;
                }
                if buf_changed || promoted {
                    let exact_p = msk.recompute_p_change(&pmf);
                    let diff = (msk.p_change - exact_p).abs();
                    if diff > max_p_diff {
                        max_p_diff = diff;
                    }
                }
            }
            if i < 200_000 || promoted {
                prev_buf.copy_from_slice(&msk.sk.buf);
                prev_c = msk.c_mart;
            }
        }
        // Check empirical transition rate vs p_change at n = 1_000_000 (AFTER promotions!)
        let p_at_1m = msk.p_change;
        let exact_1m = msk.recompute_p_change(&pmf);
        let mut hits = 0usize;
        let probes = 500_000u64;
        for j in 0..probes {
            let mut tmp = msk.sk.clone();
            let h = wyhash64(10_000_000 + j, seed);
            tmp.add_hash_64(h);
            if tmp.buf != msk.sk.buf {
                hits += 1;
            }
        }
        let emp_p = hits as f64 / probes as f64;
        let anchors_final = &msk.sk.buf[HEADER_BYTES..HEADER_BYTES + n_blk];
        let pred_rmse_pct = (exact_var_sum.sqrt() / 1_000_000.0) * 100.0;
        let ram_b = OsBmw5bInPlace::<G, K, false>::state_bytes_for(n_blk) + 16;
        let pred_mvp = (ram_b as f64 * 8.0) * (exact_var_sum / 1e12);
        println!(
            "[AUDIT {}] promo={} anchors={:?} missed={} false={} max_p_diff={:.3e} c_mart={:.1} p_1m={:.6} exact_1m={:.6} emp_p={:.6} rel_err={:+.3}% | exact_mart_pred: rmse={:.4}% mvp={:.4}",
            name, promotions, anchors_final, missed_changes, false_changes, max_p_diff, msk.c_mart, p_at_1m, exact_1m, emp_p,
            (emp_p - p_at_1m) / p_at_1m * 100.0, pred_rmse_pct, pred_mvp
        );
        assert_eq!(missed_changes, 0, "{}: missed state changes!", name);
        assert_eq!(false_changes, 0, "{}: false state changes!", name);
        assert!(max_p_diff < 1e-10, "{}: p_change drift {:.3e}!", name, max_p_diff);
        assert!((p_at_1m - exact_1m).abs() < 1e-10, "{}: p_1m != exact_1m!", name);
        assert!(((emp_p - p_at_1m) / p_at_1m).abs() < 0.08, "{}: empirical transition prob mismatch!", name);
    }

    #[test]
    fn test_osbmw_martingale_exact_p_change_and_transition_invariant() {
        audit_martingale_invariants::<512, 24>("OS-BMW-5b(8blk,Pow2)", 8, HashMode5b::Pow2);
        audit_martingale_invariants::<384, 16>("OS-BMW-5b-256B(16blk,Div3)", 16, HashMode5b::Div3);
    }
}
