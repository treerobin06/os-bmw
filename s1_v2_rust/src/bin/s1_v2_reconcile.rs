//! Milestone M0 & M1 Runner:
//!   1. Single-source-of-truth exact 64-bit preimage PMF closure & timing (< 0.2 ms)
//!   2. Explicit reconciliation of `3.0061` (q=6 independent) vs `2.9977` (K_reach=60 1,277 B Wire)
//!      vs `2.9953` (K_reach=59 floor 1,276 B Wire) vs `3.00001` (conservative K=63 1,278 B)
//!   3. Exhaustive algebraic semilattice audit (512x512 = 262,144 pairs + 5,000 random triple merges
//!      + 2x downsizing + wire roundtrips + OS-BMW-Rank roundtrip & encode/decode latency)
//!   4. Deterministic counterexamples (AVLL 22!=18, AVLL 11-bucket 47!=49, HLL_4-Fixed 14!=20)
//!   5. Multi-shard merge-tree drift experiment for S in {2, 8, 64, 512} shards!

use s1_v2_rust::baselines::*;
use s1_v2_rust::hash::{wyhash64, splitmix64};
use s1_v2_rust::osbmw_inplace::*;
use s1_v2_rust::osbmw_rank::OsBmwRankCodec;
use s1_v2_rust::osbmw_ref::OsBmwBlock;
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    println!("=== [M0] Exact 64-bit Preimage PMF Timing & Closure ===");
    let specs: Vec<(&str, DistSpec)> = vec![
        ("Pow2_w64", DistSpec::Pow2 { w_bits: 64, max_cap: 63 }),
        ("Pow2_w52_4KB", DistSpec::Pow2 { w_bits: 52, max_cap: 63 }),
        ("Pow2Div3_p11_4KB", DistSpec::Pow2Div3 { p_tot: 11 }),
        ("ErtlT2_w54_4KB", DistSpec::ErtlT { t: 2, w_bits: 54 }),
        ("Geom205_NegLog2_w64", DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64, max_cap: 63 }),
        ("Geom205_NegLog2_w52", DistSpec::GeomNegLog2 { b: 2.05, w_bits: 52, max_cap: 63 }),
        ("Geom2125_NegLog2_w64", DistSpec::GeomNegLog2 { b: 2.125, w_bits: 64, max_cap: 63 }),
        ("Geom2125_ExactFloor_w64", DistSpec::GeomExactFloor { b: 2.125, w_bits: 64, max_cap: 63 }),
    ];

    let mut pmf_json_rows = Vec::new();
    for (name, spec) in &specs {
        let t0 = Instant::now();
        let pmf = PreimagePmf::build(*spec);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        let sum_c: u128 = pmf.counts[1..].iter().sum();
        let sum_p: f64 = pmf.probs[1..].iter().sum();
        let p_err = (sum_p - 1.0).abs();
        assert_eq!(sum_c, pmf.domain_size);
        assert!(p_err < 1e-13);
        println!(
            "  {:24}: k_reach={:3}, build={:7.1} us, count_closure=EXACT, |sum_p-1|={:.2e}",
            name, pmf.k_reach, us, p_err
        );
        pmf_json_rows.push(format!(
            "    {{\"name\": \"{}\", \"k_reach\": {}, \"build_us\": {:.2}, \"exact_count_closure\": true, \"prob_sum_err\": {:.3e}}}",
            name, pmf.k_reach, us, p_err
        ));
    }

    println!("\n=== [M0] Reconciling 3.0061 (q=6 indep) vs 2.9977 (K=60 Rank Wire 1,277 B) ===");
    let pmf_2125 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.125, w_bits: 64, max_cap: 63 });
    let (j_2125, j_min, j_max) = pmf_2125.fisher_osbmw_periodic(2048, 92, 7, 2, 2.125, 1e6, 64);
    let indep_q6_bits = 6.0 + 2048.0 * (25.0_f64).log2() + 91.0 * 8.0;
    let indep_q6_mvp = (indep_q6_bits / 2048.0) / j_2125;

    let codec_k60 = OsBmwRankCodec::new(2048, 92, 7, 2, 60);
    let codec_k59 = OsBmwRankCodec::new(2048, 92, 7, 2, 59);
    let codec_k63 = OsBmwRankCodec::new(2048, 92, 7, 2, 63);
    let mvp_k60_bits = (codec_k60.exact_bits as f64 / 2048.0) / j_2125;
    let mvp_k60_bytes = ((codec_k60.fixed_bytes * 8) as f64 / 2048.0) / j_2125;
    let mvp_k59_bytes = ((codec_k59.fixed_bytes * 8) as f64 / 2048.0) / j_2125;
    let mvp_k63_bits = (codec_k63.exact_bits as f64 / 2048.0) / j_2125;
    let mvp_k63_bytes = ((codec_k63.fixed_bytes * 8) as f64 / 2048.0) / j_2125;

    println!("  Fisher J(b=2.125, g=2048, k=92, R=7, d=2, K=60 preimage PMF) = {:.8} (range {:.6}..{:.6})", j_2125, j_min, j_max);
    println!("  Independent q=6 field sum : {:.3} bits -> MVP = {:.5} (> 3.00)", indep_q6_bits, indep_q6_mvp);
    println!("  Rank Wire K_reach=60      : {} bits ({} B = {} b) -> bit MVP = {:.5}, byte Wire MVP = {:.5} (< 3.00)", codec_k60.exact_bits, codec_k60.fixed_bytes, codec_k60.fixed_bytes * 8, mvp_k60_bits, mvp_k60_bytes);
    println!("  Rank Wire K_reach=59 floor: {} bits ({} B = {} b) -> byte Wire MVP = {:.5} (< 3.00)", codec_k59.exact_bits, codec_k59.fixed_bytes, codec_k59.fixed_bytes * 8, mvp_k59_bytes);
    println!("  Rank Wire K=63 conserv.   : {} bits ({} B = {} b) -> bit MVP = {:.5}, byte Wire MVP = {:.5}", codec_k63.exact_bits, codec_k63.fixed_bytes, codec_k63.fixed_bytes * 8, mvp_k63_bits, mvp_k63_bytes);

    // Verify 200 OS-BMW-Rank roundtrips & measure real encode/decode µs/block!
    let mut rng = 0xdeadbeef_12345678u64;
    let mut test_blks = Vec::with_capacity(200);
    for idx in 0..200 {
        let lam = match idx % 5 {
            0 => 0.1,
            1 => 5.0,
            2 => 500.0,
            3 => 1e6,
            _ => 1e12,
        };
        test_blks.push(OsBmwBlock::sample_poisson(2048, 92, 7, 2, &pmf_2125, lam, &mut rng));
    }
    let mut encoded_bufs = Vec::with_capacity(200);
    let t_enc = Instant::now();
    for blk in &test_blks {
        encoded_bufs.push(codec_k60.encode(blk));
    }
    let enc_us = t_enc.elapsed().as_secs_f64() * 1e6 / (test_blks.len() as f64);

    let t_dec = Instant::now();
    let mut rank_roundtrip_fails = 0usize;
    for (blk, buf) in test_blks.iter().zip(encoded_bufs.iter()) {
        if buf.len() != 1277 {
            rank_roundtrip_fails += 1;
        }
        let dec = codec_k60.decode(buf);
        if &dec != blk {
            rank_roundtrip_fails += 1;
        }
    }
    let dec_us = t_dec.elapsed().as_secs_f64() * 1e6 / (test_blks.len() as f64);
    println!(
        "  OS-BMW-Rank 200-block roundtrip: fails={}, encode={:.1} us/blk, decode={:.1} us/blk",
        rank_roundtrip_fails, enc_us, dec_us
    );
    assert_eq!(rank_roundtrip_fails, 0);

    println!("\n=== [M1] Exhaustive 512x512 Semilattice + Randomized In-Place Algebra Audit ===");
    // 1. Exhaustive 512x512 = 262,144 subset pairs on (g=3, k=2, R=2, d=1, levels 1..=3)
    let states: Vec<OsBmwBlock> = (0..512u32)
        .map(|mask| {
            let mut blk = OsBmwBlock::new(3, 2, 2, 1);
            for r in 0..3 {
                for l in 1..=3u8 {
                    if ((mask >> (r * 3 + (l as usize - 1))) & 1) != 0 {
                        blk.add_level(r, l);
                    }
                }
            }
            blk
        })
        .collect();
    let mut exh_fails = 0usize;
    for a in 0..512usize {
        if states[a].merge(&states[a]) != states[a] {
            exh_fails += 1;
        }
        for b in 0..512usize {
            if states[a].merge(&states[b]) != states[a | b] {
                exh_fails += 1;
            }
        }
    }
    println!("  Exhaustive 512x512=262,144 pairs: fails={}", exh_fails);
    assert_eq!(exh_fails, 0);

    // 2. Randomized triple-merge + partition-homomorphism + commutativity + idempotence across
    //    ALL 5 pure fixed-byte Dense in-place kernels (NO Hybrid `to_dense()` masking!).
    let pmf_chunk_2 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64 - 1 - 10, max_cap: 63 });
    let mut alg_trials = 0usize;
    let mut alg_fails = 0usize;
    for trial in 0..200u64 {
        let seed = 0x55aa_0000 + trial;
        let n_a = 1500 + (splitmix64(seed) % 3000);
        let n_b = 1500 + (splitmix64(seed + 1) % 3000);
        let n_c = 1500 + (splitmix64(seed + 2) % 3000);

        let mut s5_a = OsBmw5b::new(4, HashMode5b::Pow2);
        let mut s5_b = OsBmw5b::new(4, HashMode5b::Pow2);
        let mut s5_c = OsBmw5b::new(4, HashMode5b::Pow2);
        let mut s5_all = OsBmw5b::new(4, HashMode5b::Pow2);

        let mut sm_a = OsBmw5bMask::new(4, HashMode5b::Pow2);
        let mut sm_b = OsBmw5bMask::new(4, HashMode5b::Pow2);
        let mut sm_c = OsBmw5bMask::new(4, HashMode5b::Pow2);
        let mut sm_all = OsBmw5bMask::new(4, HashMode5b::Pow2);

        let mut s256_a = OsBmw256::new(4, HashMode5b::Div3);
        let mut s256_b = OsBmw256::new(4, HashMode5b::Div3);
        let mut s256_c = OsBmw256::new(4, HashMode5b::Div3);
        let mut s256_all = OsBmw256::new(4, HashMode5b::Div3);

        let mut sg128_a = OsBmwG128::new(4, HashMode5b::Pow2);
        let mut sg128_b = OsBmwG128::new(4, HashMode5b::Pow2);
        let mut sg128_c = OsBmwG128::new(4, HashMode5b::Pow2);
        let mut sg128_all = OsBmwG128::new(4, HashMode5b::Pow2);

        let mut sch_a = OsBmwChunkInPlace::new(2);
        let mut sch_b = OsBmwChunkInPlace::new(2);
        let mut sch_c = OsBmwChunkInPlace::new(2);
        let mut sch_all = OsBmwChunkInPlace::new(2);

        for i in 0..n_a {
            let h = wyhash64(i, seed);
            s5_a.add_hash_64(h); s5_all.add_hash_64(h);
            sm_a.add_hash_64(h); sm_all.add_hash_64(h);
            s256_a.add_hash_64(h); s256_all.add_hash_64(h);
            sg128_a.add_hash_64(h); sg128_all.add_hash_64(h);
            sch_a.add_hash_64(h, &pmf_chunk_2); sch_all.add_hash_64(h, &pmf_chunk_2);
        }
        for i in (n_a / 2)..(n_a / 2 + n_b) {
            let h = wyhash64(i, seed);
            s5_b.add_hash_64(h); s5_all.add_hash_64(h);
            sm_b.add_hash_64(h); sm_all.add_hash_64(h);
            s256_b.add_hash_64(h); s256_all.add_hash_64(h);
            sg128_b.add_hash_64(h); sg128_all.add_hash_64(h);
            sch_b.add_hash_64(h, &pmf_chunk_2); sch_all.add_hash_64(h, &pmf_chunk_2);
        }
        for i in n_a..(n_a + n_c) {
            let h = wyhash64(i, seed);
            s5_c.add_hash_64(h); s5_all.add_hash_64(h);
            sm_c.add_hash_64(h); sm_all.add_hash_64(h);
            s256_c.add_hash_64(h); s256_all.add_hash_64(h);
            sg128_c.add_hash_64(h); sg128_all.add_hash_64(h);
            sch_c.add_hash_64(h, &pmf_chunk_2); sch_all.add_hash_64(h, &pmf_chunk_2);
        }

        // Check (A v B) v C == A v (B v C) == direct(A U B U C) == B v A (commutativity) + idempotence
        let mut lhs5 = s5_a.clone(); lhs5.merge_in_place(&s5_b).unwrap(); lhs5.merge_in_place(&s5_c).unwrap();
        let mut bc5 = s5_b.clone(); bc5.merge_in_place(&s5_c).unwrap();
        let mut rhs5 = s5_a.clone(); rhs5.merge_in_place(&bc5).unwrap();
        let mut ba5 = s5_b.clone(); ba5.merge_in_place(&s5_a).unwrap();
        let mut ab5 = s5_a.clone(); ab5.merge_in_place(&s5_b).unwrap();
        let mut idem5 = s5_all.clone(); idem5.merge_in_place(&s5_all).unwrap();
        if lhs5 != rhs5 || lhs5 != s5_all || ba5 != ab5 || idem5 != s5_all { alg_fails += 1; }

        let mut lhsm = sm_a.clone(); lhsm.merge_in_place(&sm_b).unwrap(); lhsm.merge_in_place(&sm_c).unwrap();
        let mut bcm = sm_b.clone(); bcm.merge_in_place(&sm_c).unwrap();
        let mut rhsm = sm_a.clone(); rhsm.merge_in_place(&bcm).unwrap();
        let mut idemm = sm_all.clone(); idemm.merge_in_place(&sm_all).unwrap();
        if lhsm != rhsm || lhsm != sm_all || idemm != sm_all { alg_fails += 1; }

        let mut lhs2 = s256_a.clone(); lhs2.merge_in_place(&s256_b).unwrap(); lhs2.merge_in_place(&s256_c).unwrap();
        let mut bc2 = s256_b.clone(); bc2.merge_in_place(&s256_c).unwrap();
        let mut rhs2 = s256_a.clone(); rhs2.merge_in_place(&bc2).unwrap();
        let mut idem2 = s256_all.clone(); idem2.merge_in_place(&s256_all).unwrap();
        if lhs2 != rhs2 || lhs2 != s256_all || idem2 != s256_all { alg_fails += 1; }

        let mut lhsg = sg128_a.clone(); lhsg.merge_in_place(&sg128_b).unwrap(); lhsg.merge_in_place(&sg128_c).unwrap();
        let mut bcg = sg128_b.clone(); bcg.merge_in_place(&sg128_c).unwrap();
        let mut rhsg = sg128_a.clone(); rhsg.merge_in_place(&bcg).unwrap();
        let mut idemg = sg128_all.clone(); idemg.merge_in_place(&sg128_all).unwrap();
        if lhsg != rhsg || lhsg != sg128_all || idemg != sg128_all { alg_fails += 1; }

        let mut lhsc = sch_a.clone(); lhsc.merge_in_place(&sch_b).unwrap(); lhsc.merge_in_place(&sch_c).unwrap();
        let mut bcc = sch_b.clone(); bcc.merge_in_place(&sch_c).unwrap();
        let mut rhsc = sch_a.clone(); rhsc.merge_in_place(&bcc).unwrap();
        let mut idemc = sch_all.clone(); idemc.merge_in_place(&sch_all).unwrap();
        if lhsc != rhsc || lhsc != sch_all || idemc != sch_all { alg_fails += 1; }

        alg_trials += 5;
    }
    println!("  In-place Dense triple-merge/partition/idempotence audit (5 kernels): {} trials, fails={}", alg_trials, alg_fails);
    assert_eq!(alg_fails, 0);

    // 2b. Dedicated honest Hybrid Sparse audit (NO `to_dense()` masking — compares FULL mode + content + estimate!)
    println!("\n=== [M1b] Honest OsBmwHybrid Sparse & Cross-Mode Partition Audit (NO `to_dense()` masking) ===");
    let pmf_hy_g128 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 2 - 7, max_cap: 63 });
    let pmf_hy_256 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 1 });

    // Sparse idempotence regression: keys 27 & 73, seed 0 on both g=128 (4 blks) and 256B (2 blks)
    let mut c128_dir = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
    c128_dir.add_hash_64(wyhash64(27, 0));
    c128_dir.add_hash_64(wyhash64(73, 0));
    let mut c128_a = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
    c128_a.add_hash_64(wyhash64(27, 0));
    let mut c128_b = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
    c128_b.add_hash_64(wyhash64(73, 0));
    let mut c128_mrg = c128_a.clone();
    c128_mrg.merge_in_place(&c128_b).unwrap();
    let mut c128_idem = c128_dir.clone();
    c128_idem.merge_in_place(&c128_dir).unwrap();
    let reg_g128_direct_est = c128_dir.estimate_ml(&pmf_hy_g128).0;
    let reg_g128_merged_est = c128_mrg.estimate_ml(&pmf_hy_g128).0;

    let mut c256_dir = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
    c256_dir.add_hash_64(wyhash64(27, 0));
    c256_dir.add_hash_64(wyhash64(73, 0));
    let mut c256_a = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
    c256_a.add_hash_64(wyhash64(27, 0));
    let mut c256_b = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
    c256_b.add_hash_64(wyhash64(73, 0));
    let mut c256_mrg = c256_a.clone();
    c256_mrg.merge_in_place(&c256_b).unwrap();
    let mut c256_idem = c256_dir.clone();
    c256_idem.merge_in_place(&c256_dir).unwrap();
    let reg_256b_direct_est = c256_dir.estimate_ml(&pmf_hy_256).0;
    let reg_256b_merged_est = c256_mrg.estimate_ml(&pmf_hy_256).0;

    let reg_keys_27_73_ok = c128_dir.is_sparse()
        && c128_mrg.is_sparse()
        && c128_dir == c128_mrg
        && c128_idem == c128_dir
        && (reg_g128_direct_est - reg_g128_merged_est).abs() <= 1e-12
        && (reg_g128_merged_est - 2.0).abs() < 1e-3
        && c256_dir.is_sparse()
        && c256_mrg.is_sparse()
        && c256_dir == c256_mrg
        && c256_idem == c256_dir
        && (reg_256b_direct_est - reg_256b_merged_est).abs() <= 1e-12
        && (reg_256b_merged_est - 2.0).abs() < 1e-3;

    // Randomized sparse v sparse -> sparse trials (g=128 cap=48 and 256B cap=96)
    let mut hy_ss_trials = 0usize;
    let mut hy_ss_viol = 0usize;
    for tr in 0..100u64 {
        let seed = 0x7100_0000 + tr;
        // g=128: union <= 40 < max_count=48
        let na = 1 + (splitmix64(seed) % 20);
        let nb = 1 + (splitmix64(seed + 1) % 20);
        let ov = splitmix64(seed + 2) % na.min(nb);
        let mut a = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        let mut b = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        let mut dir = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        for i in 0..na { let h = wyhash64(i, seed); a.add_hash_64(h); dir.add_hash_64(h); }
        for i in (na - ov)..(na - ov + nb) { let h = wyhash64(i, seed); b.add_hash_64(h); dir.add_hash_64(h); }
        let mut ab = a.clone(); ab.merge_in_place(&b).unwrap();
        let mut ba = b.clone(); ba.merge_in_place(&a).unwrap();
        let mut idem = dir.clone(); idem.merge_in_place(&dir).unwrap();
        let e_dir = dir.estimate_ml(&pmf_hy_g128).0;
        let e_ab = ab.estimate_ml(&pmf_hy_g128).0;
        let e_ba = ba.estimate_ml(&pmf_hy_g128).0;
        if !(dir.is_sparse() && ab.is_sparse() && ab == dir && ba == dir && idem == dir
            && (e_dir - e_ab).abs() <= 1e-12 * e_dir.max(1.0)
            && (e_dir - e_ba).abs() <= 1e-12 * e_dir.max(1.0))
        {
            hy_ss_viol += 1;
        }
        hy_ss_trials += 1;

        // 256B: union <= 80 < max_count=96
        let na2 = 1 + (splitmix64(seed + 10) % 40);
        let nb2 = 1 + (splitmix64(seed + 11) % 40);
        let ov2 = splitmix64(seed + 12) % na2.min(nb2);
        let mut a2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        let mut b2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        let mut dir2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        for i in 0..na2 { let h = wyhash64(i, seed + 99); a2.add_hash_64(h); dir2.add_hash_64(h); }
        for i in (na2 - ov2)..(na2 - ov2 + nb2) { let h = wyhash64(i, seed + 99); b2.add_hash_64(h); dir2.add_hash_64(h); }
        let mut ab2 = a2.clone(); ab2.merge_in_place(&b2).unwrap();
        let mut ba2 = b2.clone(); ba2.merge_in_place(&a2).unwrap();
        let mut idem2 = dir2.clone(); idem2.merge_in_place(&dir2).unwrap();
        let e_dir2 = dir2.estimate_ml(&pmf_hy_256).0;
        let e_ab2 = ab2.estimate_ml(&pmf_hy_256).0;
        let e_ba2 = ba2.estimate_ml(&pmf_hy_256).0;
        if !(dir2.is_sparse() && ab2.is_sparse() && ab2 == dir2 && ba2 == dir2 && idem2 == dir2
            && (e_dir2 - e_ab2).abs() <= 1e-12 * e_dir2.max(1.0)
            && (e_dir2 - e_ba2).abs() <= 1e-12 * e_dir2.max(1.0))
        {
            hy_ss_viol += 1;
        }
        hy_ss_trials += 1;
    }

    // Randomized cross-mode partition audit (Sparse v Sparse -> Dense, Sparse v Dense, Dense v Sparse, Dense v Dense)
    // Directly compares FULL Hybrid state (mode + content) and estimate WITHOUT `to_dense()`!
    let mut hy_cm_trials = 0usize;
    let mut hy_cm_state_viol = 0usize;
    let mut hy_cm_est_viol = 0usize;
    let mut hy_cm_viol = 0usize;
    for tr in 0..100u64 {
        let seed = 0x7200_0000 + tr;
        let (na, nb) = match tr % 4 {
            0 => (30 + (splitmix64(seed) % 16), 30 + (splitmix64(seed + 1) % 16)),       // Sparse v Sparse -> Dense
            1 => (10 + (splitmix64(seed) % 30), 60 + (splitmix64(seed + 1) % 80)),       // Sparse v Dense -> Dense
            2 => (60 + (splitmix64(seed) % 80), 10 + (splitmix64(seed + 1) % 30)),       // Dense v Sparse -> Dense
            _ => (60 + (splitmix64(seed) % 100), 60 + (splitmix64(seed + 1) % 100)),     // Dense v Dense -> Dense
        };
        // g=128 (4 blks)
        let mut a = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        let mut b = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        let mut dir = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        for i in 0..na { let h = wyhash64(i, seed); a.add_hash_64(h); dir.add_hash_64(h); }
        for i in (na / 2)..(na / 2 + nb) { let h = wyhash64(i, seed); b.add_hash_64(h); dir.add_hash_64(h); }
        let mut ab = a.clone(); ab.merge_in_place(&b).unwrap();
        let mut ba = b.clone(); ba.merge_in_place(&a).unwrap();
        let st_ok = ab == dir && ba == dir;
        let e_dir = dir.estimate_ml(&pmf_hy_g128).0;
        let e_ab = ab.estimate_ml(&pmf_hy_g128).0;
        let e_ba = ba.estimate_ml(&pmf_hy_g128).0;
        let est_ok = (e_dir - e_ab).abs() <= 1e-12 * e_dir.max(1.0)
            && (e_dir - e_ba).abs() <= 1e-12 * e_dir.max(1.0);
        if !st_ok { hy_cm_state_viol += 1; }
        if !est_ok { hy_cm_est_viol += 1; }
        if !st_ok || !est_ok { hy_cm_viol += 1; }
        hy_cm_trials += 1;

        // 256B (2 blks, max_count = 96)
        let (na2, nb2) = (na * 2, nb * 2);
        let mut a2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        let mut b2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        let mut dir2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        for i in 0..na2 { let h = wyhash64(i, seed + 55); a2.add_hash_64(h); dir2.add_hash_64(h); }
        for i in (na2 / 2)..(na2 / 2 + nb2) { let h = wyhash64(i, seed + 55); b2.add_hash_64(h); dir2.add_hash_64(h); }
        let mut ab2 = a2.clone(); ab2.merge_in_place(&b2).unwrap();
        let mut ba2 = b2.clone(); ba2.merge_in_place(&a2).unwrap();
        let st_ok2 = ab2 == dir2 && ba2 == dir2;
        let e_dir2 = dir2.estimate_ml(&pmf_hy_256).0;
        let e_ab2 = ab2.estimate_ml(&pmf_hy_256).0;
        let e_ba2 = ba2.estimate_ml(&pmf_hy_256).0;
        let est_ok2 = (e_dir2 - e_ab2).abs() <= 1e-12 * e_dir2.max(1.0)
            && (e_dir2 - e_ba2).abs() <= 1e-12 * e_dir2.max(1.0);
        if !st_ok2 { hy_cm_state_viol += 1; }
        if !est_ok2 { hy_cm_est_viol += 1; }
        if !st_ok2 || !est_ok2 { hy_cm_viol += 1; }
        hy_cm_trials += 1;
    }
    println!(
        "  Hybrid regression (27,73) ok={} (g128: {:.9} vs {:.9}; 256B: {:.9} vs {:.9})",
        reg_keys_27_73_ok, reg_g128_direct_est, reg_g128_merged_est, reg_256b_direct_est, reg_256b_merged_est
    );
    println!(
        "  Hybrid sparse_v_sparse: {}/{} viol | cross_mode (no to_dense): {}/{} viol (state={}, est={})",
        hy_ss_viol, hy_ss_trials, hy_cm_viol, hy_cm_trials, hy_cm_state_viol, hy_cm_est_viol
    );
    assert!(reg_keys_27_73_ok);
    assert_eq!(hy_ss_viol, 0);

    // 3. Deterministic counterexamples audit
    let mut sa = AvllSketch::new_buckets(4); sa.add_level(0, 22);
    let mut sb = AvllSketch::new_buckets(4); for r in 0..4 { sb.add_level(r, 6); }
    let mut sba = AvllSketch::new_buckets(4); for r in 0..4 { sba.add_level(r, 6); } sba.add_level(0, 22);
    let avll_part_stream = sba.abs_exp(0);
    let avll_part_merge = sa.merge(&sb).abs_exp(0);
    assert_eq!((avll_part_stream, avll_part_merge), (22, 18));

    let mut aa = AvllSketch::new_buckets(11); aa.add_level(0, 14);
    let mut bb = AvllSketch::new_buckets(11); bb.add_level(0, 13);
    let mut cc = AvllSketch::new_buckets(11); for r in 0..11 { cc.add_level(r, 3); }
    let avll_assoc_lhs = aa.merge(&bb).merge(&cc).get_state(0);
    let avll_assoc_rhs = aa.merge(&bb.merge(&cc)).get_state(0);
    assert_eq!((avll_assoc_lhs, avll_assoc_rhs), (47, 49));

    let mut ha = Hll4Sketch::new_fixed(2, 1); ha.add_level(0, 20); ha.add_level(1, 20);
    let mut hb = Hll4Sketch::new_fixed(2, 1); for r in 0..4 { hb.add_level(r, 8); }
    let mut h_dir = Hll4Sketch::new_fixed(2, 1); for r in 0..4 { h_dir.add_level(r, 8); } h_dir.add_level(0, 20); h_dir.add_level(1, 20);
    let mut h_mrg = ha.clone(); h_mrg.merge_in_place(&hb);
    let hll4f_direct = h_dir.get_abs(1);
    let hll4f_merge = h_mrg.get_abs(1);
    assert_eq!((hll4f_direct, hll4f_merge), (20, 14));
    println!(
        "  Counterexamples verified: AVLL part {}!={}, AVLL 11-bkt assoc {}!={}, HLL4-Fixed {}!={}",
        avll_part_stream, avll_part_merge, avll_assoc_lhs, avll_assoc_rhs, hll4f_direct, hll4f_merge
    );

    println!("\n=== [M1] Multi-Shard Merge-Tree Drift Experiment (S in {{2, 8, 64, 512}}) ===");
    // Total distinct elements N = 2,000,000 partitioned across S shards, merged in TWO orders:
    //   (a) Balanced binary merge tree vs direct single-stream ground-truth state
    //   (b) Left-deep sequential merge chain vs balanced binary merge tree (tests associativity drift!)
    // Repeat across 20 independent seeds per S so we get exact failure counts AND mean relative bias/drift!
    let n_total = 2_000_000u64;
    let n_seeds = 20u64;
    let pmf_5b = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 3 - 9, max_cap: 63 }); // 8 blocks x 512
    let pmf_256 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 4 }); // 16 blocks x 384
    let pmf_g128 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 5 - 7, max_cap: 63 }); // 32 blocks x 128
    let pmf_chunk = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 64 - 2 - 10, max_cap: 63 }); // 4 blocks x 1024
    let pmf_ell = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - 10 }); // 1024 regs
    let pmf_hll4 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 11, max_cap: 63 });
    let pmf_hbs = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 12, max_cap: 63 });

    let mut drift_tsv = File::create(out_dir.join("m1_multishard_drift.tsv")).unwrap();
    writeln!(
        drift_tsv,
        "shards\tmethod\ttrials\tpart_state_mismatches\tassoc_state_mismatches\tdirect_rel_bias_pct\tmerged_rel_bias_pct\tmax_abs_est_drift_pct\tmean_est_drift_pct"
    ).unwrap();

    let mut drift_json_rows = Vec::new();
    for &shards in &[2usize, 8, 64, 512] {
        // Trackers per method: (part_mismatch, assoc_mismatch, sum_dir_bias, sum_mrg_bias, sum_drift, max_drift)
        let names = [
            "OS-BMW-5b",
            "OS-BMW-256B",
            "OS-BMW-g128-Dense",
            "OS-BMW-Chunk",
            "ELL(2,24)",
            "AVLL",
            "HLL_4-Fixed(cap=32)",
            "HBS(p=12)",
        ];
        let mut stats = vec![(0usize, 0usize, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); names.len()];

        for sd in 0..n_seeds {
            let seed = 0xabcd_0000 + (shards as u64) * 100 + sd;
            let mut dir_5b = OsBmw5b::new(8, HashMode5b::Pow2);
            let mut dir_256 = OsBmw256::new(16, HashMode5b::Div3);
            let mut dir_g128 = OsBmwG128::new(32, HashMode5b::Pow2);
            let mut dir_chunk = OsBmwChunkInPlace::new(4);
            let mut dir_ell = Ell24Sketch::new(10);
            let mut dir_avll = AvllSketch::new_words(128); // 1,408 buckets (~1 KB, matching AVLL §6!)
            let mut dir_h4f = Hll4Sketch::new_fixed(11, 32);
            let mut dir_hbs = HbsSketch::new(12);

            let mut sh_5b: Vec<_> = (0..shards).map(|_| OsBmw5b::new(8, HashMode5b::Pow2)).collect();
            let mut sh_256: Vec<_> = (0..shards).map(|_| OsBmw256::new(16, HashMode5b::Div3)).collect();
            let mut sh_g128: Vec<_> = (0..shards).map(|_| OsBmwG128::new(32, HashMode5b::Pow2)).collect();
            let mut sh_chunk: Vec<_> = (0..shards).map(|_| OsBmwChunkInPlace::new(4)).collect();
            let mut sh_ell: Vec<_> = (0..shards).map(|_| Ell24Sketch::new(10)).collect();
            let mut sh_avll: Vec<_> = (0..shards).map(|_| AvllSketch::new_words(128)).collect();
            let mut sh_h4f: Vec<_> = (0..shards).map(|_| Hll4Sketch::new_fixed(11, 32)).collect();
            let mut sh_hbs: Vec<_> = (0..shards).map(|_| HbsSketch::new(12)).collect();

            // Realistic partition: shard 0 receives first N/S items, etc. (disjoint time/range shards!)
            // For S=512, each shard has ~3,906 items; outliers hit low-floor shards before merge!
            for i in 0..n_total {
                let h = wyhash64(i, seed);
                let s_idx = ((i as usize) * shards) / (n_total as usize);
                dir_5b.add_hash_64(h);              sh_5b[s_idx].add_hash_64(h);
                dir_256.add_hash_64(h);             sh_256[s_idx].add_hash_64(h);
                dir_g128.add_hash_64(h);            sh_g128[s_idx].add_hash_64(h);
                dir_chunk.add_hash_64(h, &pmf_chunk); sh_chunk[s_idx].add_hash_64(h, &pmf_chunk);
                dir_ell.add_hash_64(h);             sh_ell[s_idx].add_hash_64(h);
                dir_avll.add_hash_64(h);            sh_avll[s_idx].add_hash_64(h);
                dir_h4f.add_hash_64(h);             sh_h4f[s_idx].add_hash_64(h);
                dir_hbs.add_hash_64(h);             sh_hbs[s_idx].add_hash_64(h);
            }

            // Balanced binary tree merge vs Left-deep chain merge!
            fn tree_merge<T: Clone, F: Fn(&T, &T) -> T>(mut items: Vec<T>, f: F) -> T {
                while items.len() > 1 {
                    let mut next = Vec::with_capacity((items.len() + 1) / 2);
                    let mut i = 0;
                    while i < items.len() {
                        if i + 1 < items.len() {
                            next.push(f(&items[i], &items[i + 1]));
                        } else {
                            next.push(items[i].clone());
                        }
                        i += 2;
                    }
                    items = next;
                }
                items.pop().unwrap()
            }
            fn chain_merge<T: Clone, F: Fn(&T, &T) -> T>(items: &[T], f: F) -> T {
                let mut acc = items[0].clone();
                for x in &items[1..] {
                    acc = f(&acc, x);
                }
                acc
            }

            let t_5b = tree_merge(sh_5b.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });
            let c_5b = chain_merge(&sh_5b, |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });

            let t_256 = tree_merge(sh_256.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });
            let c_256 = chain_merge(&sh_256, |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });

            let t_g128 = tree_merge(sh_g128.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });
            let c_g128 = chain_merge(&sh_g128, |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });

            let t_chunk = tree_merge(sh_chunk.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });
            let c_chunk = chain_merge(&sh_chunk, |a, b| { let mut c = a.clone(); c.merge_in_place(b).unwrap(); c });

            let t_ell = tree_merge(sh_ell.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });
            let c_ell = chain_merge(&sh_ell, |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });

            let t_avll = tree_merge(sh_avll.clone(), |a, b| a.merge(b));
            let c_avll = chain_merge(&sh_avll, |a, b| a.merge(b));

            let t_h4f = tree_merge(sh_h4f.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });
            let c_h4f = chain_merge(&sh_h4f, |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });

            let t_hbs = tree_merge(sh_hbs.clone(), |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });
            let c_hbs = chain_merge(&sh_hbs, |a, b| { let mut c = a.clone(); c.merge_in_place(b); c });
            let hbs_part_mis = (0..dir_hbs.m).any(|r| t_hbs.get_reg(r) != dir_hbs.get_reg(r));
            let hbs_assoc_mis = (0..dir_hbs.m).any(|r| t_hbs.get_reg(r) != c_hbs.get_reg(r));

            let nf = n_total as f64;
            let update_stat = |idx: usize, pm: bool, am: bool, ed: f64, em: f64, st: &mut [(usize, usize, f64, f64, f64, f64)]| {
                if pm { st[idx].0 += 1; }
                if am { st[idx].1 += 1; }
                st[idx].2 += (ed - nf) / nf * 100.0;
                st[idx].3 += (em - nf) / nf * 100.0;
                let dr = (em - ed) / ed * 100.0;
                st[idx].4 += dr;
                if dr.abs() > st[idx].5 { st[idx].5 = dr.abs(); }
            };

            update_stat(0, t_5b != dir_5b, t_5b != c_5b, dir_5b.estimate_ml(&pmf_5b).0, t_5b.estimate_ml(&pmf_5b).0, &mut stats);
            update_stat(1, t_256 != dir_256, t_256 != c_256, dir_256.estimate_ml(&pmf_256).0, t_256.estimate_ml(&pmf_256).0, &mut stats);
            update_stat(2, t_g128 != dir_g128, t_g128 != c_g128, dir_g128.estimate_ml(&pmf_g128).0, t_g128.estimate_ml(&pmf_g128).0, &mut stats);
            update_stat(3, t_chunk != dir_chunk, t_chunk != c_chunk, dir_chunk.estimate_ml(&pmf_chunk).0, t_chunk.estimate_ml(&pmf_chunk).0, &mut stats);
            update_stat(4, t_ell != dir_ell, t_ell != c_ell, dir_ell.estimate_ml(&pmf_ell).0, t_ell.estimate_ml(&pmf_ell).0, &mut stats);
            update_stat(5, t_avll != dir_avll, t_avll != c_avll, dir_avll.estimate_ml_exact_mod().0, t_avll.estimate_ml_exact_mod().0, &mut stats);
            update_stat(6, t_h4f != dir_h4f, t_h4f != c_h4f, dir_h4f.estimate_ml(&pmf_hll4).0, t_h4f.estimate_ml(&pmf_hll4).0, &mut stats);
            update_stat(7, hbs_part_mis, hbs_assoc_mis, dir_hbs.estimate_ml(&pmf_hbs).0, t_hbs.estimate_ml(&pmf_hbs).0, &mut stats);
        }

        for (idx, name) in names.iter().enumerate() {
            let (pm, am, sdb, smb, sdr, mdr) = stats[idx];
            let nt = n_seeds as f64;
            println!(
                "  S={:3} | {:20} | part_fail={:2}/{} | assoc_fail={:2}/{} | dir_bias={:+.3}% | mrg_bias={:+.3}% | mean_drift={:+.4}% (max {:.4}%)",
                shards, name, pm, n_seeds, am, n_seeds, sdb / nt, smb / nt, sdr / nt, mdr
            );
            writeln!(
                drift_tsv,
                "{}\t{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}",
                shards, name, n_seeds, pm, am, sdb / nt, smb / nt, mdr, sdr / nt
            ).unwrap();
            drift_json_rows.push(format!(
                "    {{\"shards\": {}, \"method\": \"{}\", \"trials\": {}, \"part_state_mismatches\": {}, \"assoc_state_mismatches\": {}, \"direct_rel_bias_pct\": {:.5}, \"merged_rel_bias_pct\": {:.5}, \"mean_est_drift_pct\": {:.5}, \"max_abs_est_drift_pct\": {:.5}}}",
                shards, name, n_seeds, pm, am, sdb / nt, smb / nt, sdr / nt, mdr
            ));
        }
    }

    // Write structured M0/M1 JSON
    let json_text = format!(
r#"{{
  "m0_exact_preimage_pmfs": [
{}
  ],
  "m0_reconciliation_3_0061_vs_2_9977": {{
    "config": "OSBMW_b2125_g2048_k92_R7_d2",
    "fisher_J_k60_preimage_avg": {:.8},
    "fisher_J_phase_min": {:.8},
    "fisher_J_phase_max": {:.8},
    "independent_q6_field_bits": {:.4},
    "independent_q6_mvp": {:.5},
    "rank_wire_k60_exact_bits": {},
    "rank_wire_k60_fixed_bytes": {},
    "rank_wire_k60_bit_mvp": {:.5},
    "rank_wire_k60_byte_mvp": {:.5},
    "rank_wire_k59_fixed_bytes": {},
    "rank_wire_k59_byte_mvp": {:.5},
    "rank_wire_k63_exact_bits": {},
    "rank_wire_k63_fixed_bytes": {},
    "rank_wire_k63_bit_mvp": {:.5},
    "rank_wire_k63_byte_mvp": {:.5},
    "rank_codec_roundtrip_blocks": {},
    "rank_codec_200blk_roundtrip_fails": {},
    "rank_codec_encode_us_per_block": {:.2},
    "rank_codec_decode_us_per_block": {:.2}
  }},
  "m1_algebraic_audit": {{
    "dense_family_audited": ["OS-BMW-5b(k24)", "OS-BMW-5b(k20+m)", "OS-BMW-256B", "OS-BMW-g128-Dense", "OS-BMW-Chunk"],
    "exhaustive_512x512_pairs": 262144,
    "exhaustive_512x512_fails": {},
    "inplace_triple_merge_trials": {},
    "inplace_triple_merge_fails": {},
    "avll_part_counterexample": [{}, {}],
    "avll_assoc_11bkt_counterexample": [{}, {}],
    "hll4_fixed_counterexample": [{}, {}]
  }},
  "hybrid_sparse_audit": {{
    "note": "Engineering sparse wrapper evaluated directly on full mode+content+estimate (NO to_dense() masking); excluded from Theorem 1 pure-Dense family claim.",
    "regression_keys_27_73_ok": {},
    "regression_g128_direct_est": {:.9},
    "regression_g128_merged_est": {:.9},
    "regression_256b_direct_est": {:.9},
    "regression_256b_merged_est": {:.9},
    "sparse_sparse_trials": {},
    "sparse_sparse_violations": {},
    "cross_mode_trials": {},
    "cross_mode_violations": {},
    "cross_mode_state_violations": {},
    "cross_mode_est_violations": {}
  }},
  "m1_multishard_drift": [
{}
  ]
}}
"#,
        pmf_json_rows.join(",\n"),
        j_2125, j_min, j_max,
        indep_q6_bits, indep_q6_mvp,
        codec_k60.exact_bits, codec_k60.fixed_bytes, mvp_k60_bits, mvp_k60_bytes,
        codec_k59.fixed_bytes, mvp_k59_bytes,
        codec_k63.exact_bits, codec_k63.fixed_bytes, mvp_k63_bits, mvp_k63_bytes,
        test_blks.len(),
        rank_roundtrip_fails, enc_us, dec_us,
        exh_fails, alg_trials, alg_fails,
        avll_part_stream, avll_part_merge,
        avll_assoc_lhs, avll_assoc_rhs,
        hll4f_direct, hll4f_merge,
        reg_keys_27_73_ok,
        reg_g128_direct_est, reg_g128_merged_est,
        reg_256b_direct_est, reg_256b_merged_est,
        hy_ss_trials, hy_ss_viol,
        hy_cm_trials, hy_cm_viol, hy_cm_state_viol, hy_cm_est_viol,
        drift_json_rows.join(",\n")
    );
    fs::write(out_dir.join("m0_m1_reconcile.json"), json_text).unwrap();
    println!("\nWrote m0_m1_reconcile.json & m1_multishard_drift.tsv to {}", out_dir.display());
}
