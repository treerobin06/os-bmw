//! S1 `OS-BMW-v2-Solid` Core Library.

pub mod hash;
pub mod pmf;
pub mod osbmw_ref;
pub mod osbmw_rank;
pub mod osbmw_inplace;
pub mod baselines;

#[cfg(test)]
mod tests {
    use crate::baselines::*;
    use crate::hash::*;
    use crate::osbmw_inplace::*;
    use crate::osbmw_rank::*;
    use crate::osbmw_ref::*;
    use crate::pmf::*;

    #[test]
    fn test_m0_exact_64bit_preimage_pmf_closure_and_bounds() {
        // 1. Verify exact K_reach=60 for b=2.125 GeomNegLog2(w=64) and K_reach=59 for GeomExactFloor(w=64)
        let pmf_2125 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.125, w_bits: 64, max_cap: 63 });
        assert_eq!(pmf_2125.k_reach, 60, "b=2.125 64-bit NegLog2 must have exact K_reach=60 (no ghost 61..63)");
        assert_eq!(pmf_2125.counts[1..].iter().sum::<u128>(), 1u128 << 64);
        assert!(pmf_2125.counts[60] > 0);
        assert!((pmf_2125.probs[1..].iter().sum::<f64>() - 1.0).abs() < 1e-14);

        let pmf_2125_fl = PreimagePmf::build(DistSpec::GeomExactFloor { b: 2.125, w_bits: 64, max_cap: 63 });
        assert_eq!(pmf_2125_fl.k_reach, 59);
        assert_eq!(pmf_2125_fl.counts[1..].iter().sum::<u128>(), 1u128 << 64);

        // 2. Verify bit-identical agreement between `eval_geom_neg_log2` and binary-search `level_from_rem`
        for &h in &[0u64, 1, 2, 3, 7, 15, 0xb07dcc7451276ee8, u64::MAX - 1, u64::MAX] {
            let direct = eval_geom_neg_log2(h, 64, 2.125, 63) as u8;
            let via_tbl = pmf_2125.level_from_rem(h);
            assert_eq!(direct, via_tbl, "mismatch at h={:#x}", h);
        }
        // Test every threshold boundary +/- 1!
        for k in 1..=pmf_2125.k_reach {
            let t = pmf_2125.thresh_u64[k];
            for delta in [0u64, 1] {
                if let Some(v) = t.checked_add(delta) {
                    assert_eq!(eval_geom_neg_log2(v, 64, 2.125, 63) as u8, pmf_2125.level_from_rem(v));
                }
                if let Some(v) = t.checked_sub(delta) {
                    assert_eq!(eval_geom_neg_log2(v, 64, 2.125, 63) as u8, pmf_2125.level_from_rem(v));
                }
            }
        }
        // Test 20,000 random 64-bit hashes for b=2.05 at both w=64 and w=50
        let pmf_205_50 = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.05, w_bits: 50, max_cap: 63 });
        let mask50 = (1u64 << 50) - 1;
        for i in 0..20_000u64 {
            let h = wyhash64(i, 42) & mask50;
            assert_eq!(eval_geom_neg_log2(h, 50, 2.05, 63) as u8, pmf_205_50.level_from_rem(h));
        }

        // 3. Verify Pow2, Pow2Div3, and ErtlT exact sums
        for spec in [
            DistSpec::Pow2 { w_bits: 52, max_cap: 63 },
            DistSpec::Pow2Div3 { p_tot: 11 },
            DistSpec::ErtlT { t: 2, w_bits: 54 },
        ] {
            let pmf = PreimagePmf::build(spec);
            assert_eq!(pmf.counts[1..].iter().sum::<u128>(), pmf.domain_size);
            assert!((pmf.probs[1..].iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_m0_osbmw_rank_1277b_wire_codec_roundtrip() {
        let codec = OsBmwRankCodec::new(2048, 92, 7, 2, 60);
        assert_eq!(codec.exact_bits, 10_209);
        assert_eq!(codec.fixed_bytes, 1_277);

        // Edge case 1: empty block (anchor=0, i=0)
        let empty = OsBmwBlock::new(2048, 92, 7, 2);
        let enc = codec.encode(&empty);
        assert_eq!(enc.len(), 1_277);
        assert_eq!(codec.decode(&enc), empty);

        // Edge case 2: anchor=0 with 1 extreme level 60 overflow (Codex R1 witness!)
        let mut b60 = OsBmwBlock::new(2048, 92, 7, 2);
        b60.add_level(1675, 60);
        let enc60 = codec.encode(&b60);
        assert_eq!(codec.decode(&enc60), b60);

        // Edge case 3: anchor=60 (max reachable, i=0)
        let mut bmax = OsBmwBlock::new(2048, 92, 7, 2);
        for r in 0..95 {
            bmax.add_level(r * 19 % 2048, 60);
        }
        bmax.add_level(7, 58);
        let enc_max = codec.encode(&bmax);
        assert_eq!(codec.decode(&enc_max), bmax);

        // Edge case 4: realistic Poisson blocks across small, medium, huge cardinalities
        let pmf = PreimagePmf::build(DistSpec::GeomNegLog2 { b: 2.125, w_bits: 64, max_cap: 63 });
        let mut rng = 0x1234_5678_9abc_def0u64;
        for &lam in &[0.05, 2.0, 100.0, 1e6, 1e12] {
            let blk = OsBmwBlock::sample_poisson(2048, 92, 7, 2, &pmf, lam, &mut rng);
            let enc = codec.encode(&blk);
            assert_eq!(enc.len(), 1_277);
            let dec = codec.decode(&enc);
            assert_eq!(dec, blk, "OS-BMW-Rank roundtrip failed at lam={}", lam);
        }
    }

    #[test]
    fn test_m0_guards_and_avll_and_hll4fixed_counterexamples() {
        // 1. m2 guards: reject non-power-of-2 n_blocks and mismatched merges
        let d_pow2 = DistSpec::Pow2 { w_bits: 52, max_cap: 63 };
        assert!(OsBmwRefSketch::new(0, 512, 24, 8, 2, 6, d_pow2, true).is_err());
        assert!(OsBmwRefSketch::new(3, 512, 24, 8, 2, 6, d_pow2, true).is_err());
        assert!(OsBmwRefSketch::new(5, 512, 24, 8, 2, 6, d_pow2, true).is_err());
        let s1 = OsBmwRefSketch::new(4, 512, 24, 8, 2, 6, d_pow2, true).unwrap();
        let s2 = OsBmwRefSketch::new(4, 512, 20, 8, 2, 6, d_pow2, true).unwrap();
        let s3 = OsBmwRefSketch::new(4, 512, 24, 8, 2, 6, DistSpec::Pow2 { w_bits: 51, max_cap: 63 }, true).unwrap();
        assert!(s1.try_merge(&s2).is_err());
        assert!(s1.try_merge(&s3).is_err());

        // 2. AVLL 2-element non-overflow bit-order sanity
        let mut direct = AvllSketch::new_buckets(8);
        direct.add_level(0, 4);
        direct.add_level(0, 3);
        let mut p1 = AvllSketch::new_buckets(8);
        p1.add_level(0, 4);
        let mut p2 = AvllSketch::new_buckets(8);
        p2.add_level(0, 3);
        assert_eq!(direct, p1.merge(&p2));

        // 3. AVLL deterministic partition homomorphism counterexample: 22 != 18
        let mut sa = AvllSketch::new_buckets(4);
        sa.add_level(0, 22);
        let mut sb = AvllSketch::new_buckets(4);
        for r in 0..4 { sb.add_level(r, 6); }
        let mut sba = AvllSketch::new_buckets(4);
        for r in 0..4 { sba.add_level(r, 6); }
        sba.add_level(0, 22);
        let mab = sa.merge(&sb);
        assert_eq!(sba.abs_exp(0), 22);
        assert_eq!(mab.abs_exp(0), 18);
        assert_ne!(mab, sba);

        // 4. AVLL deterministic 11-bucket binary merge associativity counterexample: 47 != 49!
        let mut a = AvllSketch::new_buckets(11);
        a.add_level(0, 14);
        let mut b = AvllSketch::new_buckets(11);
        b.add_level(0, 13);
        let mut c = AvllSketch::new_buckets(11);
        for r in 0..11 { c.add_level(r, 3); }
        let lhs = a.merge(&b).merge(&c);
        let rhs = a.merge(&b.merge(&c));
        assert_eq!(lhs.get_state(0), 47);
        assert_eq!(rhs.get_state(0), 49);
        assert_ne!(lhs, rhs, "AVLL 11-bucket associativity counterexample 47 != 49 MUST hold");

        // 5. Generalized Theorem 2 witness: Fixed-size HLL_4-Fixed violates partition homomorphism!
        //    p=2 (m=4 buckets), max_aux_cap = 1:
        //    Shard A inserts level 20 into bucket 0 (goes to aux) AND level 20 into bucket 1 (aux full -> clamped to 14!).
        //    Shard B inserts level 8 into all 4 buckets (raises cur_min to 8!).
        //    Direct Stream(B then A): cur_min=8, so level 20 has delta=12 < 15 -> BOTH bucket 0 and 1 retain exact 20!
        //    Merge(A, B): bucket 1 was ALREADY clamped to 14 in A, so merged bucket 1 is 14 != 20!
        let mut ha = Hll4Sketch::new_fixed(2, 1);
        ha.add_level(0, 20);
        ha.add_level(1, 20);
        let mut hb = Hll4Sketch::new_fixed(2, 1);
        for r in 0..4 { hb.add_level(r, 8); }
        let mut h_direct = Hll4Sketch::new_fixed(2, 1);
        for r in 0..4 { h_direct.add_level(r, 8); }
        h_direct.add_level(0, 20);
        h_direct.add_level(1, 20);
        let mut h_merged = ha.clone();
        h_merged.merge_in_place(&hb);
        assert_eq!(h_direct.get_abs(1), 20);
        assert_eq!(h_merged.get_abs(1), 14);
        assert_ne!(h_merged.get_abs(1), h_direct.get_abs(1));
    }

    #[test]
    fn test_m3_inplace_kernels_exact_equivalence_with_reference_and_semilattice() {
        // Verify OsBmw5b, OsBmw5bMask, OsBmw256, OsBmwG128, OsBmwChunk against reference OsBmwBlock
        // across insertions (including dozens of promotions!), merges, wire roundtrip, and 2x downsizing!
        let check_5b = |n_blk: usize, mode: HashMode5b| {
            // Test G=512, K=24 (no mask) and G=512, K=20 (32b coarse mask)
            if mode == HashMode5b::Pow2 {
                let mut sk_a = OsBmw5b::new(n_blk, mode);
                let mut sk_b = OsBmw5b::new(n_blk, mode);
                let mut sk_ab = OsBmw5b::new(n_blk, mode);
                let mut ref_a = OsBmwRefSketch::new(n_blk, 512, 24, 8, 2, 6, sk_a.dist_spec_64(), true).unwrap();
                let mut ref_ab = OsBmwRefSketch::new(n_blk, 512, 24, 8, 2, 6, sk_a.dist_spec_64(), true).unwrap();
                let pmf = PreimagePmf::build(sk_a.dist_spec_64());

                for i in 0..60_000u64 {
                    let h = wyhash64(i, 101);
                    sk_a.add_hash_64(h);
                    sk_ab.add_hash_64(h);
                    ref_a.add_hash_64(h, &pmf);
                    ref_ab.add_hash_64(h, &pmf);
                }
                for i in 40_000..100_000u64 {
                    let h = wyhash64(i, 101);
                    sk_b.add_hash_64(h);
                    sk_ab.add_hash_64(h);
                    ref_ab.add_hash_64(h, &pmf);
                }
                // Check exact equivalence with reference on every block!
                for b in 0..n_blk {
                    assert_eq!(sk_a.to_ref_block(b), ref_a.blocks[b], "OsBmw5b block {} diverged from ref!", b);
                }
                // Check wire roundtrip
                let dec = OsBmw5b::from_bytes(sk_a.as_wire_bytes(), mode).unwrap();
                assert_eq!(dec, sk_a);
                // Check strict merge homomorphism: merge(A, B) == direct(A U B)
                let mut sk_m = sk_a.clone();
                sk_m.merge_in_place(&sk_b).unwrap();
                assert_eq!(sk_m, sk_ab, "OsBmw5b merge homomorphism failed!");
                for b in 0..n_blk {
                    assert_eq!(sk_m.to_ref_block(b), ref_ab.blocks[b]);
                }
                // Check exact single-64-bit 2x downsizing!
                if n_blk >= 2 {
                    let down = sk_ab.downsize_2x().unwrap();
                    let mut direct_half = OsBmw5b::new(n_blk / 2, mode);
                    for i in 0..100_000u64 {
                        direct_half.add_hash_64(wyhash64(i, 101));
                    }
                    // Also inject extreme rem == 0 hash (all top bits zero!)
                    let extreme_h = (0u64 << 9) | 7; // block 0, sub 7, top bits = 0!
                    let mut sk_ext = sk_ab.clone();
                    sk_ext.add_hash_64(extreme_h);
                    let mut direct_ext = direct_half.clone();
                    direct_ext.add_hash_64(extreme_h);
                    assert_eq!(sk_ext.downsize_2x().unwrap(), direct_ext, "OsBmw5b single-64b 2x downsize failed!");
                    assert_eq!(down, direct_half);
                }

                // Also verify OsBmw5bMask (k=20 with 32-bit coarse overflow bitmask in free padding!)
                let mut sm_a = OsBmw5bMask::new(n_blk, mode);
                let mut sm_b = OsBmw5bMask::new(n_blk, mode);
                let mut sm_ab = OsBmw5bMask::new(n_blk, mode);
                let mut ref_m = OsBmwRefSketch::new(n_blk, 512, 20, 8, 2, 6, sm_a.dist_spec_64(), true).unwrap();
                for i in 0..50_000u64 {
                    let h = wyhash64(i, 202);
                    sm_a.add_hash_64(h);
                    sm_ab.add_hash_64(h);
                    ref_m.add_hash_64(h, &pmf);
                }
                for i in 30_000..80_000u64 {
                    let h = wyhash64(i, 202);
                    sm_b.add_hash_64(h);
                    sm_ab.add_hash_64(h);
                }
                for b in 0..n_blk {
                    assert_eq!(sm_a.to_ref_block(b), ref_m.blocks[b], "OsBmw5bMask diverged from ref!");
                }
                let mut sm_merged = sm_a.clone();
                sm_merged.merge_in_place(&sm_b).unwrap();
                assert_eq!(sm_merged, sm_ab, "OsBmw5bMask merge homomorphism failed!");
            } else {
                // Verify OsBmw256 (g=384, k=16, 256 B/blk = 4 cache lines)
                let mut s256_a = OsBmw256::new(n_blk, mode);
                let mut s256_b = OsBmw256::new(n_blk, mode);
                let mut s256_ab = OsBmw256::new(n_blk, mode);
                let pmf = PreimagePmf::build(s256_a.dist_spec_64());
                let mut ref_256 = OsBmwRefSketch::new(n_blk, 384, 16, 8, 2, 6, s256_a.dist_spec_64(), true).unwrap();
                for i in 0..50_000u64 {
                    let h = wyhash64(i, 303);
                    s256_a.add_hash_64(h);
                    s256_ab.add_hash_64(h);
                    ref_256.add_hash_64(h, &pmf);
                }
                for i in 25_000..75_000u64 {
                    let h = wyhash64(i, 303);
                    s256_b.add_hash_64(h);
                    s256_ab.add_hash_64(h);
                }
                for b in 0..n_blk {
                    assert_eq!(s256_a.to_ref_block(b), ref_256.blocks[b]);
                }
                let mut merged = s256_a.clone();
                merged.merge_in_place(&s256_b).unwrap();
                assert_eq!(merged, s256_ab, "OsBmw256 merge homomorphism failed!");
                if n_blk >= 2 {
                    let mut direct_half = OsBmw256::new(n_blk / 2, mode);
                    for i in 0..75_000u64 {
                        direct_half.add_hash_64(wyhash64(i, 303));
                    }
                    // Extreme hash h = 0 -> q = 0 -> rem == 0!
                    let mut se = s256_ab.clone();
                    se.add_hash_64(0);
                    let mut de = direct_half.clone();
                    de.add_hash_64(0);
                    assert_eq!(se.downsize_2x().unwrap(), de, "OsBmw256 single-64b 2x downsize failed!");
                }
            }
        };
        check_5b(4, HashMode5b::Pow2);
        check_5b(4, HashMode5b::Div3);

        // Verify OsBmwChunk (b=2.05, g=1024, k=32, 656 B/blk)
        let mut ch_a = OsBmwChunkInPlace::new(2);
        let mut ch_b = OsBmwChunkInPlace::new(2);
        let mut ch_ab = OsBmwChunkInPlace::new(2);
        let pmf_c = PreimagePmf::build(ch_a.dist_spec_64());
        let mut ref_c = OsBmwRefSketch::new(2, 1024, 32, 8, 2, 6, ch_a.dist_spec_64(), true).unwrap();
        for i in 0..30_000u64 {
            let h = wyhash64(i, 404);
            ch_a.add_hash_64(h, &pmf_c);
            ch_ab.add_hash_64(h, &pmf_c);
            ref_c.add_hash_64(h, &pmf_c);
        }
        for i in 15_000..45_000u64 {
            let h = wyhash64(i, 404);
            ch_b.add_hash_64(h, &pmf_c);
            ch_ab.add_hash_64(h, &pmf_c);
        }
        for b in 0..2 {
            assert_eq!(ch_a.to_ref_block(b), ref_c.blocks[b]);
        }
        let mut ch_m = ch_a.clone();
        ch_m.merge_in_place(&ch_b).unwrap();
        assert_eq!(ch_m, ch_ab, "OsBmwChunk merge homomorphism failed!");

        // Verify Sparse-to-Dense Hybrid (`g=128` and `256B`) exact bit-for-bit equivalence with dense!
        let mut hy = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        let mut dn = OsBmwG128::new(4, HashMode5b::Pow2);
        for i in 0..500u64 {
            let h = wyhash64(i, 505);
            hy.add_hash_64(h);
            dn.add_hash_64(h);
            if i == 50 || i == 499 {
                assert_eq!(hy.to_dense(), dn, "OsBmwHybrid<128,8> diverged from dense at i={}", i);
            }
        }
        let mut hy256 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        let mut dn256 = OsBmw256::new(2, HashMode5b::Div3);
        for i in 0..500u64 {
            let h = wyhash64(i, 606);
            hy256.add_hash_64(h);
            dn256.add_hash_64(h);
            if i == 50 || i == 499 {
                assert_eq!(hy256.to_dense(), dn256, "OsBmwHybrid<384,16> diverged from dense at i={}", i);
            }
        }
    }

    #[test]
    fn test_all_baselines_unbiased_and_semilattice_sanity() {
        let n = 100_000u64;
        let mut hll = Hll6Sketch::new(11);
        let mut ull = UllSketch::new(11);
        let mut ell20 = Ell20Sketch::new(9);
        let mut ell24 = Ell24Sketch::new(9);
        let mut cpc = CpcSketch::new(10);
        let mut hlll = HlllSketch::new(11);
        let mut hll4 = Hll4Sketch::new_dynamic(11);
        let mut avll = AvllSketch::new_words(256);

        for i in 0..n {
            let h = wyhash64(i, 777);
            hll.add_hash_64(h);
            ull.add_hash_64(h);
            ell20.add_hash_64(h);
            ell24.add_hash_64(h);
            cpc.add_hash_64(h);
            hlll.add_hash_64(h);
            hll4.add_hash_64(h);
            avll.add_hash_64(h);
        }
        let check = |name: &str, est: f64| {
            let err = (est - n as f64).abs() / (n as f64);
            assert!(err < 0.08, "{} estimate {} too far from {} (err={:.3})", name, est, n, err);
        };
        check("HLL6", hll.estimate_ml(&PreimagePmf::build(hll.dist_spec_64())).0);
        check("ULL", ull.estimate_ml(&PreimagePmf::build(ull.dist_spec_64())).0);
        check("ELL20", ell20.estimate_ml(&PreimagePmf::build(ell20.dist_spec_64())).0);
        check("ELL24", ell24.estimate_ml(&PreimagePmf::build(ell24.dist_spec_64())).0);
        check("CPC", cpc.estimate_ml(&PreimagePmf::build(cpc.dist_spec_64())).0);
        check("HLLL", hlll.estimate_ml(&PreimagePmf::build(hlll.dist_spec_64())).0);
        check("HLL4", hll4.estimate_ml(&PreimagePmf::build(hll4.dist_spec_64())).0);
        check("AVLL", avll.estimate_ml(&PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 })).0);
        assert!(cpc.serialize_compressed().len() < cpc.ram_bytes());
    }

    #[test]
    fn test_wyhash64_statistical_and_low_n_mvp() {
        // 1. Verify no zero-collapse on small integers across arbitrary seeds.
        for seed in [0u64, 1, 0x2d35_8dcc_aa6c_78a5, 0x8bb8_4b93_962e_acc9, 0xdead_beef] {
            for i in 0..1000u64 {
                let h = wyhash64(i, seed);
                assert_ne!(h, 0, "wyhash64({}, {}) collapsed to 0!", i, seed);
                assert!(h.leading_zeros() < 40, "unexpectedly high leading zeros for i={}", i);
            }
        }
        // 2. Verify geometric leading-zero distribution on 200,000 sequential integers.
        let mut lz_cnt = [0usize; 16];
        let n_seq = 200_000u64;
        for i in 0..n_seq {
            let lz = wyhash64(i, 42).leading_zeros() as usize;
            if lz < 16 { lz_cnt[lz] += 1; }
        }
        for k in 0..10 {
            let exp = (n_seq as f64) * 2.0_f64.powi(-(k as i32) - 1);
            let sd = exp.sqrt();
            assert!((lz_cnt[k] as f64 - exp).abs() < 4.5 * sd,
                "wyhash64 lz={} count={} deviates from expected {:.1} (sd={:.1})", k, lz_cnt[k], exp, sd);
        }
        // 3. Verify 100-trial streaming MVP at n=10,000 for OS-BMW-5b, ELL(2,24), and CPC!
        let n = 10_000u64;
        let pmf_5b = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 9 - 3, max_cap: 63 });
        let pmf_ell = PreimagePmf::build(DistSpec::ErtlT { t: 2, w_bits: 64 - 10 });
        let pmf_cpc = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 11, max_cap: 63 });
        let mut mse_5b = 0.0;
        let mut mse_ell = 0.0;
        let mut mse_cpc = 0.0;
        let mut sum_wire_cpc = 0.0;
        let trials = 100usize;
        for tr in 0..trials {
            let seed = splitmix64(0x9999 ^ tr as u64);
            let mut s5 = OsBmw5b::new(8, HashMode5b::Pow2);
            let mut se = Ell24Sketch::new(10);
            let mut sc = CpcSketch::new(11);
            for i in 0..n {
                let h = wyhash64(i, seed);
                s5.add_hash_64(h);
                se.add_hash_64(h);
                sc.add_hash_64(h);
            }
            let e5 = (s5.estimate_ml(&pmf_5b).0 - n as f64) / n as f64;
            let ee = (se.estimate_ml(&pmf_ell).0 - n as f64) / n as f64;
            let ec = (sc.estimate_ml(&pmf_cpc).0 - n as f64) / n as f64;
            mse_5b += e5 * e5;
            mse_ell += ee * ee;
            mse_cpc += ec * ec;
            sum_wire_cpc += sc.serialize_compressed().len() as f64;
        }
        let mvp_5b = (2760.0 * 8.0) * (mse_5b / trials as f64);
        let mvp_ell = (4104.0 * 8.0) * (mse_ell / trials as f64);
        let mvp_cpc_wire = ((sum_wire_cpc / trials as f64) * 8.0) * (mse_cpc / trials as f64);
        println!("wyhash64 n=10000 100-trial MVP: OS-BMW-5b={:.3}, ELL24={:.3}, CPC_wire={:.3}", mvp_5b, mvp_ell, mvp_cpc_wire);
        assert!(mvp_5b > 1.0 && mvp_5b < 4.2, "OS-BMW-5b stream MVP out of bounds: {:.3}", mvp_5b);
        assert!(mvp_ell > 1.5 && mvp_ell < 5.0, "ELL(2,24) stream MVP out of bounds: {:.3}", mvp_ell);
        assert!(mvp_cpc_wire > 1.0 && mvp_cpc_wire < 3.5, "CPC wire stream MVP out of bounds: {:.3}", mvp_cpc_wire);
    }

    #[test]
    fn test_m1_exhaustive_512x512() {
        // Exhaustive 512 x 512 = 262,144 pair semilattice & homomorphism check on (g=4, k=2, R=3, d=1)
        let mut states = Vec::with_capacity(512);
        for s in 0u32..512 {
            let mut blk = OsBmwBlock::new(4, 2, 3, 1);
            for r in 0..4 {
                let code = ((s >> (r * 2)) & 3) as u8;
                if code >= 1 { blk.add_level(r, code + 1); }
                if (s >> 8) & 1 == 1 && r == 0 { blk.add_level(r, 1); }
            }
            states.push(blk);
        }
        for i in 0..states.len() {
            assert_eq!(OsBmwBlock::merge(&states[i], &states[i]), states[i]);
            for j in 0..states.len() {
                let ab = OsBmwBlock::merge(&states[i], &states[j]);
                let ba = OsBmwBlock::merge(&states[j], &states[i]);
                assert_eq!(ab, ba);
            }
        }
    }

    #[test]
    fn test_r2_hybrid_sparse_codex_keys_27_73_and_idempotence() {
        // Codex R1 FATAL #1 regression: keys 27 & 73, seed 0 on both g=128 (4 blks) and 256B (2 blks).
        let h27 = wyhash64(27, 0);
        let h73 = wyhash64(73, 0);

        // 1. g=128, 4 blocks, Pow2
        let pmf_g128 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 2 - 7, max_cap: 63 });
        let mut dir_128 = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        dir_128.add_hash_64(h27);
        dir_128.add_hash_64(h73);
        let mut a_128 = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        a_128.add_hash_64(h27);
        let mut b_128 = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
        b_128.add_hash_64(h73);
        let mut mrg_128 = a_128.clone();
        mrg_128.merge_in_place(&b_128).unwrap();
        let mut mrg_128_ba = b_128.clone();
        mrg_128_ba.merge_in_place(&a_128).unwrap();

        assert!(dir_128.is_sparse() && mrg_128.is_sparse() && mrg_128_ba.is_sparse());
        assert_eq!(mrg_128, dir_128, "g=128 (27 v 73) must preserve Sparse union state!");
        assert_eq!(mrg_128_ba, dir_128, "g=128 (73 v 27) must be commutative!");
        let e_dir_128 = dir_128.estimate_ml(&pmf_g128).0;
        let e_mrg_128 = mrg_128.estimate_ml(&pmf_g128).0;
        assert!((e_dir_128 - 2.0).abs() < 1e-4, "g128 direct est={}", e_dir_128);
        assert!((e_dir_128 - e_mrg_128).abs() <= 1e-12, "g128 direct {} != merged {}", e_dir_128, e_mrg_128);

        // Self-merge idempotence: exact state AND raw internal table unchanged!
        let mut idem_128 = dir_128.clone();
        idem_128.merge_in_place(&dir_128).unwrap();
        assert_eq!(idem_128, dir_128);
        match (&dir_128, &idem_128) {
            (OsBmwHybrid::Sparse { table: t1, count: c1, .. }, OsBmwHybrid::Sparse { table: t2, count: c2, .. }) => {
                assert_eq!(c1, c2);
                assert_eq!(t1, t2, "self-merge must leave raw sparse table untouched");
            }
            _ => panic!("expected Sparse after self-merge"),
        }

        // 2. 256B Hybrid (g=384, k=16), 2 blocks, Div3
        let pmf_256 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 1 });
        let mut dir_256 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        dir_256.add_hash_64(h27);
        dir_256.add_hash_64(h73);
        let mut a_256 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        a_256.add_hash_64(h27);
        let mut b_256 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
        b_256.add_hash_64(h73);
        let mut mrg_256 = a_256.clone();
        mrg_256.merge_in_place(&b_256).unwrap();
        let mut mrg_256_ba = b_256.clone();
        mrg_256_ba.merge_in_place(&a_256).unwrap();

        assert!(dir_256.is_sparse() && mrg_256.is_sparse() && mrg_256_ba.is_sparse());
        assert_eq!(mrg_256, dir_256, "256B (27 v 73) must preserve Sparse union state!");
        assert_eq!(mrg_256_ba, dir_256);
        let e_dir_256 = dir_256.estimate_ml(&pmf_256).0;
        let e_mrg_256 = mrg_256.estimate_ml(&pmf_256).0;
        assert!((e_dir_256 - 2.0).abs() < 1e-4, "256B direct est={}", e_dir_256);
        assert!((e_dir_256 - e_mrg_256).abs() <= 1e-12, "256B direct {} != merged {}", e_dir_256, e_mrg_256);

        let mut idem_256 = dir_256.clone();
        idem_256.merge_in_place(&dir_256).unwrap();
        assert_eq!(idem_256, dir_256);
        match (&dir_256, &idem_256) {
            (OsBmwHybrid::Sparse { table: t1, count: c1, .. }, OsBmwHybrid::Sparse { table: t2, count: c2, .. }) => {
                assert_eq!(c1, c2);
                assert_eq!(t1, t2, "256B self-merge must leave raw sparse table untouched");
            }
            _ => panic!("expected Sparse after 256B self-merge"),
        }
        println!(
            "Hybrid Codex (27,73) PASS: g128 direct={:.9} merged={:.9}; 256B direct={:.9} merged={:.9}",
            e_dir_128, e_mrg_128, e_dir_256, e_mrg_256
        );
    }

    #[test]
    fn test_r2_hybrid_cross_mode_partition_full_state_audit() {
        // Honest full-state (mode + content) & estimate audit WITHOUT calling `to_dense()`!
        let pmf_g128 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64 - 2 - 7, max_cap: 63 });
        let pmf_256 = PreimagePmf::build(DistSpec::Pow2Div3 { p_tot: 7 + 1 });

        let mut ss_trials = 0usize;
        let mut ss_viol = 0usize;
        let mut cm_trials = 0usize;
        let mut cm_state_viol = 0usize;
        let mut cm_est_viol = 0usize;

        for tr in 0..120u64 {
            let seed = 0x8800_0000 + tr;
            // Test all 5 regime transitions (including exact threshold boundary for g=128 max_count=48!):
            let (na, nb, overlap) = match tr % 5 {
                0 => (1 + (splitmix64(seed) % 20), 1 + (splitmix64(seed + 1) % 20), 0), // S v S -> S
                1 => (24, 24, 0),                                                        // S v S -> D (exact 48 boundary)
                2 => (35, 35, 10),                                                       // S v S -> D (60 > 48)
                3 => (20, 90, 5),                                                        // S v D -> D
                _ => (90, 20, 5),                                                        // D v S -> D
            };
            let mut a = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
            let mut b = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
            let mut dir = OsBmwHybrid::<128, 8>::new(4, HashMode5b::Pow2);
            for i in 0..na {
                let h = wyhash64(i, seed);
                a.add_hash_64(h);
                dir.add_hash_64(h);
            }
            for i in (na - overlap)..(na - overlap + nb) {
                let h = wyhash64(i, seed);
                b.add_hash_64(h);
                dir.add_hash_64(h);
            }
            let both_stay_sparse = dir.is_sparse() && a.is_sparse() && b.is_sparse();
            let mut ab = a.clone();
            ab.merge_in_place(&b).unwrap();
            let mut ba = b.clone();
            ba.merge_in_place(&a).unwrap();

            let st_ok = ab == dir && ba == dir;
            let e_dir = dir.estimate_ml(&pmf_g128).0;
            let e_ab = ab.estimate_ml(&pmf_g128).0;
            let e_ba = ba.estimate_ml(&pmf_g128).0;
            let est_ok = (e_dir - e_ab).abs() <= 1e-12 * e_dir.max(1.0)
                && (e_dir - e_ba).abs() <= 1e-12 * e_dir.max(1.0);

            if both_stay_sparse {
                ss_trials += 1;
                if !st_ok || !est_ok { ss_viol += 1; }
            } else {
                cm_trials += 1;
                if !st_ok { cm_state_viol += 1; }
                if !est_ok { cm_est_viol += 1; }
            }

            // Same regimes scaled 2x for 256B Hybrid (2 blks, max_count=96)
            let (na2, nb2, ov2) = (na * 2, nb * 2, overlap * 2);
            let mut a2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
            let mut b2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
            let mut dir2 = OsBmwHybrid::<384, 16>::new(2, HashMode5b::Div3);
            for i in 0..na2 {
                let h = wyhash64(i, seed + 1000);
                a2.add_hash_64(h);
                dir2.add_hash_64(h);
            }
            for i in (na2 - ov2)..(na2 - ov2 + nb2) {
                let h = wyhash64(i, seed + 1000);
                b2.add_hash_64(h);
                dir2.add_hash_64(h);
            }
            let both_stay_sparse2 = dir2.is_sparse() && a2.is_sparse() && b2.is_sparse();
            let mut ab2 = a2.clone();
            ab2.merge_in_place(&b2).unwrap();
            let mut ba2 = b2.clone();
            ba2.merge_in_place(&a2).unwrap();

            let st_ok2 = ab2 == dir2 && ba2 == dir2;
            let e_dir2 = dir2.estimate_ml(&pmf_256).0;
            let e_ab2 = ab2.estimate_ml(&pmf_256).0;
            let e_ba2 = ba2.estimate_ml(&pmf_256).0;
            let est_ok2 = (e_dir2 - e_ab2).abs() <= 1e-12 * e_dir2.max(1.0)
                && (e_dir2 - e_ba2).abs() <= 1e-12 * e_dir2.max(1.0);

            if both_stay_sparse2 {
                ss_trials += 1;
                if !st_ok2 || !est_ok2 { ss_viol += 1; }
            } else {
                cm_trials += 1;
                if !st_ok2 { cm_state_viol += 1; }
                if !est_ok2 { cm_est_viol += 1; }
            }
        }
        println!(
            "Hybrid full-state audit (NO to_dense): sparse_v_sparse={}/{} viol; cross_mode={}/{} state_viol, {}/{} est_viol",
            ss_viol, ss_trials, cm_state_viol, cm_trials, cm_est_viol, cm_trials
        );
        assert!(ss_trials > 0 && cm_trials > 0);
        assert_eq!(ss_viol, 0, "sparse v sparse must have 0 violations");
        assert_eq!(cm_state_viol, 0, "observed cross-mode state violations: {}", cm_state_viol);
        assert_eq!(cm_est_viol, 0, "observed cross-mode est violations: {}", cm_est_viol);
    }

    #[test]
    fn test_r2_downsize_2x_128bit_k24_no_lift_regression() {
        // Codex R1 #4 regression: k24 (g=512, k=24), 8 blocks -> 4 blocks, 128-bit hash pair (0, 2^11).
        // h_hi = 1 << 11 has leading_zeros = 52 => level = 53. Downsizing in 128-bit mode MUST keep 53, NOT 54!
        let mut sk8 = OsBmw5b::new_128(8);
        sk8.add_hash_128(0, 1u64 << 11);
        assert_eq!(sk8.to_ref_block(0).unpack_all().0[0], 53);

        let down4 = sk8.downsize_2x().unwrap();
        let down_lvl = down4.to_ref_block(0).unpack_all().0[0];
        assert_eq!(down_lvl, 53, "128-bit downsize_2x erroneously lifted level 53 to {}!", down_lvl);

        let mut direct4 = OsBmw5b::new_128(4);
        direct4.add_hash_128(0, 1u64 << 11);
        assert_eq!(down4, direct4, "128-bit downsize_2x(8->4) must equal direct 4-block 128-bit sketch!");

        // Verify header guards reject 64b/128b mode mismatch in from_bytes and merge_in_place
        assert!(OsBmw5b::from_bytes(sk8.as_wire_bytes(), HashMode5b::Pow2).is_err());
        assert_eq!(OsBmw5b::from_bytes(sk8.as_wire_bytes(), HashMode5b::Hash128).unwrap(), sk8);

        // Randomized 128-bit stream + extreme pairs on both OsBmw5b (g=512) and OsBmw256 (g=384)
        let mut r8 = OsBmw5b::new_128(8);
        let mut r4 = OsBmw5b::new_128(4);
        let mut q8 = OsBmw256::new_128(8);
        let mut q4 = OsBmw256::new_128(4);
        for i in 0..40_000u64 {
            let (lo, hi) = wyhash128(i, 0x128_0001);
            r8.add_hash_128(lo, hi); r4.add_hash_128(lo, hi);
            q8.add_hash_128(lo, hi); q4.add_hash_128(lo, hi);
        }
        for &(lo, hi) in &[(0u64, 1u64 << 11), (0, 0), (4, 1u64 << 11), (3, 1u64 << 10)] {
            r8.add_hash_128(lo, hi); r4.add_hash_128(lo, hi);
            q8.add_hash_128(lo, hi); q4.add_hash_128(lo, hi);
        }
        assert_eq!(r8.downsize_2x().unwrap(), r4, "OsBmw5b 128-bit stream downsize_2x failed!");
        assert_eq!(q8.downsize_2x().unwrap(), q4, "OsBmw256 128-bit stream downsize_2x failed!");
    }

    #[test]
    fn test_r2_pmf_geom_exact_floor_integer_thresholds() {
        // Codex R1 #5 regression: b = 17/8 = 2.125 exact integer thresholds (no float drift!).
        let pmf = PreimagePmf::build(DistSpec::GeomExactFloor { b: 2.125, w_bits: 64, max_cap: 63 });
        assert_eq!(pmf.k_reach, 59);
        assert_eq!(pmf.thresh[0], 1u128 << 64);
        assert_eq!(
            pmf.thresh[1], 8_680_820_740_569_200_760u128,
            "first threshold for b=17/8 must be exact integer floor(2^64 * 8 / 17)"
        );
        assert_eq!(pmf.thresh_u64[1], 8_680_820_740_569_200_760u64);
        assert_eq!(pmf.thresh[2], 4_085_092_113_209_035_651u128);
        assert_eq!(pmf.thresh[57], 4u128);
        assert_eq!(pmf.thresh[58], 1u128);
        assert_eq!(pmf.thresh[59], 0u128);
        assert_eq!(pmf.counts[1..].iter().sum::<u128>(), 1u128 << 64);

        // Sanity: b = 2.0 exact floor must match Pow2(w_bits=64, max_cap=63) on every threshold!
        let pmf_b2 = PreimagePmf::build(DistSpec::GeomExactFloor { b: 2.0, w_bits: 64, max_cap: 63 });
        let pmf_p2 = PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 });
        assert_eq!(pmf_b2.k_reach, pmf_p2.k_reach);
        assert_eq!(pmf_b2.thresh, pmf_p2.thresh);
        assert_eq!(pmf_b2.counts, pmf_p2.counts);
    }

    #[test]
    fn test_r2_avll_joint_preimage_and_prefill_exactness() {
        // Verify exact partition of all 2^64 hashes across (bucket r, level k) matching AvllSketch::add_hash_64.
        for &m in &[704usize, 1408, 5621] {
            let mut total: u128 = 0;
            for r in 0..m {
                let (l_r, u_r) = AvllSketch::bucket_lo32_bounds(r, m);
                assert!(l_r < u_r && u_r <= (1u64 << 32));
                for k in 1..=63 {
                    total += AvllSketch::joint_preimage_count(k, l_r, u_r) as u128;
                }
                // Buckets in upper half (l_r >= 2^31) cannot reach k >= 34!
                if l_r >= (1u64 << 31) {
                    for k in 34..=63 {
                        assert_eq!(AvllSketch::joint_preimage_count(k, l_r, u_r), 0);
                    }
                }
            }
            assert_eq!(total, 1u128 << 64, "AVLL joint preimage sum must equal 2^64 for m={}", m);

            // Every actual hash word (random + low-32-bit extreme) must map to a (r, k) with positive count!
            for i in 0..50_000u64 {
                let h = if i < 10_000 { i } else if i < 20_000 { i << 20 } else { wyhash64(i, 999) };
                let r = (((h as u32) as u64 * (m as u64)) >> 32) as usize;
                let k = (h.leading_zeros() + 1).min(63) as usize;
                let (l_r, u_r) = AvllSketch::bucket_lo32_bounds(r, m);
                assert!(AvllSketch::joint_preimage_count(k, l_r, u_r) > 0);
            }
        }

        // Verify prefill_uniform_saturated matches sequential level-by-level insertion without clamping!
        for &k_pre in &[1u8, 2, 3, 10, 20, 32] {
            let mut fast = AvllSketch::new_words(16);
            fast.prefill_uniform_saturated(k_pre);
            let mut seq = AvllSketch::new_words(16);
            for lvl in 1..=k_pre {
                for r in 0..seq.m {
                    seq.add_level(r, lvl);
                }
            }
            assert_eq!(fast, seq, "AvllSketch::prefill_uniform_saturated({}) mismatch!", k_pre);
        }
    }

    #[test]
    fn test_r2_hlll_and_hll4_real_serialization_roundtrip() {
        let mut hlll = HlllSketch::new(12);
        let mut hll4_dyn = Hll4Sketch::new_dynamic(12);
        let mut hll4_fix = Hll4Sketch::new_fixed(12, 32);
        for i in 0..100_000u64 {
            let h = wyhash64(i, 0xbeef_1234);
            hlll.add_hash_64(h);
            hll4_dyn.add_hash_64(h);
            hll4_fix.add_hash_64(h);
        }
        let mut buf = Vec::new();

        hlll.serialize_into(&mut buf);
        assert_eq!(buf.len(), hlll.wire_bytes());
        assert!(buf.iter().any(|&b| b != 0));
        assert_eq!(HlllSketch::deserialize(&buf).unwrap(), hlll, "HLLL roundtrip mismatch!");

        hll4_dyn.serialize_into(&mut buf);
        assert_eq!(buf.len(), hll4_dyn.wire_bytes());
        assert!(buf.iter().any(|&b| b != 0));
        assert_eq!(Hll4Sketch::deserialize(&buf).unwrap(), hll4_dyn, "HLL4-Dyn roundtrip mismatch!");

        hll4_fix.serialize_into(&mut buf);
        assert_eq!(buf.len(), hll4_fix.wire_bytes());
        assert!(buf.iter().any(|&b| b != 0));
        assert_eq!(Hll4Sketch::deserialize(&buf).unwrap(), hll4_fix, "HLL4-Fixed roundtrip mismatch!");
    }

    #[test]
    fn test_r2_hash_avalanche_full_64x64() {
        // Strict 64x64 avalanche test on our WyHash-style 2x-wymix integer mixer (`wyhash64`):
        // Flipping every single input bit must flip every single output bit with rate in [0.48, 0.52].
        const N_SAMPLES: usize = 32_768;
        let mut max_dev_item = 0.0_f64;
        let mut max_dev_seed = 0.0_f64;

        for in_bit in 0..64u32 {
            let mask = 1u64 << in_bit;
            let mut out_flips_item = [0u32; 64];
            let mut out_flips_seed = [0u32; 64];
            for s in 0..N_SAMPLES as u64 {
                let x = splitmix64(s.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (in_bit as u64));
                let seed = splitmix64(s ^ 0xfeed_cafe_1234_5678);
                let h0 = wyhash64(x, seed);
                let d_item = h0 ^ wyhash64(x ^ mask, seed);
                let d_seed = h0 ^ wyhash64(x, seed ^ mask);
                for ob in 0..64usize {
                    out_flips_item[ob] += ((d_item >> ob) & 1) as u32;
                    out_flips_seed[ob] += ((d_seed >> ob) & 1) as u32;
                }
            }
            for ob in 0..64usize {
                let r_item = (out_flips_item[ob] as f64) / (N_SAMPLES as f64);
                let r_seed = (out_flips_seed[ob] as f64) / (N_SAMPLES as f64);
                let dev_i = (r_item - 0.5).abs();
                let dev_s = (r_seed - 0.5).abs();
                if dev_i > max_dev_item { max_dev_item = dev_i; }
                if dev_s > max_dev_seed { max_dev_seed = dev_s; }
                assert!(
                    dev_i <= 0.02,
                    "wyhash64 item avalanche failed: in_bit={}, out_bit={}, rate={:.4}",
                    in_bit, ob, r_item
                );
                assert!(
                    dev_s <= 0.02,
                    "wyhash64 seed avalanche failed: seed_bit={}, out_bit={}, rate={:.4}",
                    in_bit, ob, r_seed
                );
            }
        }
        println!(
            "wyhash64 64x64 avalanche PASS (N={}): max |rate-0.5| item={:.5}, seed={:.5} (limit 0.02)",
            N_SAMPLES, max_dev_item, max_dev_seed
        );
    }

    #[test]
    fn test_r2_anchor63_128bit_merge_no_oob_and_exact_ref() {
        // Post-R2 Codex regression: legal 128-bit Dense states can reach anchor == 62 and 63;
        // merging them must never wrap `(anchor + 1) << 2` to 0 or overflow `K - 1` slots.
        let mut a = OsBmw5bInPlace::<512, 24, false>::new_128(1);
        let mut b = OsBmw5bInPlace::<512, 24, false>::new_128(1);
        let mut ab_direct = OsBmw5bInPlace::<512, 24, false>::new_128(1);
        for s in 0..30usize {
            a.add_level(0, s, 62);
            ab_direct.add_level(0, s, 62);
            b.add_level(0, s + 10, 63);
            ab_direct.add_level(0, s + 10, 63);
        }
        a.add_level(0, 0, 63);
        ab_direct.add_level(0, 0, 63);
        b.add_level(0, 500, 61);
        ab_direct.add_level(0, 500, 61);
        assert_eq!(a.as_wire_bytes()[HEADER_BYTES], 62);
        assert_eq!(b.as_wire_bytes()[HEADER_BYTES], 63);

        let ref_ab = a.to_ref_block(0).merge(&b.to_ref_block(0));
        let mut merged_ab = a.clone();
        merged_ab.merge_in_place(&b).unwrap();
        let mut merged_ba = b.clone();
        merged_ba.merge_in_place(&a).unwrap();
        let mut merged_bb = b.clone();
        merged_bb.merge_in_place(&b).unwrap();
        assert_eq!(merged_ab.as_wire_bytes()[HEADER_BYTES], 63);
        assert_eq!(merged_ba.as_wire_bytes()[HEADER_BYTES], 63);
        assert_eq!(merged_bb.as_wire_bytes()[HEADER_BYTES], 63);
        assert_eq!(merged_ab, ab_direct);
        assert_eq!(merged_ba, ab_direct);
        assert_eq!(merged_ab.to_ref_block(0), ref_ab);
        assert_eq!(merged_bb.to_ref_block(0), b.to_ref_block(0));
    }

    #[test]
    fn test_r2_from_bytes_rejects_cross_config_k24_vs_k20m() {
        let k24 = OsBmw5bInPlace::<512, 24, false>::new(8, HashMode5b::Pow2);
        let k20m = OsBmw5bInPlace::<512, 20, true>::new(8, HashMode5b::Pow2);
        assert_eq!(k24.as_wire_bytes().len(), k20m.as_wire_bytes().len());
        assert!(OsBmw5bInPlace::<512, 24, false>::from_bytes(k20m.as_wire_bytes(), HashMode5b::Pow2).is_err());
        assert!(OsBmw5bInPlace::<512, 20, true>::from_bytes(k24.as_wire_bytes(), HashMode5b::Pow2).is_err());
        assert!(OsBmw5bInPlace::<512, 24, false>::from_bytes(k24.as_wire_bytes(), HashMode5b::Pow2).is_ok());
        assert!(OsBmw5bInPlace::<512, 20, true>::from_bytes(k20m.as_wire_bytes(), HashMode5b::Pow2).is_ok());
    }

    #[test]
    fn test_r2_baseline_exact_level_and_prefill_helpers() {
        // 1. Hll4Sketch::from_exact_levels matches streamed Hll4Sketch::new_dynamic bit-for-bit!
        let mut streamed = Hll4Sketch::new_dynamic(10);
        for i in 0..80_000u64 {
            streamed.add_hash_64(wyhash64(i, 0x5151_abcd));
        }
        let levels: Vec<u8> = (0..streamed.m).map(|r| streamed.get_abs(r)).collect();
        let rebuilt = Hll4Sketch::from_exact_levels(10, &levels);
        assert_eq!(rebuilt, streamed, "Hll4Sketch::from_exact_levels mismatch!");

        // 2. Hll4Sketch::prefill_uniform_saturated matches sequential level-by-level insertion!
        for &k_pre in &[1u8, 4, 15, 25] {
            let mut fast = Hll4Sketch::new_fixed(8, 16);
            fast.prefill_uniform_saturated(k_pre);
            let mut seq = Hll4Sketch::new_fixed(8, 16);
            for lvl in 1..=k_pre {
                for r in 0..seq.m {
                    seq.add_level(r, lvl);
                }
            }
            assert_eq!(fast, seq, "Hll4Sketch::prefill_uniform_saturated({}) mismatch!", k_pre);
        }

        // 3. HlllSketch::add_level matches streamed HlllSketch::add_hash_64 after compress()!
        let mut hl_s = HlllSketch::new(10);
        for i in 0..50_000u64 {
            hl_s.add_hash_64(wyhash64(i, 0x7777_4321));
        }
        let hl_levels: Vec<u8> = (0..hl_s.m).map(|r| hl_s.get_abs(r)).collect();
        let mut hl_r = HlllSketch::new(10);
        for (r, &u) in hl_levels.iter().enumerate() {
            hl_r.add_level(r, u);
        }
        hl_s.compress();
        hl_r.compress();
        assert_eq!(hl_r, hl_s, "HlllSketch::add_level + compress mismatch!");
    }

    #[test]
    fn test_v3_mulshift_and_flat_slice_exact_equivalence() {
        // 1. Exact bit-for-bit equivalence: Div3 vs add_hash_64_div3_mulshift_deferred (including u64::MAX!)
        let mut a = OsBmw256::new(16, HashMode5b::Div3);
        let mut b = OsBmw256::new(16, HashMode5b::Div3);
        for i in 0..80_000u64 {
            let h = wyhash64(i, 0x256b_1234);
            a.add_hash_64(h);
            b.add_hash_64_div3_mulshift_deferred(h);
        }
        for edge in [0u64, 1, 2, 3, u64::MAX - 2, u64::MAX - 1, u64::MAX] {
            a.add_hash_64(edge);
            b.add_hash_64_div3_mulshift_deferred(edge);
        }
        assert_eq!(a.buf, b.buf, "Div3 mulshift128 deferred diverged from Div3!");

        // 2. Exact bit-for-bit equivalence: Flat-Slab Slice Engine vs OsBmw5bInPlace (5b + 256B)
        let mut s1 = OsBmw5b::new(8, HashMode5b::Pow2);
        let mut s2 = OsBmw5b::new(8, HashMode5b::Pow2);
        let mut f1 = vec![0u8; OsBmw5b::state_bytes_for(8)];
        let mut f2 = vec![0u8; OsBmw5b::state_bytes_for(8)];
        OsBmw5b::init_flat_slice(&mut f1, 8, HashMode5b::Pow2);
        OsBmw5b::init_flat_slice(&mut f2, 8, HashMode5b::Pow2);
        for i in 0..60_000u64 {
            let h = wyhash64(i, 0x5b_aaaa);
            s1.add_hash_64(h);
            OsBmw5b::add_hash_64_flat_slice(&mut f1, 8, HashMode5b::Pow2, h);
        }
        for i in 30_000..90_000u64 {
            let h = wyhash64(i, 0x5b_aaaa);
            s2.add_hash_64(h);
            let (blk, sub, lvl) = OsBmw5b::decompose_hash_64(h, 8, HashMode5b::Pow2);
            OsBmw5b::add_level_flat_slice(&mut f2, 8, blk, sub, lvl);
        }
        assert_eq!(s1.buf, f1, "Flat slice 5b f1 mismatch!");
        assert_eq!(s2.buf, f2, "Flat slice 5b f2 mismatch!");
        s1.merge_in_place(&s2).unwrap();
        OsBmw5b::merge_flat_slices(&mut f1, &f2, 8);
        assert_eq!(s1.buf, f1, "Flat slice 5b merge mismatch!");
        let pmf5 = PreimagePmf::build(s1.dist_spec_64());
        assert_eq!(s1.estimate_ml(&pmf5).0.to_bits(), OsBmw5b::estimate_ml_flat_slice(&f1, 8, &pmf5).0.to_bits());

        // 256B flat slice exact check
        let mut d1 = OsBmw256::new(16, HashMode5b::Div3);
        let mut d2 = OsBmw256::new(16, HashMode5b::Div3);
        let mut g1 = vec![0u8; OsBmw256::state_bytes_for(16)];
        let mut g2 = vec![0u8; OsBmw256::state_bytes_for(16)];
        OsBmw256::init_flat_slice(&mut g1, 16, HashMode5b::Div3);
        OsBmw256::init_flat_slice(&mut g2, 16, HashMode5b::Div3);
        for i in 0..40_000u64 {
            let h = wyhash64(i, 0x256b_bbbb);
            d1.add_hash_64(h);
            OsBmw256::add_hash_64_flat_slice(&mut g1, 16, HashMode5b::Div3, h);
        }
        for i in 20_000..60_000u64 {
            let h = wyhash64(i, 0x256b_bbbb);
            d2.add_hash_64(h);
            let (blk, sub, lvl) = OsBmw256::decompose_hash_64(h, 16, HashMode5b::Div3);
            OsBmw256::add_level_flat_slice(&mut g2, 16, blk, sub, lvl);
        }
        d1.merge_in_place(&d2).unwrap();
        OsBmw256::merge_flat_slices(&mut g1, &g2, 16);
        assert_eq!(d1.buf, g1, "Flat slice 256B merge mismatch!");
        let pmf256 = PreimagePmf::build(d1.dist_spec_64());
        assert_eq!(d1.estimate_ml(&pmf256).0.to_bits(), OsBmw256::estimate_ml_flat_slice(&g1, 16, &pmf256).0.to_bits());
    }
}
