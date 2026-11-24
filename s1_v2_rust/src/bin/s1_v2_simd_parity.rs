//! E-04: Direct Same-Input AVX-512 vs Forced-Scalar Full-State Byte-Parity Verifier (`s1_v2_simd_parity`).
//!
//! Calls BOTH `merge_in_place` / `merge_flat_slices` (compiled with AVX-512 when enabled)
//! AND `merge_in_place_scalar` / `merge_flat_slices_scalar` (always scalar unpack+merge+repack)
//! on IDENTICAL input byte buffers across adversarial edge cases and 10,000+ random streams/states.

use s1_v2_rust::hash::{splitmix64, wyhash128, wyhash64};
use s1_v2_rust::osbmw_inplace::{HashMode5b, OsBmw5bInPlace};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

fn check_pair<const G: usize, const K: usize, const HAS_MASK: bool>(
    a: &OsBmw5bInPlace<G, K, HAS_MASK>,
    b: &OsBmw5bInPlace<G, K, HAS_MASK>,
    checked: &mut usize,
    mismatches: &mut usize,
    label: &str,
) {
    // Forward object merge parity
    {
        let mut s_simd = a.clone();
        let mut s_scal = a.clone();
        s_simd.merge_in_place(b).expect("merge_in_place");
        s_scal.merge_in_place_scalar(b).expect("merge_in_place_scalar");
        *checked += 1;
        if s_simd.as_wire_bytes() != s_scal.as_wire_bytes() {
            *mismatches += 1;
            eprintln!("MISMATCH fwd obj [{}] a vs b", label);
        }
        // Reverse commutative parity
        let mut r_simd = b.clone();
        let mut r_scal = b.clone();
        r_simd.merge_in_place(a).expect("merge_in_place rev");
        r_scal.merge_in_place_scalar(a).expect("merge_in_place_scalar rev");
        *checked += 1;
        if r_simd.as_wire_bytes() != r_scal.as_wire_bytes() || r_simd.as_wire_bytes() != s_simd.as_wire_bytes() {
            *mismatches += 1;
            eprintln!("MISMATCH rev obj [{}] b vs a", label);
        }
    }

    // Flat-slice parity (only for HAS_MASK == false / DataFusion slab configs, plus masked slices)
    {
        let mut f_simd = a.as_wire_bytes().to_vec();
        let mut f_scal = a.as_wire_bytes().to_vec();
        OsBmw5bInPlace::<G, K, HAS_MASK>::merge_flat_slices(&mut f_simd, b.as_wire_bytes(), a.n_blk);
        OsBmw5bInPlace::<G, K, HAS_MASK>::merge_flat_slices_scalar(&mut f_scal, b.as_wire_bytes(), a.n_blk);
        *checked += 1;
        if f_simd != f_scal {
            *mismatches += 1;
            eprintln!("MISMATCH fwd flat [{}] a vs b", label);
        }
        let mut rf_simd = b.as_wire_bytes().to_vec();
        let mut rf_scal = b.as_wire_bytes().to_vec();
        OsBmw5bInPlace::<G, K, HAS_MASK>::merge_flat_slices(&mut rf_simd, a.as_wire_bytes(), b.n_blk);
        OsBmw5bInPlace::<G, K, HAS_MASK>::merge_flat_slices_scalar(&mut rf_scal, a.as_wire_bytes(), b.n_blk);
        *checked += 1;
        if rf_simd != rf_scal || rf_simd != f_simd {
            *mismatches += 1;
            eprintln!("MISMATCH rev flat [{}] b vs a", label);
        }
    }
}

fn run_adversarial_suite(checked: &mut usize, mismatches: &mut usize) {
    // 1. Empty + empty, empty + populated
    {
        let e1 = OsBmw5bInPlace::<512, 24, false>::new(8, HashMode5b::Pow2);
        let e2 = OsBmw5bInPlace::<512, 24, false>::new(8, HashMode5b::Pow2);
        check_pair(&e1, &e2, checked, mismatches, "empty_empty");
        let mut p1 = e1.clone();
        p1.add_level(0, 0, 12);
        p1.add_level(3, 511, 25);
        check_pair(&e1, &p1, checked, mismatches, "empty_pop");
    }

    // 2. Equal-anchor with full K-1=23 overflows on both sides -> tie-break raise
    {
        let mut a = OsBmw5bInPlace::<512, 24, false>::new(2, HashMode5b::Pow2);
        let mut b = OsBmw5bInPlace::<512, 24, false>::new(2, HashMode5b::Pow2);
        for r in 0..23 {
            a.add_level(0, r * 2, 16);
            b.add_level(0, r * 2 + 1, 16);
        }
        for r in 0..40 {
            a.add_level(1, r * 7, 15);
            b.add_level(1, r * 7 + 1, 15);
        }
        check_pair(&a, &b, checked, mismatches, "eq_anchor_tie_raise");
    }

    // 3. Exact boundary gap == R = 8, gap = 7 (< R), gap = 9 (> R), skewed fast-path vs scalar unpack
    for &(anchor_a, anchor_b, tag) in &[(16u8, 8u8, "gap_eq_R8"), (15u8, 8u8, "gap_7"), (17u8, 8u8, "gap_9"), (28u8, 12u8, "gap_16_skew")] {
        let mut a = OsBmw5bInPlace::<512, 24, false>::new(1, HashMode5b::Pow2);
        let mut b = OsBmw5bInPlace::<512, 24, false>::new(1, HashMode5b::Pow2);
        for r in 0..24 {
            a.add_level(0, r * 3, anchor_a);
        }
        if anchor_b > 8 {
            for r in 0..24 {
                b.add_level(0, r * 3 + 1, anchor_b);
            }
        }
        // Inject boundary history bits & overflows right around win_lo = anchor_a - 7
        for r in 0..16 {
            b.add_level(0, 100 + r, anchor_a - 7);
            b.add_level(0, 150 + r, anchor_a - 8);
            b.add_level(0, 200 + r, anchor_a + 1);
            b.add_level(0, 200 + r, anchor_a);
            b.add_level(0, 200 + r, anchor_a - 1);
        }
        check_pair(&a, &b, checked, mismatches, tag);
    }

    // 4. Max height saturation u = 63 with full history h=3 and >K registers at u=63
    {
        let mut a = OsBmw5bInPlace::<512, 24, false>::new(1, HashMode5b::Pow2);
        let mut b = OsBmw5bInPlace::<512, 24, false>::new(1, HashMode5b::Pow2);
        for r in 0..32 {
            a.add_level(0, r, 63);
            a.add_level(0, r, 62);
            a.add_level(0, r, 61);
            b.add_level(0, r + 16, 63);
            b.add_level(0, r + 16, 62);
            b.add_level(0, r + 16, 61);
        }
        check_pair(&a, &b, checked, mismatches, "u63_saturation");
    }
}

fn run_random_suite<const G: usize, const K: usize, const HAS_MASK: bool>(
    n_blk: usize,
    mode: HashMode5b,
    n_pairs: usize,
    seed_salt: u64,
    label: &str,
    checked: &mut usize,
    mismatches: &mut usize,
) {
    for tr in 0..n_pairs {
        let s = splitmix64(seed_salt ^ (tr as u64));
        let mut a = OsBmw5bInPlace::<G, K, HAS_MASK>::new(n_blk, mode);
        let mut b = OsBmw5bInPlace::<G, K, HAS_MASK>::new(n_blk, mode);
        let na = match tr % 5 {
            0 => 50usize,
            1 => 800,
            2 => 6_000,
            3 => 40_000,
            _ => 120_000,
        };
        let nb = match (tr / 5) % 5 {
            0 => 50usize,
            1 => 800,
            2 => 6_000,
            3 => 40_000,
            _ => 500,
        };
        for i in 0..na as u64 {
            let h = wyhash64(i, s ^ 0xA1A1);
            if mode == HashMode5b::Hash128 {
                let (lo, hi) = wyhash128(i, s ^ 0xA1A1);
                a.add_hash_128(lo, hi);
            } else {
                a.add_hash_64(h);
            }
        }
        for i in 0..nb as u64 {
            let h = wyhash64(i, s ^ 0xB2B2);
            if mode == HashMode5b::Hash128 {
                let (lo, hi) = wyhash128(i, s ^ 0xB2B2);
                b.add_hash_128(lo, hi);
            } else {
                b.add_hash_64(h);
            }
        }
        check_pair(&a, &b, checked, mismatches, label);
    }
}

fn main() {
    let out_dir = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    let avx512_compiled = cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2"
    ));

    let mut checked = 0usize;
    let mut mismatches = 0usize;

    run_adversarial_suite(&mut checked, &mut mismatches);
    // 3,000 random pairs x 4 merge ops (obj fwd/rev + flat fwd/rev) across 6 kernel configs = 12,000 direct checks!
    run_random_suite::<512, 24, false>(8, HashMode5b::Pow2, 700, 0x1111_0001, "5b_g512_k24_8blk", &mut checked, &mut mismatches);
    run_random_suite::<512, 24, false>(16, HashMode5b::Pow2, 400, 0x2222_0002, "5b_g512_k24_16blk", &mut checked, &mut mismatches);
    run_random_suite::<512, 20, true>(8, HashMode5b::Pow2, 500, 0x3333_0003, "5b_g512_k20_mask", &mut checked, &mut mismatches);
    run_random_suite::<384, 16, false>(16, HashMode5b::Div3, 600, 0x4444_0004, "256b_div3_16blk", &mut checked, &mut mismatches);
    run_random_suite::<384, 16, false>(16, HashMode5b::MulShift384, 400, 0x5555_0005, "256b_mulshift_16blk", &mut checked, &mut mismatches);
    run_random_suite::<128, 8, false>(8, HashMode5b::Pow2, 400, 0x6666_0006, "g128_k8_8blk", &mut checked, &mut mismatches);

    let out_path = out_dir.join("m1_simd_scalar_parity.json");
    let mut f = File::create(&out_path).unwrap();
    writeln!(
        f,
        "{{\n  \"avx512_compiled\": {},\n  \"arch\": \"{}\",\n  \"total_merges_checked\": {},\n  \"mismatches\": {},\n  \"status\": \"{}\"\n}}",
        avx512_compiled,
        std::env::consts::ARCH,
        checked,
        mismatches,
        if mismatches == 0 { "PASS" } else { "FAIL" }
    )
    .unwrap();
    println!(
        "=== [E-04 SIMD vs Scalar Parity] avx512_compiled={} checked={} mismatches={} -> {} ===",
        avx512_compiled,
        checked,
        mismatches,
        out_path.display()
    );
    assert_eq!(mismatches, 0, "SIMD vs scalar byte mismatch detected!");
}
