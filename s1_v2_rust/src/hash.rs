//! Unified hash functions (WyHash-style 2x-wymix 64-bit & 128-bit integer mixers; NOT official
//! WyHash v4.2 KAT) and self-describing 8-byte header.
//!
//! Every sketch (OS-BMW variants and all 8 baselines) shares the EXACT SAME single 64-bit hash
//! `wyhash64(item, seed)` in the primary benchmark protocol, and `wyhash128(item, seed)` in the
//! 128-bit ablation. Every serialized/in-memory budget accounts for the 8-byte `SketchHeader`.

pub const HEADER_BYTES: usize = 8;

/// Standard 8-byte self-describing header counted in ALL equal-byte RAM and Wire comparisons.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SketchHeader {
    pub magic: u8,     // 0x53 ('S')
    pub kind: u8,      // sketch family ID
    pub p_or_k: u8,    // precision p (blocks = 2^p) or parameter
    pub flags: u8,     // bit 0 = sparse mode, bits 1..7 = variant flags
    pub aux: u32,      // extra self-describing config (g, k, R, d packed or sparse count)
}

impl SketchHeader {
    pub const MAGIC: u8 = 0x53;

    #[inline(always)]
    pub const fn new(kind: u8, p_or_k: u8, flags: u8, aux: u32) -> Self {
        Self { magic: Self::MAGIC, kind, p_or_k, flags, aux }
    }

    #[inline(always)]
    pub const fn to_u64(self) -> u64 {
        (self.magic as u64)
            | ((self.kind as u64) << 8)
            | ((self.p_or_k as u64) << 16)
            | ((self.flags as u64) << 24)
            | ((self.aux as u64) << 32)
    }

    #[inline(always)]
    pub const fn from_u64(w: u64) -> Self {
        Self {
            magic: (w & 0xFF) as u8,
            kind: ((w >> 8) & 0xFF) as u8,
            p_or_k: ((w >> 16) & 0xFF) as u8,
            flags: ((w >> 24) & 0xFF) as u8,
            aux: (w >> 32) as u32,
        }
    }
}

/// 128-bit multiply-XOR mixing primitive (`wymix`).
/// Compiles to 1 `mulx` + `xor` on x86-64 (~1.5 ns).
#[inline(always)]
pub fn wymix(a: u64, b: u64) -> u64 {
    let prod = (a as u128).wrapping_mul(b as u128);
    (prod as u64) ^ ((prod >> 64) as u64)
}

/// WyHash-style 2x-wymix 64-bit integer mixer (NOT official WyHash v4.2 KAT).
/// Uses a loop-hoistable seed pre-mix followed by two 128-bit multiply-XOR rounds;
/// shared identically across OS-BMW and all baselines (avalanche verified in unit tests).
#[inline(always)]
pub fn wyhash64(item: u64, seed: u64) -> u64 {
    const S0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const S1: u64 = 0x8bb8_4b93_962e_acc9;
    const S2: u64 = 0x4b33_a62e_d433_d4a3;
    let s = seed ^ wymix(seed ^ S0, S1);
    let t = wymix(item ^ S1, item.rotate_left(32) ^ s ^ S2);
    wymix(S1 ^ 8, t ^ S0)
}

/// 128-bit domain-separated WyHash-style 2x-wymix integer mixer for the 128-bit ablation
/// (NOT official WyHash v4.2 KAT).
#[inline(always)]
pub fn wyhash128(item: u64, seed: u64) -> (u64, u64) {
    const S3: u64 = 0x8ebc_6af0_9c88_c6e3;
    const S4: u64 = 0x5899_65cc_7537_4cc3;
    const S5: u64 = 0x1d8e_4e27_c47d_124f;
    let h_lo = wyhash64(item, seed);
    let s2 = seed ^ wymix(seed ^ S3, S4);
    let t2 = wymix(item ^ S4, item.rotate_left(29) ^ s2 ^ S5);
    let h_hi = wymix(S4 ^ 16, t2 ^ S3);
    (h_lo, h_hi)
}

/// Deterministic PRNG step (SplitMix64) for Monte Carlo and Poisson sampling.
#[inline(always)]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Fast uniform f64 in (0, 1) from SplitMix64 state.
#[inline(always)]
pub fn rng_u01(rng: &mut u64) -> f64 {
    *rng = rng.wrapping_add(1);
    let v = splitmix64(*rng);
    ((v >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

/// Exact constant for `floor(2^64 / 3) = (2^64 - 1) / 3 = 0x5555_5555_5555_5555`.
pub const Q_MAX_DIV3: u64 = 0x5555_5555_5555_5555;

/// Single 64-bit hash decomposition for `OS-BMW-5b-256B` (`g = 384 = 3 * 128`, `n_blk = 2^p_blk`).
/// Uses `q = h / 3` (compiles to 1 `mulx` + shift, zero `udiv`!), returning `(blk, sub, lvl)`
/// with exact 2x-downsizing compatibility (identical to Ertl Alg. 6 on the `rem == 0` top level).
#[inline(always)]
pub fn hash_to_256b(h: u64, p_blk: u32) -> (usize, usize, u8) {
    // Clamp the 1-in-2^64 value h == u64::MAX so (q, r3) is exact uniform on [0, Q_MAX_DIV3) x {0,1,2}.
    let hc = if h == u64::MAX { h - 1 } else { h };
    let q = hc / 3;
    let r3 = (hc - q * 3) as usize;
    let sub128 = (q & 127) as usize;
    let sub = r3 * 128 + sub128;
    let blk = ((q >> 7) & ((1u64 << p_blk) - 1)) as usize;
    let p_tot = 7 + p_blk;
    // Sentinel bit at bit (p_tot - 1) prevents leading_zeros from reading into (blk, sub) bits.
    // Because q < 2^62 * (4/3) < 2^63, bit 63 of q is ALWAYS 0, so leading_zeros() >= 1!
    let sentinel = 1u64 << (p_tot - 1);
    let lvl = (q | sentinel).leading_zeros() as u8;
    // lvl in 1..=(64 - p_tot):
    //   q in [2^62, Q_MAX_DIV3)  -> lz = 1  (prob 1/4)
    //   q in [2^61, 2^62)        -> lz = 2  (prob 3/8)
    //   q in [2^60, 2^61)        -> lz = 3  (prob 3/16)
    //   ...
    //   rem == 0                 -> lz = 64 - p_tot (top level!)
    (blk, sub, lvl)
}

/// Single 64-bit hash decomposition for pure power-of-2 OS-BMW (`g = 2^s_sub`, `n_blk = 2^p_blk`).
/// Exact Ertl-style MSB-first NLZ with sentinel bit at `p_tot - 1`, giving 100% lossless 2x downsizing!
#[inline(always)]
pub fn hash_to_pow2(h: u64, p_blk: u32, s_sub: u32) -> (usize, usize, u8) {
    let sub = (h & ((1u64 << s_sub) - 1)) as usize;
    let blk = ((h >> s_sub) & ((1u64 << p_blk) - 1)) as usize;
    let p_tot = p_blk + s_sub;
    let lvl = if p_tot == 0 {
        (h.leading_zeros() + 1).min(63) as u8
    } else {
        let sentinel = 1u64 << (p_tot - 1);
        ((h | sentinel).leading_zeros() + 1).min(63) as u8
    };
    (blk, sub, lvl)
}
