//! Reference OS-BMW Block & Sketch (Exact Semilattice Oracle with Strict m2 Guards).
//!
//! Key guarantees:
//!   - Uses `PreimagePmf` (single source of truth exact 64-bit preimage counts, zero ghost levels).
//!   - Enforces strict `m2` guards: `n_blocks` must be a power of 2 in `1..=(1<<20)`; `merge()`
//!     strictly checks `dist`, `q_bits`, `g`, `k`, `r_win`, `d`, `n_blocks` compatibility.
//!   - Supports BOTH single 64-bit hash (`hash_to_pow2` / `hash_to_256b` / `ErtlT` / `Geom`) with
//!     exact lossless 2x downsizing (Ertl Alg. 6 top-level adjustment) AND 128-bit domain-separated hash.

use crate::hash::{hash_to_256b, hash_to_pow2, rng_u01};
use crate::pmf::{DistSpec, PreimagePmf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OsBmwBlock {
    pub g: usize,
    pub k: usize,
    pub r_win: u8,
    pub d: u8,
    pub anchor: u8,
    pub slots: Vec<u32>,
    pub ov_u: Vec<u8>,
    pub ov_h: Vec<u32>,
}

impl OsBmwBlock {
    #[inline]
    pub fn n_in(r_win: u8, d: u8) -> u32 {
        let r = r_win as u32;
        let dd = d as u32;
        if dd >= r - 1 {
            1u32 << r
        } else {
            (r - dd + 1) * (1u32 << dd)
        }
    }

    #[inline]
    pub fn n_slot(r_win: u8, d: u8, k: usize) -> u32 {
        Self::n_in(r_win, d) + if k > 1 { 1 } else { 0 }
    }

    #[inline]
    pub fn encode_normal(tau: u8, hist: u32, d: u8) -> u32 {
        if tau == 0 {
            return 0;
        }
        let t = tau as u32;
        let dd = d as u32;
        let base = if t - 1 <= dd {
            1u32 << (t - 1)
        } else {
            (t - dd) * (1u32 << dd)
        };
        let nb = (t - 1).min(dd);
        let mask = if nb >= 32 { u32::MAX } else { (1u32 << nb) - 1 };
        base + (hist & mask)
    }

    #[inline]
    pub fn decode_normal(code: u32, d: u8) -> (u8, u32) {
        if code == 0 {
            return (0, 0);
        }
        let dd = d as u32;
        let thr = 1u32 << dd;
        if code < thr {
            let t_m1 = 31 - code.leading_zeros();
            let hist = code - (1u32 << t_m1);
            ((t_m1 + 1) as u8, hist)
        } else {
            let block = code >> dd;
            let tau = (block + dd) as u8;
            let hist = code & ((1u32 << dd) - 1);
            (tau, hist)
        }
    }

    pub fn new(g: usize, k: usize, r_win: u8, d: u8) -> Self {
        assert!(g >= 1 && k >= 1 && k <= g && r_win >= 1);
        Self {
            g,
            k,
            r_win,
            d,
            anchor: 0,
            slots: vec![0; g],
            ov_u: Vec::with_capacity(k.saturating_sub(1)),
            ov_h: Vec::with_capacity(k.saturating_sub(1)),
        }
    }

    pub fn unpack_all(&self) -> (Vec<u8>, Vec<u32>) {
        if self.g == 1 && self.k == 1 {
            return (vec![self.anchor], vec![self.slots[0]]);
        }
        let n_in = Self::n_in(self.r_win, self.d);
        let base_lo = (self.anchor as i32) - (self.r_win as i32);
        let mut us = vec![0u8; self.g];
        let mut hs = vec![0u32; self.g];
        let mut ov_idx = 0usize;
        for r in 0..self.g {
            let code = self.slots[r];
            if self.k > 1 && code == n_in {
                us[r] = self.ov_u[ov_idx];
                hs[r] = self.ov_h[ov_idx];
                ov_idx += 1;
            } else if code > 0 {
                let (tau, h) = Self::decode_normal(code, self.d);
                us[r] = (base_lo + tau as i32) as u8;
                hs[r] = h;
            }
        }
        debug_assert_eq!(ov_idx, self.ov_u.len());
        (us, hs)
    }

    pub fn pack_from_uh(g: usize, k: usize, r_win: u8, d: u8, us: &[u8], hs: &[u32]) -> Self {
        if g == 1 && k == 1 {
            let u = us[0];
            let nb = if u == 0 { 0 } else { (u - 1).min(d) as u32 };
            let mask = if nb >= 32 { u32::MAX } else { (1u32 << nb) - 1 };
            return Self {
                g: 1,
                k: 1,
                r_win,
                d,
                anchor: u,
                slots: vec![hs[0] & mask],
                ov_u: Vec::new(),
                ov_h: Vec::new(),
            };
        }
        let mut sorted_u = us.to_vec();
        sorted_u.sort_unstable_by(|a, b| b.cmp(a));
        let anchor = sorted_u[k - 1];
        let n_in = Self::n_in(r_win, d);
        let base_lo = (anchor as i32) - (r_win as i32);
        let win_lo = (base_lo + 1).max(1);

        let mut slots = vec![0u32; g];
        let mut ov_u = Vec::with_capacity(k.saturating_sub(1));
        let mut ov_h = Vec::with_capacity(k.saturating_sub(1));

        for r in 0..g {
            let u = us[r];
            if (u as i32) < win_lo {
                slots[r] = 0;
            } else if u <= anchor {
                let tau = ((u as i32) - base_lo) as u8;
                let nb = ((tau - 1).min(d)).min(u - 1) as u32;
                let mask = if nb >= 32 { u32::MAX } else { (1u32 << nb) - 1 };
                slots[r] = Self::encode_normal(tau, hs[r] & mask, d);
            } else {
                let nb = (d as i32).min((u as i32) - win_lo).min((u as i32) - 1).max(0) as u32;
                let mask = if nb >= 32 { u32::MAX } else { (1u32 << nb) - 1 };
                slots[r] = n_in;
                ov_u.push(u);
                ov_h.push(hs[r] & mask);
            }
        }
        debug_assert!(ov_u.len() <= k - 1);
        Self { g, k, r_win, d, anchor, slots, ov_u, ov_h }
    }

    #[inline(always)]
    pub fn merge_sub(u1: u8, h1: u32, u2: u8, h2: u32, d: u8) -> (u8, u32) {
        if u1 == 0 {
            return (u2, h2);
        }
        if u2 == 0 {
            return (u1, h1);
        }
        let d_mask: u32 = if d >= 32 { u32::MAX } else { (1u32 << d) - 1 };
        if u1 == u2 {
            (u1, (h1 | h2) & d_mask)
        } else if u1 > u2 {
            let delta = (u1 - u2) as u32;
            let mut h = h1;
            if delta <= d as u32 {
                h |= 1u32 << (delta - 1);
                if delta < 32 {
                    h |= (h2 << delta) & d_mask;
                }
            }
            (u1, h & d_mask)
        } else {
            let delta = (u2 - u1) as u32;
            let mut h = h2;
            if delta <= d as u32 {
                h |= 1u32 << (delta - 1);
                if delta < 32 {
                    h |= (h1 << delta) & d_mask;
                }
            }
            (u2, h & d_mask)
        }
    }

    pub fn add_level(&mut self, r: usize, lvl: u8) -> bool {
        if self.g == 1 && self.k == 1 {
            if (lvl as i32) < (self.anchor as i32) - (self.d as i32) {
                return true;
            }
            let (nu, nh) = Self::merge_sub(self.anchor, self.slots[0], lvl, 0, self.d);
            let nb = if nu == 0 { 0 } else { (nu - 1).min(self.d) as u32 };
            let mask = if nb >= 32 { u32::MAX } else { (1u32 << nb) - 1 };
            self.anchor = nu;
            self.slots[0] = nh & mask;
            return false;
        }
        let base_lo = (self.anchor as i32) - (self.r_win as i32);
        if (lvl as i32) <= base_lo {
            return true;
        }
        let (mut us, mut hs) = self.unpack_all();
        let (nu, nh) = Self::merge_sub(us[r], hs[r], lvl, 0, self.d);
        us[r] = nu;
        hs[r] = nh;
        *self = Self::pack_from_uh(self.g, self.k, self.r_win, self.d, &us, &hs);
        false
    }

    pub fn merge(&self, other: &Self) -> Self {
        assert_eq!(self.g, other.g, "OsBmwBlock::merge g mismatch");
        assert_eq!(self.k, other.k, "OsBmwBlock::merge k mismatch");
        assert_eq!(self.r_win, other.r_win, "OsBmwBlock::merge r_win mismatch");
        assert_eq!(self.d, other.d, "OsBmwBlock::merge d mismatch");
        let (u1, h1) = self.unpack_all();
        let (u2, h2) = other.unpack_all();
        let mut um = vec![0u8; self.g];
        let mut hm = vec![0u32; self.g];
        for r in 0..self.g {
            let (u, h) = Self::merge_sub(u1[r], h1[r], u2[r], h2[r], self.d);
            um[r] = u;
            hm[r] = h;
        }
        Self::pack_from_uh(self.g, self.k, self.r_win, self.d, &um, &hm)
    }

    /// Shift top boundary level `top_lvl` up by 1 (inserting a 0 bit below it) for exact single-64-bit 2x downsizing
    /// on the `j = 0` half-block (exact analogue of Ertl Alg. 6!).
    pub fn lift_top_level(&self, top_lvl: u8) -> Self {
        let (mut us, mut hs) = self.unpack_all();
        let d_mask: u32 = if self.d >= 32 { u32::MAX } else { (1u32 << self.d) - 1 };
        let mut changed = false;
        for r in 0..self.g {
            if us[r] == top_lvl {
                us[r] = top_lvl + 1;
                hs[r] = (hs[r] << 1) & d_mask;
                changed = true;
            }
        }
        if !changed {
            return self.clone();
        }
        Self::pack_from_uh(self.g, self.k, self.r_win, self.d, &us, &hs)
    }

    /// Exact Poissonized block sample using `PreimagePmf`.
    pub fn sample_poisson(g: usize, k: usize, r_win: u8, d: u8, pmf: &PreimagePmf, lam: f64, rng: &mut u64) -> Self {
        let km = pmf.k_reach;
        let mut cdf = vec![0.0_f64; km + 1];
        for u in 0..=km {
            let x = lam * pmf.tail[u];
            cdf[u] = if x > 745.0 { 0.0 } else { (-x).exp() };
        }
        let mut us = vec![0u8; g];
        let mut hs = vec![0u32; g];
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
            if u >= 2 && d > 0 {
                let nb = (u - 1).min(d as usize);
                let mut h = 0u32;
                for j in 0..nb {
                    let lvl = u - 1 - j;
                    let x = lam * pmf.probs[lvl];
                    let p1 = if x > 700.0 { 1.0 } else { -(-x).exp_m1() };
                    if rng_u01(rng) < p1 {
                        h |= 1u32 << j;
                    }
                }
                hs[r] = h;
            }
        }
        Self::pack_from_uh(g, k, r_win, d, &us, &hs)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OsBmwRefSketch {
    pub blocks: Vec<OsBmwBlock>,
    pub g: usize,
    pub k: usize,
    pub r_win: u8,
    pub d: u8,
    pub q_bits: u32,
    pub dist: DistSpec,
    pub single_hash_64: bool,
}

impl OsBmwRefSketch {
    pub fn new(
        n_blocks: usize,
        g: usize,
        k: usize,
        r_win: u8,
        d: u8,
        q_bits: u32,
        dist: DistSpec,
        single_hash_64: bool,
    ) -> Result<Self, String> {
        if !n_blocks.is_power_of_two() || n_blocks == 0 || n_blocks > (1 << 20) {
            return Err(format!("n_blocks={} must be power of 2 in 1..=2^20", n_blocks));
        }
        if k == 0 || k > g || r_win == 0 {
            return Err(format!("invalid (g={}, k={}, R={})", g, k, r_win));
        }
        match dist {
            DistSpec::Pow2 { max_cap, .. }
            | DistSpec::GeomNegLog2 { max_cap, .. }
            | DistSpec::GeomExactFloor { max_cap, .. } => {
                if max_cap == 0 || max_cap > 63 {
                    return Err(format!("max_cap={} must be in 1..=63", max_cap));
                }
            }
            _ => {}
        }
        if single_hash_64 {
            let p_blk = n_blocks.trailing_zeros();
            match dist {
                DistSpec::Pow2 { w_bits, .. } => {
                    if !g.is_power_of_two() {
                        return Err(format!("single_hash_64 Pow2 requires power-of-two g, got {}", g));
                    }
                    let used_bits = p_blk + g.trailing_zeros();
                    if used_bits >= 64 {
                        return Err(format!("single_hash_64 Pow2 p_blk+s_sub={} must be < 64", used_bits));
                    }
                    let exact_w = 64u32 - used_bits;
                    if w_bits != exact_w {
                        return Err(format!(
                            "single_hash_64 Pow2 requires exact remaining width w_bits={} == 64-p_blk-s_sub (got {})",
                            exact_w, w_bits
                        ));
                    }
                }
                DistSpec::Pow2Div3 { p_tot } => {
                    if g != 384 {
                        return Err(format!("single_hash_64 Pow2Div3 requires g=384, got {}", g));
                    }
                    let expected_p_tot = 7 + p_blk;
                    if p_tot >= 64 || p_tot != expected_p_tot {
                        return Err(format!(
                            "single_hash_64 Pow2Div3 requires p_tot={} < 64 == 7+p_blk (got {})",
                            expected_p_tot, p_tot
                        ));
                    }
                }
                DistSpec::GeomNegLog2 { w_bits, .. } | DistSpec::GeomExactFloor { w_bits, .. } => {
                    if !g.is_power_of_two() {
                        return Err(format!("single_hash_64 Geom requires power-of-two g, got {}", g));
                    }
                    let max_w = 64u32.saturating_sub(p_blk + g.trailing_zeros());
                    if w_bits == 0 || w_bits > max_w {
                        return Err(format!(
                            "single_hash_64 Geom w_bits={} must be in 1..={}",
                            w_bits, max_w
                        ));
                    }
                }
                DistSpec::ErtlT { t, w_bits } => {
                    if !g.is_power_of_two() {
                        return Err(format!("single_hash_64 ErtlT requires power-of-two g, got {}", g));
                    }
                    let max_w = 64u32.saturating_sub(p_blk + g.trailing_zeros());
                    if w_bits <= t || w_bits > max_w {
                        return Err(format!(
                            "single_hash_64 ErtlT requires {} < w_bits <= {} (got {})",
                            t, max_w, w_bits
                        ));
                    }
                }
            }
        }
        let pmf = PreimagePmf::build(dist);
        let q_max = if q_bits >= 63 { usize::MAX } else { (1usize << q_bits) - 1 };
        if pmf.k_reach > q_max {
            return Err(format!("k_reach={} exceeds 2^{}-1", pmf.k_reach, q_bits));
        }
        let blocks = (0..n_blocks).map(|_| OsBmwBlock::new(g, k, r_win, d)).collect();
        Ok(Self { blocks, g, k, r_win, d, q_bits, dist, single_hash_64 })
    }

    /// Strictly guarded merge: verifies all schema, geometry, and distribution fields.
    pub fn try_merge(&self, other: &Self) -> Result<Self, String> {
        if self.blocks.len() != other.blocks.len() {
            return Err(format!("n_blocks mismatch: {} vs {}", self.blocks.len(), other.blocks.len()));
        }
        if self.g != other.g || self.k != other.k || self.r_win != other.r_win || self.d != other.d {
            return Err("block geometry (g/k/R/d) mismatch".to_string());
        }
        if self.q_bits != other.q_bits || self.dist != other.dist || self.single_hash_64 != other.single_hash_64 {
            return Err("distribution/q_bits/hash_mode mismatch".to_string());
        }
        let blocks = self.blocks.iter().zip(other.blocks.iter()).map(|(a, b)| a.merge(b)).collect();
        Ok(Self {
            blocks,
            g: self.g,
            k: self.k,
            r_win: self.r_win,
            d: self.d,
            q_bits: self.q_bits,
            dist: self.dist,
            single_hash_64: self.single_hash_64,
        })
    }

    pub fn add_hash_64(&mut self, h: u64, pmf: &PreimagePmf) -> bool {
        assert!(self.single_hash_64, "add_hash_64 requires single_hash_64 == true");
        assert_eq!(
            pmf.spec, self.dist,
            "add_hash_64: PreimagePmf {:?} does not match sketch dist {:?}",
            pmf.spec, self.dist
        );
        let p_blk = self.blocks.len().trailing_zeros();
        let (blk, sub, lvl) = match self.dist {
            DistSpec::Pow2 { w_bits, max_cap } => {
                debug_assert!(self.g.is_power_of_two());
                let (b, s, raw_lvl) = hash_to_pow2(h, p_blk, self.g.trailing_zeros());
                let eff_cap = (w_bits as usize + 1).min(max_cap).min(pmf.k_reach);
                (b, s, (raw_lvl as usize).min(eff_cap) as u8)
            }
            DistSpec::Pow2Div3 { .. } => {
                debug_assert_eq!(self.g, 384);
                let (b, s, raw_lvl) = hash_to_256b(h, p_blk);
                (b, s, (raw_lvl as usize).min(pmf.k_reach) as u8)
            }
            DistSpec::GeomNegLog2 { w_bits, .. } | DistSpec::GeomExactFloor { w_bits, .. } => {
                debug_assert!(self.g.is_power_of_two());
                let s_sub = self.g.trailing_zeros();
                let sub = (h & ((1u64 << s_sub) - 1)) as usize;
                let blk = ((h >> s_sub) & ((1u64 << p_blk) - 1)) as usize;
                let rem = h >> (64 - w_bits);
                let lvl = pmf.level_from_rem(rem);
                (blk, sub, lvl)
            }
            DistSpec::ErtlT { t, w_bits } => {
                debug_assert!(self.g.is_power_of_two());
                let s_sub = self.g.trailing_zeros();
                let sub = (h & ((1u64 << s_sub) - 1)) as usize;
                let blk = ((h >> s_sub) & ((1u64 << p_blk) - 1)) as usize;
                let rem = h >> (64 - w_bits);
                let sub_t = (rem & ((1u64 << t) - 1)) as u32;
                let hi = rem >> t;
                let hi_bits = w_bits - t;
                let nlz = if hi == 0 { hi_bits } else { hi.leading_zeros() - (64 - hi_bits) };
                let lvl = (((nlz << t) + sub_t + 1) as usize).min(pmf.k_reach) as u8;
                (blk, sub, lvl)
            }
        };
        self.blocks[blk].add_level(sub, lvl)
    }

    pub fn add_hash_128(&mut self, h_lo: u64, h_hi: u64, pmf: &PreimagePmf) -> bool {
        assert!(!self.single_hash_64, "add_hash_128 requires single_hash_64 == false");
        assert_eq!(
            pmf.spec, self.dist,
            "add_hash_128: PreimagePmf {:?} does not match sketch dist {:?}",
            pmf.spec, self.dist
        );
        let n_blk = self.blocks.len();
        let blk = (h_lo as usize) & (n_blk - 1);
        let sub = if self.g.is_power_of_two() {
            ((h_lo >> 20) as usize) & (self.g - 1)
        } else {
            (((h_lo >> 20) as u128 * (self.g as u128)) >> 44) as usize
        };
        let lvl = pmf.level_from_rem(h_hi);
        self.blocks[blk].add_level(sub, lvl)
    }

    /// Exact lossless 2x downsizing (works for BOTH 128-bit hash AND single-64-bit Pow2 / Pow2Div3!).
    pub fn downsize_2x(&self) -> Result<Self, String> {
        let n = self.blocks.len();
        if n < 2 || !n.is_power_of_two() {
            return Err(format!("Cannot downsize n_blocks={}", n));
        }
        let half = n / 2;
        if !self.single_hash_64 {
            let blocks = (0..half).map(|j| self.blocks[j].merge(&self.blocks[j + half])).collect();
            return Ok(Self { blocks, ..self.clone() });
        }
        match self.dist {
            DistSpec::Pow2 { w_bits, max_cap } => {
                // Top reachable level at precision P is `top = w_bits + 1`.
                // When downsizing P -> P-1, block `j` (upper bit 0) lifts `top` -> `top + 1`
                // ONLY when `w_bits + 1 < max_cap` (top level is not yet saturated at `max_cap`).
                let top = (w_bits as usize + 1).min(max_cap) as u8;
                let new_w = (w_bits + 1).min(64);
                let new_dist = DistSpec::Pow2 { w_bits: new_w, max_cap };
                let new_pmf = PreimagePmf::build(new_dist);
                let q_max = if self.q_bits >= 63 { usize::MAX } else { (1usize << self.q_bits) - 1 };
                if new_pmf.k_reach > q_max {
                    return Err(format!("downsized k_reach={} exceeds 2^{}-1", new_pmf.k_reach, self.q_bits));
                }
                let blocks = (0..half)
                    .map(|j| {
                        let lifted_lo = if (w_bits as usize + 1) < max_cap && (w_bits as usize + 1) < 63 {
                            self.blocks[j].lift_top_level(top)
                        } else {
                            self.blocks[j].clone()
                        };
                        lifted_lo.merge(&self.blocks[j + half])
                    })
                    .collect();
                Ok(Self {
                    blocks,
                    dist: new_dist,
                    ..self.clone()
                })
            }
            DistSpec::Pow2Div3 { p_tot } => {
                if p_tot <= 1 {
                    return Err("Cannot downsize Pow2Div3 with p_tot <= 1".to_string());
                }
                let new_dist = DistSpec::Pow2Div3 { p_tot: p_tot - 1 };
                let new_pmf = PreimagePmf::build(new_dist);
                let q_max = if self.q_bits >= 63 { usize::MAX } else { (1usize << self.q_bits) - 1 };
                if new_pmf.k_reach > q_max {
                    return Err(format!("downsized k_reach={} exceeds 2^{}-1", new_pmf.k_reach, self.q_bits));
                }
                let top = (64 - p_tot) as u8;
                let blocks = (0..half)
                    .map(|j| self.blocks[j].lift_top_level(top).merge(&self.blocks[j + half]))
                    .collect();
                Ok(Self {
                    blocks,
                    dist: new_dist,
                    ..self.clone()
                })
            }
            _ => Err("Exact single-64-bit 2x downsizing requires dyadic Pow2/Pow2Div3 or fixed-W/128-bit hash".to_string()),
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        assert_eq!(
            pmf.spec, self.dist,
            "estimate_ml: PreimagePmf {:?} does not match sketch dist {:?}",
            pmf.spec, self.dist
        );
        let mut ones_cnt = vec![0u32; pmf.k_reach + 1];
        let mut zero_lin = 0.0_f64;
        for blk in &self.blocks {
            let (us, hs) = blk.unpack_all();
            let win_lo = if blk.g == 1 && blk.k == 1 {
                1usize
            } else {
                let base_lo = (blk.anchor as i32) - (blk.r_win as i32);
                (base_lo + 1).max(1) as usize
            };
            for r in 0..blk.g {
                let u = us[r] as usize;
                assert!(
                    u <= pmf.k_reach,
                    "register level u={} exceeds pmf.k_reach={}",
                    u, pmf.k_reach
                );
                if u == 0 {
                    zero_lin += pmf.tail[win_lo - 1];
                } else {
                    zero_lin += pmf.tail[u];
                    ones_cnt[u] += 1;
                    let lo = win_lo.max(u.saturating_sub(blk.d as usize));
                    let h = hs[r];
                    for lvl in lo..u {
                        let bit_idx = u - 1 - lvl;
                        if ((h >> bit_idx) & 1) != 0 {
                            ones_cnt[lvl] += 1;
                        } else {
                            zero_lin += pmf.probs[lvl];
                        }
                    }
                }
            }
        }
        pmf.solve_mle(&ones_cnt, zero_lin, (self.blocks.len() * self.g) as f64)
    }
}
