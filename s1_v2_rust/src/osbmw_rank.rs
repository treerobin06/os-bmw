//! Milestone M0/M4: `OS-BMW-Rank` Deterministic Fixed-Byte Combinatorial Wire Codec.
//!
//! Proves Theorem 4 constructively by encoding ANY valid `OsBmwBlock(g=2048, k=92, R=7, d=2)`
//! under exact 64-bit preimage support `K_reach = 60` into a strictly fixed-size **1,277-byte**
//! (10,216-bit) wire payload with ZERO entropy coding, ZERO failure probability, and 100%
//! bijective roundtrip `decode(encode(blk)) == blk` (Wire MVP = 2.99767 < 3.00).
//!
//! Also supports arbitrary `(g, k, R, d, K_reach)` (e.g., `K_reach = 59` -> 1,276 B -> 2.99532;
//! conservative `K_reach = 63` -> 1,278 B -> 3.00001).

use crate::osbmw_ref::OsBmwBlock;

/// Compact big-unsigned integer (little-endian `u64` limbs) for exact combinatorial ranking.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BigU {
    pub limbs: Vec<u64>,
}

impl BigU {
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    pub fn from_u64(v: u64) -> Self {
        if v == 0 {
            Self::zero()
        } else {
            Self { limbs: vec![v] }
        }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    pub fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 64 + (64 - top.leading_zeros() as usize),
        }
    }

    pub fn cmp_big(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {
                for i in (0..self.limbs.len()).rev() {
                    match self.limbs[i].cmp(&other.limbs[i]) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
            ord => ord,
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        let n = self.limbs.len().max(other.limbs.len());
        if self.limbs.len() < n {
            self.limbs.resize(n, 0);
        }
        let mut carry = 0u128;
        for i in 0..n {
            let a = self.limbs[i] as u128;
            let b = if i < other.limbs.len() { other.limbs[i] as u128 } else { 0 };
            let s = a + b + carry;
            self.limbs[i] = s as u64;
            carry = s >> 64;
        }
        if carry > 0 {
            self.limbs.push(carry as u64);
        }
    }

    pub fn sub_assign(&mut self, other: &Self) {
        debug_assert!(self.cmp_big(other) != std::cmp::Ordering::Less);
        let mut borrow = 0i128;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i128;
            let b = if i < other.limbs.len() { other.limbs[i] as i128 } else { 0 };
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1i128 << 64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            self.limbs[i] = d as u64;
        }
        debug_assert_eq!(borrow, 0);
        self.trim();
    }

    pub fn mul_add_u64(&mut self, mult: u64, add: u64) {
        if self.is_zero() {
            if add > 0 {
                self.limbs.push(add);
            }
            return;
        }
        let m = mult as u128;
        let mut carry = add as u128;
        for limb in &mut self.limbs {
            let prod = (*limb as u128) * m + carry;
            *limb = prod as u64;
            carry = prod >> 64;
        }
        if carry > 0 {
            self.limbs.push(carry as u64);
        }
    }

    pub fn div_rem_u64(&mut self, div: u64) -> u64 {
        debug_assert!(div > 0);
        let d = div as u128;
        let mut rem = 0u128;
        for limb in self.limbs.iter_mut().rev() {
            let cur = (rem << 64) | (*limb as u128);
            *limb = (cur / d) as u64;
            rem = cur % d;
        }
        self.trim();
        rem as u64
    }

    pub fn mul_big(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let a128 = a as u128;
            let mut carry = 0u128;
            for (j, &b) in other.limbs.iter().enumerate() {
                let prod = a128 * (b as u128) + (out[i + j] as u128) + carry;
                out[i + j] = prod as u64;
                carry = prod >> 64;
            }
            if carry > 0 {
                out[i + other.limbs.len()] = carry as u64;
            }
        }
        let mut res = Self { limbs: out };
        res.trim();
        res
    }

    pub fn pow_u64(base: u64, mut exp: usize) -> Self {
        let mut res = Self::from_u64(1);
        let mut cur = Self::from_u64(base);
        while exp > 0 {
            if (exp & 1) != 0 {
                res = res.mul_big(&cur);
            }
            exp >>= 1;
            if exp > 0 {
                cur = cur.mul_big(&cur);
            }
        }
        res
    }

    pub fn to_fixed_le_bytes(&self, n_bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; n_bytes];
        for (i, &w) in self.limbs.iter().enumerate() {
            let le = w.to_le_bytes();
            let base = i * 8;
            for j in 0..8 {
                if base + j < n_bytes {
                    out[base + j] = le[j];
                } else {
                    assert_eq!(le[j], 0, "BigU overflowed fixed byte length {}", n_bytes);
                }
            }
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let n_limbs = (bytes.len() + 7) / 8;
        let mut limbs = vec![0u64; n_limbs];
        for (i, &b) in bytes.iter().enumerate() {
            limbs[i / 8] |= (b as u64) << ((i % 8) * 8);
        }
        let mut res = Self { limbs };
        res.trim();
        res
    }
}

/// Fixed-size 9-limb unsigned integer for fast binomial table `C(n, r)` up to `C(2048, 91) < 2^516`!
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct U576([u64; 9]);

impl U576 {
    const ZERO: Self = Self([0; 9]);
    const ONE: Self = Self([1, 0, 0, 0, 0, 0, 0, 0, 0]);

    #[inline]
    fn add(self, other: Self) -> Self {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for i in 0..9 {
            let s = (self.0[i] as u128) + (other.0[i] as u128) + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        debug_assert_eq!(carry, 0);
        Self(out)
    }

    #[inline]
    fn sub_assign(&mut self, other: Self) {
        let mut borrow = 0i128;
        for i in 0..9 {
            let mut d = (self.0[i] as i128) - (other.0[i] as i128) - borrow;
            if d < 0 {
                d += 1i128 << 64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            self.0[i] = d as u64;
        }
        debug_assert_eq!(borrow, 0);
    }

    #[inline]
    fn ge(self, other: Self) -> bool {
        for i in (0..9).rev() {
            if self.0[i] != other.0[i] {
                return self.0[i] > other.0[i];
            }
        }
        true
    }

    fn to_big(self) -> BigU {
        let mut limbs = self.0.to_vec();
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        BigU { limbs }
    }

    fn from_big(b: &BigU) -> Self {
        assert!(b.limbs.len() <= 9);
        let mut out = [0u64; 9];
        for (i, &w) in b.limbs.iter().enumerate() {
            out[i] = w;
        }
        Self(out)
    }
}

/// Precomputed exact combinatorial rank/unrank codec for `OS-BMW(g, k, R, d)` with upper support `k_reach`.
pub struct OsBmwRankCodec {
    pub g: usize,
    pub k: usize,
    pub r_win: u8,
    pub d: u8,
    pub k_reach: usize,
    pub total_states: BigU,
    pub exact_bits: usize,
    pub fixed_bytes: usize,
    /// Flat Pascal triangle `comb[n * k + r] = C(n, r)` for `n in 0..=g`, `r in 0..k`.
    comb: Vec<U576>,
    /// Cumulative state offsets per `(a, i)` cell (in `(a, i)` lexicographic order): `(a, i, offset, n_norm, v_ov)`.
    cells: Vec<(usize, usize, BigU, u32, u32)>,
    /// Lookup from `(a, i)` to index in `cells` (`usize::MAX` if unreachable).
    cell_idx: Vec<usize>,
    /// Exact normal-slot compact state map for each anchor `a`:
    /// `norm_code_to_compact[a][raw_code]` and inverse `compact_to_norm_code[a][compact]`.
    norm_to_compact: Vec<Vec<u32>>,
    compact_to_norm: Vec<Vec<u32>>,
    /// Exact overflow `(u, h)` compact encoding for each anchor `a`.
    ov_val_list: Vec<Vec<(u8, u32)>>,
}

impl OsBmwRankCodec {
    pub fn new(g: usize, k: usize, r_win: u8, d: u8, k_reach: usize) -> Self {
        assert!(k >= 1 && k <= g && k <= 128);
        // 1. Build Pascal triangle C(n, r) for n=0..=g, r=0..k
        let mut comb = vec![U576::ZERO; (g + 1) * k];
        for n in 0..=g {
            comb[n * k] = U576::ONE;
            let rmax = n.min(k - 1);
            for r in 1..=rmax {
                comb[n * k + r] = comb[(n - 1) * k + (r - 1)].add(comb[(n - 1) * k + r]);
            }
        }

        // 2. Build exact valid normal-slot and overflow alphabets conditional on anchor `a in 0..=k_reach`
        let n_in = OsBmwBlock::n_in(r_win, d) as usize;
        let mut norm_to_compact = Vec::with_capacity(k_reach + 1);
        let mut compact_to_norm = Vec::with_capacity(k_reach + 1);
        let mut ov_val_list = Vec::with_capacity(k_reach + 1);

        for a in 0..=k_reach {
            let base_lo = (a as i32) - (r_win as i32);
            let win_lo = (base_lo + 1).max(1) as usize;
            let mut n2c = vec![u32::MAX; n_in];
            let mut c2n = Vec::new();
            // State 0 (below window) is always valid
            n2c[0] = 0;
            c2n.push(0u32);
            for u in win_lo..=a {
                let tau = ((u as i32) - base_lo) as u8;
                let nb = ((tau - 1).min(d)).min((u - 1) as u8) as u32;
                for h in 0..(1u32 << nb) {
                    let raw = OsBmwBlock::encode_normal(tau, h, d);
                    let idx = c2n.len() as u32;
                    n2c[raw as usize] = idx;
                    c2n.push(raw);
                }
            }
            norm_to_compact.push(n2c);
            compact_to_norm.push(c2n);

            let mut ovs = Vec::new();
            for u in (a + 1)..=k_reach {
                let nb = (d as i32).min((u as i32) - (win_lo as i32)).min((u as i32) - 1).max(0) as u32;
                for h in 0..(1u32 << nb) {
                    ovs.push((u as u8, h));
                }
            }
            ov_val_list.push(ovs);
        }

        // 3. Precompute cumulative cell offsets for every `(a, i)`
        let mut cells = Vec::new();
        let mut cell_idx = vec![usize::MAX; (k_reach + 1) * k];
        let mut acc = BigU::zero();
        for a in 0..=k_reach {
            let n_norm = compact_to_norm[a].len() as u32;
            let v_ov = ov_val_list[a].len() as u32;
            for i in 0..k {
                if i > 0 && v_ov == 0 {
                    continue;
                }
                let c_gi = comb[g * k + i].to_big();
                let norm_pow = BigU::pow_u64(n_norm as u64, g - i);
                let ov_pow = BigU::pow_u64(v_ov as u64, i);
                let count = c_gi.mul_big(&norm_pow).mul_big(&ov_pow);
                let idx = cells.len();
                cell_idx[a * k + i] = idx;
                cells.push((a, i, acc.clone(), n_norm, v_ov));
                acc.add_assign(&count);
            }
        }
        let exact_bits = acc.bit_len();
        let fixed_bytes = (exact_bits + 7) / 8;

        Self {
            g,
            k,
            r_win,
            d,
            k_reach,
            total_states: acc,
            exact_bits,
            fixed_bytes,
            comb,
            cells,
            cell_idx,
            norm_to_compact,
            compact_to_norm,
            ov_val_list,
        }
    }

    /// Encode `OsBmwBlock` into a fixed-length byte vector of EXACT length `self.fixed_bytes` (1,277 B for K=60).
    pub fn encode(&self, blk: &OsBmwBlock) -> Vec<u8> {
        assert_eq!(blk.g, self.g);
        assert_eq!(blk.k, self.k);
        assert_eq!(blk.r_win, self.r_win);
        assert_eq!(blk.d, self.d);
        let a = blk.anchor as usize;
        assert!(a <= self.k_reach, "anchor {} > k_reach {}", a, self.k_reach);
        let i = blk.ov_u.len();
        assert!(i < self.k);
        let cidx = self.cell_idx[a * self.k + i];
        assert_ne!(cidx, usize::MAX);
        let (_, _, ref offset, n_norm, v_ov) = self.cells[cidx];

        let n_in = OsBmwBlock::n_in(self.r_win, self.d);
        let n2c = &self.norm_to_compact[a];

        // 1. Collect overflow positions + compact normal slot symbols
        let mut ov_pos = Vec::with_capacity(i);
        let mut norm_syms = Vec::with_capacity(self.g - i);
        for (r, &s) in blk.slots.iter().enumerate() {
            if self.k > 1 && s == n_in {
                ov_pos.push(r);
            } else {
                let c = n2c[s as usize];
                debug_assert_ne!(c, u32::MAX, "invalid normal slot {} for anchor {}", s, a);
                norm_syms.push(c);
            }
        }
        debug_assert_eq!(ov_pos.len(), i);

        // 2. Combinatorial rank of `ov_pos[0..i]` in `[0, C(g, i))`:
        //    rank = sum_{j=0..i} C(ov_pos[j], j + 1)
        let mut comb_rank = U576::ZERO;
        for (j, &pos) in ov_pos.iter().enumerate() {
            if pos >= j + 1 {
                comb_rank = comb_rank.add(self.comb[pos * self.k + (j + 1)]);
            }
        }

        // 3. Mixed-radix accumulate:
        //    val = ((comb_rank * n_norm^{g-i} + norm_code) * v_ov^i + ov_code) + offset
        //    Chunk normal digits so `base^chunk <= 2^60` (e.g., 24^13 < 2^60 -> 13x fewer big-int ops!).
        let mut val = comb_rank.to_big();
        if n_norm > 1 {
            let (chunk_len, chunk_base) = max_chunk_u64(n_norm as u64);
            let mut idx = 0usize;
            while idx + chunk_len <= norm_syms.len() {
                let mut acc = 0u64;
                for &sym in &norm_syms[idx..idx + chunk_len] {
                    acc = acc * (n_norm as u64) + (sym as u64);
                }
                val.mul_add_u64(chunk_base, acc);
                idx += chunk_len;
            }
            if idx < norm_syms.len() {
                let mut acc = 0u64;
                let mut mult = 1u64;
                for &sym in &norm_syms[idx..] {
                    acc = acc * (n_norm as u64) + (sym as u64);
                    mult *= n_norm as u64;
                }
                val.mul_add_u64(mult, acc);
            }
        }

        // 4. Encode `i` overflow `(ov_u, ov_h)` values in base `v_ov`
        if i > 0 && v_ov > 1 {
            let ov_list = &self.ov_val_list[a];
            let mut ov_syms = Vec::with_capacity(i);
            for j in 0..i {
                let target = (blk.ov_u[j], blk.ov_h[j]);
                let sym = ov_list
                    .binary_search(&target)
                    .expect("overflow (u, h) not in valid domain for anchor a");
                ov_syms.push(sym as u64);
            }
            let (chunk_len, chunk_base) = max_chunk_u64(v_ov as u64);
            let mut idx = 0usize;
            while idx + chunk_len <= ov_syms.len() {
                let mut acc = 0u64;
                for &sym in &ov_syms[idx..idx + chunk_len] {
                    acc = acc * (v_ov as u64) + sym;
                }
                val.mul_add_u64(chunk_base, acc);
                idx += chunk_len;
            }
            if idx < ov_syms.len() {
                let mut acc = 0u64;
                let mut mult = 1u64;
                for &sym in &ov_syms[idx..] {
                    acc = acc * (v_ov as u64) + sym;
                    mult *= v_ov as u64;
                }
                val.mul_add_u64(mult, acc);
            }
        }

        val.add_assign(offset);
        val.to_fixed_le_bytes(self.fixed_bytes)
    }

    /// Decode a fixed-length `self.fixed_bytes` slice back into the exact `OsBmwBlock`.
    pub fn decode(&self, bytes: &[u8]) -> OsBmwBlock {
        assert_eq!(bytes.len(), self.fixed_bytes);
        let mut val = BigU::from_le_bytes(bytes);

        // 1. Binary search cell `(a, i)` by cumulative offset
        let mut lo = 0usize;
        let mut hi = self.cells.len();
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if val.cmp_big(&self.cells[mid].2) != std::cmp::Ordering::Less {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (a, i, ref offset, n_norm, v_ov) = self.cells[lo];
        val.sub_assign(offset);

        // 2. Decode `i` overflow values (reverse order from least-significant chunk!)
        let mut ov_syms = vec![0usize; i];
        if i > 0 && v_ov > 1 {
            let (chunk_len, chunk_base) = max_chunk_u64(v_ov as u64);
            let rem_len = i % chunk_len;
            let mut end = i;
            if rem_len > 0 {
                let mut mult = 1u64;
                for _ in 0..rem_len {
                    mult *= v_ov as u64;
                }
                let mut rem = val.div_rem_u64(mult);
                for j in (end - rem_len..end).rev() {
                    ov_syms[j] = (rem % (v_ov as u64)) as usize;
                    rem /= v_ov as u64;
                }
                end -= rem_len;
            }
            while end > 0 {
                let mut rem = val.div_rem_u64(chunk_base);
                for j in (end - chunk_len..end).rev() {
                    ov_syms[j] = (rem % (v_ov as u64)) as usize;
                    rem /= v_ov as u64;
                }
                end -= chunk_len;
            }
        }

        // 3. Decode `g - i` normal slot symbols (reverse order)
        let n_normal = self.g - i;
        let mut norm_syms = vec![0u32; n_normal];
        if n_norm > 1 {
            let (chunk_len, chunk_base) = max_chunk_u64(n_norm as u64);
            let rem_len = n_normal % chunk_len;
            let mut end = n_normal;
            if rem_len > 0 {
                let mut mult = 1u64;
                for _ in 0..rem_len {
                    mult *= n_norm as u64;
                }
                let mut rem = val.div_rem_u64(mult);
                for j in (end - rem_len..end).rev() {
                    norm_syms[j] = (rem % (n_norm as u64)) as u32;
                    rem /= n_norm as u64;
                }
                end -= rem_len;
            }
            while end > 0 {
                let mut rem = val.div_rem_u64(chunk_base);
                for j in (end - chunk_len..end).rev() {
                    norm_syms[j] = (rem % (n_norm as u64)) as u32;
                    rem /= n_norm as u64;
                }
                end -= chunk_len;
            }
        }

        // 4. Unrank `i` overflow positions from `val in [0, C(g, i))`
        let mut comb_rank = U576::from_big(&val);
        let mut ov_pos = vec![0usize; i];
        let mut pos_hi = self.g;
        for j in (0..i).rev() {
            let r_order = j + 1;
            // Find largest `p in j..pos_hi` such that C(p, r_order) <= comb_rank
            let mut p_lo = j;
            let mut p_hi = pos_hi;
            while p_lo + 1 < p_hi {
                let mid = (p_lo + p_hi) >> 1;
                if comb_rank.ge(self.comb[mid * self.k + r_order]) {
                    p_lo = mid;
                } else {
                    p_hi = mid;
                }
            }
            ov_pos[j] = p_lo;
            comb_rank.sub_assign(self.comb[p_lo * self.k + r_order]);
            pos_hi = p_lo;
        }
        debug_assert_eq!(comb_rank, U576::ZERO);

        // 5. Reassemble `OsBmwBlock`
        let n_in = OsBmwBlock::n_in(self.r_win, self.d);
        let c2n = &self.compact_to_norm[a];
        let mut slots = vec![0u32; self.g];
        let mut ov_ptr = 0usize;
        let mut norm_ptr = 0usize;
        for r in 0..self.g {
            if ov_ptr < i && ov_pos[ov_ptr] == r {
                slots[r] = n_in;
                ov_ptr += 1;
            } else {
                slots[r] = c2n[norm_syms[norm_ptr] as usize];
                norm_ptr += 1;
            }
        }
        let ov_list = &self.ov_val_list[a];
        let mut ov_u = Vec::with_capacity(i);
        let mut ov_h = Vec::with_capacity(i);
        for &sym in &ov_syms {
            let (u, h) = ov_list[sym];
            ov_u.push(u);
            ov_h.push(h);
        }
        OsBmwBlock {
            g: self.g,
            k: self.k,
            r_win: self.r_win,
            d: self.d,
            anchor: a as u8,
            slots,
            ov_u,
            ov_h,
        }
    }
}

#[inline]
fn max_chunk_u64(base: u64) -> (usize, u64) {
    let mut len = 1usize;
    let mut pow = base as u128;
    let limit = 1u128 << 62;
    while pow * (base as u128) <= limit {
        pow *= base as u128;
        len += 1;
    }
    (len, pow as u64)
}
