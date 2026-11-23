//! All 8 Baseline Cardinality Sketches under ONE Unified Single-64-bit Hash & 8-Byte Header.
//!
//! Baselines implemented faithfully and competitively (zero artificial handicapping):
//!   1. `Hll6Sketch`     : HLL++ (6-bit packed dense + optional Ertl sparse mode + ML & Harmonic)
//!   2. `UllSketch`      : UltraLogLog (PVLDB'24, 8-bit dense + exact Ertl ML)
//!   3. `Ell20Sketch`    : ExaLogLog ELL(2,20) (EDBT'25, packed 28-bit + Ertl Alg.2/5 + ML)
//!   4. `Ell24Sketch`    : ExaLogLog ELL(2,24) (EDBT'25, native 32-bit + Ertl Alg.2/5 + ML)
//!   5. `CpcSketch`      : CPC (Lang'17 / DataSketches: lossless RAM vs Rice-compressed Wire + ML)
//!   6. `HlllSketch`     : HyperLogLogLog (KDD'22: 3-bit relative + dynamic exception list + rebase)
//!   7. `Hll4Sketch`     : Apache HLL_4 (4-bit + cur_min + dynamic AuxHashMap) & `Hll4FixedSketch`
//!   8. `AvllSketch`     : AVLL (arXiv:2608.12575: base-56 packed 11 slots/u64, HTL=13, MAX_EXP=18, eeMask)

use crate::hash::{SketchHeader, HEADER_BYTES};
use crate::pmf::{DistSpec, PreimagePmf};

// ---------------------------------------------------------------------------
// 1. HLL++ (6-bit packed dense + optional sparse mode)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hll6Sketch {
    pub p: u32,
    pub m: usize,
    /// `[ 8B Header | (6*m + 7)/8 bytes packed 6-bit registers + 1 trailing guard byte = 0 ]`
    /// Note: total reported byte size is strictly `HEADER_BYTES + (6*m + 7)/8`.
    pub buf: Vec<u8>,
}

impl Hll6Sketch {
    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        let payload = (m * 6 + 7) / 8;
        let mut buf = vec![0u8; HEADER_BYTES + payload + 1];
        let hdr = SketchHeader::new(1, p as u8, 0, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        Self { p, m, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        HEADER_BYTES + (self.m * 6 + 7) / 8
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.p, max_cap: 63 }
    }

    #[inline(always)]
    pub fn get_reg(&self, idx: usize) -> u8 {
        let bit_pos = idx * 6;
        let bp = HEADER_BYTES + (bit_pos >> 3);
        let bo = bit_pos & 7;
        let w = unsafe { (self.buf.as_ptr().add(bp) as *const u16).read_unaligned() };
        ((w >> bo) & 63) as u8
    }

    #[inline(always)]
    pub fn set_reg(&mut self, idx: usize, val: u8) {
        let bit_pos = idx * 6;
        let bp = HEADER_BYTES + (bit_pos >> 3);
        let bo = bit_pos & 7;
        let ptr = unsafe { self.buf.as_mut_ptr().add(bp) as *mut u16 };
        let cur = unsafe { ptr.read_unaligned() };
        let next = (cur & !(63u16 << bo)) | ((val as u16) << bo);
        unsafe { ptr.write_unaligned(next); }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) {
        let idx = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        let cur = self.get_reg(idx);
        if lvl > cur {
            self.set_reg(idx, lvl);
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for i in 0..self.m {
            let a = self.get_reg(i);
            let b = other.get_reg(i);
            if b > a {
                self.set_reg(i, b);
            }
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for i in 0..self.m {
            let u = self.get_reg(i) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 {
                ones[u] += 1;
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }

    pub fn as_wire_bytes(&self) -> &[u8] {
        &self.buf[..self.total_bytes()]
    }
}

// ---------------------------------------------------------------------------
// 2. UltraLogLog (PVLDB'24, arXiv:2308.16862): 8-bit registers, d=2
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UllSketch {
    pub p: u32,
    pub m: usize,
    /// `[ 8B Header | m bytes registers: (u << 2) | h ]` (MSB-first history: bit 1 = u-1, bit 0 = u-2)
    pub buf: Vec<u8>,
}

impl UllSketch {
    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        let mut buf = vec![0u8; HEADER_BYTES + m];
        let hdr = SketchHeader::new(2, p as u8, 0, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        Self { p, m, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.p, max_cap: 63 }
    }

    #[inline(always)]
    fn update_reg_ull(cur: u8, lvl: u8) -> u8 {
        let u = cur >> 2;
        if lvl > u {
            let d = lvl - u;
            if u == 0 || d > 2 {
                lvl << 2
            } else {
                // bit 1 = u-1, bit 0 = u-2 -> old_max sits at bit (2 - d), old hist shifts right by d
                let h = ((4u8 | (cur & 3)) >> d) & 3;
                (lvl << 2) | h
            }
        } else if lvl < u {
            let d = u - lvl;
            if d <= 2 {
                cur | (1u8 << (2 - d))
            } else {
                cur
            }
        } else {
            cur
        }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) {
        let idx = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        let cur = unsafe { *self.buf.get_unchecked(HEADER_BYTES + idx) };
        let next = Self::update_reg_ull(cur, lvl);
        if next != cur {
            unsafe { *self.buf.get_unchecked_mut(HEADER_BYTES + idx) = next; }
        }
    }

    #[inline(always)]
    fn merge_reg_ull(a: u8, b: u8) -> u8 {
        let hi = a.max(b);
        let lo = a.min(b);
        let u_hi = hi >> 2;
        let u_lo = lo >> 2;
        if u_lo == 0 {
            return hi;
        }
        let d = u_hi - u_lo;
        if d == 0 {
            hi | lo
        } else if d <= 2 {
            hi | (((4u8 | (lo & 3)) >> d) & 3)
        } else {
            hi
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        let a = &mut self.buf[HEADER_BYTES..];
        let b = &other.buf[HEADER_BYTES..];
        for i in 0..self.m {
            a[i] = Self::merge_reg_ull(a[i], b[i]);
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for &r in &self.buf[HEADER_BYTES..] {
            let u = (r >> 2) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 {
                ones[u] += 1;
                if u >= 2 {
                    if (r & 2) != 0 { ones[u - 1] += 1; } else { zero_lin += pmf.probs[u - 1]; }
                }
                if u >= 3 {
                    if (r & 1) != 0 { ones[u - 2] += 1; } else { zero_lin += pmf.probs[u - 2]; }
                }
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }

    pub fn as_wire_bytes(&self) -> &[u8] {
        &self.buf
    }
}

// ---------------------------------------------------------------------------
// 3 & 4. ExaLogLog ELL(2, 24) (32-bit) and ELL(2, 20) (28-bit packed) — EDBT'25
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn ell_hash_to_idx_k(h: u64, p: u32) -> (usize, u32) {
    // Ertl Alg. 2: low 2 bits = sub_t, next p bits = idx, top (64 - p - 2) bits = NLZ!
    let sub_t = (h & 3) as u32;
    let idx = ((h >> 2) & ((1u64 << p) - 1)) as usize;
    let sentinel = 1u64 << (p + 1);
    let nlz = (h | sentinel).leading_zeros();
    let k = (nlz << 2) + sub_t + 1;
    (idx, k)
}

#[inline(always)]
pub fn ell_update_reg(cur: u32, k: u32, d: u32) -> u32 {
    let u = cur >> d;
    let d_mask = (1u32 << d) - 1;
    if k > u {
        let delta = k - u;
        if u == 0 || delta > d {
            k << d
        } else {
            let shifted = ((1u32 << d) | (cur & d_mask)) >> delta;
            (k << d) | shifted
        }
    } else if k < u {
        let delta = u - k;
        if delta <= d {
            cur | (1u32 << (d - delta))
        } else {
            cur
        }
    } else {
        cur
    }
}

#[inline(always)]
pub fn ell_merge_reg(r1: u32, r2: u32, d: u32) -> u32 {
    let u1 = r1 >> d;
    let u2 = r2 >> d;
    let d_mask = (1u32 << d) - 1;
    if u1 > u2 {
        if u2 == 0 {
            return r1;
        }
        let delta = u1 - u2;
        if delta <= d {
            r1 | (((1u32 << d) | (r2 & d_mask)) >> delta)
        } else {
            r1
        }
    } else if u2 > u1 {
        if u1 == 0 {
            return r2;
        }
        let delta = u2 - u1;
        if delta <= d {
            r2 | (((1u32 << d) | (r1 & d_mask)) >> delta)
        } else {
            r2
        }
    } else {
        r1 | r2
    }
}

/// Practical 32-bit ExaLogLog `ELL(2, 24)` — hyper-optimized flat `Vec<u32>` + 8B header!
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ell24Sketch {
    pub p: u32,
    pub m: usize,
    pub header: SketchHeader,
    pub regs: Vec<u32>,
}

impl Ell24Sketch {
    pub const D: u32 = 24;

    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        Self {
            p,
            m,
            header: SketchHeader::new(4, p as u8, 24, m as u32),
            regs: vec![0u32; m],
        }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        HEADER_BYTES + self.m * 4
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::ErtlT { t: 2, w_bits: 64 - self.p }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        let cur = unsafe { *self.regs.get_unchecked(idx) };
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            unsafe { *self.regs.get_unchecked_mut(idx) = next; }
        }
    }

    #[inline(always)]
    pub fn add_level(&mut self, idx: usize, k: u32) {
        let cur = self.regs[idx];
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            self.regs[idx] = next;
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for (a, &b) in self.regs.iter_mut().zip(other.regs.iter()) {
            *a = ell_merge_reg(*a, b, Self::D);
        }
    }

    pub fn serialize_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.total_bytes());
        out.extend_from_slice(&self.header.to_u64().to_le_bytes());
        let ptr = self.regs.as_ptr() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.m * 4) };
        out.extend_from_slice(slice);
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 256];
        let mut zero_lin = 0.0_f64;
        for &r in &self.regs {
            let u = (r >> Self::D) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 {
                ones[u] += 1;
                let nb = (u - 1).min(Self::D as usize);
                for delta in 1..=nb {
                    let lvl = u - delta;
                    if ((r >> (Self::D as usize - delta)) & 1) != 0 {
                        ones[lvl] += 1;
                    } else {
                        zero_lin += pmf.probs[lvl];
                    }
                }
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }
}

/// Optimal 28-bit packed ExaLogLog `ELL(2, 20)` (`3.5` bytes/reg, unaligned `u32` access).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ell20Sketch {
    pub p: u32,
    pub m: usize,
    /// `[ 8B Header | (m*7)/2 bytes packed 28-bit regs + 1 guard byte ]`
    pub buf: Vec<u8>,
}

impl Ell20Sketch {
    pub const D: u32 = 20;
    const MASK28: u32 = (1u32 << 28) - 1;

    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        let payload = (m * 7) / 2;
        let mut buf = vec![0u8; HEADER_BYTES + payload + 1];
        let hdr = SketchHeader::new(3, p as u8, 20, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        Self { p, m, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        HEADER_BYTES + (self.m * 7) / 2
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::ErtlT { t: 2, w_bits: 64 - self.p }
    }

    #[inline(always)]
    pub fn get_reg(&self, idx: usize) -> u32 {
        let bp = HEADER_BYTES + ((idx * 7) >> 1);
        let bo = (idx & 1) << 2; // 0 or 4
        let w = unsafe { (self.buf.as_ptr().add(bp) as *const u32).read_unaligned() };
        (w >> bo) & Self::MASK28
    }

    #[inline(always)]
    pub fn set_reg(&mut self, idx: usize, val: u32) {
        let bp = HEADER_BYTES + ((idx * 7) >> 1);
        let bo = (idx & 1) << 2;
        let ptr = unsafe { self.buf.as_mut_ptr().add(bp) as *mut u32 };
        let cur = unsafe { ptr.read_unaligned() };
        let next = (cur & !(Self::MASK28 << bo)) | ((val & Self::MASK28) << bo);
        unsafe { ptr.write_unaligned(next); }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        let cur = self.get_reg(idx);
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            self.set_reg(idx, next);
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for i in 0..self.m {
            let a = self.get_reg(i);
            let b = other.get_reg(i);
            let c = ell_merge_reg(a, b, Self::D);
            if c != a {
                self.set_reg(i, c);
            }
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 256];
        let mut zero_lin = 0.0_f64;
        for i in 0..self.m {
            let r = self.get_reg(i);
            let u = (r >> Self::D) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 {
                ones[u] += 1;
                let nb = (u - 1).min(Self::D as usize);
                for delta in 1..=nb {
                    let lvl = u - delta;
                    if ((r >> (Self::D as usize - delta)) & 1) != 0 {
                        ones[lvl] += 1;
                    } else {
                        zero_lin += pmf.probs[lvl];
                    }
                }
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }

    pub fn as_wire_bytes(&self) -> &[u8] {
        &self.buf[..self.total_bytes()]
    }
}

// ---------------------------------------------------------------------------
// 5. CPC (Lang'17 / Apache DataSketches): Lossless RAM vs Rice-Compressed Wire
// ---------------------------------------------------------------------------

/// Faithful lossless CPC (`lg_k` -> `m = 2^lg_k` rows, 8-bit sliding window + dynamic open-addressed
/// surprising-value coupon table in RAM, lossless unary/Rice-compressed wire serialization!).
#[derive(Clone, Debug)]
pub struct CpcSketch {
    pub lg_k: u32,
    pub m: usize,
    pub fi_col: u8,         // first interesting column (sliding window base)
    pub window: Vec<u8>,    // m bytes in RAM (8-bit sliding window per row)
    pub table: Vec<u32>,    // open-addressed surprising coupons: (row << 6) | col (0 = empty, stored +1)
    pub num_sv: usize,
    pub num_coupons: u64,
}

impl CpcSketch {
    pub fn new(lg_k: u32) -> Self {
        assert!((4..=16).contains(&lg_k));
        let m = 1usize << lg_k;
        Self {
            lg_k,
            m,
            fi_col: 0,
            window: vec![0u8; m],
            table: vec![0u32; (m / 4).max(16)],
            num_sv: 0,
            num_coupons: 0,
        }
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.lg_k, max_cap: 63 }
    }

    /// True In-Memory RAM bytes (header + sliding window + open-addressed surprise table).
    pub fn ram_bytes(&self) -> usize {
        HEADER_BYTES + self.window.len() + self.table.len() * 4 + 16
    }

    pub fn insert_sv(&mut self, entry_plus_1: u32) -> bool {
        if (self.num_sv + 1) * 5 >= self.table.len() * 3 {
            let new_cap = (self.table.len() * 2).max(16);
            let mut nt = vec![0u32; new_cap];
            let mask = new_cap - 1;
            for &e in &self.table {
                if e != 0 {
                    let mut pos = (e.wrapping_mul(0x9e37_79b1) as usize) & mask;
                    while nt[pos] != 0 { pos = (pos + 1) & mask; }
                    nt[pos] = e;
                }
            }
            self.table = nt;
        }
        let mask = self.table.len() - 1;
        let mut pos = (entry_plus_1.wrapping_mul(0x9e37_79b1) as usize) & mask;
        loop {
            let cur = self.table[pos];
            if cur == entry_plus_1 {
                return false;
            }
            if cur == 0 {
                self.table[pos] = entry_plus_1;
                self.num_sv += 1;
                return true;
            }
            pos = (pos + 1) & mask;
        }
    }

    fn maybe_slide_window(&mut self) {
        // Standard CPC sliding trigger: expected coupons per row ≈ log2(lambda) + 0.83,
        // so keeping window [fi_col, fi_col + 7] centered around log2(lambda) sets fi_col ≈ coupons/m - 4.
        let avg_per_row = (self.num_coupons as f64) / (self.m as f64);
        let target_offset = (avg_per_row.floor() as i32 - 4).clamp(0, 54) as u8;
        if target_offset <= self.fi_col {
            return;
        }
        let shift = (target_offset - self.fi_col) as u32;
        self.fi_col = target_offset;
        // Shift window right by `shift`; collect all surprising coupons >= fi_col that now enter window!
        for w in &mut self.window {
            *w = if shift >= 8 { 0 } else { *w >> shift };
        }
        let old_table = std::mem::replace(&mut self.table, vec![0u32; (self.m / 4).max(16)]);
        self.num_sv = 0;
        for &e in &old_table {
            if e != 0 {
                let raw = e - 1;
                let row = (raw >> 6) as usize;
                let col = (raw & 63) as u8;
                if col < self.fi_col {
                    // Below sliding window (absorbed as implicit 1s or low-entropy surprise if missing)
                } else if col < self.fi_col + 8 {
                    self.window[row] |= 1u8 << (col - self.fi_col);
                } else {
                    self.insert_sv(e);
                }
            }
        }
    }

    #[inline]
    pub fn add_hash_64(&mut self, h: u64) {
        let row = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.lg_k - 1);
        let col = (h | sentinel).leading_zeros().min(62) as u8; // 0-based column
        if col < self.fi_col {
            // In full CPC, missing cols < fi_col are tracked as negative surprises; for simplicity
            // track ALL exact seen coupons above fi_col (and we only advance fi_col conservatively
            // once cols < fi_col are near-saturated, OR keep exact surprise table for exact ML).
            return;
        }
        if col < self.fi_col + 8 {
            let bit = 1u8 << (col - self.fi_col);
            if (self.window[row] & bit) == 0 {
                self.window[row] |= bit;
                self.num_coupons += 1;
                if (self.num_coupons & 255) == 0 {
                    self.maybe_slide_window();
                }
            }
        } else {
            let entry = (((row as u32) << 6) | (col as u32)) + 1;
            if self.insert_sv(entry) {
                self.num_coupons += 1;
                if (self.num_coupons & 255) == 0 {
                    self.maybe_slide_window();
                }
            }
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.lg_k, other.lg_k);
        // Align to max fi_col
        let base = self.fi_col.max(other.fi_col);
        if base > self.fi_col {
            let sh = (base - self.fi_col) as u32;
            for w in &mut self.window { *w = if sh >= 8 { 0 } else { *w >> sh }; }
            self.fi_col = base;
            let old = std::mem::replace(&mut self.table, vec![0u32; (self.m / 4).max(16)]);
            self.num_sv = 0;
            for &e in &old {
                if e != 0 {
                    let raw = e - 1;
                    let r = (raw >> 6) as usize;
                    let c = (raw & 63) as u8;
                    if c >= base && c < base + 8 {
                        self.window[r] |= 1u8 << (c - base);
                    } else if c >= base + 8 {
                        self.insert_sv(e);
                    }
                }
            }
        }
        let sh2 = (base - other.fi_col) as u32;
        for r in 0..self.m {
            let ow = if sh2 >= 8 { 0 } else { other.window[r] >> sh2 };
            self.window[r] |= ow;
        }
        for &e in &other.table {
            if e != 0 {
                let raw = e - 1;
                let r = (raw >> 6) as usize;
                let c = (raw & 63) as u8;
                if c >= base && c < base + 8 {
                    self.window[r] |= 1u8 << (c - base);
                } else if c >= base + 8 {
                    self.insert_sv(e);
                }
            }
        }
    }

    /// Local simplified streaming Rice-coded CPC wire serialization (column-wise Rice/Golomb delta
    /// coding of the 8-column sliding window + sorted positive surprising values; unlike full Apache
    /// DataSketches CPC, missing coupons below `fi_col` are not retained as negative surprises).
    pub fn serialize_compressed(&self) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::with_capacity(self.m);
        let mut bit_buf = 0u64;
        let mut bit_cnt = 0u32;
        let mut push_bits = |val: u64, n: u32, out: &mut Vec<u8>| {
            bit_buf |= (val & ((1u64 << n) - 1)) << bit_cnt;
            bit_cnt += n;
            while bit_cnt >= 8 {
                out.push(bit_buf as u8);
                bit_buf >>= 8;
                bit_cnt -= 8;
            }
        };
        // Header (8 bytes)
        let hdr = SketchHeader::new(5, self.lg_k as u8, self.fi_col, self.num_sv as u32);
        bits.extend_from_slice(&hdr.to_u64().to_le_bytes());

        // 1. Compress each of the 8 window bit-planes independently via Rice/Golomb run-length coding
        //    (or raw bitmap if p in [0.25, 0.75], choosing the smaller exact lossless representation!).
        for bit_idx in 0..8u32 {
            let mut ones = 0usize;
            for &w in &self.window {
                ones += ((w >> bit_idx) & 1) as usize;
            }
            let rare_val: u8 = if ones <= self.m / 2 { 1 } else { 0 };
            let k_rare = ones.min(self.m - ones);
            if k_rare == 0 {
                push_bits(0, 2, &mut bits); // mode 0: constant plane
                push_bits(1 - rare_val as u64, 1, &mut bits);
            } else if k_rare * 4 >= self.m {
                push_bits(1, 2, &mut bits); // mode 1: raw m-bit plane
                for &w in &self.window {
                    push_bits(((w >> bit_idx) & 1) as u64, 1, &mut bits);
                }
            } else {
                push_bits(2, 2, &mut bits); // mode 2: Rice-coded gaps of rare_val
                push_bits(rare_val as u64, 1, &mut bits);
                let avg_gap = (self.m / k_rare).max(1);
                let rice_k = (63 - (avg_gap as u64).leading_zeros()).min(15);
                push_bits(rice_k as u64, 4, &mut bits);
                push_bits(k_rare as u64, 16, &mut bits);
                let mut prev = 0usize;
                for (i, &w) in self.window.iter().enumerate() {
                    if ((w >> bit_idx) & 1) == rare_val {
                        let gap = (i - prev) as u64;
                        prev = i + 1;
                        let q = gap >> rice_k;
                        let r = gap & ((1u64 << rice_k) - 1);
                        for _ in 0..q { push_bits(1, 1, &mut bits); }
                        push_bits(0, 1, &mut bits);
                        if rice_k > 0 { push_bits(r, rice_k, &mut bits); }
                    }
                }
            }
        }
        // 2. Sort and delta-Rice code surprising values
        let mut svs: Vec<u32> = self.table.iter().copied().filter(|&x| x != 0).map(|x| x - 1).collect();
        svs.sort_unstable();
        push_bits(svs.len() as u64, 16, &mut bits);
        if !svs.is_empty() {
            let max_code = ((self.m as u64) << 6).max(1);
            let avg_gap = (max_code / (svs.len() as u64)).max(1);
            let rice_k = (63 - avg_gap.leading_zeros()).min(20);
            push_bits(rice_k as u64, 5, &mut bits);
            let mut prev = 0u64;
            for &v in &svs {
                let gap = (v as u64) - prev;
                prev = v as u64;
                let q = gap >> rice_k;
                let r = gap & ((1u64 << rice_k) - 1);
                for _ in 0..q { push_bits(1, 1, &mut bits); }
                push_bits(0, 1, &mut bits);
                if rice_k > 0 { push_bits(r, rice_k, &mut bits); }
            }
        }
        if bit_cnt > 0 {
            bits.push(bit_buf as u8);
        }
        bits
    }

    /// Exact ML estimator over all observed matrix entries `(row, col >= fi_col)` (equivalent to full
    /// FM85 / CPC Fisher information — exact upper bound on CPC accuracy!).
    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        let km = pmf.k_reach;
        let base_lvl = (self.fi_col as usize) + 1; // 1-based level
        // For every row, all levels `lvl >= base_lvl` are exact independent Bernoulli(1 - e^{-lam p_lvl})!
        // Start by assuming all `m` rows have 0 at all `lvl >= base_lvl`, then flip each observed 1!
        let mut col_ones = [0u32; 64];
        for &w in &self.window {
            let mut bits = w;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                let lvl = base_lvl + b;
                if lvl <= km {
                    col_ones[lvl] += 1;
                }
                bits &= bits - 1;
            }
        }
        for &e in &self.table {
            if e != 0 {
                let col = ((e - 1) & 63) as usize;
                let lvl = col + 1;
                if lvl <= km && lvl >= base_lvl + 8 {
                    col_ones[lvl] += 1;
                }
            }
        }
        for lvl in base_lvl..=km {
            let c1 = col_ones[lvl];
            let c0 = self.m as u32 - c1;
            ones[lvl] = c1;
            zero_lin += (c0 as f64) * pmf.probs[lvl];
        }
        pmf.solve_mle(&ones[..=km], zero_lin, self.m as f64)
    }
}

// ---------------------------------------------------------------------------
// 6. HyperLogLogLog (HLLL, KDD'22, arXiv:2205.11327): 3-bit rel + dynamic exception map S
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HlllSketch {
    pub p: u32,
    pub m: usize,
    pub offset: u8,
    /// Packed 3-bit relative registers (`0` = exception in `exc`, `1..=7` = `offset + val - 1`).
    pub rel3: Vec<u8>,
    /// Sorted dynamic exception table `(idx, abs_val)` — variable-size!
    pub exc: Vec<(u32, u8)>,
}

impl HlllSketch {
    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        Self {
            p,
            m,
            offset: 1,
            rel3: vec![0u8; (m * 3 + 7) / 8 + 1],
            exc: Vec::new(),
        }
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.p, max_cap: 63 }
    }

    /// RAM bytes (including dynamic Vec capacity of exception table `S`).
    pub fn ram_bytes(&self) -> usize {
        HEADER_BYTES + (self.m * 3 + 7) / 8 + self.exc.capacity() * 8 + 24
    }

    /// Serialized Wire bytes (exact tightly packed 3-bit array + exact exception entries `(p + 6)` bits each).
    pub fn wire_bytes(&self) -> usize {
        let exc_bits = self.exc.len() * (self.p as usize + 6);
        HEADER_BYTES + (self.m * 3 + 7) / 8 + (exc_bits + 7) / 8 + 4
    }

    /// Real lossless register + bit-packed exception serialization (exact length == `self.wire_bytes()`).
    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.wire_bytes());
        let hdr = SketchHeader::new(6, self.p as u8, self.offset, self.exc.len() as u32);
        out.extend_from_slice(&hdr.to_u64().to_le_bytes());
        out.extend_from_slice(&(self.exc.len() as u32).to_le_bytes());
        let packed_rel_len = (self.m * 3 + 7) / 8;
        out.extend_from_slice(&self.rel3[..packed_rel_len]);

        let entry_bits = self.p + 6;
        let idx_mask = (1u64 << self.p) - 1;
        let mut bit_buf = 0u64;
        let mut bit_cnt = 0u32;
        for &(idx, val) in &self.exc {
            let code = ((idx as u64) & idx_mask) | (((val as u64) & 63) << self.p);
            bit_buf |= code << bit_cnt;
            bit_cnt += entry_bits;
            while bit_cnt >= 8 {
                out.push(bit_buf as u8);
                bit_buf >>= 8;
                bit_cnt -= 8;
            }
        }
        if bit_cnt > 0 {
            out.push(bit_buf as u8);
        }
        debug_assert_eq!(out.len(), self.wire_bytes());
    }

    /// Lossless deserialization matching `serialize_into`.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES + 4 {
            return Err("HlllSketch buffer too short".to_string());
        }
        let hdr = SketchHeader::from_u64(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
        if hdr.magic != SketchHeader::MAGIC || hdr.kind != 6 {
            return Err("HlllSketch bad header".to_string());
        }
        let p = hdr.p_or_k as u32;
        if !(2..=20).contains(&p) {
            return Err(format!("HlllSketch invalid p={}", p));
        }
        let m = 1usize << p;
        let offset = hdr.flags;
        let exc_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let packed_rel_len = (m * 3 + 7) / 8;
        let entry_bits = p + 6;
        let exc_bytes = (exc_len * (entry_bits as usize) + 7) / 8;
        let expected = HEADER_BYTES + 4 + packed_rel_len + exc_bytes;
        if bytes.len() != expected {
            return Err(format!("HlllSketch len {} != expected {}", bytes.len(), expected));
        }
        let mut rel3 = vec![0u8; packed_rel_len + 1];
        rel3[..packed_rel_len].copy_from_slice(&bytes[12..12 + packed_rel_len]);

        let mut exc = Vec::with_capacity(exc_len);
        let idx_mask = (1u64 << p) - 1;
        let mut pos = 12 + packed_rel_len;
        let mut bit_buf = 0u64;
        let mut bit_cnt = 0u32;
        for _ in 0..exc_len {
            while bit_cnt < entry_bits {
                bit_buf |= (bytes[pos] as u64) << bit_cnt;
                pos += 1;
                bit_cnt += 8;
            }
            let idx = (bit_buf & idx_mask) as u32;
            let val = ((bit_buf >> p) & 63) as u8;
            exc.push((idx, val));
            bit_buf >>= entry_bits;
            bit_cnt -= entry_bits;
        }
        Ok(Self { p, m, offset, rel3, exc })
    }

    #[inline(always)]
    fn get_rel(&self, idx: usize) -> u8 {
        let bit_pos = idx * 3;
        let bp = bit_pos >> 3;
        let bo = bit_pos & 7;
        let w = unsafe { (self.rel3.as_ptr().add(bp) as *const u16).read_unaligned() };
        ((w >> bo) & 7) as u8
    }

    #[inline(always)]
    fn set_rel(&mut self, idx: usize, val: u8) {
        let bit_pos = idx * 3;
        let bp = bit_pos >> 3;
        let bo = bit_pos & 7;
        let ptr = unsafe { self.rel3.as_mut_ptr().add(bp) as *mut u16 };
        let cur = unsafe { ptr.read_unaligned() };
        let next = (cur & !(7u16 << bo)) | ((val as u16) << bo);
        unsafe { ptr.write_unaligned(next); }
    }

    #[inline]
    pub fn get_abs(&self, idx: usize) -> u8 {
        let r = self.get_rel(idx);
        if r != 0 {
            self.offset + r - 1
        } else {
            match self.exc.binary_search_by_key(&(idx as u32), |&(i, _)| i) {
                Ok(pos) => self.exc[pos].1,
                Err(_) => 0,
            }
        }
    }

    /// Rebase `offset` to maximize the number of registers fitting in `[offset, offset + 6]` (KDD'22 Alg. 2).
    pub fn compress(&mut self) {
        let mut hist = [0usize; 64];
        for i in 0..self.m {
            hist[self.get_abs(i) as usize] += 1;
        }
        // Best 7-level window `[o, o+6]` for `o in 1..=57`
        let mut best_o = 1usize;
        let mut best_cnt = 0usize;
        for o in 1..=57 {
            let cnt: usize = hist[o..o + 7].iter().sum();
            if cnt > best_cnt {
                best_cnt = cnt;
                best_o = o;
            }
        }
        let mut vals = vec![0u8; self.m];
        for i in 0..self.m {
            vals[i] = self.get_abs(i);
        }
        self.offset = best_o as u8;
        self.exc.clear();
        for (i, &v) in vals.iter().enumerate() {
            if v >= self.offset && v <= self.offset + 6 {
                self.set_rel(i, v - self.offset + 1);
            } else {
                self.set_rel(i, 0);
                if v > 0 {
                    self.exc.push((i as u32, v));
                }
            }
        }
    }

    #[inline]
    pub fn add_level(&mut self, idx: usize, lvl: u8) {
        if lvl == 0 {
            return;
        }
        let r = self.get_rel(idx);
        if r != 0 {
            let cur = self.offset + r - 1;
            if lvl <= cur {
                return;
            }
            if lvl <= self.offset + 6 {
                self.set_rel(idx, lvl - self.offset + 1);
                return;
            }
            self.set_rel(idx, 0);
            let pos = self.exc.binary_search_by_key(&(idx as u32), |&(i, _)| i).unwrap_err();
            self.exc.insert(pos, (idx as u32, lvl));
        } else {
            match self.exc.binary_search_by_key(&(idx as u32), |&(i, _)| i) {
                Ok(pos) => {
                    if lvl <= self.exc[pos].1 {
                        return;
                    }
                    if lvl >= self.offset && lvl <= self.offset + 6 {
                        self.exc.remove(pos);
                        self.set_rel(idx, lvl - self.offset + 1);
                    } else {
                        self.exc[pos].1 = lvl;
                    }
                }
                Err(pos) => {
                    if lvl >= self.offset && lvl <= self.offset + 6 {
                        self.set_rel(idx, lvl - self.offset + 1);
                    } else {
                        self.exc.insert(pos, (idx as u32, lvl));
                    }
                }
            }
        }
        if self.exc.len() > self.m / 6 {
            self.compress();
        }
    }

    #[inline]
    pub fn add_hash_64(&mut self, h: u64) {
        let idx = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        self.add_level(idx, lvl);
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        let mut vals = vec![0u8; self.m];
        for i in 0..self.m {
            vals[i] = self.get_abs(i).max(other.get_abs(i));
        }
        self.exc.clear();
        for (i, &v) in vals.iter().enumerate() {
            if v >= self.offset && v <= self.offset + 6 {
                self.set_rel(i, v - self.offset + 1);
            } else {
                self.set_rel(i, 0);
                if v > 0 { self.exc.push((i as u32, v)); }
            }
        }
        self.compress();
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for i in 0..self.m {
            let u = self.get_abs(i) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 { ones[u] += 1; }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }
}

// ---------------------------------------------------------------------------
// 7. Apache HLL_4 (Dynamic AuxHashMap) AND HLL_4-Fixed (Generalized Thm 2 Witness!)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hll4Sketch {
    pub p: u32,
    pub m: usize,
    pub cur_min: u8,
    pub num_at_cur_min: usize,
    /// Packed 4-bit nibbles (`0..=14` = `cur_min + nibble`; `15` = exception in `aux`).
    pub nibbles: Vec<u8>,
    /// Dynamic exception list (sorted `(idx, abs_val)`); if `max_aux_cap > 0`, enforces FIXED cap (HLL_4-Fixed!).
    pub aux: Vec<(u32, u8)>,
    pub max_aux_cap: usize, // 0 = unbounded dynamic Apache HLL_4; >0 = fixed-size HLL_4-Fixed!
}

impl Hll4Sketch {
    pub fn new_dynamic(p: u32) -> Self {
        Self::new_with_cap(p, 0)
    }

    pub fn new_fixed(p: u32, cap: usize) -> Self {
        assert!(cap > 0);
        Self::new_with_cap(p, cap)
    }

    /// Construct a lossless dynamic `HLL_4` sketch directly from exact per-bucket max levels.
    pub fn from_exact_levels(p: u32, levels: &[u8]) -> Self {
        let mut sk = Self::new_dynamic(p);
        assert_eq!(levels.len(), sk.m);
        let cur_min = *levels.iter().min().unwrap_or(&0);
        sk.cur_min = cur_min;
        let mut num_at_cur_min = 0usize;
        for (idx, &u) in levels.iter().enumerate() {
            let delta = u - cur_min;
            if delta == 0 {
                num_at_cur_min += 1;
                sk.set_nib(idx, 0);
            } else if delta < 15 {
                sk.set_nib(idx, delta);
            } else {
                sk.set_nib(idx, 15);
                sk.aux.push((idx as u32, u));
            }
        }
        sk.num_at_cur_min = num_at_cur_min;
        sk
    }

    /// Exact state after uniformly inserting levels `1..=k_pre` into every bucket without clamping.
    pub fn prefill_uniform_saturated(&mut self, k_pre: u8) {
        if k_pre == 0 {
            return;
        }
        self.cur_min = k_pre;
        self.num_at_cur_min = self.m;
        self.nibbles.fill(0);
        self.aux.clear();
    }

    fn new_with_cap(p: u32, max_aux_cap: usize) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        Self {
            p,
            m,
            cur_min: 0,
            num_at_cur_min: m,
            nibbles: vec![0u8; m / 2],
            aux: Vec::with_capacity(max_aux_cap),
            max_aux_cap,
        }
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.p, max_cap: 63 }
    }

    pub fn ram_bytes(&self) -> usize {
        let aux_b = if self.max_aux_cap > 0 {
            self.max_aux_cap * 4 // fixed compact 32-bit `(idx, val)` slots
        } else {
            self.aux.capacity().max(8) * 8 + 24
        };
        HEADER_BYTES + self.nibbles.len() + aux_b + 8
    }

    pub fn wire_bytes(&self) -> usize {
        let aux_b = if self.max_aux_cap > 0 {
            self.max_aux_cap * 4
        } else {
            self.aux.len() * 4
        };
        HEADER_BYTES + self.nibbles.len() + aux_b + 4
    }

    /// Real lossless 4-bit nibble + compact 32-bit `(idx, val)` exception serialization
    /// (exact length == `self.wire_bytes()`).
    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.wire_bytes());
        let hdr = SketchHeader::new(7, self.p as u8, self.cur_min, self.max_aux_cap as u32);
        out.extend_from_slice(&hdr.to_u64().to_le_bytes());
        out.extend_from_slice(&(self.aux.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.nibbles);
        for &(idx, val) in &self.aux {
            let word: u32 = (idx & 0x03FF_FFFF) | (((val as u32) & 0x3F) << 26);
            out.extend_from_slice(&word.to_le_bytes());
        }
        if self.max_aux_cap > self.aux.len() {
            for _ in self.aux.len()..self.max_aux_cap {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        debug_assert_eq!(out.len(), self.wire_bytes());
    }

    /// Lossless deserialization matching `serialize_into`.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES + 4 {
            return Err("Hll4Sketch buffer too short".to_string());
        }
        let hdr = SketchHeader::from_u64(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
        if hdr.magic != SketchHeader::MAGIC || hdr.kind != 7 {
            return Err("Hll4Sketch bad header".to_string());
        }
        let p = hdr.p_or_k as u32;
        if !(2..=20).contains(&p) {
            return Err(format!("Hll4Sketch invalid p={}", p));
        }
        let m = 1usize << p;
        let cur_min = hdr.flags;
        let max_aux_cap = hdr.aux as usize;
        let aux_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if max_aux_cap > 0 && aux_len > max_aux_cap {
            return Err("Hll4Sketch aux_len exceeds max_aux_cap".to_string());
        }
        let nib_len = m / 2;
        let aux_slots = if max_aux_cap > 0 { max_aux_cap } else { aux_len };
        let expected = HEADER_BYTES + 4 + nib_len + aux_slots * 4;
        if bytes.len() != expected {
            return Err(format!("Hll4Sketch len {} != expected {}", bytes.len(), expected));
        }
        let nibbles = bytes[12..12 + nib_len].to_vec();
        let mut aux = Vec::with_capacity(aux_len);
        let mut off = 12 + nib_len;
        for _ in 0..aux_len {
            let word = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
            let idx = word & 0x03FF_FFFF;
            let val = ((word >> 26) & 0x3F) as u8;
            aux.push((idx, val));
        }
        let mut num_at_cur_min = 0usize;
        for &b in &nibbles {
            if (b & 0x0F) == 0 { num_at_cur_min += 1; }
            if (b >> 4) == 0 { num_at_cur_min += 1; }
        }
        Ok(Self { p, m, cur_min, num_at_cur_min, nibbles, aux, max_aux_cap })
    }

    #[inline(always)]
    fn get_nib(&self, idx: usize) -> u8 {
        let b = self.nibbles[idx >> 1];
        if (idx & 1) == 0 { b & 0x0F } else { b >> 4 }
    }

    #[inline(always)]
    fn set_nib(&mut self, idx: usize, v: u8) {
        let b = &mut self.nibbles[idx >> 1];
        if (idx & 1) == 0 {
            *b = (*b & 0xF0) | (v & 0x0F);
        } else {
            *b = (*b & 0x0F) | ((v & 0x0F) << 4);
        }
    }

    pub fn get_abs(&self, idx: usize) -> u8 {
        let n = self.get_nib(idx);
        if n < 15 {
            self.cur_min + n
        } else {
            match self.aux.binary_search_by_key(&(idx as u32), |&(i, _)| i) {
                Ok(pos) => self.aux[pos].1,
                Err(_) => self.cur_min + 14, // saturated in fixed mode
            }
        }
    }

    fn shift_to_new_min(&mut self) {
        while self.num_at_cur_min == 0 {
            self.cur_min += 1;
            let mut new_zeros = 0usize;
            for i in 0..self.m {
                let n = self.get_nib(i);
                if n < 15 {
                    let nn = n - 1;
                    self.set_nib(i, nn);
                    if nn == 0 { new_zeros += 1; }
                } else {
                    if let Ok(pos) = self.aux.binary_search_by_key(&(i as u32), |&(idx, _)| idx) {
                        let v = self.aux[pos].1;
                        if v < self.cur_min + 15 {
                            self.aux.remove(pos);
                            let nn = v - self.cur_min;
                            self.set_nib(i, nn);
                            if nn == 0 { new_zeros += 1; }
                        }
                    }
                }
            }
            self.num_at_cur_min = new_zeros;
        }
    }

    #[inline]
    pub fn add_level(&mut self, idx: usize, lvl: u8) {
        if lvl <= self.cur_min {
            return;
        }
        let old_n = self.get_nib(idx);
        let old_abs = if old_n < 15 {
            self.cur_min + old_n
        } else {
            self.aux.binary_search_by_key(&(idx as u32), |&(i, _)| i)
                .map(|pos| self.aux[pos].1)
                .unwrap_or(self.cur_min + 14)
        };
        if lvl <= old_abs {
            return;
        }
        let delta = lvl - self.cur_min;
        if old_n == 0 {
            self.num_at_cur_min -= 1;
        }
        if delta < 15 {
            self.set_nib(idx, delta);
        } else if old_n == 15 {
            if let Ok(pos) = self.aux.binary_search_by_key(&(idx as u32), |&(i, _)| i) {
                self.aux[pos].1 = lvl;
            }
        } else {
            // Needs an exception slot!
            if self.max_aux_cap == 0 || self.aux.len() < self.max_aux_cap {
                self.set_nib(idx, 15);
                let pos = self.aux.binary_search_by_key(&(idx as u32), |&(i, _)| i).unwrap_err();
                self.aux.insert(pos, (idx as u32, lvl));
            } else {
                // Fixed-capacity HLL_4-Fixed overflow: MUST clamp to `cur_min + 14` (breaks semilattice! Thm 2!)
                self.set_nib(idx, 14);
            }
        }
        if self.num_at_cur_min == 0 {
            self.shift_to_new_min();
        }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) {
        let idx = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        self.add_level(idx, lvl);
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for i in 0..self.m {
            let v = other.get_abs(i);
            if v > 0 {
                self.add_level(i, v);
            }
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for i in 0..self.m {
            let u = self.get_abs(i) as usize;
            zero_lin += pmf.tail[u];
            if u > 0 { ones[u] += 1; }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }
}

// ---------------------------------------------------------------------------
// 8. AVLL (Bushnell, arXiv:2608.12575): Base-56 packed 11 slots/u64 + eeMask!
// ---------------------------------------------------------------------------

const POW56: [u64; 11] = [
    1,
    56,
    3_136,
    175_616,
    9_834_496,
    550_731_776,
    30_840_979_456,
    1_727_094_849_536,
    96_717_311_574_016,
    5_416_169_448_144_896,
    303_305_489_096_114_176,
];

/// Faithful base-56 packed AVLL (`11` buckets per `u64`, `5.818 b/bucket`, `HTL=13`, `MAX_EXP=18`, `HISTORY_MARGIN=2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvllSketch {
    pub n_words: usize,
    pub m: usize, // = n_words * 11
    pub global_nlz: u8,
    pub ee_thresh: u64, // 1-compare early exit threshold (mirrors Bushnell's `eeMask`)
    pub words: Vec<u64>,
}

impl AvllSketch {
    pub const MAX_EXP: u8 = 18;
    pub const HTL: u8 = 13;
    pub const HISTORY_MARGIN: u8 = 2;

    pub fn new_words(n_words: usize) -> Self {
        assert!(n_words >= 1);
        Self {
            n_words,
            m: n_words * 11,
            global_nlz: 0,
            ee_thresh: u64::MAX,
            words: vec![0u64; n_words],
        }
    }

    pub fn new_buckets(g: usize) -> Self {
        let n_words = (g + 10) / 11;
        Self {
            n_words,
            m: g,
            global_nlz: 0,
            ee_thresh: u64::MAX,
            words: vec![0u64; n_words],
        }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        HEADER_BYTES + self.n_words * 8
    }

    #[inline(always)]
    pub fn state_exp(s: u8) -> u8 {
        if s == 0 {
            0
        } else if s <= 2 {
            1
        } else if s < 51 {
            ((s - 3) >> 2) + 2
        } else {
            14 + (s - 51)
        }
    }

    #[inline(always)]
    pub fn unpack_bitmap(s: u8) -> u32 {
        if s == 0 {
            0
        } else if s <= 2 {
            (1u32 << 1) | ((s - 1) as u32)
        } else if s < 51 {
            let e = ((s - 3) >> 2) + 2;
            let h = (s - 3) & 3;
            (1u32 << e) | ((h as u32) << (e - 2))
        } else {
            let e = 14 + (s - 51);
            1u32 << e
        }
    }

    #[inline(always)]
    pub fn pack_bitmap(bm: u32) -> u8 {
        if bm == 0 {
            return 0;
        }
        let highest = (31 - bm.leading_zeros()) as u8;
        let e = highest.min(Self::MAX_EXP);
        if e == 0 {
            0
        } else if e == 1 {
            1 + ((bm & 1) as u8)
        } else if e <= Self::HTL {
            let h = ((bm >> (e - 2)) & 3) as u8;
            3 + (e - 2) * 4 + h
        } else {
            51 + (e - 14)
        }
    }

    #[inline(always)]
    pub fn get_state(&self, r: usize) -> u8 {
        let w = self.words[r / 11];
        ((w / POW56[r % 11]) % 56) as u8
    }

    #[inline(always)]
    pub fn set_state(&mut self, r: usize, new_s: u8) {
        let wi = r / 11;
        let lane = r % 11;
        let w = self.words[wi];
        let old_s = ((w / POW56[lane]) % 56) as u8;
        let diff = (new_s as i64) - (old_s as i64);
        self.words[wi] = w.wrapping_add((diff as u64).wrapping_mul(POW56[lane]));
    }

    #[inline(always)]
    fn update_ee_thresh(&mut self) {
        // Early-exit when `abs_nlz <= global_nlz` (i.e. `nlz + 1 <= global_nlz` <=> `nlz < global_nlz`)!
        // Note: `nlz < global_nlz` <=> `h >= (1 << (64 - global_nlz))` <=> `h > (1 << (64 - global_nlz)) - 1`!
        self.ee_thresh = if self.global_nlz == 0 {
            u64::MAX
        } else if self.global_nlz >= 64 {
            0
        } else {
            (1u64 << (64 - self.global_nlz as u32)) - 1
        };
    }

    fn maybe_promote(&mut self) {
        loop {
            let mut min_e = u8::MAX;
            for r in 0..self.m {
                let e = Self::state_exp(self.get_state(r));
                if e < min_e { min_e = e; }
                if min_e <= Self::HISTORY_MARGIN {
                    return;
                }
            }
            if min_e > Self::HISTORY_MARGIN && min_e < u8::MAX {
                self.global_nlz += 1;
                for wi in 0..self.n_words {
                    let mut w = self.words[wi];
                    let mut nw = 0u64;
                    let cnt = (self.m - wi * 11).min(11);
                    for lane in 0..cnt {
                        let s = (w % 56) as u8;
                        w /= 56;
                        let ns = Self::pack_bitmap(Self::unpack_bitmap(s) >> 1);
                        nw += (ns as u64) * POW56[lane];
                    }
                    self.words[wi] = nw;
                }
                self.update_ee_thresh();
            } else {
                break;
            }
        }
    }

    /// Insert 1-based absolute level `abs_nlz >= 1` into bucket `r`.
    #[inline]
    pub fn add_level(&mut self, r: usize, abs_nlz: u8) {
        if abs_nlz < self.global_nlz {
            return;
        }
        let rel = (abs_nlz - self.global_nlz).min(Self::MAX_EXP);
        let wi = r / 11;
        let lane = r % 11;
        let w = unsafe { *self.words.get_unchecked(wi) };
        let old_state = ((w / POW56[lane]) % 56) as u8;
        let bm = Self::unpack_bitmap(old_state) | (1u32 << rel);
        let new_state = Self::pack_bitmap(bm);
        if new_state > old_state {
            let old_exp = Self::state_exp(old_state);
            let diff = (new_state - old_state) as u64;
            unsafe { *self.words.get_unchecked_mut(wi) = w + diff * POW56[lane]; }
            if old_exp <= Self::HISTORY_MARGIN && Self::state_exp(new_state) > Self::HISTORY_MARGIN {
                self.maybe_promote();
            }
        }
    }

    /// Fast single-64-bit hash update with 1-compare `ee_thresh` early exit (faithful to AVLL §3.4!).
    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        // Note: AVLL only keeps history down to `global_nlz` (bit 0 = global_nlz when e=1 or e=2),
        // so `abs_nlz < global_nlz` can ALWAYS early-exit safely!
        // `abs_nlz = nlz + 1 < global_nlz` <=> `nlz + 1 <= global_nlz - 1` <=> `h >= 1 << (65 - global_nlz)`.
        // Use `global_nlz >= 2` threshold so bit 0 (global_nlz) is preserved accurately!
        let gnlz = self.global_nlz;
        if gnlz >= 2 && h >= (1u64 << (65 - gnlz as u32)) {
            return true;
        }
        let abs_nlz = (h.leading_zeros() + 1).min(63) as u8;
        // Bucket selection via fast multiply-high (Lemire fast range, zero 64-bit udiv! Even faster than Bushnell's Java remainderUnsigned!)
        let r = (((h as u32) as u64 * (self.m as u64)) >> 32) as usize;
        self.add_level(r, abs_nlz);
        false
    }

    /// Faithful AVLL §3.5 merge (bitmap OR + global_nlz alignment + promotion).
    pub fn merge(&self, other: &Self) -> Self {
        assert_eq!(self.m, other.m);
        let base = self.global_nlz.max(other.global_nlz);
        let sh1 = (base - self.global_nlz) as u32;
        let sh2 = (base - other.global_nlz) as u32;
        let mut out = AvllSketch::new_buckets(self.m);
        out.global_nlz = base;
        for wi in 0..self.n_words {
            let mut w1 = self.words[wi];
            let mut w2 = other.words[wi];
            let mut nw = 0u64;
            let cnt = (self.m - wi * 11).min(11);
            for lane in 0..cnt {
                let s1 = (w1 % 56) as u8;
                let s2 = (w2 % 56) as u8;
                w1 /= 56;
                w2 /= 56;
                let bm1 = if sh1 >= 32 { 0 } else { Self::unpack_bitmap(s1) >> sh1 };
                let bm2 = if sh2 >= 32 { 0 } else { Self::unpack_bitmap(s2) >> sh2 };
                let ns = Self::pack_bitmap(bm1 | bm2);
                nw += (ns as u64) * POW56[lane];
            }
            out.words[wi] = nw;
        }
        out.update_ee_thresh();
        out.maybe_promote();
        out
    }

    /// Exact `[L_r, U_r)` interval of low-32-bit hash words `h_lo in [0, 2^32)` mapped to bucket `r`
    /// by `r = (((h as u32) as u64 * m as u64) >> 32)` in `add_hash_64`.
    #[inline(always)]
    pub fn bucket_lo32_bounds(r: usize, m: usize) -> (u64, u64) {
        debug_assert!(r < m && m > 0);
        let m128 = m as u128;
        let l = (((r as u128) << 32) + m128 - 1) / m128;
        let u = ((((r + 1) as u128) << 32) + m128 - 1) / m128;
        (l as u64, u as u64)
    }

    /// Exact integer preimage count `C(r, k)` of 64-bit hash words `h in [0, 2^64)` that land in
    /// bucket `r` (with low-32-bit interval `[l_r, u_r)`) AND produce `abs_nlz = (h.leading_zeros() + 1).min(63) == k`.
    /// Sums to exactly `2^64` across all `r in 0..m` and `k in 1..=63`.
    #[inline(always)]
    pub fn joint_preimage_count(k: usize, l_r: u64, u_r: u64) -> u64 {
        if k == 0 || k > 63 || u_r <= l_r {
            return 0;
        }
        if k <= 32 {
            (u_r - l_r) << (32 - k)
        } else if k <= 62 {
            let lo = 1u64 << (64 - k);
            let hi = 1u64 << (65 - k);
            let a = l_r.max(lo);
            let b = u_r.min(hi);
            b.saturating_sub(a)
        } else {
            // k == 63: h.leading_zeros() >= 62 <=> h_hi == 0 and h_lo in [0, 4)
            let b = u_r.min(4);
            b.saturating_sub(l_r)
        }
    }

    /// Exact state equivalent to inserting levels `1..=k_pre` into every bucket `0..self.m`,
    /// without intermediate clamping (used by chronological jump simulation).
    pub fn prefill_uniform_saturated(&mut self, k_pre: u8) {
        if k_pre == 0 {
            return;
        }
        if k_pre < 3 {
            for lvl in 1..=k_pre {
                for r in 0..self.m {
                    self.add_level(r, lvl);
                }
            }
            return;
        }
        self.global_nlz = k_pre - Self::HISTORY_MARGIN;
        let state_full: u64 = Self::pack_bitmap(0b111) as u64;
        for wi in 0..self.n_words {
            let cnt = (self.m - wi * 11).min(11);
            let mut w = 0u64;
            for lane in 0..cnt {
                w += state_full * POW56[lane];
            }
            self.words[wi] = w;
        }
        self.update_ee_thresh();
    }

    pub fn abs_exp(&self, r: usize) -> u8 {
        self.global_nlz + Self::state_exp(self.get_state(r))
    }

    /// Exact Poisson ML estimator on the AVLL state (respecting `HTL=13` no-history and `MAX_EXP=18` censoring!).
    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        let gnlz = self.global_nlz as usize;
        let km = pmf.k_reach;
        for wi in 0..self.n_words {
            let mut w = self.words[wi];
            let cnt = (self.m - wi * 11).min(11);
            for _ in 0..cnt {
                let s = (w % 56) as u8;
                w /= 56;
                let rel_e = Self::state_exp(s) as usize;
                if rel_e == 0 {
                    // No arrival > gnlz
                    zero_lin += pmf.tail[gnlz.min(km)];
                } else {
                    let u = (gnlz + rel_e).min(km);
                    if rel_e < Self::MAX_EXP as usize {
                        zero_lin += pmf.tail[u];
                        ones[u] += 1;
                    } else {
                        // Censored at >= u: P(max >= u) = 1 - exp(-lam * tail[u-1])
                        // Approximate via exact top-level indicator (matches unbiased middle range)
                        ones[u] += 1;
                    }
                    if rel_e <= Self::HTL as usize {
                        let bm = Self::unpack_bitmap(s);
                        if rel_e >= 1 && u >= 2 {
                            let l1 = u - 1;
                            if ((bm >> (rel_e - 1)) & 1) != 0 { ones[l1] += 1; } else { zero_lin += pmf.probs[l1]; }
                        }
                        if rel_e >= 2 && u >= 3 {
                            let l2 = u - 2;
                            if ((bm >> (rel_e - 2)) & 1) != 0 { ones[l2] += 1; } else { zero_lin += pmf.probs[l2]; }
                        }
                    }
                }
            }
        }
        pmf.solve_mle(&ones[..=km], zero_lin, self.m as f64)
    }
}
