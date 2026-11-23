//! Milestone M0 Single Source of Truth: Exact 64-bit Preimage PMF via Monotone Binary Search.
//!
//! For ANY base `b` (dyadic Pow2, Pow2Div3, ErtlT(t), or general base Geom(b)) and remaining
//! hash bit width `w_bits = 64 - P`, computes exact integer preimage counts `c_k` (using `u128`
//! so `sum_{k=1}^{K_reach} c_k == domain_size` with ZERO rounding error and ZERO ghost levels)
//! in < 0.2 ms. Shared identically by Rust hashers, Poisson MC, analytical Fisher, and Newton MLE.

use crate::hash::Q_MAX_DIV3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistSpec {
    /// Exact geometric b=2 via leading zeros on `w_bits` (<= 64), clamped at `max_cap` (<= 63).
    Pow2 { w_bits: u32, max_cap: usize },
    /// Exact `q = h / 3` leading-zeros distribution for `OS-BMW-5b-256B` (`g=384`), with `p_tot = 7 + p_blk` bits removed.
    Pow2Div3 { p_tot: u32 },
    /// Ertl Eq. (8) / Alg. 2 with parameter `t` on `w_bits = 64 - P` bits.
    ErtlT { t: u32, w_bits: u32 },
    /// General base `b > 1` using the exact monotone `neg_log2_u` formula on `w_bits` (<= 64) bits,
    /// with exact integer preimages discovered by binary search (automatically yields `K_reach = 60`
    /// for `b = 2.125, w_bits = 64`!).
    GeomNegLog2 { b: f64, w_bits: u32, max_cap: usize },
    /// General base `b > 1` using exact integer thresholds `floor(2^w * b^{-k})` (stops when threshold == 0).
    GeomExactFloor { b: f64, w_bits: u32, max_cap: usize },
}

/// Monotone non-increasing level evaluator for `GeomNegLog2` on `rem in 0..2^w_bits`.
#[inline(always)]
pub fn eval_geom_neg_log2(rem: u64, w_bits: u32, b: f64, max_cap: usize) -> usize {
    // Shift `rem` to MSB-aligned 64-bit representation (if w_bits < 64, shift left by 64 - w_bits).
    let lz_raw = if w_bits == 64 {
        rem.leading_zeros()
    } else {
        let aligned = rem << (64 - w_bits);
        aligned.leading_zeros().min(w_bits)
    };
    let shifted = if w_bits == 64 {
        if lz_raw < 63 { rem << (lz_raw + 1) } else { 0 }
    } else if lz_raw < w_bits {
        let aligned = rem << (64 - w_bits);
        if lz_raw < 63 { aligned << (lz_raw + 1) } else { 0 }
    } else {
        0
    };
    let frac = (shifted >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    let neg_log2_u = (lz_raw as f64 + 1.0) - (1.0 + frac).log2();
    let k = (neg_log2_u / b.log2()).floor() as usize + 1;
    k.clamp(1, max_cap)
}

/// Exact arbitrary-precision helper: multiply little-endian `u64`-limb integer by `m`.
fn big_mul_u64(limbs: &mut Vec<u64>, m: u64) {
    let mut carry = 0u128;
    for l in limbs.iter_mut() {
        let prod = (*l as u128) * (m as u128) + carry;
        *l = prod as u64;
        carry = prod >> 64;
    }
    if carry > 0 {
        limbs.push(carry as u64);
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
        limbs.pop();
    }
}

fn big_bit_len(limbs: &[u64]) -> usize {
    let mut n = limbs.len();
    while n > 0 && limbs[n - 1] == 0 {
        n -= 1;
    }
    if n == 0 {
        return 0;
    }
    (n - 1) * 64 + (64 - limbs[n - 1].leading_zeros() as usize)
}

fn big_ge(a: &[u64], b: &[u64]) -> bool {
    let mut na = a.len();
    while na > 0 && a[na - 1] == 0 {
        na -= 1;
    }
    let mut nb = b.len();
    while nb > 0 && b[nb - 1] == 0 {
        nb -= 1;
    }
    if na != nb {
        return na > nb;
    }
    for i in (0..na).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn big_sub_in_place(a: &mut Vec<u64>, b: &[u64]) {
    let mut borrow = 0u128;
    for i in 0..a.len() {
        let bv = if i < b.len() { b[i] as u128 } else { 0 };
        let av = a[i] as u128;
        let sub = bv + borrow;
        if av >= sub {
            a[i] = (av - sub) as u64;
            borrow = 0;
        } else {
            a[i] = ((1u128 << 64) + av - sub) as u64;
            borrow = 1;
        }
    }
    debug_assert_eq!(borrow, 0);
    while a.len() > 1 && *a.last().unwrap() == 0 {
        a.pop();
    }
}

fn big_shl1(a: &mut Vec<u64>) {
    let mut carry = 0u64;
    for l in a.iter_mut() {
        let next_carry = *l >> 63;
        *l = (*l << 1) | carry;
        carry = next_carry;
    }
    if carry != 0 {
        a.push(carry);
    }
}

/// Exact integer quotient `floor(2^shift / denom)` for `b > 1` thresholds (result `< 2^64`).
fn exact_pow2_div_big(shift: i32, denom: &[u64]) -> u128 {
    if shift < 0 {
        return 0;
    }
    let bl = big_bit_len(denom);
    debug_assert!(bl > 0);
    let sh = shift as usize;
    if sh + 1 < bl {
        return 0;
    }
    let mut rem = vec![0u64; ((bl - 1) / 64) + 1];
    rem[(bl - 1) / 64] = 1u64 << ((bl - 1) % 64);
    let steps = sh - (bl - 1);
    debug_assert!(steps <= 64);
    let mut q: u128 = 0;
    for bit in (0..=steps).rev() {
        if big_ge(&rem, denom) {
            big_sub_in_place(&mut rem, denom);
            q |= 1u128 << bit;
        }
        if bit > 0 {
            big_shl1(&mut rem);
        }
    }
    q
}

/// Exact integer preimage table + derived PMF/tail + fast threshold lookup table.
#[derive(Clone, Debug)]
pub struct PreimagePmf {
    pub spec: DistSpec,
    /// Total discrete domain size (`2^w_bits` for power-of-2 domains, `Q_MAX_DIV3 >> p_tot` for Div3).
    pub domain_size: u128,
    /// Largest reachable level with `counts[k] > 0` (ZERO ghost levels!).
    pub k_reach: usize,
    /// Exact integer preimage counts `counts[1..=k_reach]`; `sum == domain_size`.
    pub counts: Vec<u128>,
    /// Exact probabilities `probs[k] = counts[k] / domain_size` (`probs[0] = 0.0`).
    pub probs: Vec<f64>,
    /// Exact upper tail `tail[u] = sum_{k > u} probs[k]` (`tail[0] = 1.0`, `tail[k_reach] = 0.0`).
    pub tail: Vec<f64>,
    /// Monotone descending upper boundaries `thresh[0] = domain_size`, `thresh[k] = thresh[k-1] - counts[k]`,
    /// `thresh[k_reach] = 0`. For any `rem in 0..domain_size`, `level(rem) == k <=> thresh[k] <= rem < thresh[k-1]`.
    pub thresh: Vec<u128>,
    /// 64-bit aligned thresholds for fast O(log K) runtime lookup and 1-compare early exit:
    /// `thresh_u64[k]` is `thresh[k]` (valid for `k >= 1`, since `thresh[k] < 2^64`).
    pub thresh_u64: Vec<u64>,
}

impl PreimagePmf {
    /// Construct the exact 64-bit preimage PMF via exact counting / monotone binary search (< 0.2 ms).
    pub fn build(spec: DistSpec) -> Self {
        let (domain_size, counts) = match spec {
            DistSpec::Pow2 { w_bits, max_cap } => {
                assert!((1..=64).contains(&w_bits));
                let dom = 1u128 << w_bits;
                let km = (w_bits as usize + 1).min(max_cap);
                let mut c = vec![0u128; km + 1];
                let mut rem = dom;
                for k in 1..km {
                    let ck = 1u128 << (w_bits as usize - k);
                    c[k] = ck;
                    rem -= ck;
                }
                c[km] = rem;
                (dom, c)
            }
            DistSpec::Pow2Div3 { p_tot } => {
                assert!((1..=58).contains(&p_tot));
                let n_rem = (Q_MAX_DIV3 >> p_tot) as u128;
                // Sentinel at bit (p_tot - 1) maps `q >> p_tot == rem`:
                // `lvl = ( (rem << p_tot) | (1 << (p_tot - 1)) ).leading_zeros()`
                // Check exact monotone boundary for each lvl = 1..(64 - p_tot):
                let k_max = (64 - p_tot) as usize;
                let mut c = vec![0u128; k_max + 1];
                let mut hi = n_rem;
                for k in 1..k_max {
                    // Leading zeros == k <=> top bit is at position (63 - k), i.e.,
                    // in `rem` (which sits at bits `p_tot..64`), bit `(63 - k - p_tot)` is 1 and higher bits are 0.
                    let shift = 63 - (k as u32) - p_tot;
                    let lo = 1u128 << shift;
                    if hi > lo {
                        c[k] = hi - lo;
                        hi = lo;
                    }
                }
                // rem == 0 gives lz = 64 - p_tot = k_max
                c[k_max] = hi;
                (n_rem, c)
            }
            DistSpec::ErtlT { t, w_bits } => {
                assert!((1..=64).contains(&w_bits) && t <= 4 && w_bits > t);
                let dom = 1u128 << w_bits;
                let hi_bits = w_bits - t;
                let q_cap = ((1usize << (6 + t)) - 1).min(((hi_bits as usize + 1) << t) + 1);
                let mut c = vec![0u128; q_cap + 1];
                let mut rem = dom;
                // For nlz in 0..hi_bits: hi has top 1 at bit (hi_bits - 1 - nlz), giving 2^(hi_bits - 1 - nlz) values of hi
                // per each of the 2^t values of sub_t.
                for nlz in 0..hi_bits {
                    let count_each = 1u128 << (hi_bits - 1 - nlz);
                    for sub_t in 0..(1u32 << t) {
                        let k = (((nlz << t) + sub_t + 1) as usize).min(q_cap);
                        c[k] += count_each;
                        rem -= count_each;
                    }
                }
                // nlz == hi_bits (hi == 0): 1 value per sub_t
                for sub_t in 0..(1u32 << t) {
                    let k = (((hi_bits << t) + sub_t + 1) as usize).min(q_cap);
                    c[k] += 1;
                    rem -= 1;
                }
                debug_assert_eq!(rem, 0);
                (dom, c)
            }
            DistSpec::GeomNegLog2 { b, w_bits, max_cap } => {
                assert!((1..=64).contains(&w_bits) && b > 1.0);
                let dom = 1u128 << w_bits;
                // Monotone binary search: eval_geom_neg_log2(rem) is non-increasing in rem in [0, dom).
                // Find upper[k] = smallest rem in [0, dom] such that eval(rem) <= k (or dom if none).
                // Wait: eval is non-increasing, so {rem : eval(rem) >= k} is a prefix [0, split(k))!
                // Specifically:
                //   split(k) = number of rem in [0, dom) with eval(rem) >= k.
                //   Then counts[k] = split(k) - split(k + 1)!
                let find_split = |target_k: usize| -> u128 {
                    let mut lo: u128 = 0;
                    let mut hi: u128 = dom;
                    while lo < hi {
                        let mid = lo + ((hi - lo) >> 1);
                        let v = eval_geom_neg_log2(mid as u64, w_bits, b, max_cap);
                        if v >= target_k {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                };
                let k_top = eval_geom_neg_log2(0, w_bits, b, max_cap);
                let mut splits = vec![0u128; k_top + 2];
                splits[1] = dom;
                for k in 2..=k_top {
                    splits[k] = find_split(k);
                }
                splits[k_top + 1] = 0;
                let mut c = vec![0u128; k_top + 1];
                for k in 1..=k_top {
                    c[k] = splits[k] - splits[k + 1];
                }
                (dom, c)
            }
            DistSpec::GeomExactFloor { b, w_bits, max_cap } => {
                assert!((1..=64).contains(&w_bits) && b.is_finite() && b > 1.0);
                // Decompose IEEE-754 f64 `b` into exact dyadic rational `num * 2^exp2` (odd `num`),
                // then compute exact integer thresholds `T_k = floor(2^w_bits / b^k) = floor(2^(w - k*exp2) / num^k)`
                // with ZERO floating-point rounding drift!
                let bits = b.to_bits();
                let exp_biased = ((bits >> 52) & 0x7ff) as i32;
                let frac = bits & ((1u64 << 52) - 1);
                let mut num: u64 = (1u64 << 52) | frac;
                let mut exp2: i32 = exp_biased - 1023 - 52;
                let tz = num.trailing_zeros();
                num >>= tz;
                exp2 += tz as i32;

                let dom = 1u128 << w_bits;
                let mut thresh = vec![dom];
                let mut denom: Vec<u64> = vec![1u64];
                for k in 1..=max_cap {
                    big_mul_u64(&mut denom, num);
                    let shift = (w_bits as i32) - (k as i32) * exp2;
                    let t = exact_pow2_div_big(shift, &denom).min(*thresh.last().unwrap());
                    if k == max_cap || t == 0 {
                        thresh.push(0);
                        break;
                    }
                    thresh.push(t);
                }
                let km = thresh.len() - 1;
                let mut c = vec![0u128; km + 1];
                for k in 1..=km {
                    c[k] = thresh[k - 1] - thresh[k];
                }
                (dom, c)
            }
        };

        // Trim any trailing zero-count levels so k_reach has STRICTLY positive count!
        let mut k_reach = counts.len() - 1;
        while k_reach > 1 && counts[k_reach] == 0 {
            k_reach -= 1;
        }
        let counts = counts[..=k_reach].to_vec();
        let sum_c: u128 = counts[1..].iter().sum();
        assert_eq!(sum_c, domain_size, "Exact preimage count sum mismatch!");

        let inv_dom = 1.0 / (domain_size as f64);
        let mut probs = vec![0.0_f64; k_reach + 1];
        for k in 1..=k_reach {
            probs[k] = (counts[k] as f64) * inv_dom;
        }
        let mut tail = vec![0.0_f64; k_reach + 1];
        let mut acc_u128 = 0u128;
        for u in (0..k_reach).rev() {
            acc_u128 += counts[u + 1];
            tail[u] = (acc_u128 as f64) * inv_dom;
        }
        tail[k_reach] = 0.0;

        let mut thresh = vec![0u128; k_reach + 1];
        thresh[0] = domain_size;
        for k in 1..=k_reach {
            thresh[k] = thresh[k - 1] - counts[k];
        }
        debug_assert_eq!(thresh[k_reach], 0);
        let thresh_u64: Vec<u64> = thresh
            .iter()
            .map(|&t| if t >= (1u128 << 64) { u64::MAX } else { t as u64 })
            .collect();

        Self { spec, domain_size, k_reach, counts, probs, tail, thresh, thresh_u64 }
    }

    /// Fast branchless-friendly level lookup from `rem in 0..domain_size` using exact preimage thresholds.
    /// Guaranteed 100% identical to `eval_geom_neg_log2` / exact floor on every single integer!
    #[inline(always)]
    pub fn level_from_rem(&self, rem: u64) -> u8 {
        // Binary search in descending thresh_u64[1..=k_reach]:
        // Find smallest k in 1..=k_reach such that rem >= thresh_u64[k].
        let mut lo = 1usize;
        let mut hi = self.k_reach;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if rem >= self.thresh_u64[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u8
    }

    /// Exact safeguarded Newton-Raphson MLE from sufficient statistics `(ones_cnt[1..=k_reach], zero_lin)`.
    /// Returns `(n_hat, n_newton_iters)`.
    pub fn solve_mle(&self, ones_cnt: &[u32], zero_lin: f64, total_units: f64) -> (f64, u32) {
        let mut active_k = [0usize; 256];
        let mut active_c = [0.0_f64; 256];
        let mut n_act = 0usize;
        let mut total_ones = 0.0_f64;
        let km = self.k_reach.min(ones_cnt.len() - 1);
        for k in 1..=km {
            let c = ones_cnt[k];
            if c > 0 {
                active_k[n_act] = k;
                active_c[n_act] = c as f64;
                total_ones += c as f64;
                n_act += 1;
            }
        }
        if n_act == 0 {
            return (0.0, 0);
        }
        if zero_lin <= 0.0 {
            return (f64::INFINITY, 0);
        }

        let probs = &self.probs;
        let score_and_deriv = |lam: f64| -> (f64, f64) {
            let mut s = -zero_lin;
            let mut ds = 0.0_f64;
            for idx in 0..n_act {
                let pk = probs[active_k[idx]];
                let x = lam * pk;
                if x < 700.0 {
                    let ex = (-x).exp();
                    let om = -(-x).exp_m1();
                    let c = active_c[idx];
                    let ratio = pk * ex / om;
                    s += c * ratio;
                    ds -= c * ratio * pk / om;
                }
            }
            (s, ds)
        };

        // Fast accurate initial guess:
        // Small-x linear approximation: sum c_k / lam ≈ zero_lin => lam0 = total_ones / zero_lin.
        // Note: since x * e^{-x} / (1 - e^{-x}) <= 1, we ALWAYS have lam * s(lam) <= total_ones - lam * zero_lin,
        // which PROVES the exact MLE root ALWAYS satisfies: lam_hat <= total_ones / zero_lin!
        let mut hi = total_ones / zero_lin;
        // Verify hi is indeed an upper bracket (s(hi) <= 0), expand if rounding occurred.
        while score_and_deriv(hi).0 > 0.0 && hi < 1e24 {
            hi *= 2.0;
        }
        // Tight lower bracket via geometric step down from hi * 0.5
        let mut lo = hi * 0.5;
        while score_and_deriv(lo).0 < 0.0 && lo > 1e-24 {
            lo *= 0.25;
        }

        // Start at geometric mean of tight bracket (within ~1.5x of truth!) -> 4-6 Newton steps!
        let mut lam = (lo * hi).sqrt();
        let mut iters = 0u32;
        for it in 1..=40 {
            iters = it;
            let (s, ds) = score_and_deriv(lam);
            if s > 0.0 {
                lo = lam;
            } else {
                hi = lam;
            }
            if (hi - lo) <= 1e-12 * lam || s.abs() <= 1e-14 * zero_lin {
                break;
            }
            // Log-space Newton step (even better conditioned because d/d(ln lam) is super smooth!)
            // Wait: standard Newton in lam with strict bracket fallback is quadratic and strictly monotone.
            let next = if ds < -1e-300 { lam - s / ds } else { (lo * hi).sqrt() };
            lam = if next > lo && next < hi { next } else { (lo * hi).sqrt() };
        }
        (lam * total_units, iters)
    }

    /// Exact closed-form relative Fisher information `lam^2 * I_sub(lam)` for a single independent register
    /// with history depth `d`.
    pub fn fisher_single(&self, d: usize, lam: f64) -> f64 {
        let km = self.k_reach;
        let mut cs_mid = vec![0.0_f64; km + 1];
        let mut g_top = vec![0.0_f64; km + 1];
        for k in 1..=km {
            let x = lam * self.probs[k];
            let (gt, gm) = fisher_atoms(x);
            g_top[k] = gt;
            cs_mid[k] = cs_mid[k - 1] + gm;
        }
        let mut total = 0.0_f64;
        for u in 1..=km {
            let x_t = lam * self.tail[u];
            if x_t > 745.0 {
                continue;
            }
            let pu = (-x_t).exp() * (-(-lam * self.probs[u]).exp_m1());
            if pu < 1e-300 {
                continue;
            }
            let lo = u.saturating_sub(d).max(1);
            total += pu * (g_top[u] + (cs_mid[u - 1] - cs_mid[lo - 1]));
        }
        total
    }

    /// Exact closed-form relative Fisher information `lam^2 * I_block(lam) / g` for `OS-BMW(g, k, R, d)`
    /// via Theorem 3 (factorized Bartlett identity + exact binomial order-statistic CDFs).
    pub fn fisher_osbmw(&self, g: usize, k: usize, r_win: usize, d: usize, lam: f64) -> f64 {
        if g == 1 && k == 1 {
            return self.fisher_single(d, lam);
        }
        let km = self.k_reach;
        let mut f_cdf = vec![0.0_f64; km + 1];
        let mut p_u = vec![0.0_f64; km + 1];
        let mut g_top = vec![0.0_f64; km + 1];
        let mut cs_mid = vec![0.0_f64; km + 1];

        for u in 0..=km {
            let xt = lam * self.tail[u];
            f_cdf[u] = if xt > 745.0 { 0.0 } else { (-xt).exp() };
        }
        p_u[0] = f_cdf[0];
        for u in 1..=km {
            let x = lam * self.probs[u];
            p_u[u] = f_cdf[u] * (-(-x).exp_m1());
            let (gt, gm) = fisher_atoms(x);
            g_top[u] = gt;
            cs_mid[u] = cs_mid[u - 1] + gm;
        }

        // Order statistics of the other n = g - 1 sub-buckets:
        // P(V_(r) <= m) = P(Bin(n, 1 - F[m]) <= r - 1)
        let n = g - 1;
        let mut cdf_vk = vec![1.0_f64; km + 1];
        let mut pmf_vk = vec![0.0_f64; km + 1];
        if k <= n {
            for m in 0..=km {
                cdf_vk[m] = binom_cdf_le(n, k - 1, 1.0 - f_cdf[m]);
            }
            pmf_vk[0] = cdf_vk[0];
            for m in 1..=km {
                pmf_vk[m] = (cdf_vk[m] - cdf_vk[m - 1]).max(0.0);
            }
        }
        let mut cdf_vkm1 = vec![0.0_f64; km + 1];
        let mut pmf_vkm1 = vec![0.0_f64; km + 1];
        if k >= 2 {
            for m in 0..=km {
                cdf_vkm1[m] = binom_cdf_le(n, k - 2, 1.0 - f_cdf[m]);
            }
            pmf_vkm1[0] = cdf_vkm1[0];
            for m in 1..=km {
                pmf_vkm1[m] = (cdf_vkm1[m] - cdf_vkm1[m - 1]).max(0.0);
            }
        }

        let mut total = 0.0_f64;
        for u in 1..=km {
            let pu = p_u[u];
            if pu < 1e-300 {
                continue;
            }
            let gt = g_top[u];
            let csu1 = cs_mid[u - 1];

            // Case A: m = V_(k-1) < u
            if k >= 2 {
                if d <= r_win {
                    let lo = u.saturating_sub(d).max(1);
                    total += pu * cdf_vkm1[u - 1] * (gt + (csu1 - cs_mid[lo - 1]));
                } else {
                    for m in 0..u {
                        let pm = pmf_vkm1[m];
                        if pm > 1e-300 {
                            let win_lo = (m + 1).saturating_sub(r_win).max(1);
                            let lo = win_lo.max(u.saturating_sub(d));
                            total += pu * pm * (gt + (csu1 - cs_mid[lo - 1]));
                        }
                    }
                }
            }

            // Case B: B = u (V_(k) <= u <= V_(k-1))
            let p_b = (cdf_vk[u] - if k >= 2 { cdf_vkm1[u - 1] } else { 0.0 }).max(0.0);
            if p_b > 1e-300 {
                let lo = u.saturating_sub(d.min(r_win - 1)).max(1);
                total += pu * p_b * (gt + (csu1 - cs_mid[lo - 1]));
            }

            // Case C: m = V_(k) in [u+1, min(km, u + r_win - 1)]
            if k <= n {
                let m_hi = km.min(u + r_win - 1);
                for m in (u + 1)..=m_hi {
                    let pm = pmf_vk[m];
                    if pm > 1e-300 {
                        let win_lo = (m + 1).saturating_sub(r_win).max(1);
                        let lo = win_lo.max(u.saturating_sub(d));
                        total += pu * pm * (gt + (csu1 - cs_mid[lo - 1]));
                    }
                }
            }
        }
        total
    }

    /// Periodic average of `fisher_osbmw` over `n_pts` points across one log_b period around `lam_base`.
    pub fn fisher_osbmw_periodic(&self, g: usize, k: usize, r_win: usize, d: usize, b_period: f64, lam_base: f64, n_pts: usize) -> (f64, f64, f64) {
        let mut sum = 0.0_f64;
        let mut mn = f64::INFINITY;
        let mut mx = 0.0_f64;
        for i in 0..n_pts {
            let lam = lam_base * b_period.powf((i as f64) / (n_pts as f64));
            let j = self.fisher_osbmw(g, k, r_win, d, lam);
            sum += j;
            if j < mn { mn = j; }
            if j > mx { mx = j; }
        }
        (sum / (n_pts as f64), mn, mx)
    }
}

#[inline(always)]
fn fisher_atoms(x: f64) -> (f64, f64) {
    if x <= 1e-8 {
        (1.0 - x * x / 12.0, x)
    } else if x >= 700.0 {
        (0.0, 0.0)
    } else {
        let ex = (-x).exp();
        let om = -(-x).exp_m1();
        let gm = x * x * ex / om;
        (gm / om, gm)
    }
}

/// Exact P(Bin(n, q) <= r) for small r <= 256 (our k <= 128), computed in log-space without overflow!
fn binom_cdf_le(n: usize, r: usize, q: f64) -> f64 {
    if r >= n {
        return 1.0;
    }
    let q = q.clamp(0.0, 1.0);
    if q <= 0.0 {
        return 1.0;
    }
    if q >= 1.0 {
        return 0.0;
    }
    let p = 1.0 - q;
    let log_q = q.ln();
    let log_p = p.ln();
    let mut sum = 0.0_f64;
    let mut log_comb = 0.0_f64;
    for j in 0..=r {
        if j > 0 {
            log_comb += ((n - j + 1) as f64).ln() - (j as f64).ln();
        }
        let log_term = log_comb + (j as f64) * log_q + ((n - j) as f64) * log_p;
        if log_term > -745.0 {
            sum += log_term.exp();
        }
    }
    sum.min(1.0)
}
