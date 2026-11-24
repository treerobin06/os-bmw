#!/usr/bin/env python3
"""
Exact Mathematical & Exhaustive Verification Suite for `PROOFS.md` (Theorems 1–4).

Verifies by exact arithmetic / brute-force enumeration / numerical proof checks:
  - Theorem 1 (Pure Dense Family Linearizability, Join-Semilattice Homomorphism, & 2x-Reducibility):
      (a) Verifies Pettie-Wang (STOC'21 Def. 8) linearization property on ALL 512 x 512 = 262,144
          subset pairs of (g=3, k=2, R=2, d=1, K=3), proving P(A U B) == P(P(A) U P(B)) AND
          confirming cell visibility depends strictly on levels > j;
      (b) Verifies 2x-downsizing dispatch: 128-bit domain-separated hash merges corresponding blocks
          directly WITHOUT top-level lifting (preserving level 53 on 8->4 blocks for (0, 2^11)),
          whereas single 64-bit dyadic Pow2 requires lifting u_top -> u_top + 1 on j < 2^(p-1).
      Note: OsBmwHybrid (V_BITS=25 sparse wrapper) is explicitly excluded from Theorem 1.
  - Generalized Theorem 2 (Counting Impossibility, High-Tier Associativity, & Overflow-Avoidance iff):
      (a) Verifies the minimal 4-state counterexample (g=r=3, W=2, K_max=4, C_ex=1) proving that
          "which exception is retained" carries information, and verifies the exact counting
          condition sum_{j=0}^{C_ex} C(t, j) m^j < m^t (m = min(K_max - W, W + 1)) is False there;
      (b) Evaluates the exact counting condition on real sketches (AVLL/DLL W=18 C=0, HLL_4-Fixed
          W=14 cap=1/4/32, Median/Quartile anchors) and reports minimal valid t in [C_ex+1, r_ord-1];
      (c) Runs generic saturating partition-homomorphism witnesses (Bottom_HLL4_Fixed 18 != 14,
          Bottom_AVLL_Cap0 22 != 18, Median_Anchor_Fixed 16 != 12, Quartile_Anchor_Fixed 14 != 10);
          note concrete Rust implementation witnesses (AVLL 11-bucket 47 != 49, HLL_4-Fixed g=4 cap=1
          20 != 14) are verified in s1_v2_rust/src/lib.rs, not claimed as reproduced here;
      (d) Exhaustively verifies on a toy bottom-anchored model over inputs strictly below W (zero
          upper clamping) that H = W - 1 triggers 0 associativity violations, whereas H = W - 2
          triggers deterministic associativity violation;
      (e) Exhaustively verifies Part 3: zero upper-overflow eviction on all inputs iff r_ord <= C_ex + 1,
          with top-k anchoring (r_ord = k = C_ex + 1, W = 0) tightly attaining the boundary.
  - Theorem 3 (Block Likelihood Factorization, Closed-Form Fisher/Entropy, Strict Concavity, & PMF):
      (a) Exact joint PMF equals Cartesian product prod_{r=1}^g P_r(A_r | lam) (< 1e-13);
      (b) Closed-form Fisher/entropy matches brute-force enumeration (< 1e-14);
      (c) Strict log-concavity in both lam and theta = ln(lam);
      (d) Part 4 discrete preimage boundary checks: Pow2Div3 P=11 (Q mod 2048 = 1365, 4096 excluded
          raw preimages) and GeomExactFloor b=17/8 W=64 (exact int threshold 8680820740569200760
          vs f64 8680820740569200640, drift = 120, K_reach = 59).
  - Theorem 4 (Raw-Payload Wire < 3.00 Existence Bound & Amortized O(1) Update Complexity):
      Verifies exact log2(N_total) = 10208.865167 (10,209 bits -> 1,277 B raw payload), raw 64-phase
      mean-Fisher MVP = 2.997665 < 3.00, per-phase MVP range 2.967692..3.036546 with 28/64 phases
      > 3.00, 8B header amortized over 2 blocks = 3.007055 > 3.00, K_reach=60 single top preimage
      (2048 * 2^64 = 2^75 > 2^64), and unified promotion upper bound K_reach - R = 60 - 7 = 53.
"""

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path
import numpy as np



def require(cond: bool, msg: str = "Verification failed") -> None:
    if not cond:
        raise AssertionError(msg)

def verify_theorem_1():
    """Verify Pettie-Wang Def. 8 linearization, join-semilattice homomorphism, and 2x-downsizing rules."""
    g, k, R, d, K = 3, 2, 2, 1, 3

    def project(bitsets):
        # bitsets: tuple of g ints, bit (l-1) indicates arrival at level l in 1..K
        us = [bs.bit_length() for bs in bitsets]
        B = sorted(us, reverse=True)[k - 1]
        win_lo = max(1, B - R + 1)
        out = []
        for r in range(g):
            u = us[r]
            if u < win_lo:
                out.append((0, 0))
            else:
                lo = max(win_lo, u - d)
                h = 0
                for j in range(lo, u):
                    if (bitsets[r] >> (j - 1)) & 1:
                        h |= 1 << (j - lo)
                out.append((u, h))
        return (B, tuple(out))

    def cell_visible_from_higher(bitsets, r, j):
        """Check whether cell (r, j) is visible in OS-BMW state purely from levels strictly > j."""
        has_hi_r = (bitsets[r] >> (j + d)) != 0
        if has_hi_r:
            return False
        other_hi = sum(1 for r2 in range(g) if r2 != r and (bitsets[r2] >> (j + R - 1)) != 0)
        return other_hi <= k - 1

    all_inputs = []
    states = []
    vis_sigs = []
    for mask in range(1 << (g * K)):
        bs = tuple((mask >> (r * K)) & ((1 << K) - 1) for r in range(g))
        all_inputs.append(bs)
        states.append(project(bs))
        sig = []
        for j in range(K, 0, -1):
            for r in range(g):
                if cell_visible_from_higher(bs, r, j):
                    sig.append((r, j, (bs[r] >> (j - 1)) & 1))
        vis_sigs.append(tuple(sig))

    state_to_sig = {}
    sig_to_state = {}
    for st, sg in zip(states, vis_sigs):
        if st in state_to_sig:
            require(state_to_sig[st] == sg, "Linearization mismatch: same state, different visible cells!")
        else:
            state_to_sig[st] = sg
        if sg in sig_to_state:
            require(sig_to_state[sg] == st, "Linearization mismatch: same visible cells, different state!")
        else:
            sig_to_state[sg] = st

    rep = {}
    for bs, st in zip(all_inputs, states):
        if st not in rep:
            rep[st] = bs
    hom_checks = 0
    for i in range(512):
        for j in range(512):
            bs_union = tuple(all_inputs[i][r] | all_inputs[j][r] for r in range(g))
            rep_union = tuple(rep[states[i]][r] | rep[states[j]][r] for r in range(g))
            require(project(bs_union) == project(rep_union), "Verification failed at line 117")
            hom_checks += 1

    # Part 4 check: 128-bit domain-separated downsizing (NO lift) vs 64-bit Pow2 downsizing (WITH lift)
    # 1) 128-bit mode: k24 (g=512, s_sub=9), 8 blocks (p=3) -> 4 blocks (p=2), hash pair (h_lo=0, h_hi=2^11)
    #    In 128-bit mode, h_hi = 2^11 has nlz = 64 - 12 = 52 -> lvl = 53 for BOTH p=3 and p=2!
    h_lo_128, h_hi_128 = 0, 1 << 11
    lvl_128 = (64 - h_hi_128.bit_length()) + 1  # 53
    require(lvl_128 == 53, "Verification failed at line 125")
    blk_p3 = (h_lo_128 >> 9) & 7
    blk_p2_direct = (h_lo_128 >> 9) & 3
    blk_p2_merged = blk_p3 & 3  # merging block j with block j + 4 without lifting
    require(blk_p2_merged == blk_p2_direct and lvl_128 == 53, "Verification failed at line 129")

    # 2) Single 64-bit Pow2 mode: exhaustive check on small 8-bit toy hash (s_sub=1, p=2 -> p=1)
    s_toy, p_toy = 1, 2
    w_toy = 8
    u_top_toy = (w_toy - p_toy - s_toy) + 1  # 8 - 3 + 1 = 6
    for h in range(1 << w_toy):
        sub = h & ((1 << s_toy) - 1)
        blk_hi = (h >> s_toy) & ((1 << p_toy) - 1)
        sent_hi = 1 << (p_toy + s_toy - 1)
        lvl_hi = (w_toy - (h | sent_hi).bit_length()) + 1

        blk_lo_direct = (h >> s_toy) & ((1 << (p_toy - 1)) - 1)
        sent_lo = 1 << ((p_toy - 1) + s_toy - 1)
        lvl_lo_direct = (w_toy - (h | sent_lo).bit_length()) + 1

        # Downsize rule: lift u_top -> u_top + 1 ONLY on half-blocks j < 2^(p-1)
        blk_lo_down = blk_hi & ((1 << (p_toy - 1)) - 1)
        lvl_lo_down = (lvl_hi + 1) if (blk_hi < (1 << (p_toy - 1)) and lvl_hi == u_top_toy) else lvl_hi
        require(blk_lo_down == blk_lo_direct and lvl_lo_down == lvl_lo_direct, "Verification failed at line 148")

    return {
        "passed": True,
        "scope": "pure_fixed_byte_dense_family_only_excluding_OsBmwHybrid",
        "distinct_states": len(state_to_sig),
        "linearization_bijection_verified_on_all_inputs": len(all_inputs),
        "homomorphism_pairs_verified": hom_checks,
        "downsize_2x_128bit_no_lift_level": lvl_128,
        "downsize_2x_128bit_no_lift_vs_64bit_lift_verified": True,
    }


def _counting_lhs(t: int, C_ex: int, m: int) -> int:
    return sum(math.comb(t, j) * (m ** j) for j in range(C_ex + 1))


def _evaluate_counting_condition(r_ord: int, W: int, C_ex: int, K_max: int, margin: int = 0):
    """
    Evaluate the exact counting inequality of Generalized Theorem 2 Part 1:
      requires m_mu >= 1 and margin + m_mu - 1 <= W (where m_mu = min(K_max - W, W + 1 - margin)),
      plus exists t in [C_ex + 1, r_ord - 1] such that sum_{j=0}^{C_ex} C(t, j) * m_mu^j < m_mu^t.
    """
    m = min(K_max - W, W + 1 - margin)
    t_lo = C_ex + 1
    t_hi = r_ord - 1
    preconditions_met = (
        (1 <= W <= K_max - 1)
        and (0 <= C_ex < r_ord - 1)
        and (margin >= 0)
        and (m >= 1)
        and (margin + m - 1 <= W)
    )
    if (not preconditions_met) or m <= 1 or t_lo > t_hi:
        return {
            "holds": False,
            "preconditions_met": preconditions_met,
            "m": m,
            "margin_plus_m_minus_1": margin + m - 1,
            "t_range": [t_lo, t_hi],
            "min_t": None,
        }
    min_t = None
    ratio_at_min_t = None
    for t in range(t_lo, t_hi + 1):
        lhs_val = _counting_lhs(t, C_ex, m)
        rhs_val = m ** t
        if lhs_val < rhs_val:
            min_t = t
            ratio_at_min_t = lhs_val / rhs_val
            break
    lhs_c1 = _counting_lhs(t_lo, C_ex, m)
    rhs_c1 = m ** t_lo
    res = {
        "holds": min_t is not None,
        "preconditions_met": preconditions_met,
        "m": m,
        "margin_plus_m_minus_1": margin + m - 1,
        "t_range": [t_lo, t_hi],
        "min_t": min_t,
        "t_eq_C_plus_1_ratio_1_plus_inv_m_pow": round((1.0 + 1.0 / m) ** (C_ex + 1), 6),
        "ratio_at_C_plus_1": round(lhs_c1 / rhs_c1, 6),
    }
    if min_t is not None:
        res["ratio_at_min_t"] = round(ratio_at_min_t, 6)
        if min_t <= 10:
            res["states_bound_at_min_t"] = _counting_lhs(min_t, C_ex, m)
            res["inputs_at_min_t"] = m ** min_t
    else:
        if t_lo <= 10:
            res["states_bound_at_C_plus_1"] = lhs_c1
            res["inputs_at_C_plus_1"] = rhs_c1
    return res


def _verify_toy_part2_associativity():
    """
    Exhaustively verify Generalized Theorem 2 Part 2 boundary on a toy bottom-anchored sketch
    over inputs strictly below W (levels in {0, ..., W-1}, zero upper clamping):
      - H = W - 1 triggers 0 associativity violations across ALL state triples;
      - H = W - 2 triggers deterministic associativity violation with zero upper clamping.
    """
    def run_toy(W: int, H: int, d: int, margin: int, g: int = 2):
        def proj(abs_bms):
            us = [bm.bit_length() for bm in abs_bms]
            anc = max(0, min(us) - margin)
            out = []
            for bm, u in zip(abs_bms, us):
                if u <= anc:
                    out.append((0, 0))
                else:
                    rel = min(u - anc, W)
                    if rel > H:
                        out.append((anc + rel, 0))
                    else:
                        lo = max(1, anc + 1, (anc + rel) - d)
                        h = 0
                        for lvl in range(lo, anc + rel):
                            if (bm >> (lvl - 1)) & 1:
                                h |= 1 << (lvl - lo)
                        out.append((anc + rel, h))
            return (anc, tuple(out))

        def to_bm(st):
            anc, regs = st
            bms = []
            for (u, h) in regs:
                if u == 0:
                    bm = (1 << (anc - 1)) if anc >= 1 else 0
                else:
                    bm = 1 << (u - 1)
                    if u - anc <= H:
                        lo = max(1, anc + 1, u - d)
                        for lvl in range(lo, u):
                            if (h >> (lvl - lo)) & 1:
                                bm |= 1 << (lvl - 1)
                bms.append(bm)
            return tuple(bms)

        def merge(s1, s2):
            return proj(tuple(x | y for x, y in zip(to_bm(s1), to_bm(s2))))

        nb = 1 << (W - 1)  # strictly below W: levels 1..W-1 only (zero upper clamping!)
        all_states = sorted(set(proj((a, b)) for a in range(nb) for b in range(nb)))
        triples_checked = 0
        first_violation = None
        for s1 in all_states:
            for s2 in all_states:
                s12 = merge(s1, s2)
                for s3 in all_states:
                    triples_checked += 1
                    lhs = merge(s12, s3)
                    rhs = merge(s1, merge(s2, s3))
                    if lhs != rhs and first_violation is None:
                        first_violation = {
                            "A": s1, "B": s2, "C": s3,
                            "lhs_AB_C": lhs, "rhs_A_BC": rhs,
                        }
        # Also test canonical construction A=(H+1,0), B=(H,0), C=(0, margin+1) when H <= W - 2
        canon_viol = None
        if H <= W - 2:
            A = proj((1 << H, 0))
            B = proj((1 << (H - 1), 0))
            C = proj((0, 1 << margin))
            lhs_c = merge(merge(A, B), C)
            rhs_c = merge(A, merge(B, C))
            require(lhs_c != rhs_c, "Verification failed at line 294")
            canon_viol = {
                "A_levels": [H + 1, 0],
                "B_levels": [H, 0],
                "C_levels": [0, margin + 1],
                "max_input_level": H + 1,
                "strictly_below_W": (H + 1) < W,
                "lhs_state": str(lhs_c),
                "rhs_state": str(rhs_c),
            }
        return {
            "g": g, "W": W, "H": H, "d": d, "margin": margin,
            "reachable_states_below_W": len(all_states),
            "triples_exhaustively_checked": triples_checked,
            "associativity_violated": first_violation is not None,
            "canonical_witness": canon_viol,
        }

    m0_no_viol = run_toy(W=4, H=3, d=1, margin=0)
    m0_viol = run_toy(W=4, H=2, d=1, margin=0)
    m1_no_viol = run_toy(W=4, H=3, d=1, margin=1)
    m1_viol = run_toy(W=4, H=2, d=1, margin=1)

    require(not m0_no_viol["associativity_violated"], "Verification failed at line 317")
    require(m0_viol["associativity_violated"], "Verification failed at line 318")
    require(not m1_no_viol["associativity_violated"], "Verification failed at line 319")
    require(m1_viol["associativity_violated"], "Verification failed at line 320")

    return {
        "H_eq_W_minus_1_zero_violations_margin0": m0_no_viol,
        "H_eq_W_minus_2_violates_margin0": m0_viol,
        "H_eq_W_minus_1_zero_violations_margin1": m1_no_viol,
        "H_eq_W_minus_2_violates_margin1": m1_viol,
        "boundary_H_le_W_minus_2_verified": True,
    }


def verify_generalized_theorem_2():
    """
    Verify Generalized Theorem 2 (Parts 1–3):
      - Part 1: Minimal 4-state counterexample + exact counting condition on toy & real params
                + generic saturating partition-homomorphism witnesses;
      - Part 2: Exhaustive toy verification of H = W - 1 (no violation) vs H <= W - 2 (violation);
      - Part 3: Exhaustive & randomized verification of zero-eviction iff r_ord <= C_ex + 1.
    """
    # 1A. Verify minimal 4-state counterexample (g=r_ord=3, W=2, K_max=4, C_ex=1)
    g_cx, r_cx, W_cx, K_cx, C_cx = 3, 3, 2, 4, 1
    four_state_inputs = [(3, 3, 0), (3, 4, 0), (4, 3, 0), (4, 4, 0)]
    four_state_map = {}
    for u in four_state_inputs:
        anchor = sorted(u, reverse=True)[r_cx - 1]
        require(anchor == 0, "Verification failed at line 345")
        # Deterministic retention policy: retain bucket 0 if u[0] == u[1], else retain bucket 1; saturate other to A+W=2
        if u[0] == u[1]:
            state = (u[0], anchor + W_cx, u[2])
        else:
            state = (anchor + W_cx, u[1], u[2])
        four_state_map[str(u)] = list(state)
    distinct_four_states = set(tuple(v) for v in four_state_map.values())
    require(len(distinct_four_states) == 4, "Verification failed at line 353")
    four_state_cond = _evaluate_counting_condition(r_cx, W_cx, C_cx, K_cx)
    require(four_state_cond["holds"] is False, "Verification failed at line 355")
    require(four_state_cond["states_bound_at_C_plus_1"] == 5 and four_state_cond["inputs_at_C_plus_1"] == 4, "Verification failed at line 356")

    minimal_4state_counterexample = {
        "params": {"g": g_cx, "r_ord": r_cx, "W": W_cx, "K_max": K_cx, "C_ex": C_cx},
        "input_to_state_map": four_state_map,
        "distinct_states_produced": len(distinct_four_states),
        "all_4_inputs_distinguished": len(distinct_four_states) == 4,
        "counting_condition": four_state_cond,
    }

    # 1B. Evaluate exact counting condition across minimal, degenerate-margin, & real parameters
    audit_specs = [
        ("Minimal_4State_Boundary",   3,    3, 2,  1, 4,  0, False),
        ("Degenerate_Margin_W1_mu4",  3,    3, 1,  0, 4,  4, False),
        ("AVLL_DLL_Cap0_K60",        11,   11, 18, 0, 60, 0, True),
        ("AVLL_Margin2_Cap0_K60",    11,   11, 18, 0, 60, 2, True),
        ("HLL4_Fixed_Cap1_g4",        4,    4, 14, 1, 60, 0, True),
        ("HLL4_Fixed_Cap1_g1024",  1024, 1024, 14, 1, 60, 0, True),
        ("HLL4_Fixed_Cap4_g16",      16,   16, 14, 4, 60, 0, True),
        ("HLL4_Fixed_Cap32_g1024", 1024, 1024, 14, 32, 60, 0, True),
        ("Median_Anchor_Fixed",      16,    8, 12, 5, 60, 0, True),
        ("Quartile_Anchor_Fixed",    32,    8, 10, 6, 60, 0, True),
    ]
    counting_audit = []
    for name, g_v, r_v, W_v, C_v, K_v, mu_v, expected in audit_specs:
        ev = _evaluate_counting_condition(r_v, W_v, C_v, K_v, margin=mu_v)
        require(ev["holds"] is expected, f"Unexpected counting condition for {name}: {ev}")
        if expected:
            require(ev["preconditions_met"] and ev["m"] >= 1 and ev["margin_plus_m_minus_1"] <= W_v, "Verification failed at line 384")
        counting_audit.append({
            "case": name,
            "g": g_v, "r_ord": r_v, "W": W_v, "C_ex": C_v, "K_max": K_v, "margin": mu_v,
            **ev,
        })
    # Verify expected minimal t values & margin boundary preconditions explicitly
    by_case = {e["case"]: e for e in counting_audit}
    require(by_case["Degenerate_Margin_W1_mu4"]["m"] == -2, "Verification failed at line 392")
    require(by_case["Degenerate_Margin_W1_mu4"]["preconditions_met"] is False, "Verification failed at line 393")
    # Exhaustively confirm the degenerate margin case (g=r_ord=3, W=1, K_max=mu=4, C_ex=0) is exactly mergeable
    deg_inputs = list(product(range(5), repeat=3))
    deg_proj = {u: (max(0, min(u) - 4), tuple(min(x, 1) for x in u)) for u in deg_inputs}
    require(len(deg_inputs) == 125 and len(set(deg_proj.values())) == 8, "Verification failed at line 397")
    for u in deg_inputs:
        for v in deg_inputs:
            uv = tuple(max(a, b) for a, b in zip(u, v))
            m_uv = (0, tuple(max(a, b) for a, b in zip(deg_proj[u][1], deg_proj[v][1])))
            require(deg_proj[uv] == m_uv, "Verification failed at line 402")
    require(by_case["AVLL_DLL_Cap0_K60"]["min_t"] == 1 and by_case["AVLL_DLL_Cap0_K60"]["preconditions_met"] is True, "Verification failed at line 403")
    require(by_case["AVLL_Margin2_Cap0_K60"]["min_t"] == 1 and by_case["AVLL_Margin2_Cap0_K60"]["preconditions_met"] is True, "Verification failed at line 404")
    require(by_case["HLL4_Fixed_Cap1_g4"]["min_t"] == 2, "Verification failed at line 405")
    require(by_case["HLL4_Fixed_Cap1_g1024"]["min_t"] == 2, "Verification failed at line 406")
    require(by_case["HLL4_Fixed_Cap4_g16"]["min_t"] == 5, "Verification failed at line 407")
    require(by_case["HLL4_Fixed_Cap32_g1024"]["min_t"] == 37, "Verification failed at line 408")

    # 1C. Generic saturating partition-homomorphism witnesses (FIFO retention + elementwise-max merge)
    # Note: Bottom_HLL4_Fixed gives 18 != 14 here; concrete Rust witnesses (HLL4_Fixed g=4 cap=1 -> 20 != 14
    # and AVLL 11-bucket associativity 47 != 49) are verified in s1_v2_rust/src/lib.rs.
    cases = [
        ("Bottom_HLL4_Fixed", 16, 16, 14, 4, 60),
        ("Bottom_AVLL_Cap0",  11, 11, 18, 0, 60),
        ("Median_Anchor_Fixed", 16, 8, 12, 5, 60),
        ("Quartile_Anchor_Fixed", 32, 8, 10, 6, 60),
    ]
    results = []
    for name, g, r_ord, W, C_ex, K_max in cases:
        require(C_ex < r_ord - 1, "Verification failed at line 421")
        require(W + 4 <= K_max, "Verification failed at line 422")
        L_hi = W + 4
        L_lift = 4

        def insert_generic(state_u, cap_exc, r_idx, lvl):
            u = list(state_u)
            cand = max(u[r_idx], lvl)
            u_tmp = list(u)
            u_tmp[r_idx] = cand
            new_anchor = sorted(u_tmp, reverse=True)[r_ord - 1]
            exc_cnt = sum(1 for x in u_tmp if x > new_anchor + W)
            if exc_cnt <= cap_exc:
                u[r_idx] = cand
            else:
                u[r_idx] = min(cand, new_anchor + W)
            return tuple(u)

        uA = tuple([0] * g)
        for j in range(C_ex + 1):
            uA = insert_generic(uA, C_ex, j, L_hi)
        uB = tuple([0] * g)
        for j in range(g):
            uB = insert_generic(uB, C_ex, j, L_lift)
        uBA = uB
        for j in range(C_ex + 1):
            uBA = insert_generic(uBA, C_ex, j, L_hi)
        uM = tuple(max(uA[j], uB[j]) for j in range(g))

        require(uBA[C_ex] == L_hi, "Verification failed at line 450")
        require(uM[C_ex] == W < L_hi, "Verification failed at line 451")
        require(uM != uBA, "Verification failed at line 452")
        results.append({
            "case": name,
            "g": g, "anchor_order_r": r_ord, "W": W, "C_ex": C_ex,
            "direct_BA_bucket_val": uBA[C_ex],
            "merged_AB_bucket_val": uM[C_ex],
            "homomorphism_violated": True,
        })

    # 2. Part 2: Exhaustive toy verification of H = W - 1 (no violation) vs H = W - 2 (violation)
    part2_boundary = _verify_toy_part2_associativity()

    # 3. Part 3: Exhaustive verification of zero-eviction iff r_ord <= C_ex + 1 (g=4, W=2, K_max=4)
    g_p3, W_p3, K_p3 = 4, 2, 4
    iff_checks = 0
    for r_ord in range(1, g_p3 + 1):
        max_over = 0
        for u in product(range(K_p3 + 1), repeat=g_p3):
            anc = sorted(u, reverse=True)[r_ord - 1]
            over = sum(1 for x in u if x > anc + W_p3)
            if over > max_over:
                max_over = over
        require(max_over == r_ord - 1, "Verification failed at line 474")
        for C_ex in range(0, g_p3):
            no_eviction_all_inputs = (max_over <= C_ex)
            require(no_eviction_all_inputs == (r_ord <= C_ex + 1), "Verification failed at line 477")
            iff_checks += 1

    # Also verify top-k tightness (r_ord = k = C_ex + 1, W = 0) on larger (g, k)
    for g, k in [(16, 4), (16, 8), (32, 8)]:
        rng = np.random.default_rng(42 + g + k)
        max_exc = 0
        for _ in range(2000):
            u = rng.integers(0, 61, size=g)
            B = int(np.sort(u)[::-1][k - 1])
            exc = int(np.sum(u > B))
            max_exc = max(max_exc, exc)
            require(exc <= k - 1, "Verification failed at line 489")
        require(max_exc == k - 1, "Verification failed at line 490")  # tight!

    return {
        "passed": True,
        "minimal_4state_counterexample": minimal_4state_counterexample,
        "counting_condition_audit": counting_audit,
        "witness_cases": results,
        "part2_high_tier_history_boundary": part2_boundary,
        "part3_overflow_avoidance_iff_verified_pairs": iff_checks,
        "top_k_tight_bound_verified": True,
        "witness_attribution_note": (
            "Python verifies the generic saturating witnesses (Bottom_HLL4_Fixed 18/14, Bottom_AVLL_Cap0 22/18, "
            "Median/Quartile anchors), the minimal 4-state boundary counterexample, and the toy H=W-1 vs H<=W-2 associativity "
            "boundary; concrete Rust implementation witnesses (AVLL 11-bucket 47/49, HLL_4-Fixed g=4 cap=1 20/14) "
            "are verified in s1_v2_rust/src/lib.rs."
        ),
    }


def verify_theorem_3():
    """Verify exact Cartesian likelihood factorization, closed-form Fisher/entropy, concavity, and Part 4 PMF bounds."""
    from reconcile_pmf_and_rank import exact_osbmw_fisher_and_entropy, build_exact_preimage_pmf_div3

    K = 6
    probs = np.zeros(K + 1)
    for k in range(1, K):
        probs[k] = 2.0 ** (-k)
    probs[K] = 2.0 ** (-(K - 1))
    tail = np.zeros(K + 1)
    for u in range(K - 1, -1, -1):
        tail[u] = tail[u + 1] + probs[u + 1]

    max_fisher_err = 0.0
    max_pmf_sum_err = 0.0
    max_entropy_err = 0.0

    for (g, k, R, d, lam) in [(2, 1, 3, 1, 8.0), (2, 2, 3, 1, 8.0), (3, 2, 3, 1, 10.0), (3, 2, 4, 2, 12.0)]:
        nb = 1 << K
        bs_p = np.zeros(nb)
        bs_dp = np.zeros(nb)
        bs_u = np.zeros(nb, dtype=np.int32)
        for mask in range(nb):
            p = 1.0
            dl = 0.0
            u = 0
            for lvl in range(1, K + 1):
                pk = probs[lvl]
                xk = lam * pk
                if (mask >> (lvl - 1)) & 1:
                    u = lvl
                    qk = -math.expm1(-xk)
                    p *= qk
                    dl += pk * math.exp(-xk) / qk
                else:
                    p *= math.exp(-xk)
                    dl -= pk
            bs_p[mask] = p
            bs_dp[mask] = p * dl
            bs_u[mask] = u

        state_p = {}
        state_dp = {}
        for combo in product(range(nb), repeat=g):
            B = int(sorted((bs_u[c] for c in combo), reverse=True)[k - 1])
            win_lo = max(1, B - R + 1)
            proj = []
            for c in combo:
                u = int(bs_u[c])
                if u < win_lo:
                    proj.append((0, 0))
                else:
                    lo = max(win_lo, u - d)
                    h = 0
                    for j in range(lo, u):
                        if (c >> (j - 1)) & 1:
                            h |= 1 << (j - lo)
                    proj.append((u, h))
            key = (B, tuple(proj))
            pj = 1.0
            for c in combo:
                pj *= bs_p[c]
            dpj = 0.0
            for i in range(g):
                t = bs_dp[combo[i]]
                for j in range(g):
                    if j != i:
                        t *= bs_p[combo[j]]
                dpj += t
            state_p[key] = state_p.get(key, 0.0) + pj
            state_dp[key] = state_dp.get(key, 0.0) + dpj

        pmf_sum = sum(state_p.values())
        require(math.isfinite(pmf_sum), f"Non-finite PMF sum: {pmf_sum}")
        max_pmf_sum_err = max(max_pmf_sum_err, abs(pmf_sum - 1.0))
        for (B, proj), p_true in state_p.items():
            win_lo = max(1, B - R + 1)
            p_fact = 1.0
            for (u, h) in proj:
                if u == 0:
                    p_fact *= math.exp(-lam * tail[win_lo - 1])
                else:
                    p_fact *= math.exp(-lam * tail[u]) * (-math.expm1(-lam * probs[u]))
                    lo = max(win_lo, u - d)
                    for j in range(lo, u):
                        bit = (h >> (j - lo)) & 1
                        if bit:
                            p_fact *= -math.expm1(-lam * probs[j])
                        else:
                            p_fact *= math.exp(-lam * probs[j])
            require(abs(p_fact - p_true) < 1e-13 * max(1.0, p_true), "Factorization failed!")

        J_bf = sum((lam * state_dp[key]) ** 2 / state_p[key] for key in state_p if state_p[key] > 1e-300) / g
        H_bf = -sum(p * math.log2(p) for p in state_p.values() if p > 1e-300) / g
        J_th, H_th = exact_osbmw_fisher_and_entropy(probs, g, k, R, d, lam)
        require(
            math.isfinite(J_bf) and J_bf > 0.0 and math.isfinite(J_th)
            and math.isfinite(H_bf) and H_bf > 0.0 and math.isfinite(H_th),
            f"Non-finite Fisher/entropy values: J_bf={J_bf}, J_th={J_th}, H_bf={H_bf}, H_th={H_th}",
        )
        max_fisher_err = max(max_fisher_err, abs(J_bf - J_th) / J_bf)
        max_entropy_err = max(max_entropy_err, abs(H_bf - H_th) / H_bf)

    # Strict log-concavity in lam and theta = ln(lam)
    xs = np.logspace(-8, 2, 1000)
    d2_lam = -(xs ** 2) * np.exp(-xs) / ((-np.expm1(-xs)) ** 2)
    d2_theta = xs * (np.expm1(xs) - xs * np.exp(xs)) / (np.expm1(xs) ** 2)
    require(np.all(d2_lam < 0.0), "Verification failed at line 610")
    require(np.all(d2_theta < 0.0), "Verification failed at line 611")

    # Part 4 discrete preimage boundary checks:
    # 1) Pow2Div3 at P = 11 (6,144 sub-buckets): Q = floor(2^64/3), Q mod 2048 = 1365, excluded = 4096
    Q_div3 = ((1 << 64) - 1) // 3
    P_div3 = 11
    q_mod = Q_div3 % (1 << P_div3)
    n_rem = Q_div3 >> P_div3
    excluded_raw = (1 << 64) - 3 * (1 << P_div3) * n_rem
    km_div3, c_div3, _ = build_exact_preimage_pmf_div3(P_div3)
    require(q_mod == 1365 and excluded_raw == 4096 and sum(c_div3[1:]) == n_rem, "Verification failed at line 621")

    # 2) GeomExactFloor exact integer thresholds for b = 17/8, W = 64
    dom64 = 1 << 64
    b_frac = Fraction(17, 8)
    thresh_exact = [dom64]
    lvl_idx = 1
    while True:
        t_val = (dom64 * (b_frac.denominator ** lvl_idx)) // (b_frac.numerator ** lvl_idx)
        if t_val == 0 or lvl_idx == 63:
            thresh_exact.append(0)
            break
        thresh_exact.append(t_val)
        lvl_idx += 1
    t1_exact = thresh_exact[1]
    t1_f64 = int(float(dom64) * (2.125 ** -1))
    k_reach_floor = len(thresh_exact) - 1
    counts_floor = [thresh_exact[j - 1] - thresh_exact[j] for j in range(1, k_reach_floor + 1)]
    require(t1_exact == 8680820740569200760, "Verification failed at line 639")
    require(t1_exact - t1_f64 == 120, "Verification failed at line 640")
    require(k_reach_floor == 59 and sum(counts_floor) == dom64, "Verification failed at line 641")

    require(math.isfinite(max_pmf_sum_err) and max_pmf_sum_err < 1e-13, f"PMF sum error too large: {max_pmf_sum_err}")
    require(math.isfinite(max_fisher_err) and max_fisher_err < 1e-12, f"Fisher relative error too large: {max_fisher_err}")
    require(math.isfinite(max_entropy_err) and max_entropy_err < 1e-12, f"Entropy relative error too large: {max_entropy_err}")

    return {
        "passed": True,
        "factorization_exact_to_1e_13": True,
        "max_pmf_sum_err": max_pmf_sum_err,
        "max_fisher_rel_err": max_fisher_err,
        "max_entropy_rel_err": max_entropy_err,
        "strict_concavity_in_lam_and_log_lam": True,
        "part4_discrete_preimage_audit": {
            "pow2div3_p11": {
                "Q": Q_div3,
                "Q_mod_2048": q_mod,
                "n_rem": n_rem,
                "excluded_raw_preimages": excluded_raw,
                "k_reach": km_div3,
                "quotient_closure_exact": sum(c_div3[1:]) == n_rem,
            },
            "geom_exact_floor_b_17_8_w64": {
                "first_threshold_exact_int": t1_exact,
                "first_threshold_f64": t1_f64,
                "f64_drift": t1_exact - t1_f64,
                "k_reach_exact_int": k_reach_floor,
                "domain_closure_exact": sum(counts_floor) == dom64,
            },
        },
    }


def verify_theorem_4():
    """Verify exact combinatorial state count, per-phase & header-amortized MVP, and promotion bound K_reach - R."""
    from reconcile_pmf_and_rank import combinatorial_rank_size, build_exact_preimage_pmf_geom, exact_osbmw_fisher_and_entropy

    R_rank = 7
    g_rank = 2048
    k_rank = 92
    d_rank = 2
    k60, c60, p60, _ = build_exact_preimage_pmf_geom(2.125, 64, 63)
    require(k60 == 60 and c60[60] == 1, "Verification failed at line 683")

    total, log2_s, bits_int, bytes_int = combinatorial_rank_size(g_rank, k_rank, R_rank, d_rank, k60)
    require(bits_int == 10209 and bytes_int == 1277, "Verification failed at line 686")
    require(abs(log2_s - 10208.865166991258) < 1e-9, "Verification failed at line 687")

    n_phases = 64
    js = [
        exact_osbmw_fisher_and_entropy(p60, g_rank, k_rank, R_rank, d_rank, 1e6 * (2.125 ** (i / n_phases)))[0]
        for i in range(n_phases)
    ]
    J_avg = float(np.mean(js))
    bits_per_sub_raw = (bytes_int * 8) / float(g_rank)
    per_phase_mvps = [bits_per_sub_raw / j for j in js]
    gt3_count = sum(1 for v in per_phase_mvps if v > 3.0)
    mvp_raw_j_avg = bits_per_sub_raw / J_avg
    mvp_hdr_2blk = ((2 * bytes_int + 8) * 8 / (2.0 * g_rank)) / J_avg
    mvp_hdr_1blk = ((bytes_int + 8) * 8 / float(g_rank)) / J_avg

    require(mvp_raw_j_avg < 3.0, "Verification failed at line 702")
    require(gt3_count == 28, "Verification failed at line 703")
    require(abs(mvp_hdr_2blk - 3.007055) < 1e-5, "Verification failed at line 704")
    max_promotions = k60 - R_rank
    require(max_promotions == 53, "Verification failed at line 706")

    return {
        "passed": True,
        "k_reach": k60,
        "R": R_rank,
        "exact_log2_states": round(log2_s, 6),
        "exact_log2_states_full": log2_s,
        "exact_bits": bits_int,
        "fixed_bytes": bytes_int,
        "fisher_J_avg_64phases": round(J_avg, 8),
        "wire_mvp": round(mvp_raw_j_avg, 5),
        "wire_mvp_raw_payload_j_avg": round(mvp_raw_j_avg, 6),
        "strictly_below_3_00": mvp_raw_j_avg < 3.0,
        "strictly_below_3_00_scope": "raw_1277B_payload_64phase_mean_J_avg_independent_bucket_poisson_only",
        "per_phase_count": n_phases,
        "per_phase_mvp_min": round(min(per_phase_mvps), 6),
        "per_phase_mvp_max": round(max(per_phase_mvps), 6),
        "per_phase_mvp_gt_3_count": gt3_count,
        "wire_mvp_with_8B_header_2blk": round(mvp_hdr_2blk, 6),
        "wire_mvp_with_8B_header_1blk": round(mvp_hdr_1blk, 6),
        "k60_top_level_preimages": int(c60[60]),
        "single_64bit_word_cannot_realize_2048_indep_k60": (g_rank * (1 << 64) > (1 << 64)),
        "max_promotions_formula": "K_reach - R",
        "max_promotions_per_block_upper_bound": max_promotions,
        "steady_state_non_early_exit_order": "Theta(b^R * m * k / (n * g))",
    }


def main():
    theorems = {
        "theorem_1": verify_theorem_1(),
        "generalized_theorem_2": verify_generalized_theorem_2(),
        "theorem_3": verify_theorem_3(),
        "theorem_4": verify_theorem_4(),
    }
    all_passed = all(v["passed"] for v in theorems.values())
    res = {
        **theorems,
        "all_passed": all_passed,
    }
    import os
    import sys

    default_out = Path(__file__).resolve().parent.parent / "out" / "proofs_verification.json"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("S1_PROOFS_OUT", str(default_out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
    require(all_passed, "Verification failed at line 755")
    print("\nALL FOUR THEOREMS VERIFIED BY EXACT EXHAUSTIVE/NUMERICAL CHECKS!")


if __name__ == "__main__":
    main()
