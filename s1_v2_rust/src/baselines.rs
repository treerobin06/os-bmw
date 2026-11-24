//! All 9 Baseline Cardinality Sketches under ONE Unified Single-64-bit Hash & 8-Byte Header.
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
//!   9. `HbsSketch`      : Huffman-Bucket Sketch (Karppa 2026, arXiv:2603.10930: lossless Huffman-coded HLL buckets)

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

// Exact Ertl Optimal FGRA estimator tables & helpers for UltraLogLog (PVLDB'24, arXiv:2308.16862; Apache-2.0 hash4j)
const ULL_FGRA_ETA_0: f64 = 4.663135422063788;
const ULL_FGRA_ETA_1: f64 = 2.1378502137958524;
const ULL_FGRA_ETA_2: f64 = 2.781144650979996;
const ULL_FGRA_ETA_3: f64 = 0.9824082545153715;
const ULL_FGRA_TAU: f64 = 0.8194911375910897;
const ULL_FGRA_POW_2_TAU: f64 = 1.7647834144840502;
const ULL_FGRA_POW_2_MINUS_TAU: f64 = 0.5666417713316728;
const ULL_FGRA_POW_4_MINUS_TAU: f64 = 0.32108289701789583;
const ULL_FGRA_MINUS_INV_TAU: f64 = -1.0 / ULL_FGRA_TAU;
const ULL_FGRA_ETA_X: f64 = ULL_FGRA_ETA_0 - ULL_FGRA_ETA_1 - ULL_FGRA_ETA_2 + ULL_FGRA_ETA_3;
const ULL_FGRA_ETA23X: f64 = (ULL_FGRA_ETA_2 - ULL_FGRA_ETA_3) / ULL_FGRA_ETA_X;
const ULL_FGRA_ETA13X: f64 = (ULL_FGRA_ETA_1 - ULL_FGRA_ETA_3) / ULL_FGRA_ETA_X;
const ULL_FGRA_ETA3012XX: f64 = (ULL_FGRA_ETA_3 * ULL_FGRA_ETA_0 - ULL_FGRA_ETA_1 * ULL_FGRA_ETA_2) / (ULL_FGRA_ETA_X * ULL_FGRA_ETA_X);
const ULL_FGRA_POW_4_MINUS_TAU_ETA_23: f64 = ULL_FGRA_POW_4_MINUS_TAU * (ULL_FGRA_ETA_2 - ULL_FGRA_ETA_3);
const ULL_FGRA_POW_4_MINUS_TAU_ETA_01: f64 = ULL_FGRA_POW_4_MINUS_TAU * (ULL_FGRA_ETA_0 - ULL_FGRA_ETA_1);
const ULL_FGRA_POW_4_MINUS_TAU_ETA_3: f64 = ULL_FGRA_POW_4_MINUS_TAU * ULL_FGRA_ETA_3;
const ULL_FGRA_POW_4_MINUS_TAU_ETA_1: f64 = ULL_FGRA_POW_4_MINUS_TAU * ULL_FGRA_ETA_1;
const ULL_FGRA_POW_2_MINUS_TAU_ETA_X: f64 = ULL_FGRA_POW_2_MINUS_TAU * ULL_FGRA_ETA_X;
const ULL_FGRA_PHI_1: f64 = ULL_FGRA_ETA_0 / (ULL_FGRA_POW_2_TAU * (2.0 * ULL_FGRA_POW_2_TAU - 1.0));
const ULL_FGRA_P_INITIAL: f64 = ULL_FGRA_ETA_X * (ULL_FGRA_POW_4_MINUS_TAU / (2.0 - ULL_FGRA_POW_2_MINUS_TAU));
const ULL_FGRA_POW_2_MINUS_TAU_ETA_02: f64 = ULL_FGRA_POW_2_MINUS_TAU * (ULL_FGRA_ETA_0 - ULL_FGRA_ETA_2);
const ULL_FGRA_POW_2_MINUS_TAU_ETA_13: f64 = ULL_FGRA_POW_2_MINUS_TAU * (ULL_FGRA_ETA_1 - ULL_FGRA_ETA_3);
const ULL_FGRA_POW_2_MINUS_TAU_ETA_2: f64 = ULL_FGRA_POW_2_MINUS_TAU * ULL_FGRA_ETA_2;
const ULL_FGRA_POW_2_MINUS_TAU_ETA_3: f64 = ULL_FGRA_POW_2_MINUS_TAU * ULL_FGRA_ETA_3;
const ULL_FGRA_ESTIMATION_FACTORS: [f64; 24] = [
    94.59941722950778,
    455.6358404615186,
    2159.476860400962,
    10149.51036338182,
    47499.52712820488,
    221818.76564766388,
    1034754.6840013304,
    4824374.384717942,
    2.2486750611989766E7,
    1.0479810199493326E8,
    4.8837185623048025E8,
    2.275794725435168E9,
    1.0604938814719946E10,
    4.9417362104242645E10,
    2.30276227770117E11,
    1.0730444972228585E12,
    5.0001829613164E12,
    2.329988778511272E13,
    1.0857295240912981E14,
    5.059288069986326E14,
    2.3575295235667005E15,
    1.0985627213141412E16,
    5.119087674515589E16,
    2.3853948339571715E17,
];
const ULL_FGRA_REGISTER_CONTRIBUTIONS: [f64; 236] = [
    0.8484061093359406, 0.38895829052007685, 0.5059986252327467, 0.17873835725405993,
    0.48074234060273024, 0.22040001471443574, 0.2867199572932749, 0.10128061935935387,
    0.2724086914332655, 0.12488785473931466, 0.16246750447680292, 0.057389829555353204,
    0.15435814343988866, 0.0707666752272979, 0.09206087452057209, 0.03251947467566813,
    0.08746577181824695, 0.0400993542020493, 0.05216553700867983, 0.018426892732996067,
    0.04956175987398336, 0.022721969094305374, 0.029559172293066274, 0.01044144713836362,
    0.02808376340530896, 0.012875216815740723, 0.01674946174724118, 0.005916560101748389,
    0.015913433441643893, 0.0072956356627506685, 0.009490944673308844, 0.0033525700962450116,
    0.009017216113341773, 0.004134011914931561, 0.0053779657012946284, 0.0018997062578498703,
    0.005109531310944485, 0.002342503834183061, 0.00304738001114257, 0.001076452918957914,
    0.0028952738727082267, 0.0013273605219527246, 0.0017267728074345586, 6.09963188753462E-4,
    0.0016405831157217021, 7.521379173550258E-4, 9.78461602292084E-4, 3.4563062172237723E-4,
    9.2962292270938E-4, 4.2619276177576713E-4, 5.544372155028133E-4, 1.958487477192352E-4,
    5.267631795945699E-4, 2.4149862146135835E-4, 3.141672858847145E-4, 1.1097608132071735E-4,
    2.9848602115777116E-4, 1.3684320663902123E-4, 1.7802030736817869E-4, 6.288368329501905E-5,
    1.6913464774658265E-4, 7.754107700464113E-5, 1.0087374230011362E-4, 3.563252169014952E-5,
    9.583875639268212E-5, 4.393801322487549E-5, 5.715927601779108E-5, 2.0190875207520577E-5,
    5.430624268457414E-5, 2.4897113642537945E-5, 3.2388833410757184E-5, 1.144099329232623E-5,
    3.0772185549154786E-5, 1.4107744575453657E-5, 1.8352865935237916E-5, 6.482944704957522E-6,
    1.7436805727319977E-5, 7.99403737572986E-6, 1.0399500462555932E-5, 3.67350727106242E-6,
    9.880422483694849E-6, 4.529755498675165E-6, 5.892791363067244E-6, 2.081562667074589E-6,
    5.5986600976661345E-6, 2.5667486794686803E-6, 3.339101736056405E-6, 1.1795003568090263E-6,
    3.1724346748254955E-6, 1.4544270182973653E-6, 1.8920745223756656E-6, 6.683541714686068E-7,
    1.7976340035771381E-6, 8.241391019206623E-7, 1.072128458850476E-6, 3.7871739159788393E-7,
    1.0186145159929963E-6, 4.6699164053601817E-7, 6.075127690181302E-7, 2.1459709360913574E-7,
    5.77189533646426E-7, 2.6461697039041317E-7, 3.442421115430427E-7, 1.2159967724530947E-7,
    3.27059699739513E-7, 1.4994302882644454E-7, 1.9506195985170504E-7, 6.890345650764188E-8,
    1.853256875916027E-7, 8.49639834530526E-8, 1.1053025444979778E-7, 3.904357664636507E-8,
    1.0501327589016596E-7, 4.814414208323267E-8, 6.263105916717392E-8, 2.2123721430020238E-8,
    5.9504908663745294E-8, 2.7280481949286693E-8, 3.548937430686624E-8, 1.2536224699555158E-8,
    3.371796684815404E-8, 1.545826061452554E-8, 2.0109761920695445E-8, 7.103548569567803E-9,
    1.910600846054063E-8, 8.759296176321385E-9, 1.139503111580109E-8, 4.0251673442004705E-9,
    1.082626247715867E-8, 4.963383100969499E-9, 6.456900615837058E-9, 2.28082795382416E-9,
    6.134612546958812E-9, 2.812460192131048E-9, 3.65874960227048E-9, 1.292412391857717E-9,
    3.476127720042246E-9, 1.5936574250689536E-9, 2.0732003554895977E-9, 7.323348470132607E-10,
    1.9697191686598677E-9, 9.030328662369446E-10, 1.1747619217600795E-9, 4.1497151491950363E-10,
    1.1161251587553774E-9, 5.116961428952198E-10, 6.656691762391315E-10, 2.351401942661752E-10,
    6.324431369849931E-10, 2.899484087937328E-10, 3.771959611450379E-10, 1.3324025619025952E-10,
    3.5836869940773545E-10, 1.6429687995368037E-10, 2.1373498756237659E-10, 7.549949478033437E-11,
    2.0306667462222755E-10, 9.309747508122088E-11, 1.2111117194789844E-10, 4.2781167456975155E-11,
    1.1506606020637118E-10, 5.275291818652914E-11, 6.86266490006118E-11, 2.424159650745726E-11,
    6.520123617549523E-11, 2.9892007004129765E-11, 3.888672595026375E-11, 1.3736301184893309E-11,
    3.6945743959497274E-11, 1.693805979747882E-11, 2.2034843273746723E-11, 7.783562034953282E-12,
    2.093500180037604E-11, 9.597812206565218E-12, 1.248586262365167E-11, 4.4104913787558985E-12,
    1.186264650299681E-11, 5.4385213096368525E-12, 7.075011313669894E-12, 2.499168647301308E-12,
    6.721871027139603E-12, 3.081693348317683E-12, 4.008996942969544E-12, 1.4161333491633975E-12,
    3.808892905481426E-12, 1.7462161775917615E-12, 2.271665129027518E-12, 8.024403094117999E-13,
    2.1582778227746425E-12, 9.89479027998621E-13, 1.2872203525845489E-12, 4.54696198313039E-13,
    1.2229703685228866E-12, 5.606801491206791E-13, 7.293928206826874E-13, 2.5764985922987735E-13,
    6.92986095905959E-13, 3.1770479284824887E-13, 4.1330443990824427E-13, 1.4599517261737423E-13,
    3.926748688923721E-13, 1.8002480658009348E-13, 2.3419555992885186E-13, 8.272696321778206E-14,
    2.225059832666067E-13, 1.0200957528418621E-13, 1.327049869160979E-13, 4.687655297461429E-14,
    1.2608118449008524E-13, 5.780288643182276E-14, 7.519618885068399E-14, 2.656221301145837E-14,
    7.144286571105751E-14, 3.2753529955811655E-14, 4.2609301647742677E-14, 1.5051259431302017E-14,
    4.0482511975524363E-14, 1.8559518231526075E-14, 2.4144210160882415E-14, 8.528672304925501E-15,
    2.293908229376684E-14, 1.0516598285774437E-14, 1.3681118012966618E-14, 4.832701981970378E-15,
    1.2998242223663023E-14, 5.959143881034847E-15, 7.752292944665042E-15, 2.7384108113817744E-15,
    7.365346997814574E-15, 3.376699844369893E-15, 4.392773006047039E-15, 1.5516979527951759E-15,
    4.173513269314059E-15, 1.9133791810691354E-15, 2.4891286772044455E-15, 8.792568765435867E-16,
];

#[inline]
fn ull_fgra_small_range_estimate(c0: i64, c4: i64, c8: i64, c10: i64, m: i64) -> f64 {
    let alpha = m + 3 * (c0 + c4 + c8 + c10);
    let beta = m - c0 - c4;
    let gamma = 4 * c0 + 2 * c4 + 3 * c8 + c10;
    let disc = (beta * beta + 4 * alpha * gamma) as f64;
    let quad_root_z = (disc.sqrt() - (beta as f64)) / ((2 * alpha) as f64);
    let root_z = quad_root_z * quad_root_z;
    root_z * root_z
}

#[inline]
fn ull_fgra_large_range_estimate(c4w0: i64, c4w1: i64, c4w2: i64, c4w3: i64, m: i64) -> f64 {
    let alpha = m + 3 * (c4w0 + c4w1 + c4w2 + c4w3);
    let beta = c4w0 + c4w1 + 2 * (c4w2 + c4w3);
    let gamma = m + 2 * c4w0 + c4w2 - c4w3;
    let disc = (beta * beta + 4 * alpha * gamma) as f64;
    ((disc.sqrt() - (beta as f64)) / ((2 * alpha) as f64)).sqrt()
}

#[inline(always)]
fn ull_fgra_psi_prime(z: f64, z_sq: f64) -> f64 {
    (z + ULL_FGRA_ETA23X) * (z_sq + ULL_FGRA_ETA13X) + ULL_FGRA_ETA3012XX
}

fn ull_fgra_sigma(z: f64) -> f64 {
    if z <= 0.0 {
        return ULL_FGRA_ETA_3;
    }
    if z >= 1.0 {
        return f64::INFINITY;
    }
    let mut pow_z = z;
    let mut next_pow_z = pow_z * pow_z;
    let mut s = 0.0;
    let mut pow_tau = ULL_FGRA_ETA_X;
    loop {
        let old_s = s;
        let next_next_pow_z = next_pow_z * next_pow_z;
        s += pow_tau * (pow_z - next_pow_z) * ull_fgra_psi_prime(next_pow_z, next_next_pow_z);
        if !(s > old_s) {
            return s / z;
        }
        pow_z = next_pow_z;
        next_pow_z = next_next_pow_z;
        pow_tau *= ULL_FGRA_POW_2_TAU;
    }
}

fn ull_fgra_phi(z: f64, z_sq: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    if z >= 1.0 {
        return ULL_FGRA_PHI_1;
    }
    let mut previous_pow_z = z_sq;
    let mut pow_z = z;
    let mut next_pow_z = pow_z.sqrt();
    let mut p = ULL_FGRA_P_INITIAL / (1.0 + next_pow_z);
    let mut ps = ull_fgra_psi_prime(pow_z, previous_pow_z);
    let mut s = next_pow_z * (ps + ps) * p;
    loop {
        previous_pow_z = pow_z;
        pow_z = next_pow_z;
        let old_s = s;
        next_pow_z = pow_z.sqrt();
        let next_ps = ull_fgra_psi_prime(pow_z, previous_pow_z);
        p *= ULL_FGRA_POW_2_MINUS_TAU / (1.0 + next_pow_z);
        s += next_pow_z * ((next_ps + next_ps) - (pow_z + next_pow_z) * ps) * p;
        if !(s > old_s) {
            return s;
        }
        ps = next_ps;
    }
}

fn ull_fgra_large_range_contrib(c4w0: i64, c4w1: i64, c4w2: i64, c4w3: i64, m: i64, w: u32) -> f64 {
    let z = ull_fgra_large_range_estimate(c4w0, c4w1, c4w2, c4w3, m);
    let root_z = z.sqrt();
    let mut s = ull_fgra_phi(root_z, z) * ((c4w0 + c4w1 + c4w2 + c4w3) as f64);
    s += z * (1.0 + root_z)
        * ((c4w0 as f64) * ULL_FGRA_ETA_0
            + (c4w1 as f64) * ULL_FGRA_ETA_1
            + (c4w2 as f64) * ULL_FGRA_ETA_2
            + (c4w3 as f64) * ULL_FGRA_ETA_3);
    s += root_z
        * (((c4w0 + c4w1) as f64) * (z * ULL_FGRA_POW_2_MINUS_TAU_ETA_02 + ULL_FGRA_POW_2_MINUS_TAU_ETA_2)
            + ((c4w2 + c4w3) as f64) * (z * ULL_FGRA_POW_2_MINUS_TAU_ETA_13 + ULL_FGRA_POW_2_MINUS_TAU_ETA_3));
    s * ULL_FGRA_POW_2_MINUS_TAU.powi(w as i32) / ((1.0 + root_z) * (1.0 + z))
}

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

    #[inline(always)]
    pub fn state_bytes_for(p: u32) -> usize {
        HEADER_BYTES + (1usize << p)
    }

    #[inline(always)]
    pub fn decompose_hash_64(h: u64, p: u32) -> (usize, u8) {
        let m = 1usize << p;
        let idx = (h & ((m as u64) - 1)) as usize;
        let sentinel = 1u64 << (p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        (idx, lvl)
    }

    #[inline]
    pub fn init_flat_slice(buf: &mut [u8], p: u32) {
        let m = 1usize << p;
        debug_assert_eq!(buf.len(), HEADER_BYTES + m);
        buf.fill(0);
        let hdr = SketchHeader::new(2, p as u8, 0, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
    }

    #[inline(always)]
    pub fn add_level_flat_slice(buf: &mut [u8], idx: usize, lvl: u8) {
        let cur = unsafe { *buf.get_unchecked(HEADER_BYTES + idx) };
        let next = Self::update_reg_ull(cur, lvl);
        if next != cur {
            unsafe { *buf.get_unchecked_mut(HEADER_BYTES + idx) = next; }
        }
    }

    #[inline(always)]
    pub fn add_hash_64_flat_slice(buf: &mut [u8], p: u32, h: u64) {
        let (idx, lvl) = Self::decompose_hash_64(h, p);
        Self::add_level_flat_slice(buf, idx, lvl);
    }

    #[inline]
    pub fn merge_flat_slices(dst: &mut [u8], src: &[u8], m: usize) {
        debug_assert_eq!(dst.len(), HEADER_BYTES + m);
        debug_assert_eq!(src.len(), HEADER_BYTES + m);
        let a = &mut dst[HEADER_BYTES..HEADER_BYTES + m];
        let b = &src[HEADER_BYTES..HEADER_BYTES + m];
        for i in 0..m {
            a[i] = Self::merge_reg_ull(a[i], b[i]);
        }
    }

    pub fn estimate_ml_flat_slice(buf: &[u8], m: usize, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for &r in &buf[HEADER_BYTES..HEADER_BYTES + m] {
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
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, m as f64)
    }

    pub fn estimate_fgra(&self) -> f64 {
        Self::estimate_fgra_regs(&self.buf[HEADER_BYTES..], self.p)
    }

    pub fn estimate_fgra_flat_slice(buf: &[u8], p: u32) -> f64 {
        let m = 1usize << p;
        Self::estimate_fgra_regs(&buf[HEADER_BYTES..HEADER_BYTES + m], p)
    }

    pub fn estimate_fgra_regs(regs: &[u8], p: u32) -> f64 {
        debug_assert!((3..=26).contains(&p));
        let m = regs.len();
        debug_assert_eq!(m, 1usize << p);
        let mut c0: i64 = 0;
        let mut c4: i64 = 0;
        let mut c8: i64 = 0;
        let mut c10: i64 = 0;
        let mut c4w0: i64 = 0;
        let mut c4w1: i64 = 0;
        let mut c4w2: i64 = 0;
        let mut c4w3: i64 = 0;
        let mut sum: f64 = 0.0;
        let large_u = (65 - p) as u8;
        for &r_rust in regs {
            if r_rust == 0 {
                c0 += 1;
            } else {
                let u = r_rust >> 2;
                if u < 3 {
                    match r_rust {
                        4 => c4 += 1,
                        8 => c8 += 1,
                        10 => c10 += 1,
                        _ => {}
                    }
                } else if u < large_u {
                    let r2 = (r_rust - 12) as usize;
                    sum += ULL_FGRA_REGISTER_CONTRIBUTIONS[r2];
                } else {
                    match r_rust & 3 {
                        0 => c4w0 += 1,
                        1 => c4w1 += 1,
                        2 => c4w2 += 1,
                        _ => c4w3 += 1,
                    }
                }
            }
        }
        if c0 == m as i64 {
            return 0.0;
        }
        if c0 > 0 || c4 > 0 || c8 > 0 || c10 > 0 {
            let z = ull_fgra_small_range_estimate(c0, c4, c8, c10, m as i64);
            if c0 > 0 { sum += (c0 as f64) * ull_fgra_sigma(z); }
            if c4 > 0 { sum += (c4 as f64) * ULL_FGRA_POW_2_MINUS_TAU_ETA_X * ull_fgra_psi_prime(z, z * z); }
            if c8 > 0 { sum += (c8 as f64) * (z * ULL_FGRA_POW_4_MINUS_TAU_ETA_01 + ULL_FGRA_POW_4_MINUS_TAU_ETA_1); }
            if c10 > 0 { sum += (c10 as f64) * (z * ULL_FGRA_POW_4_MINUS_TAU_ETA_23 + ULL_FGRA_POW_4_MINUS_TAU_ETA_3); }
        }
        if c4w0 > 0 || c4w1 > 0 || c4w2 > 0 || c4w3 > 0 {
            sum += ull_fgra_large_range_contrib(c4w0, c4w1, c4w2, c4w3, m as i64, 65 - p);
        }
        if !sum.is_finite() || sum <= 0.0 {
            return if c0 == m as i64 { 0.0 } else { f64::INFINITY };
        }
        ULL_FGRA_ESTIMATION_FACTORS[(p - 3) as usize] * sum.powf(ULL_FGRA_MINUS_INV_TAU)
    }
}

/// Block-min filtered `ULL(p=12)` ablation baseline (`E-01`):
/// `[8B Header | 8B block min_u (B=8 x 512 regs) | m bytes ULL regs]` = `4,112 B` (+8 B over ULL).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UllFiltered8B {
    pub p: u32,
    pub m: usize,
    pub blk_shift: u32,
    pub blk_size: usize,
    pub buf: Vec<u8>,
}

impl UllFiltered8B {
    pub const B: usize = 8;

    pub fn new(p: u32) -> Self {
        assert!((3..=20).contains(&p));
        let m = 1usize << p;
        let blk_size = m / Self::B;
        let blk_shift = p - 3;
        let mut buf = vec![0u8; HEADER_BYTES + Self::B + m];
        let hdr = SketchHeader::new(12, p as u8, Self::B as u8, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        Self { p, m, blk_shift, blk_size, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        let (idx, lvl) = UllSketch::decompose_hash_64(h, self.p);
        let blk = idx >> self.blk_shift;
        let mu = unsafe { *self.buf.get_unchecked(HEADER_BYTES + blk) };
        if lvl.wrapping_add(2) < mu {
            return true;
        }
        let reg_off = HEADER_BYTES + Self::B + idx;
        let cur = unsafe { *self.buf.get_unchecked(reg_off) };
        let next = UllSketch::update_reg_ull(cur, lvl);
        if next != cur {
            unsafe { *self.buf.get_unchecked_mut(reg_off) = next; }
            if (cur >> 2) == mu && (next >> 2) > mu {
                let start = HEADER_BYTES + Self::B + blk * self.blk_size;
                let mut mmin = u8::MAX;
                for &r in &self.buf[start..start + self.blk_size] {
                    let u = r >> 2;
                    if u < mmin {
                        mmin = u;
                        if mmin == mu {
                            break;
                        }
                    }
                }
                self.buf[HEADER_BYTES + blk] = mmin;
            }
        }
        false
    }

    pub fn regs(&self) -> &[u8] {
        &self.buf[HEADER_BYTES + Self::B..]
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        let base = HEADER_BYTES + Self::B;
        for i in 0..self.m {
            self.buf[base + i] = UllSketch::merge_reg_ull(self.buf[base + i], other.buf[base + i]);
        }
        for blk in 0..Self::B {
            let start = base + blk * self.blk_size;
            let mut mmin = u8::MAX;
            for &r in &self.buf[start..start + self.blk_size] {
                let u = r >> 2;
                if u < mmin {
                    mmin = u;
                }
            }
            self.buf[HEADER_BYTES + blk] = mmin;
        }
    }

    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        for &r in self.regs() {
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

    #[inline(always)]
    pub fn state_bytes_for(p: u32) -> usize {
        HEADER_BYTES + (1usize << p) * 4
    }

    #[inline]
    pub fn init_flat_slice(buf: &mut [u8], p: u32) {
        let m = 1usize << p;
        debug_assert_eq!(buf.len(), HEADER_BYTES + m * 4);
        buf.fill(0);
        let hdr = SketchHeader::new(4, p as u8, 24, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
    }

    #[inline(always)]
    pub fn add_level_flat_slice(buf: &mut [u8], idx: usize, k: u32) {
        let off = HEADER_BYTES + (idx << 2);
        let ptr = unsafe { buf.as_mut_ptr().add(off) as *mut u32 };
        let cur = u32::from_le(unsafe { ptr.read_unaligned() });
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            unsafe { ptr.write_unaligned(next.to_le()); }
        }
    }

    #[inline(always)]
    pub fn add_hash_64_flat_slice(buf: &mut [u8], p: u32, h: u64) {
        let (idx, k) = ell_hash_to_idx_k(h, p);
        Self::add_level_flat_slice(buf, idx, k);
    }

    #[inline]
    pub fn merge_flat_slices(dst: &mut [u8], src: &[u8], m: usize) {
        debug_assert_eq!(dst.len(), HEADER_BYTES + m * 4);
        debug_assert_eq!(src.len(), HEADER_BYTES + m * 4);
        let dp = unsafe { dst.as_mut_ptr().add(HEADER_BYTES) as *mut u32 };
        let sp = unsafe { src.as_ptr().add(HEADER_BYTES) as *const u32 };
        for i in 0..m {
            let a = u32::from_le(unsafe { dp.add(i).read_unaligned() });
            let b = u32::from_le(unsafe { sp.add(i).read_unaligned() });
            let merged = ell_merge_reg(a, b, Self::D);
            unsafe { dp.add(i).write_unaligned(merged.to_le()); }
        }
    }

    pub fn estimate_ml_flat_slice(buf: &[u8], m: usize, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 256];
        let mut zero_lin = 0.0_f64;
        let sp = unsafe { buf.as_ptr().add(HEADER_BYTES) as *const u32 };
        for i in 0..m {
            let r = u32::from_le(unsafe { sp.add(i).read_unaligned() });
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
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, m as f64)
    }
}

/// Pure 8-byte block-min filtered `ELL(2,24)` ablation baseline (`E-01`):
/// `[8B Header | 8B min_u (B=8 x 128 regs) | m*4 B regs]` = `4,112 B` at p=10 (+8 B over ELL(2,24)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ell24Filtered8B {
    pub p: u32,
    pub m: usize,
    pub blk_shift: u32,
    pub blk_size: usize,
    pub buf: Vec<u8>,
}

impl Ell24Filtered8B {
    pub const D: u32 = 24;
    pub const B: usize = 8;

    pub fn new(p: u32) -> Self {
        assert!((3..=20).contains(&p));
        let m = 1usize << p;
        let blk_size = m / Self::B;
        let blk_shift = p - 3;
        let mut buf = vec![0u8; HEADER_BYTES + Self::B + m * 4];
        let hdr = SketchHeader::new(13, p as u8, Self::B as u8, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        Self { p, m, blk_shift, blk_size, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    fn get_reg(&self, idx: usize) -> u32 {
        let ptr = unsafe { self.buf.as_ptr().add(HEADER_BYTES + Self::B + (idx << 2)) as *const u32 };
        u32::from_le(unsafe { ptr.read_unaligned() })
    }

    #[inline(always)]
    fn set_reg(&mut self, idx: usize, val: u32) {
        let ptr = unsafe { self.buf.as_mut_ptr().add(HEADER_BYTES + Self::B + (idx << 2)) as *mut u32 };
        unsafe { ptr.write_unaligned(val.to_le()); }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        let blk = idx >> self.blk_shift;
        let mu = unsafe { *self.buf.get_unchecked(HEADER_BYTES + blk) } as u32;
        if k + Self::D < mu {
            return true;
        }
        let cur = self.get_reg(idx);
        let old_u = cur >> Self::D;
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            self.set_reg(idx, next);
            let new_u = next >> Self::D;
            if old_u == mu && new_u > old_u {
                let start = blk * self.blk_size;
                let mut mmin = u32::MAX;
                for j in start..(start + self.blk_size) {
                    let u = self.get_reg(j) >> Self::D;
                    if u < mmin {
                        mmin = u;
                        if mmin == mu {
                            break;
                        }
                    }
                }
                unsafe { *self.buf.get_unchecked_mut(HEADER_BYTES + blk) = mmin.min(255) as u8; }
            }
        }
        false
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for i in 0..self.m {
            let mrg = ell_merge_reg(self.get_reg(i), other.get_reg(i), Self::D);
            self.set_reg(i, mrg);
        }
        for blk in 0..Self::B {
            let start = blk * self.blk_size;
            let mut mmin = u32::MAX;
            for j in start..(start + self.blk_size) {
                let u = self.get_reg(j) >> Self::D;
                if u < mmin {
                    mmin = u;
                }
            }
            self.buf[HEADER_BYTES + blk] = mmin.min(255) as u8;
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
}

/// 16-block (`64` regs/blk) counted block-min filtered `ELL(2,24)` ablation baseline (`E-01`):
/// `[8B Header | 16B min_u | 16B min_cnt | m*4 B regs]` = `4,136 B` at p=10 (+32 B; rescans only when last min reg increments).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ell24Filtered16B {
    pub p: u32,
    pub m: usize,
    pub blk_shift: u32,
    pub blk_size: usize,
    pub buf: Vec<u8>,
}

impl Ell24Filtered16B {
    pub const D: u32 = 24;
    pub const B: usize = 16;
    const META_BYTES: usize = Self::B * 2; // 16B min_u + 16B min_cnt

    pub fn new(p: u32) -> Self {
        assert!((4..=20).contains(&p));
        let m = 1usize << p;
        let blk_size = m / Self::B;
        assert!(blk_size <= 255, "blk_size must fit in u8 min_cnt for p<=11");
        let blk_shift = p - 4;
        let mut buf = vec![0u8; HEADER_BYTES + Self::META_BYTES + m * 4];
        let hdr = SketchHeader::new(14, p as u8, Self::B as u8, m as u32);
        buf[0..8].copy_from_slice(&hdr.to_u64().to_le_bytes());
        for b in 0..Self::B {
            buf[HEADER_BYTES + Self::B + b] = blk_size as u8;
        }
        Self { p, m, blk_shift, blk_size, buf }
    }

    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    fn get_reg(&self, idx: usize) -> u32 {
        let ptr = unsafe { self.buf.as_ptr().add(HEADER_BYTES + Self::META_BYTES + (idx << 2)) as *const u32 };
        u32::from_le(unsafe { ptr.read_unaligned() })
    }

    #[inline(always)]
    fn set_reg(&mut self, idx: usize, val: u32) {
        let ptr = unsafe { self.buf.as_mut_ptr().add(HEADER_BYTES + Self::META_BYTES + (idx << 2)) as *mut u32 };
        unsafe { ptr.write_unaligned(val.to_le()); }
    }

    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        let blk = idx >> self.blk_shift;
        let mu = unsafe { *self.buf.get_unchecked(HEADER_BYTES + blk) } as u32;
        if k + Self::D < mu {
            return true;
        }
        let cur = self.get_reg(idx);
        let old_u = cur >> Self::D;
        let next = ell_update_reg(cur, k, Self::D);
        if next != cur {
            self.set_reg(idx, next);
            let new_u = next >> Self::D;
            if old_u == mu && new_u > old_u {
                let cnt_ptr = unsafe { self.buf.get_unchecked_mut(HEADER_BYTES + Self::B + blk) };
                *cnt_ptr = cnt_ptr.saturating_sub(1);
                if *cnt_ptr == 0 {
                    let start = blk * self.blk_size;
                    let mut mmin = u32::MAX;
                    let mut cnt = 0u8;
                    for j in start..(start + self.blk_size) {
                        let u = self.get_reg(j) >> Self::D;
                        if u < mmin {
                            mmin = u;
                            cnt = 1;
                        } else if u == mmin {
                            cnt += 1;
                        }
                    }
                    self.buf[HEADER_BYTES + blk] = mmin.min(255) as u8;
                    self.buf[HEADER_BYTES + Self::B + blk] = cnt;
                }
            }
        }
        false
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        for i in 0..self.m {
            let mrg = ell_merge_reg(self.get_reg(i), other.get_reg(i), Self::D);
            self.set_reg(i, mrg);
        }
        for blk in 0..Self::B {
            let start = blk * self.blk_size;
            let mut mmin = u32::MAX;
            let mut cnt = 0u8;
            for j in start..(start + self.blk_size) {
                let u = self.get_reg(j) >> Self::D;
                if u < mmin {
                    mmin = u;
                    cnt = 1;
                } else if u == mmin {
                    cnt += 1;
                }
            }
            self.buf[HEADER_BYTES + blk] = mmin.min(255) as u8;
            self.buf[HEADER_BYTES + Self::B + blk] = cnt;
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

    pub fn contains_sv(&self, entry_plus_1: u32) -> bool {
        let mask = self.table.len() - 1;
        let mut pos = (entry_plus_1.wrapping_mul(0x9e37_79b1) as usize) & mask;
        loop {
            let cur = self.table[pos];
            if cur == entry_plus_1 {
                return true;
            }
            if cur == 0 {
                return false;
            }
            pos = (pos + 1) & mask;
        }
    }

    pub fn remove_sv(&mut self, entry_plus_1: u32) -> bool {
        let mask = self.table.len() - 1;
        let mut pos = (entry_plus_1.wrapping_mul(0x9e37_79b1) as usize) & mask;
        loop {
            let cur = self.table[pos];
            if cur == 0 {
                return false;
            }
            if cur == entry_plus_1 {
                // Backward-shift deletion (Knuth Algorithm R): zero tombstones, exact probe invariant.
                let mut hole = pos;
                let mut next = (hole + 1) & mask;
                while self.table[next] != 0 {
                    let cand = self.table[next];
                    let ideal = (cand.wrapping_mul(0x9e37_79b1) as usize) & mask;
                    if (hole.wrapping_sub(ideal) & mask) < (next.wrapping_sub(ideal) & mask) {
                        self.table[hole] = cand;
                        hole = next;
                    }
                    next = (next + 1) & mask;
                }
                self.table[hole] = 0;
                self.num_sv -= 1;
                return true;
            }
            pos = (pos + 1) & mask;
        }
    }

    #[inline]
    pub fn get_bit(&self, row: usize, col: u8) -> bool {
        if col < self.fi_col {
            let entry = (((row as u32) << 6) | (col as u32)) + 1;
            !self.contains_sv(entry)
        } else if col < self.fi_col + 8 {
            ((self.window[row] >> (col - self.fi_col)) & 1) != 0
        } else {
            let entry = (((row as u32) << 6) | (col as u32)) + 1;
            self.contains_sv(entry)
        }
    }

    #[inline]
    fn canonical_fi_col(&self) -> u8 {
        (((self.num_coupons / (self.m as u64)) as i32) - 4).clamp(0, 54) as u8
    }

    /// Losslessly advances `fi_col` to `target_offset`, retaining all missing coupons `< target_offset`
    /// as negative surprises in `table` and all present coupons `>= target_offset + 8` as positive surprises.
    fn slide_window_to(&mut self, target_offset: u8) {
        if target_offset <= self.fi_col {
            return;
        }
        let old_fi = self.fi_col;
        let new_fi = target_offset;
        let shift = (new_fi - old_fi) as u32;
        let old_window = std::mem::replace(&mut self.window, vec![0u8; self.m]);
        let old_table = std::mem::replace(&mut self.table, vec![0u32; (self.m / 4).max(16)]);
        self.fi_col = new_fi;
        self.num_sv = 0;

        // 1. Keep surviving window bits in [new_fi, old_fi + 8).
        if shift < 8 {
            for r in 0..self.m {
                self.window[r] = old_window[r] >> shift;
            }
        }

        // 2. Record missing bits in [old_fi, min(new_fi, old_fi + 8)) as negative surprises.
        let win_drop_end = new_fi.min(old_fi + 8);
        for c in old_fi..win_drop_end {
            let bit = 1u8 << (c - old_fi);
            for r in 0..self.m {
                if (old_window[r] & bit) == 0 {
                    let entry = (((r as u32) << 6) | (c as u32)) + 1;
                    self.insert_sv(entry);
                }
            }
        }

        // 3. If new_fi jumps past old_fi + 8, columns in [old_fi + 8, new_fi) transition from
        //    positive-surprise encoding to negative-surprise encoding.
        let gap_start = old_fi + 8;
        let mut gap_present: Vec<bool> = if new_fi > gap_start {
            vec![false; ((new_fi - gap_start) as usize) * self.m]
        } else {
            Vec::new()
        };

        // 4. Re-classify every entry from old_table.
        for &e in &old_table {
            if e != 0 {
                let raw = e - 1;
                let row = (raw >> 6) as usize;
                let col = (raw & 63) as u8;
                if col < old_fi {
                    // Already a negative surprise below old_fi (< new_fi).
                    self.insert_sv(e);
                } else if col >= gap_start {
                    // Positive surprise from old_table.
                    if col < new_fi {
                        let idx = ((col - gap_start) as usize) * self.m + row;
                        gap_present[idx] = true;
                    } else if col < new_fi + 8 {
                        self.window[row] |= 1u8 << (col - new_fi);
                    } else {
                        self.insert_sv(e);
                    }
                }
            }
        }

        if new_fi > gap_start {
            for c in gap_start..new_fi {
                let base_idx = ((c - gap_start) as usize) * self.m;
                for r in 0..self.m {
                    if !gap_present[base_idx + r] {
                        let entry = (((r as u32) << 6) | (c as u32)) + 1;
                        self.insert_sv(entry);
                    }
                }
            }
        }
    }

    #[inline]
    pub fn add_hash_64(&mut self, h: u64) {
        let row = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.lg_k - 1);
        let col = (h | sentinel).leading_zeros().min(62) as u8; // 0-based column
        let added = if col < self.fi_col {
            // Exact negative-surprise tracking: inserting a previously missing coupon removes it.
            let entry = (((row as u32) << 6) | (col as u32)) + 1;
            self.remove_sv(entry)
        } else if col < self.fi_col + 8 {
            let bit = 1u8 << (col - self.fi_col);
            if (self.window[row] & bit) == 0 {
                self.window[row] |= bit;
                true
            } else {
                false
            }
        } else {
            let entry = (((row as u32) << 6) | (col as u32)) + 1;
            self.insert_sv(entry)
        };
        if added {
            self.num_coupons += 1;
            if self.fi_col < 54 && self.num_coupons >= ((self.fi_col as u64) + 5) * (self.m as u64) {
                let target = self.canonical_fi_col();
                self.slide_window_to(target);
            }
        }
    }

    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.lg_k, other.lg_k);
        // 1. Align self's window up to max(self.fi_col, other.fi_col) losslessly.
        let base = self.fi_col.max(other.fi_col);
        if base > self.fi_col {
            self.slide_window_to(base);
        }

        // 2. Resolve negative surprises below `base`: any bit missing in `self` but present in `other`
        //    becomes 1 in the union and increments `num_coupons`.
        if base > 0 && self.num_sv > 0 {
            let mut resolved: Vec<u32> = Vec::new();
            for &e in &self.table {
                if e != 0 {
                    let raw = e - 1;
                    let r = (raw >> 6) as usize;
                    let c = (raw & 63) as u8;
                    if c < base && other.get_bit(r, c) {
                        resolved.push(e);
                    }
                }
            }
            for e in resolved {
                if self.remove_sv(e) {
                    self.num_coupons += 1;
                }
            }
        }

        // 3. Union `other.window` bits that fall in [base, base + 8).
        let sh2 = (base - other.fi_col) as u32;
        if sh2 < 8 {
            for r in 0..self.m {
                let ow = other.window[r] >> sh2;
                let newly_set = ow & !self.window[r];
                if newly_set != 0 {
                    self.window[r] |= ow;
                    self.num_coupons += newly_set.count_ones() as u64;
                }
            }
        }

        // 4. Union positive surprises from `other.table` (entries with c >= other.fi_col + 8).
        let other_pos_min = other.fi_col + 8;
        for &e in &other.table {
            if e != 0 {
                let raw = e - 1;
                let r = (raw >> 6) as usize;
                let c = (raw & 63) as u8;
                if c >= other_pos_min {
                    if c >= base && c < base + 8 {
                        let bit = 1u8 << (c - base);
                        if (self.window[r] & bit) == 0 {
                            self.window[r] |= bit;
                            self.num_coupons += 1;
                        }
                    } else if c >= base + 8 {
                        if self.insert_sv(e) {
                            self.num_coupons += 1;
                        }
                    }
                }
            }
        }

        // 5. Advance `fi_col` to the exact canonical offset determined by merged `num_coupons`.
        let target = self.canonical_fi_col();
        if target > self.fi_col {
            self.slide_window_to(target);
        }
    }

    /// Local streaming Rice-coded CPC wire serialization (column-wise Rice/Golomb delta coding of
    /// the 8-column sliding window + canonically sorted negative (`col < fi_col`) and positive
    /// (`col >= fi_col + 8`) surprising values).
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
                assert!(
                    k_rare <= u16::MAX as usize,
                    "CpcSketch wire serialization k_rare={} exceeds u16::MAX",
                    k_rare
                );
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
        assert!(
            svs.len() <= u16::MAX as usize,
            "CpcSketch wire serialization svs.len()={} exceeds u16::MAX",
            svs.len()
        );
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

    /// Exact ML estimator over all observed matrix entries `(row, col)` across all columns `0..km`
    /// (including negative surprises below `fi_col`, window bits, and positive surprises above `fi_col + 7`).
    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        let km = pmf.k_reach;
        let fi = self.fi_col as usize;
        let base_lvl = fi + 1; // 1-based level
        let mut col_ones = [0u32; 64];
        // Columns 0..fi default to all 1s, decremented by negative surprises stored in `table`.
        for c in 0..fi {
            let lvl = c + 1;
            if lvl <= km {
                col_ones[lvl] = self.m as u32;
            }
        }
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
                if lvl <= km {
                    if col < fi {
                        col_ones[lvl] -= 1;
                    } else if col >= fi + 8 {
                        col_ones[lvl] += 1;
                    }
                }
            }
        }
        for lvl in 1..=km {
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
    pub fn update_ee_thresh(&mut self) {
        // Early-exit when `abs_nlz < global_nlz` (bit 0 = global_nlz when e=1 or e=2).
        // `abs_nlz = (h.leading_zeros() + 1).min(63) < global_nlz`
        // <=> `h >= 1 << (65 - global_nlz)` (for 2 <= global_nlz <= 64)
        // <=> `h > (1 << (65 - global_nlz)) - 1`.
        self.ee_thresh = if self.global_nlz < 2 {
            u64::MAX
        } else if self.global_nlz >= 65 {
            0
        } else {
            (1u64 << (65 - self.global_nlz as u32)) - 1
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

    /// Fast single-64-bit hash update with 1-compare `ee_thresh` early exit and faithful
    /// Bushnell (arXiv:2608.12575) unsigned modulo bucketing `r = (h % m) as usize`.
    #[inline(always)]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        if h > self.ee_thresh {
            return true;
        }
        let abs_nlz = (h.leading_zeros() + 1).min(63) as u8;
        let r = (h % (self.m as u64)) as usize;
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

    /// Count of unsigned 64-bit hashes `h in [0, hi)` satisfying `h % m == r` (`0 <= r < m`).
    #[inline(always)]
    fn count_mod_below(hi: u128, r: u128, m: u128) -> u128 {
        if hi <= r {
            0
        } else {
            ((hi - 1 - r) / m) + 1
        }
    }

    /// Exact integer preimage count `C(r, k)` of 64-bit hash words `h in [0, 2^64)` that land in
    /// bucket `r = (h % m) as usize` AND produce `abs_nlz = (h.leading_zeros() + 1).min(63) == k`.
    /// Sums to exactly `2^64` across all `r in 0..m` and `k in 1..=63`.
    #[inline(always)]
    pub fn joint_preimage_count(k: usize, r: usize, m: usize) -> u64 {
        debug_assert!(r < m && m > 0);
        if k == 0 || k > 63 {
            return 0;
        }
        let (lo, hi): (u128, u128) = if k <= 62 {
            (1u128 << (64 - k), 1u128 << (65 - k))
        } else {
            // k == 63: h.leading_zeros() >= 62 <=> h in [0, 4)
            (0, 4)
        };
        let m128 = m as u128;
        let r128 = r as u128;
        (Self::count_mod_below(hi, r128, m128) - Self::count_mod_below(lo, r128, m128)) as u64
    }

    /// Exact integer preimage count of 64-bit hashes landing in bucket `r = (h % m)` with
    /// `abs_nlz > u` (`u in 0..=63`), i.e., `sum_{k = u + 1}^{63} joint_preimage_count(k, r, m)`.
    #[inline(always)]
    pub fn tail_preimage_count(u: usize, r: usize, m: usize) -> u64 {
        debug_assert!(r < m && m > 0);
        if u >= 63 {
            0
        } else {
            Self::count_mod_below(1u128 << (64 - u), r as u128, m as u128) as u64
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

    /// Legacy approximate Pow2 Poisson ML estimator (retained only for unit-test comparison against
    /// `estimate_ml_exact_mod`; ignores per-bucket discrete modulo preimage counts at `2^64`).
    pub fn estimate_ml_approx_pow2(&self, pmf: &PreimagePmf) -> (f64, u32) {
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
                    zero_lin += pmf.tail[gnlz.min(km)];
                } else {
                    let u = (gnlz + rel_e).min(km);
                    if rel_e < Self::MAX_EXP as usize {
                        zero_lin += pmf.tail[u];
                        ones[u] += 1;
                    } else if u < km && u >= 2 {
                        ones[u - 1] += 1;
                    } else {
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

    /// Exact-modulo Poisson ML estimator using per-bucket integer preimage counts
    /// `C(r, k) = Self::joint_preimage_count(k, r, self.m)` and exact tails
    /// `T(r, u) = Self::tail_preimage_count(u, r, self.m)` (exact through `n = 2^64`).
    ///
    /// Note: Bushnell's original AVLL paper uses Wang mixing + HLDLC; we pair faithful
    /// `h % m` bucketing with exact modulo preimage counts `C(r, k)` under an unclamped
    /// Poisson likelihood (approximate for states affected by historical old-anchor clamping)
    /// under our unified 64-bit `wyhash` for controlled apples-to-apples comparison.
    pub fn estimate_ml_exact_mod(&self) -> (f64, u32) {
        const INV_2_64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
        let m = self.m;
        let m128 = m as u128;
        let gnlz = self.global_nlz as usize;

        let mut q_reg = [0u64; 64];
        let mut q_cens = [0u64; 64];
        for k in 1..=63usize {
            let len_reg: u128 = if k <= 62 { 1u128 << (64 - k) } else { 4 };
            q_reg[k] = (len_reg / m128) as u64;
            let len_cens: u128 = 1u128 << (65 - k);
            q_cens[k] = (len_cens / m128) as u64;
        }

        let mut reg_counts = [[0u32; 2]; 64];
        let mut cens_counts = [[0u32; 2]; 64];
        let mut zero_mass_u128: u128 = 0;

        let mut r = 0usize;
        for wi in 0..self.n_words {
            let mut w = self.words[wi];
            let cnt = (m - wi * 11).min(11);
            for _ in 0..cnt {
                let s = (w % 56) as u8;
                w /= 56;
                let rel_e = Self::state_exp(s) as usize;
                if rel_e == 0 {
                    zero_mass_u128 += Self::tail_preimage_count(gnlz.min(63), r, m) as u128;
                } else {
                    let u = (gnlz + rel_e).min(63);
                    if rel_e < Self::MAX_EXP as usize {
                        zero_mass_u128 += Self::tail_preimage_count(u, r, m) as u128;
                        let c = Self::joint_preimage_count(u, r, m);
                        if c > 0 {
                            reg_counts[u][(c > q_reg[u]) as usize] += 1;
                        }
                    } else {
                        // Right-censored at MAX_EXP: observed max_level >= u <=> arrival in [u, 63]
                        let c = Self::tail_preimage_count(u - 1, r, m);
                        if c > 0 {
                            cens_counts[u][(c > q_cens[u]) as usize] += 1;
                        }
                    }
                    if rel_e <= Self::HTL as usize {
                        let bm = Self::unpack_bitmap(s);
                        if rel_e >= 1 && u >= 2 {
                            let l1 = u - 1;
                            let c1 = Self::joint_preimage_count(l1, r, m);
                            if ((bm >> (rel_e - 1)) & 1) != 0 {
                                if c1 > 0 {
                                    reg_counts[l1][(c1 > q_reg[l1]) as usize] += 1;
                                }
                            } else {
                                zero_mass_u128 += c1 as u128;
                            }
                        }
                        if rel_e >= 2 && u >= 3 {
                            let l2 = u - 2;
                            let c2 = Self::joint_preimage_count(l2, r, m);
                            if ((bm >> (rel_e - 2)) & 1) != 0 {
                                if c2 > 0 {
                                    reg_counts[l2][(c2 > q_reg[l2]) as usize] += 1;
                                }
                            } else {
                                zero_mass_u128 += c2 as u128;
                            }
                        }
                    }
                }
                r += 1;
            }
        }

        // Collect compact non-zero (probability p_i, multiplicity c_i) terms
        let mut terms: [(f64, f64); 256] = [(0.0, 0.0); 256];
        let mut n_terms = 0usize;
        let mut total_ones: u64 = 0;
        for k in 1..=63usize {
            for hi in 0..2usize {
                let cnt = reg_counts[k][hi];
                if cnt > 0 {
                    let c = q_reg[k] + (hi as u64);
                    if c > 0 {
                        terms[n_terms] = ((c as f64) * INV_2_64, cnt as f64);
                        n_terms += 1;
                        total_ones += cnt as u64;
                    }
                }
                let ccnt = cens_counts[k][hi];
                if ccnt > 0 {
                    let c = q_cens[k] + (hi as u64);
                    if c > 0 {
                        terms[n_terms] = ((c as f64) * INV_2_64, ccnt as f64);
                        n_terms += 1;
                        total_ones += ccnt as u64;
                    }
                }
            }
        }
        if total_ones == 0 {
            return (0.0, 0);
        }
        if zero_mass_u128 == 0 {
            return (18_446_744_073_709_551_616.0, 0);
        }
        let z0 = (zero_mass_u128 as f64) * INV_2_64;
        let mut lo = 0.0_f64;
        let mut hi = (total_ones as f64) / z0;
        let mut n = 0.5 * hi;
        let mut iters = 0u32;

        for iter in 1..=64u32 {
            iters = iter;
            let mut g = -z0;
            let mut h = 0.0_f64;
            for &(p, c) in &terms[..n_terms] {
                let x = n * p;
                if x <= 40.0 {
                    let em1 = x.exp_m1();
                    if em1 > 0.0 {
                        let ex = 1.0 + em1;
                        let inv_em1 = 1.0 / em1;
                        g += c * p * inv_em1;
                        h -= c * p * p * ex * inv_em1 * inv_em1;
                    }
                } else {
                    let emx = (-x).exp();
                    let denom = 1.0 - emx;
                    g += c * p * emx / denom;
                    h -= c * p * p * emx / (denom * denom);
                }
            }
            if g > 0.0 {
                lo = n;
            } else {
                hi = n;
            }
            if (hi - lo) <= 1e-12 * n.max(1.0) || g.abs() <= 1e-15 * z0 {
                break;
            }
            let cand = if h < 0.0 && h.is_finite() {
                n - g / h
            } else {
                f64::NAN
            };
            if cand.is_finite() && cand > lo && cand < hi {
                n = cand;
            } else if lo > 0.0 {
                n = (lo * hi).sqrt();
            } else {
                n = 0.5 * hi;
            }
        }
        (n, iters)
    }

    /// Exact-modulo Poisson ML estimator on the AVLL state (ignores `_pmf` and uses exact `h % m` preimages).
    pub fn estimate_ml(&self, _pmf: &PreimagePmf) -> (f64, u32) {
        self.estimate_ml_exact_mod()
    }
}

// ---------------------------------------------------------------------------
// 9. Huffman-Bucket Sketch (Karppa 2026, arXiv:2603.10930)
//    Lossless Huffman-coded HLL buckets with global canonical codebook,
//    per-bucket r_min/c_min early exit, intra-bucket checkpoints, O(log n) rebuilds.
// ---------------------------------------------------------------------------

const HBS_BUCKET_SIZE: usize = 128;
const HBS_INLINE_WORDS: usize = 7; // 448 bits = 56 B inline/bucket (~326 b steady-state mean, >+8.5σ slack)
const HBS_CKPT_STRIDE: usize = 32; // checkpoints at j = 32, 64, 96 (max 31 sequential decodes per peek)
const HBS_NO_OV: u16 = u16::MAX;

#[derive(Clone, Debug)]
struct HbsCodebook {
    /// Bit-reversed canonical codewords for LSB-first bitstream packing (`len` low bits valid).
    code_rev: [u64; 64],
    /// Codeword bit length per rank symbol `0..64` (`1..=60`).
    code_len: [u8; 64],
    /// Fast 8-bit LSB-first prefix lookup table: `(sym, len)` when `len <= 8`, else `(0, 0)`.
    fast_lut: [(u8, u8); 256],
    /// Canonical symbol ordering for rare `len > 8` decoding.
    sorted_syms: [u8; 64],
    /// Number of symbols at each bit length `1..=64`.
    len_count: [u8; 65],
    max_len: u8,
}

impl HbsCodebook {
    fn from_code_lens(code_len: [u8; 64]) -> Self {
        let mut len_count = [0u8; 65];
        let mut max_len = 0u8;
        for &l in &code_len {
            assert!((1..=60).contains(&l), "HBS codeword length {} out of 1..=60", l);
            len_count[l as usize] += 1;
            if l > max_len {
                max_len = l;
            }
        }
        // Canonical ordering: sort by (length, symbol)
        let mut next_off = [0usize; 65];
        let mut acc = 0usize;
        for l in 1..=max_len as usize {
            next_off[l] = acc;
            acc += len_count[l] as usize;
        }
        let mut sorted_syms = [0u8; 64];
        for sym in 0..64usize {
            let l = code_len[sym] as usize;
            sorted_syms[next_off[l]] = sym as u8;
            next_off[l] += 1;
        }
        // Assign MSB-first canonical codes and bit-reverse for LSB-first bitstream
        let mut code_rev = [0u64; 64];
        let mut fast_lut = [(0u8, 0u8); 256];
        let mut code: u64 = 0;
        let mut idx = 0usize;
        for l in 1..=max_len as usize {
            let cnt = len_count[l] as usize;
            for _ in 0..cnt {
                let sym = sorted_syms[idx] as usize;
                idx += 1;
                let rev = code.reverse_bits() >> (64 - l);
                code_rev[sym] = rev;
                if l <= 8 {
                    let step = 1usize << l;
                    let mut p = rev as usize;
                    while p < 256 {
                        fast_lut[p] = (sym as u8, l as u8);
                        p += step;
                    }
                }
                code += 1;
            }
            code <<= 1;
        }
        Self {
            code_rev,
            code_len,
            fast_lut,
            sorted_syms,
            len_count,
            max_len,
        }
    }

    /// Build canonical Huffman codebook from the Poissonized HLL rank distribution at load `lambda = n/m`.
    fn from_lambda(lambda: f64) -> Self {
        let lam = lambda.max(0.0);
        let mut weights = [0.0_f64; 64];
        // Floor weights at 1e-11: Fibonacci bound guarantees max Huffman depth <= 55 (< 60) with <1e-8 bit entropy impact.
        const MIN_W: f64 = 1e-11;
        if lam <= 0.0 {
            weights[0] = 1.0;
            for k in 1..64 {
                weights[k] = MIN_W * (0.99_f64).powi(k as i32);
            }
        } else {
            weights[0] = (if lam > 700.0 { 0.0 } else { (-lam).exp() }).max(MIN_W);
            let mut pow2_neg = 0.5_f64; // 2^{-1}
            for k in 1..=62usize {
                let a = lam * pow2_neg; // lam * 2^{-k}
                // P(R = k) = exp(-a) - exp(-2a) = exp(-a) * (1 - exp(-a))
                let pk = if a > 700.0 {
                    0.0
                } else {
                    (-a).exp() * (-(-a).exp_m1())
                };
                weights[k] = pk.max(MIN_W);
                pow2_neg *= 0.5;
            }
            let a62 = lam * (2.0_f64).powi(-62);
            let p63 = if a62 > 700.0 { 1.0 } else { -(-a62).exp_m1() };
            weights[63] = p63.max(MIN_W);
        }

        // Standard Huffman tree over 64 symbols (63 merges, O(64^2) ~ 1.5 us, zero heap alloc)
        let mut parent = [0u16; 127];
        let mut active_w = [f64::INFINITY; 127];
        let mut is_active = [false; 127];
        for i in 0..64 {
            active_w[i] = weights[i];
            is_active[i] = true;
        }
        let mut next_node = 64usize;
        for _ in 0..63 {
            let mut m1 = usize::MAX;
            let mut m2 = usize::MAX;
            let mut w1 = f64::INFINITY;
            let mut w2 = f64::INFINITY;
            for i in 0..next_node {
                if !is_active[i] {
                    continue;
                }
                let w = active_w[i];
                if w < w1 {
                    w2 = w1;
                    m2 = m1;
                    w1 = w;
                    m1 = i;
                } else if w < w2 {
                    w2 = w;
                    m2 = i;
                }
            }
            is_active[m1] = false;
            is_active[m2] = false;
            parent[m1] = next_node as u16;
            parent[m2] = next_node as u16;
            active_w[next_node] = w1 + w2;
            is_active[next_node] = true;
            next_node += 1;
        }
        let root = (next_node - 1) as u16;
        let mut code_len = [0u8; 64];
        for sym in 0..64 {
            let mut cur = sym as u16;
            let mut depth = 0u8;
            while cur != root {
                cur = parent[cur as usize];
                depth += 1;
            }
            assert!(depth <= 60, "HBS Huffman depth {} exceeded 60 at lambda={}", depth, lam);
            code_len[sym] = depth;
        }
        Self::from_code_lens(code_len)
    }

    #[inline(always)]
    fn read_u64_at(words: &[u64], bit_pos: usize) -> u64 {
        let wi = bit_pos >> 6;
        let bo = bit_pos & 63;
        let mut v = if wi < words.len() { words[wi] >> bo } else { 0 };
        if bo != 0 && wi + 1 < words.len() {
            v |= words[wi + 1] << (64 - bo);
        }
        v
    }

    #[inline(always)]
    fn decode_one(&self, words: &[u64], bit_pos: usize) -> (u8, usize) {
        let v = Self::read_u64_at(words, bit_pos);
        let (sym, len) = self.fast_lut[(v & 0xFF) as usize];
        if len != 0 {
            return (sym, len as usize);
        }
        let msb = v.reverse_bits();
        let mut first_code: u64 = 0;
        let mut sym_off: usize = 0;
        for l in 1..=self.max_len as usize {
            let cnt = self.len_count[l] as u64;
            if l > 8 && cnt > 0 {
                let pref = msb >> (64 - l);
                if pref >= first_code && pref - first_code < cnt {
                    return (self.sorted_syms[sym_off + (pref - first_code) as usize], l);
                }
            }
            first_code = (first_code + cnt) << 1;
            sym_off += cnt as usize;
        }
        unreachable!("corrupt HBS Huffman bitstream")
    }

    /// Active RAM footprint of the global codebook + fast LUTs (Karppa §6.1 big-sketch tables).
    #[inline(always)]
    fn table_bytes() -> usize {
        // code_rev (512B) + code_len (64B) + fast_lut (512B) + sorted_syms (64B) + len_count (65B) + max_len (1B)
        512 + 64 + 512 + 64 + 65 + 1
    }
}

#[derive(Clone, Copy, Debug)]
struct HbsBucketMeta {
    bit_len: u16,
    ckpt: [u16; 3], // bit offsets of registers 32, 64, 96 inside bucket
    r_min: u8,
    c_min: u8,
    ov_slot: u16,   // HBS_NO_OV when stored inline in 448-bit cache-line slab
}

/// Faithful lossless Karppa (2026) Huffman-Bucket Sketch (`arXiv:2603.10930`).
///
/// Partitions `m = 2^p` HLL registers into `m / B` buckets (`B = min(m, 128)`) encoded with a
/// global canonical Huffman codebook rebuilt on `2x` cardinality doublings, with per-bucket
/// `(r_min, c_min)` early exit, 3 intra-bucket checkpoints (`j = 32, 64, 96`), and 448-bit inline
/// cache-line bucket storage + lossless dynamic overflow fallback.
#[derive(Clone, Debug)]
pub struct HbsSketch {
    pub p: u32,
    pub m: usize,
    pub bucket_size: usize,
    pub n_buckets: usize,
    codebook: HbsCodebook,
    inline_words: Vec<u64>,       // n_buckets * HBS_INLINE_WORDS
    meta: Vec<HbsBucketMeta>,     // n_buckets * 12 bytes
    overflows: Vec<Vec<u64>>,     // rare >448-bit bucket backing buffers
    harm_sum: f64,                // sum_{r=0}^{m-1} 2^{-reg[r]} for O(1) global n_hat maintenance
    n_hat: f64,
    n_old: f64,
    pub rebuild_count: u32,
}

impl HbsSketch {
    #[inline(always)]
    fn alpha_m(m: usize) -> f64 {
        match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / (m as f64)),
        }
    }

    #[inline(always)]
    fn pow2_neg(r: u8) -> f64 {
        f64::from_bits((1023u64 - (r as u64)) << 52)
    }

    pub fn new(p: u32) -> Self {
        assert!((2..=20).contains(&p));
        let m = 1usize << p;
        let bucket_size = m.min(HBS_BUCKET_SIZE);
        let n_buckets = m / bucket_size;
        let codebook = HbsCodebook::from_lambda(0.0);
        let inline_words = vec![0u64; n_buckets * HBS_INLINE_WORDS];
        let meta = vec![
            HbsBucketMeta {
                bit_len: 0,
                ckpt: [0; 3],
                r_min: 0,
                c_min: bucket_size as u8,
                ov_slot: HBS_NO_OV,
            };
            n_buckets
        ];
        let mut sk = Self {
            p,
            m,
            bucket_size,
            n_buckets,
            codebook,
            inline_words,
            meta,
            overflows: Vec::new(),
            harm_sum: m as f64,
            n_hat: 0.0,
            n_old: 0.0,
            rebuild_count: 0,
        };
        let zeros = vec![0u8; bucket_size];
        for b in 0..n_buckets {
            sk.encode_bucket_from_regs(b, &zeros);
        }
        sk
    }

    /// Construct an exact lossless `HbsSketch` directly from `m = 2^p` HLL register values.
    pub fn from_hll_registers(p: u32, regs: &[u8]) -> Self {
        let m = 1usize << p;
        assert_eq!(regs.len(), m);
        let bucket_size = m.min(HBS_BUCKET_SIZE);
        let n_buckets = m / bucket_size;
        let mut harm_sum = 0.0_f64;
        for &r in regs {
            harm_sum += Self::pow2_neg(r);
        }
        let raw_est = Self::alpha_m(m) * (m as f64) * (m as f64) / harm_sum;
        let z_cnt = regs.iter().filter(|&&r| r == 0).count();
        let n_hat = if z_cnt > 0 && raw_est <= 2.5 * (m as f64) {
            (m as f64) * ((m as f64) / (z_cnt as f64)).ln()
        } else {
            raw_est
        };
        let codebook = HbsCodebook::from_lambda(n_hat / (m as f64));
        let inline_words = vec![0u64; n_buckets * HBS_INLINE_WORDS];
        let meta = vec![
            HbsBucketMeta {
                bit_len: 0,
                ckpt: [0; 3],
                r_min: 0,
                c_min: bucket_size as u8,
                ov_slot: HBS_NO_OV,
            };
            n_buckets
        ];
        let mut sk = Self {
            p,
            m,
            bucket_size,
            n_buckets,
            codebook,
            inline_words,
            meta,
            overflows: Vec::new(),
            harm_sum,
            n_hat,
            n_old: n_hat,
            rebuild_count: 0,
        };
        for b in 0..n_buckets {
            let start = b * bucket_size;
            sk.encode_bucket_from_regs(b, &regs[start..start + bucket_size]);
        }
        sk
    }

    pub fn dist_spec_64(&self) -> DistSpec {
        DistSpec::Pow2 { w_bits: 64 - self.p, max_cap: 63 }
    }

    /// Active mergeable RAM bytes: 8B header + global codebook/LUTs + globals + inline bucket slab + bucket metadata + dynamic overflows.
    pub fn ram_bytes(&self) -> usize {
        let ov_bytes: usize = self.overflows.iter().map(|v| v.capacity() * 8).sum();
        HEADER_BYTES
            + HbsCodebook::table_bytes()
            + 24 // harm_sum, n_hat, n_old
            + self.inline_words.len() * 8
            + self.meta.len() * std::mem::size_of::<HbsBucketMeta>()
            + ov_bytes
    }

    /// Compact cold-wire bytes (Karppa §6.1 small-sketch wire representation):
    /// 8B header + 48B packed 6-bit canonical code lengths + exact concatenated Huffman bit payload.
    pub fn wire_bytes(&self) -> usize {
        let total_bits: usize = self.meta.iter().map(|m| m.bit_len as usize).sum();
        HEADER_BYTES + 48 + (total_bits + 7) / 8
    }

    #[inline(always)]
    fn bucket_words(&self, b: usize) -> &[u64] {
        let ov = self.meta[b].ov_slot;
        if ov == HBS_NO_OV {
            let off = b * HBS_INLINE_WORDS;
            &self.inline_words[off..off + HBS_INLINE_WORDS]
        } else {
            &self.overflows[ov as usize]
        }
    }

    #[inline(always)]
    fn write_bits_append(words: &mut [u64], pos: &mut usize, rev_bits: u64, len: usize) {
        let wi = *pos >> 6;
        let bo = *pos & 63;
        words[wi] |= rev_bits << bo;
        if bo + len > 64 {
            words[wi + 1] |= rev_bits >> (64 - bo);
        }
        *pos += len;
    }

    fn encode_bucket_from_regs(&mut self, b: usize, regs: &[u8]) {
        debug_assert_eq!(regs.len(), self.bucket_size);
        let mut total_bits = 0usize;
        let mut r_min = u8::MAX;
        let mut c_min = 0u8;
        let mut ckpt = [0u16; 3];
        for (j, &r) in regs.iter().enumerate() {
            if j == HBS_CKPT_STRIDE { ckpt[0] = total_bits as u16; }
            else if j == 2 * HBS_CKPT_STRIDE { ckpt[1] = total_bits as u16; }
            else if j == 3 * HBS_CKPT_STRIDE { ckpt[2] = total_bits as u16; }
            total_bits += self.codebook.code_len[r as usize] as usize;
            if r < r_min {
                r_min = r;
                c_min = 1;
            } else if r == r_min {
                c_min = c_min.saturating_add(1);
            }
        }
        let need_words = (total_bits + 63) / 64;
        if need_words <= HBS_INLINE_WORDS {
            let off = b * HBS_INLINE_WORDS;
            for w in &mut self.inline_words[off..off + HBS_INLINE_WORDS] {
                *w = 0;
            }
            let mut pos = 0usize;
            for &r in regs {
                let len = self.codebook.code_len[r as usize] as usize;
                let rev = self.codebook.code_rev[r as usize];
                Self::write_bits_append(&mut self.inline_words[off..off + HBS_INLINE_WORDS], &mut pos, rev, len);
            }
            // Note: leave any prior overflow Vec cleared if previously overflowed
            let old_ov = self.meta[b].ov_slot;
            if old_ov != HBS_NO_OV {
                self.overflows[old_ov as usize].clear();
                self.overflows[old_ov as usize].shrink_to_fit();
            }
            self.meta[b] = HbsBucketMeta {
                bit_len: total_bits as u16,
                ckpt,
                r_min,
                c_min,
                ov_slot: HBS_NO_OV,
            };
        } else {
            let mut buf = vec![0u64; need_words.max(HBS_INLINE_WORDS + 2)];
            let mut pos = 0usize;
            for &r in regs {
                let len = self.codebook.code_len[r as usize] as usize;
                let rev = self.codebook.code_rev[r as usize];
                Self::write_bits_append(&mut buf, &mut pos, rev, len);
            }
            let ov_slot = if self.meta[b].ov_slot != HBS_NO_OV {
                let s = self.meta[b].ov_slot as usize;
                self.overflows[s] = buf;
                self.meta[b].ov_slot
            } else {
                let s = self.overflows.len() as u16;
                self.overflows.push(buf);
                s
            };
            self.meta[b] = HbsBucketMeta {
                bit_len: total_bits as u16,
                ckpt,
                r_min,
                c_min,
                ov_slot,
            };
        }
    }

    #[inline]
    fn decode_bucket_into(&self, b: usize, out: &mut [u8]) {
        let words = self.bucket_words(b);
        let mut pos = 0usize;
        for slot in out.iter_mut().take(self.bucket_size) {
            let (sym, len) = self.codebook.decode_one(words, pos);
            *slot = sym;
            pos += len;
        }
    }

    /// Karppa Alg. 1 `Peek(b, j)`: returns `(rank, start_bit, code_len)` using intra-bucket checkpoints.
    #[inline]
    fn peek_pos(&self, b: usize, j: usize) -> (u8, usize, usize) {
        let m = self.meta[b];
        let words = self.bucket_words(b);
        let (mut cur_j, mut pos) = if j >= 3 * HBS_CKPT_STRIDE {
            (3 * HBS_CKPT_STRIDE, m.ckpt[2] as usize)
        } else if j >= 2 * HBS_CKPT_STRIDE {
            (2 * HBS_CKPT_STRIDE, m.ckpt[1] as usize)
        } else if j >= HBS_CKPT_STRIDE {
            (HBS_CKPT_STRIDE, m.ckpt[0] as usize)
        } else {
            (0usize, 0usize)
        };
        while cur_j < j {
            let (_, len) = self.codebook.decode_one(words, pos);
            pos += len;
            cur_j += 1;
        }
        let (sym, len) = self.codebook.decode_one(words, pos);
        (sym, pos, len)
    }

    #[inline]
    pub fn get_reg(&self, idx: usize) -> u8 {
        let b = idx / self.bucket_size;
        let j = idx % self.bucket_size;
        self.peek_pos(b, j).0
    }

    /// Replace bits `[start .. start + old_len)` in `words` with `(new_rev, new_len)`, shifting suffix cleanly.
    #[inline]
    fn replace_bits_slice(
        words: &mut [u64],
        old_total: usize,
        start: usize,
        old_len: usize,
        new_rev: u64,
        new_len: usize,
    ) {
        let tail_src = start + old_len;
        let tail_len = old_total - tail_src;
        let tail_dst = start + new_len;
        let new_total = tail_dst + tail_len;

        // Extract suffix [tail_src .. old_total) into tiny stack/temp buffer
        let tail_words = (tail_len + 63) / 64;
        let mut stack_tail = [0u64; 16];
        let mut heap_tail = Vec::new();
        let tail_buf: &mut [u64] = if tail_words <= 16 {
            &mut stack_tail[..tail_words]
        } else {
            heap_tail.resize(tail_words, 0);
            &mut heap_tail[..]
        };
        for w in 0..tail_words {
            let src_bit = tail_src + (w << 6);
            let rem = (old_total - src_bit).min(64);
            let val = HbsCodebook::read_u64_at(words, src_bit);
            let mask = if rem == 64 { u64::MAX } else { (1u64 << rem) - 1 };
            tail_buf[w] = val & mask;
        }

        // Clear words from bit `start` onward
        let swi = start >> 6;
        let sbo = start & 63;
        if sbo == 0 {
            words[swi] = 0;
        } else {
            words[swi] &= (1u64 << sbo) - 1;
        }
        for w in (swi + 1)..words.len() {
            words[w] = 0;
        }

        // Write new codeword at `start`
        let mut pos = start;
        Self::write_bits_append(words, &mut pos, new_rev, new_len);

        // Write saved suffix at `tail_dst`
        for w in 0..tail_words {
            let chunk_bits = (tail_len - (w << 6)).min(64);
            Self::write_bits_append(words, &mut pos, tail_buf[w], chunk_bits);
        }
        debug_assert_eq!(pos, new_total);
    }

    /// Karppa Alg. 1 `Poke(r, b, j)`: replace codeword in bucket `b` and update checkpoints & overflow if needed.
    fn poke_at(&mut self, b: usize, j: usize, start: usize, old_len: usize, new_r: u8) {
        let new_len = self.codebook.code_len[new_r as usize] as usize;
        let new_rev = self.codebook.code_rev[new_r as usize];
        let old_total = self.meta[b].bit_len as usize;
        let new_total = old_total + new_len - old_len;
        let need_words = (new_total + 63) / 64;

        let ov = self.meta[b].ov_slot;
        if ov == HBS_NO_OV {
            if need_words <= HBS_INLINE_WORDS {
                let off = b * HBS_INLINE_WORDS;
                Self::replace_bits_slice(
                    &mut self.inline_words[off..off + HBS_INLINE_WORDS],
                    old_total,
                    start,
                    old_len,
                    new_rev,
                    new_len,
                );
            } else {
                // Promote bucket to dynamic overflow buffer (lossless!)
                let off = b * HBS_INLINE_WORDS;
                let mut buf = vec![0u64; need_words.max(HBS_INLINE_WORDS + 2)];
                buf[..HBS_INLINE_WORDS].copy_from_slice(&self.inline_words[off..off + HBS_INLINE_WORDS]);
                Self::replace_bits_slice(&mut buf, old_total, start, old_len, new_rev, new_len);
                let slot = self.overflows.len() as u16;
                self.overflows.push(buf);
                self.meta[b].ov_slot = slot;
            }
        } else {
            let buf = &mut self.overflows[ov as usize];
            if buf.len() < need_words {
                buf.resize(need_words + 2, 0);
            }
            Self::replace_bits_slice(buf, old_total, start, old_len, new_rev, new_len);
        }

        self.meta[b].bit_len = new_total as u16;
        let delta = (new_len as i32) - (old_len as i32);
        if delta != 0 {
            if j < HBS_CKPT_STRIDE {
                self.meta[b].ckpt[0] = (self.meta[b].ckpt[0] as i32 + delta) as u16;
            }
            if j < 2 * HBS_CKPT_STRIDE {
                self.meta[b].ckpt[1] = (self.meta[b].ckpt[1] as i32 + delta) as u16;
            }
            if j < 3 * HBS_CKPT_STRIDE {
                self.meta[b].ckpt[2] = (self.meta[b].ckpt[2] as i32 + delta) as u16;
            }
        }
    }

    fn recompute_bucket_min(&mut self, b: usize) {
        let mut regs = [0u8; HBS_BUCKET_SIZE];
        self.decode_bucket_into(b, &mut regs[..self.bucket_size]);
        let mut r_min = u8::MAX;
        let mut c_min = 0u8;
        for &r in &regs[..self.bucket_size] {
            if r < r_min {
                r_min = r;
                c_min = 1;
            } else if r == r_min {
                c_min = c_min.saturating_add(1);
            }
        }
        self.meta[b].r_min = r_min;
        self.meta[b].c_min = c_min;
    }

    /// Karppa Alg. 1 lines 24-27: rebuild global Huffman codebook & re-encode all buckets on 2x doubling.
    pub fn rebuild_codebook(&mut self) {
        let mut all_regs = vec![0u8; self.m];
        for b in 0..self.n_buckets {
            let off = b * self.bucket_size;
            self.decode_bucket_into(b, &mut all_regs[off..off + self.bucket_size]);
        }
        self.codebook = HbsCodebook::from_lambda(self.n_hat / (self.m as f64));
        self.overflows.clear();
        for m in &mut self.meta {
            m.ov_slot = HBS_NO_OV;
        }
        for b in 0..self.n_buckets {
            let off = b * self.bucket_size;
            self.encode_bucket_from_regs(b, &all_regs[off..off + self.bucket_size]);
        }
        self.n_old = self.n_hat;
        self.rebuild_count += 1;
    }

    /// Karppa Alg. 1 `Insert(y)` under identical 64-bit hash decomposition to `Hll6Sketch`.
    /// Returns `true` when the `r <= r_min` bucket early exit fires.
    #[inline]
    pub fn add_hash_64(&mut self, h: u64) -> bool {
        let idx = (h & ((self.m as u64) - 1)) as usize;
        let sentinel = 1u64 << (self.p - 1);
        let lvl = ((h | sentinel).leading_zeros() + 1).min(63) as u8;
        let b = idx / self.bucket_size;
        if lvl <= self.meta[b].r_min {
            return true;
        }
        let j = idx % self.bucket_size;
        let (old_r, start, old_len) = self.peek_pos(b, j);
        if lvl <= old_r {
            return false;
        }
        self.poke_at(b, j, start, old_len, lvl);
        if old_r == self.meta[b].r_min {
            self.meta[b].c_min -= 1;
            if self.meta[b].c_min == 0 {
                self.recompute_bucket_min(b);
            }
        }
        self.harm_sum += Self::pow2_neg(lvl) - Self::pow2_neg(old_r);
        self.n_hat = Self::alpha_m(self.m) * (self.m as f64) * (self.m as f64) / self.harm_sum;
        let rebuild_thresh = (self.n_old * 2.0).max((self.m as f64) * 0.25);
        if self.n_hat >= rebuild_thresh {
            self.rebuild_codebook();
        }
        false
    }

    /// Karppa Alg. 1 `Merge(S1, S2)`: lossless decode -> pointwise max -> conditional codebook rebuild -> re-encode.
    pub fn merge_in_place(&mut self, other: &Self) {
        assert_eq!(self.p, other.p);
        let mut regs = vec![0u8; self.m];
        let mut tmp_a = [0u8; HBS_BUCKET_SIZE];
        let mut tmp_b = [0u8; HBS_BUCKET_SIZE];
        let mut harm_sum = 0.0_f64;
        for b in 0..self.n_buckets {
            self.decode_bucket_into(b, &mut tmp_a[..self.bucket_size]);
            other.decode_bucket_into(b, &mut tmp_b[..self.bucket_size]);
            let off = b * self.bucket_size;
            for j in 0..self.bucket_size {
                let r = tmp_a[j].max(tmp_b[j]);
                regs[off + j] = r;
                harm_sum += Self::pow2_neg(r);
            }
        }
        self.harm_sum = harm_sum;
        self.n_hat = Self::alpha_m(self.m) * (self.m as f64) * (self.m as f64) / harm_sum;
        let max_old = self.n_old.max(other.n_old);
        if self.n_hat >= (max_old * 2.0).max((self.m as f64) * 0.25) {
            self.codebook = HbsCodebook::from_lambda(self.n_hat / (self.m as f64));
            self.n_old = self.n_hat;
            self.rebuild_count += 1;
        } else if other.n_old > self.n_old {
            self.codebook = other.codebook.clone();
            self.n_old = other.n_old;
        }
        self.overflows.clear();
        for m in &mut self.meta {
            m.ov_slot = HBS_NO_OV;
        }
        for b in 0..self.n_buckets {
            let off = b * self.bucket_size;
            self.encode_bucket_from_regs(b, &regs[off..off + self.bucket_size]);
        }
    }

    /// Exact Ertl Poisson ML estimator over decoded HLL registers (bit-identical to `Hll6Sketch::estimate_ml`).
    pub fn estimate_ml(&self, pmf: &PreimagePmf) -> (f64, u32) {
        let mut ones = [0u32; 64];
        let mut zero_lin = 0.0_f64;
        let mut tmp = [0u8; HBS_BUCKET_SIZE];
        for b in 0..self.n_buckets {
            self.decode_bucket_into(b, &mut tmp[..self.bucket_size]);
            for &rv in &tmp[..self.bucket_size] {
                let u = rv as usize;
                zero_lin += pmf.tail[u];
                if u > 0 {
                    ones[u] += 1;
                }
            }
        }
        pmf.solve_mle(&ones[..=pmf.k_reach], zero_lin, self.m as f64)
    }

    /// Compact cold-wire serialization (8B header + 48B 6-bit code lengths + contiguous Huffman bitstream).
    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        let total_bits: usize = self.meta.iter().map(|m| m.bit_len as usize).sum();
        let payload_bytes = (total_bits + 7) / 8;
        out.clear();
        out.reserve(HEADER_BYTES + 48 + payload_bytes);
        let hdr = SketchHeader::new(9, self.p as u8, 0, self.m as u32);
        out.extend_from_slice(&hdr.to_u64().to_le_bytes());
        // Pack 64 code lengths at 6 bits each -> 48 bytes
        let mut bit_buf = 0u64;
        let mut bit_cnt = 0u32;
        for &l in &self.codebook.code_len {
            bit_buf |= ((l as u64) & 63) << bit_cnt;
            bit_cnt += 6;
            while bit_cnt >= 8 {
                out.push((bit_buf & 0xFF) as u8);
                bit_buf >>= 8;
                bit_cnt -= 8;
            }
        }
        // Concatenate all bucket codewords sequentially into a dense byte bitstream
        let mut cur_byte = 0u64;
        let mut cur_bits = 0u32;
        for b in 0..self.n_buckets {
            let blen = self.meta[b].bit_len as usize;
            let words = self.bucket_words(b);
            let mut rem = blen;
            let mut wi = 0usize;
            while rem > 0 {
                let take = rem.min(64) as u32;
                let mask = if take == 64 { u64::MAX } else { (1u64 << take) - 1 };
                let mut w = words[wi] & mask;
                let mut left = take;
                while left > 0 {
                    let room = (64 - cur_bits).min(left);
                    let cmask = if room == 64 { u64::MAX } else { (1u64 << room) - 1 };
                    cur_byte |= (w & cmask) << cur_bits;
                    cur_bits += room;
                    w = if room == 64 { 0 } else { w >> room };
                    left -= room;
                    while cur_bits >= 8 {
                        out.push((cur_byte & 0xFF) as u8);
                        cur_byte >>= 8;
                        cur_bits -= 8;
                    }
                }
                rem -= take as usize;
                wi += 1;
            }
        }
        if cur_bits > 0 {
            out.push((cur_byte & 0xFF) as u8);
        }
        debug_assert_eq!(out.len(), HEADER_BYTES + 48 + payload_bytes);
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES + 48 {
            return Err("HbsSketch buffer too short".to_string());
        }
        let hdr = SketchHeader::from_u64(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
        if hdr.magic != SketchHeader::MAGIC || hdr.kind != 9 {
            return Err("HbsSketch bad header".to_string());
        }
        let p = hdr.p_or_k as u32;
        if !(2..=20).contains(&p) {
            return Err(format!("HbsSketch invalid p={}", p));
        }
        let m = 1usize << p;
        let mut code_len = [0u8; 64];
        let mut bit_buf = 0u64;
        let mut bit_cnt = 0u32;
        let mut pos = HEADER_BYTES;
        for slot in &mut code_len {
            while bit_cnt < 6 {
                bit_buf |= (bytes[pos] as u64) << bit_cnt;
                pos += 1;
                bit_cnt += 8;
            }
            *slot = (bit_buf & 63) as u8;
            bit_buf >>= 6;
            bit_cnt -= 6;
        }
        let codebook = HbsCodebook::from_code_lens(code_len);
        let payload = &bytes[HEADER_BYTES + 48..];
        let mut words = vec![0u64; (payload.len() + 7) / 8 + 1];
        for (i, chunk) in payload.chunks(8).enumerate() {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            words[i] = u64::from_le_bytes(arr);
        }
        let mut regs = vec![0u8; m];
        let mut bit_pos = 0usize;
        for slot in &mut regs {
            let (sym, len) = codebook.decode_one(&words, bit_pos);
            *slot = sym;
            bit_pos += len;
        }
        let mut sk = Self::from_hll_registers(p, &regs);
        // Preserve wire codebook identically
        sk.codebook = codebook;
        sk.overflows.clear();
        for meta in &mut sk.meta {
            meta.ov_slot = HBS_NO_OV;
        }
        for b in 0..sk.n_buckets {
            let off = b * sk.bucket_size;
            sk.encode_bucket_from_regs(b, &regs[off..off + sk.bucket_size]);
        }
        Ok(sk)
    }
}
