//! Milestone M4: Real Apache DataFusion 55.3 SQL `GROUP BY` End-to-End Engine Integration
//!
//! Registers vectorized `AggregateUDFImpl` + `GroupsAccumulator` UDAFs for:
//!   - `osbmw5b_cd`       (OS-BMW-5b, g=512, k=24, 8 blks = 2,760 B/group, AVX-512 merge)
//!   - `osbmw5b16_cd`     (OS-BMW-5b, g=512, k=24, 16 blks = 8 + 16x344 = 5,512 B/group, equal-accuracy control)
//!   - `osbmw5b_mask_cd`  (OS-BMW-5b(k20+mask), g=512, k=20+32b mask, 8 blks = 2,760 B/group)
//!   - `osbmw256_cd`      (OS-BMW-5b-256B, g=384, k=16, 16 blks = 4,104 B/group, exact 4KB equal-RAM)
//!   - `ell24_cd`         (ExaLogLog(2,24), p=10, 4,104 B/group, EDBT'25 SOTA)
//!   - `ell20_cd`         (ExaLogLog(2,20), p=10, 3,592 B/group, EDBT'25)
//!   - `ull_cd`           (UltraLogLog, p=12, 4,104 B/group, PVLDB'24)
//!   - `hll6_cd`          (HLL++ 6-bit, p=12, 3,080 B/group)
//!   - `avll_cd`          (AVLL base-56, 512 words = 4,104 B/group, 2608.12575)
//!   - `approx_distinct`  (DataFusion 55.3 native built-in 16,384 B/group HLL)
//!
//! Executes real SQL:
//!   `SELECT group_id, <fn>(val) AS est_distinct FROM events GROUP BY group_id`
//! across:
//!   - W1_1T_16Kg_32M  (single-thread, target_partitions=1 on CPU 32: 16,384 groups x 32M rows)
//!   - W2_8T_8Kg_40M   (8-thread parallel Partial->Repartition->Final on CPUs 32-39: 8,192 groups x 40M rows)
//!   - W3_8T_Zipf_4Kg_40M (8-thread parallel Zipf-1.0 skewed groups on CPUs 32-39: 4,096 groups x 40M rows)
//!
//! Stage-consistency & physical memory instrumentation:
//!   - Per-group stage-consistency check: every (workload, method) is re-executed on the SAME Arrow
//!     buffers through a second `SessionContext` with a different `target_partitions`
//!     (W1: 1 vs 4 partitions; W2/W3: 1 vs 8 partitions), i.e. single-stage `AggregateMode::Single`
//!     vs multi-stage `Partial -> Repartition -> FinalPartitioned`. The per-group f64 estimate bit
//!     patterns are compared and the number of differing groups is reported (`stage_mismatch_groups`).
//!   - Real process peak RSS: before every timed run `malloc_trim(0)` + write `5` to
//!     `/proc/self/clear_refs` (resets VmHWM) and record VmRSS; after the run read VmHWM.
//!     `rss_hwm_delta_mb` = max over timed runs of (VmHWM_after - VmRSS_before).
//!   - `logical_state_mb` (renamed from `peak_state_mb`): live-group x logical state bytes
//!     (`tracked` for custom UDAFs, `budget_formula` for the built-in approx_distinct).

use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanArray, Float64Array, UInt32Array,
    UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result as DfResult, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionConfig;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, EmitTo, GroupsAccumulator, Signature, Volatility,
};
use datafusion::prelude::*;
use s1_v2_rust::baselines::{
    ell_hash_to_idx_k, AvllSketch, Ell20Sketch, Ell24Sketch, Hll6Sketch, UllSketch,
};
use s1_v2_rust::hash::{splitmix64, wyhash64, HEADER_BYTES};
use s1_v2_rust::osbmw_inplace::{
    HashMode5b, OsBmw256, OsBmw5b, OsBmw5bInPlace, OsBmw5bMask,
};
use s1_v2_rust::pmf::{DistSpec, PreimagePmf};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const SQL_HASH_SEED: u64 = 0x2027_5101_cafe_babe;

/// Unified trait bridging any fixed-size sketch in `s1_v2_rust` into DataFusion's `GroupsAccumulator`.
trait SqlSketch: Clone + Send + Sync + 'static {
    fn new_sketch() -> Self;
    fn add_hash(&mut self, h: u64);
    fn merge_serialized(&mut self, bytes: &[u8]);
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        let _ = scratch;
        self.merge_serialized(bytes);
    }
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        None
    }
    fn serialize_into(&self, out: &mut Vec<u8>);
    fn estimate(&self, pmf: &PreimagePmf) -> f64;
    fn state_bytes() -> usize;
    fn build_pmf() -> PreimagePmf;
}

impl SqlSketch for OsBmw5b {
    #[inline]
    fn new_sketch() -> Self {
        OsBmw5b::new(8, HashMode5b::Pow2)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            let other = OsBmw5b::from_bytes(bytes, HashMode5b::Pow2).expect("osbmw5b bytes");
            self.merge_in_place(&other).expect("osbmw5b merge");
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), scratch.buf.len(), "osbmw5b byte length mismatch");
            assert_eq!(&bytes[0..8], &scratch.buf[0..8], "osbmw5b header mismatch");
            scratch.buf.copy_from_slice(bytes);
            self.merge_in_place(scratch).expect("osbmw5b merge");
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 8 * OsBmw5b::BLOCK_BYTES // 2,760 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(OsBmw5b::new(8, HashMode5b::Pow2).dist_spec_64())
    }
}

/// OS-BMW-5b(k24) with 16 blocks (8 + 16 x 344 = 5,512 B/group): equal-accuracy control.
#[derive(Clone)]
struct OsBmw5b16(OsBmw5b);

impl OsBmw5b16 {
    const N_BLK: usize = 16;
}

impl SqlSketch for OsBmw5b16 {
    #[inline]
    fn new_sketch() -> Self {
        OsBmw5b16(OsBmw5b::new(Self::N_BLK, HashMode5b::Pow2))
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.0.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.0.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            let other = OsBmw5b::from_bytes(bytes, HashMode5b::Pow2).expect("osbmw5b16 bytes");
            assert_eq!(other.n_blk, Self::N_BLK, "osbmw5b16 block count mismatch");
            self.0.merge_in_place(&other).expect("osbmw5b16 merge");
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.0.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), scratch.0.buf.len(), "osbmw5b16 byte length mismatch");
            assert_eq!(&bytes[0..8], &scratch.0.buf[0..8], "osbmw5b16 header mismatch");
            scratch.0.buf.copy_from_slice(bytes);
            self.0.merge_in_place(&scratch.0).expect("osbmw5b16 merge");
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.0.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.0.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.0.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + Self::N_BLK * OsBmw5b::BLOCK_BYTES // 5,512 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(OsBmw5b::new(Self::N_BLK, HashMode5b::Pow2).dist_spec_64())
    }
}

impl SqlSketch for OsBmw5bMask {
    #[inline]
    fn new_sketch() -> Self {
        OsBmw5bMask::new(8, HashMode5b::Pow2)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            let other = OsBmw5bMask::from_bytes(bytes, HashMode5b::Pow2).expect("osbmw5bmask bytes");
            self.merge_in_place(&other).expect("osbmw5bmask merge");
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), scratch.buf.len(), "osbmw5bmask byte length mismatch");
            assert_eq!(&bytes[0..8], &scratch.buf[0..8], "osbmw5bmask header mismatch");
            scratch.buf.copy_from_slice(bytes);
            self.merge_in_place(scratch).expect("osbmw5bmask merge");
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 8 * OsBmw5bMask::BLOCK_BYTES // 2,760 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(OsBmw5bMask::new(8, HashMode5b::Pow2).dist_spec_64())
    }
}

impl SqlSketch for OsBmw256 {
    #[inline]
    fn new_sketch() -> Self {
        OsBmw256::new(16, HashMode5b::Div3)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            let other = OsBmw256::from_bytes(bytes, HashMode5b::Div3).expect("osbmw256 bytes");
            self.merge_in_place(&other).expect("osbmw256 merge");
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), scratch.buf.len(), "osbmw256 byte length mismatch");
            assert_eq!(&bytes[0..8], &scratch.buf[0..8], "osbmw256 header mismatch");
            scratch.buf.copy_from_slice(bytes);
            self.merge_in_place(scratch).expect("osbmw256 merge");
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 16 * OsBmw256::BLOCK_BYTES // 4,104 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(OsBmw256::new(16, HashMode5b::Div3).dist_spec_64())
    }
}

/// OS-BMW-5b-256B with zero-critical-path MulShift384 (`HashMode5b::MulShift384`, 4,104 B/group).
#[derive(Clone)]
struct OsBmw256Ms(OsBmw256);

impl SqlSketch for OsBmw256Ms {
    #[inline]
    fn new_sketch() -> Self {
        OsBmw256Ms(OsBmw256::new(16, HashMode5b::MulShift384))
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.0.add_hash_64_mulshift384(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.0.add_hash_64_mulshift384(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            let other = OsBmw256::from_bytes(bytes, HashMode5b::MulShift384).expect("osbmw256ms bytes");
            self.0.merge_in_place(&other).expect("osbmw256ms merge");
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.0.add_hash_64_mulshift384(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), scratch.0.buf.len(), "osbmw256ms byte length mismatch");
            assert_eq!(&bytes[0..8], &scratch.0.buf[0..8], "osbmw256ms header mismatch");
            scratch.0.buf.copy_from_slice(bytes);
            self.0.merge_in_place(&scratch.0).expect("osbmw256ms merge");
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.0.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.0.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.0.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 16 * OsBmw256::BLOCK_BYTES // 4,104 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(OsBmw256::new(16, HashMode5b::MulShift384).dist_spec_64())
    }
}

impl SqlSketch for Ell24Sketch {
    #[inline]
    fn new_sketch() -> Self {
        Ell24Sketch::new(10)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == HEADER_BYTES + 1024 * 4 {
            let mut other = Ell24Sketch::new(10);
            for (dst, chunk) in other.regs.iter_mut().zip(bytes[HEADER_BYTES..].chunks_exact(4)) {
                *dst = u32::from_le_bytes(chunk.try_into().unwrap());
            }
            self.merge_in_place(&other);
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == HEADER_BYTES + 1024 * 4 {
            for (dst, chunk) in scratch.regs.iter_mut().zip(bytes[HEADER_BYTES..].chunks_exact(4)) {
                *dst = u32::from_le_bytes(chunk.try_into().unwrap());
            }
            self.merge_in_place(scratch);
        }
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        self.serialize_bytes(out);
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 1024 * 4 // 4,104 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(Ell24Sketch::new(10).dist_spec_64())
    }
}

impl SqlSketch for Ell20Sketch {
    #[inline]
    fn new_sketch() -> Self {
        Ell20Sketch::new(10)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), self.total_bytes(), "Ell20 serialized length mismatch");
            let mut other = Ell20Sketch::new(10);
            other.buf[..bytes.len()].copy_from_slice(bytes);
            self.merge_in_place(&other);
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), self.total_bytes(), "Ell20 serialized length mismatch");
            scratch.buf[..bytes.len()].copy_from_slice(bytes);
            self.merge_in_place(scratch);
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        Ell20Sketch::new(10).total_bytes() // 3,592 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(Ell20Sketch::new(10).dist_spec_64())
    }
}

impl SqlSketch for UllSketch {
    #[inline]
    fn new_sketch() -> Self {
        UllSketch::new(12)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == self.buf.len() {
            let mut other = UllSketch::new(12);
            other.buf.copy_from_slice(bytes);
            self.merge_in_place(&other);
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == self.buf.len() {
            scratch.buf.copy_from_slice(bytes);
            self.merge_in_place(scratch);
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 4096 // 4,104 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(UllSketch::new(12).dist_spec_64())
    }
}

impl SqlSketch for Hll6Sketch {
    #[inline]
    fn new_sketch() -> Self {
        Hll6Sketch::new(12)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), self.total_bytes(), "Hll6 serialized length mismatch");
            let mut other = Hll6Sketch::new(12);
            other.buf[..bytes.len()].copy_from_slice(bytes);
            self.merge_in_place(&other);
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if !bytes.is_empty() {
            assert_eq!(bytes.len(), self.total_bytes(), "Hll6 serialized length mismatch");
            scratch.buf[..bytes.len()].copy_from_slice(bytes);
            self.merge_in_place(scratch);
        }
    }
    #[inline]
    fn as_serialized_slice(&self) -> Option<&[u8]> {
        Some(self.as_wire_bytes())
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.as_wire_bytes());
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        Hll6Sketch::new(12).total_bytes() // 3,080 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(Hll6Sketch::new(12).dist_spec_64())
    }
}

impl SqlSketch for AvllSketch {
    #[inline]
    fn new_sketch() -> Self {
        AvllSketch::new_words(512)
    }
    #[inline]
    fn add_hash(&mut self, h: u64) {
        self.add_hash_64(h);
    }
    fn merge_serialized(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == HEADER_BYTES + 512 * 8 {
            let mut other = AvllSketch::new_words(512);
            other.global_nlz = bytes[0];
            other.update_ee_thresh();
            for (dst, chunk) in other.words.iter_mut().zip(bytes[HEADER_BYTES..].chunks_exact(8)) {
                *dst = u64::from_le_bytes(chunk.try_into().unwrap());
            }
            let merged = self.merge(&other);
            *self = merged;
        }
    }
    fn merge_serialized_with_scratch(&mut self, bytes: &[u8], scratch: &mut Self) {
        if bytes.len() == 8 {
            self.add_hash_64(u64::from_le_bytes(bytes.try_into().unwrap()));
        } else if bytes.len() == HEADER_BYTES + 512 * 8 {
            scratch.global_nlz = bytes[0];
            scratch.update_ee_thresh();
            for (dst, chunk) in scratch.words.iter_mut().zip(bytes[HEADER_BYTES..].chunks_exact(8)) {
                *dst = u64::from_le_bytes(chunk.try_into().unwrap());
            }
            let merged = self.merge(scratch);
            *self = merged;
        }
    }
    fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(HEADER_BYTES + self.words.len() * 8);
        out.push(self.global_nlz);
        out.extend_from_slice(&[0u8; 7]);
        for &w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    fn estimate(&self, pmf: &PreimagePmf) -> f64 {
        self.estimate_ml(pmf).0
    }
    fn state_bytes() -> usize {
        HEADER_BYTES + 512 * 8 // 4,104 B
    }
    fn build_pmf() -> PreimagePmf {
        PreimagePmf::build(DistSpec::Pow2 { w_bits: 64, max_cap: 63 })
    }
}

// ---------------------------------------------------------------------------
// Generic Vectorized DataFusion GroupsAccumulator & Scalar Accumulator
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct PromoteStats {
    partial_dense: Arc<AtomicUsize>,
    partial_total: Arc<AtomicUsize>,
    final_dense: Arc<AtomicUsize>,
    final_total: Arc<AtomicUsize>,
}

impl PromoteStats {
    fn new() -> Self {
        Self {
            partial_dense: Arc::new(AtomicUsize::new(0)),
            partial_total: Arc::new(AtomicUsize::new(0)),
            final_dense: Arc::new(AtomicUsize::new(0)),
            final_total: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn reset(&self) {
        self.partial_dense.store(0, Ordering::SeqCst);
        self.partial_total.store(0, Ordering::SeqCst);
        self.final_dense.store(0, Ordering::SeqCst);
        self.final_total.store(0, Ordering::SeqCst);
    }
}

#[derive(Clone, Debug)]
struct StageTimers {
    enabled: bool,
    update_ns: Arc<AtomicU64>,
    merge_ns: Arc<AtomicU64>,
    eval_ns: Arc<AtomicU64>,
    state_ns: Arc<AtomicU64>,
}

impl StageTimers {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            update_ns: Arc::new(AtomicU64::new(0)),
            merge_ns: Arc::new(AtomicU64::new(0)),
            eval_ns: Arc::new(AtomicU64::new(0)),
            state_ns: Arc::new(AtomicU64::new(0)),
        }
    }
    fn reset(&self) {
        self.update_ns.store(0, Ordering::SeqCst);
        self.merge_ns.store(0, Ordering::SeqCst);
        self.eval_ns.store(0, Ordering::SeqCst);
        self.state_ns.store(0, Ordering::SeqCst);
    }
}

#[cfg(target_os = "linux")]
fn allowed_cpu_ids() -> Vec<usize> {
    let mut out = Vec::new();
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
                for part in rest.trim().split(',') {
                    if let Some((a, b)) = part.split_once('-') {
                        if let (Ok(lo), Ok(hi)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                            for c in lo..=hi {
                                out.push(c);
                            }
                        }
                    } else if let Ok(c) = part.trim().parse::<usize>() {
                        out.push(c);
                    }
                }
            }
        }
    }
    out
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpu_ids() -> Vec<usize> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn read_mean_cpu_freq_mhz(cpus: &[usize]) -> Option<f64> {
    let mut sum_khz = 0u64;
    let mut cnt = 0u64;
    for &c in cpus {
        let p = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", c);
        if let Ok(s) = fs::read_to_string(&p) {
            if let Ok(khz) = s.trim().parse::<u64>() {
                sum_khz += khz;
                cnt += 1;
            }
        }
    }
    if cnt > 0 { Some((sum_khz as f64) / (cnt as f64) / 1000.0) } else { None }
}

#[cfg(not(target_os = "linux"))]
fn read_mean_cpu_freq_mhz(_cpus: &[usize]) -> Option<f64> {
    None
}

struct SketchGroupsAcc<S: SqlSketch> {
    groups: Vec<S>,
    merge_scratch: Option<S>,
    pmf: Arc<PreimagePmf>,
    live_bytes_tracker: Arc<AtomicUsize>,
    peak_bytes_tracker: Arc<AtomicUsize>,
    promote_stats: PromoteStats,
    stage_timers: StageTimers,
    reported_bytes: usize,
}

impl<S: SqlSketch> SketchGroupsAcc<S> {
    fn new(
        pmf: Arc<PreimagePmf>,
        live_bytes_tracker: Arc<AtomicUsize>,
        peak_bytes_tracker: Arc<AtomicUsize>,
        promote_stats: PromoteStats,
        stage_timers: StageTimers,
    ) -> Self {
        Self {
            groups: Vec::new(),
            merge_scratch: None,
            pmf,
            live_bytes_tracker,
            peak_bytes_tracker,
            promote_stats,
            stage_timers,
            reported_bytes: 0,
        }
    }

    #[inline]
    fn ensure_groups(&mut self, total_num_groups: usize) {
        if total_num_groups > self.groups.len() {
            let added = total_num_groups - self.groups.len();
            self.groups.resize_with(total_num_groups, S::new_sketch);
            let add_bytes = added * S::state_bytes();
            self.reported_bytes += add_bytes;
            let now = self.live_bytes_tracker.fetch_add(add_bytes, Ordering::Relaxed) + add_bytes;
            self.peak_bytes_tracker.fetch_max(now, Ordering::Relaxed);
        }
    }
}

impl<S: SqlSketch> Drop for SketchGroupsAcc<S> {
    fn drop(&mut self) {
        if self.reported_bytes > 0 {
            self.live_bytes_tracker.fetch_sub(self.reported_bytes, Ordering::Relaxed);
            self.reported_bytes = 0;
        }
    }
}

impl<S: SqlSketch> GroupsAccumulator for SketchGroupsAcc<S> {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> DfResult<()> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        self.ensure_groups(total_num_groups);
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        let vals = arr.values();
        match opt_filter {
            None => {
                if arr.null_count() == 0 {
                    for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                        let h = wyhash64(v, SQL_HASH_SEED);
                        self.groups[gid].add_hash(h);
                    }
                } else {
                    for (row, &gid) in group_indices.iter().enumerate() {
                        if arr.is_valid(row) {
                            let h = wyhash64(vals[row], SQL_HASH_SEED);
                            self.groups[gid].add_hash(h);
                        }
                    }
                }
            }
            Some(filter) => {
                for (row, &gid) in group_indices.iter().enumerate() {
                    if filter.is_valid(row) && filter.value(row) && arr.is_valid(row) {
                        let h = wyhash64(vals[row], SQL_HASH_SEED);
                        self.groups[gid].add_hash(h);
                    }
                }
            }
        }
        if let Some(t0) = t_stage {
            self.stage_timers.update_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> DfResult<()> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        self.ensure_groups(total_num_groups);
        let states = values[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| DataFusionError::Internal("expected BinaryArray".to_string()))?;
        let scratch = self.merge_scratch.get_or_insert_with(S::new_sketch);
        for (row, &gid) in group_indices.iter().enumerate() {
            if states.is_valid(row) {
                self.groups[gid].merge_serialized_with_scratch(states.value(row), scratch);
            }
        }
        if let Some(t0) = t_stage {
            self.stage_timers.merge_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> DfResult<ArrayRef> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        let emitted = emit_to.take_needed(&mut self.groups);
        let n_emit = emitted.len();
        self.promote_stats.final_dense.fetch_add(n_emit, Ordering::Relaxed);
        self.promote_stats.final_total.fetch_add(n_emit, Ordering::Relaxed);
        let freed = n_emit * S::state_bytes();
        let out: Float64Array = emitted.into_iter().map(|s| Some(s.estimate(&self.pmf))).collect();
        self.reported_bytes = self.reported_bytes.saturating_sub(freed);
        self.live_bytes_tracker.fetch_sub(freed, Ordering::Relaxed);
        if let Some(t0) = t_stage {
            self.stage_timers.eval_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(Arc::new(out))
    }

    fn state(&mut self, emit_to: EmitTo) -> DfResult<Vec<ArrayRef>> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        let emitted = emit_to.take_needed(&mut self.groups);
        let n_emit = emitted.len();
        self.promote_stats.partial_dense.fetch_add(n_emit, Ordering::Relaxed);
        self.promote_stats.partial_total.fetch_add(n_emit, Ordering::Relaxed);
        let freed = n_emit * S::state_bytes();
        let mut builder = BinaryBuilder::with_capacity(emitted.len(), emitted.len() * S::state_bytes());
        let mut scratch = Vec::new();
        for s in emitted {
            if let Some(slice) = s.as_serialized_slice() {
                builder.append_value(slice);
            } else {
                s.serialize_into(&mut scratch);
                builder.append_value(&scratch);
            }
            drop(s);
        }
        self.reported_bytes = self.reported_bytes.saturating_sub(freed);
        self.live_bytes_tracker.fetch_sub(freed, Ordering::Relaxed);
        let res = Arc::new(builder.finish());
        if let Some(t0) = t_stage {
            self.stage_timers.state_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(vec![res])
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> DfResult<Vec<ArrayRef>> {
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        let mut builder = BinaryBuilder::with_capacity(arr.len(), arr.len() * 8);
        for row in 0..arr.len() {
            let ok = arr.is_valid(row) && opt_filter.map_or(true, |f| f.is_valid(row) && f.value(row));
            if ok {
                let h = wyhash64(arr.value(row), SQL_HASH_SEED);
                builder.append_value(h.to_le_bytes());
            } else {
                builder.append_value([]);
            }
        }
        Ok(vec![Arc::new(builder.finish())])
    }

    fn size(&self) -> usize {
        self.groups.len() * S::state_bytes()
    }
}

// ---------------------------------------------------------------------------
// V3/V4 Symmetric Exact-Cell Homomorphic Sparse + Flat Slab GroupsAccumulator
// ---------------------------------------------------------------------------
// Applied identically to OS-BMW, ELL(2,24) (cf. Ertl §4.3 sparse replay), and ULL:
//   - Sparse groups store exact u32 cells/tokens in 128-byte pooled chunks (`CHUNK_CAP = 30`
//     cells/chunk) inside a single contiguous `sparse_pool: Vec<SparseChunk>`.
//   - Dense groups live back-to-back inside a single contiguous `dense_slab: Vec<u8>` and run
//     zero-copy flat-slice updates & merges directly in-place.
//   - Because OS-BMW, ELL(2,24), and ULL are all exact join-semilattices, replaying exact
//     u32 tokens into Dense produces bit-for-bit identical canonical dense bytes & estimates
//     across ANY partitioning (`p=1` vs `p=4/8/16/32` -> 0 stage mismatch guaranteed!).
//   - Note on sparse wire framing: sparse `state()` appends exact tokens (`0xE5` + 4c B), so
//     merging a 1-token sparse frame (5B) with itself appends to 9B before compaction/promotion,
//     while canonical dense projection and ML evaluation are strictly idempotent!

const TAG_SINGLE_HASH: u8 = 0xE1;
const TAG_EXACT_CELL_SPARSE: u8 = 0xE5;
const NO_CHUNK: u32 = u32::MAX;
const DENSE_TAG: u32 = u32::MAX;
const SPARSE_CHUNK_CAP: usize = 30; // 4B next + 4B used + 30*4B cells = 128B exact (2 cache lines!)

trait FlatSlabSketchOps: Clone + Send + Sync + 'static {
    fn state_bytes(&self) -> usize;
    fn expected_hdr(&self) -> [u8; 8];
    fn init_flat_slice(&self, slice: &mut [u8]);
    fn decompose_hash_to_cell(&self, h: u64) -> u32;
    fn add_cell_flat_slice(&self, slice: &mut [u8], cell: u32);
    fn add_hash_64_flat_slice(&self, slice: &mut [u8], h: u64);
    fn merge_flat_slices(&self, dst: &mut [u8], src: &[u8]);
    fn estimate_ml_flat_slice(&self, slice: &[u8], pmf: &PreimagePmf) -> f64;
    fn debug_label(&self) -> &'static str;
}

#[derive(Clone, Copy)]
struct OsBmwSlabOps<const G: usize, const K: usize> {
    n_blk: usize,
    mode: HashMode5b,
    state_bytes: usize,
    expected_hdr: [u8; 8],
}

impl<const G: usize, const K: usize> OsBmwSlabOps<G, K> {
    fn new(n_blk: usize, mode: HashMode5b) -> Self {
        let state_bytes = OsBmw5bInPlace::<G, K, false>::state_bytes_for(n_blk);
        let mut hdr_probe = vec![0u8; state_bytes];
        OsBmw5bInPlace::<G, K, false>::init_flat_slice(&mut hdr_probe, n_blk, mode);
        let expected_hdr: [u8; 8] = hdr_probe[0..8].try_into().unwrap();
        Self {
            n_blk,
            mode,
            state_bytes,
            expected_hdr,
        }
    }
}

impl<const G: usize, const K: usize> FlatSlabSketchOps for OsBmwSlabOps<G, K> {
    #[inline(always)]
    fn state_bytes(&self) -> usize {
        self.state_bytes
    }
    #[inline(always)]
    fn expected_hdr(&self) -> [u8; 8] {
        self.expected_hdr
    }
    #[inline(always)]
    fn init_flat_slice(&self, slice: &mut [u8]) {
        OsBmw5bInPlace::<G, K, false>::init_flat_slice(slice, self.n_blk, self.mode);
    }
    #[inline(always)]
    fn decompose_hash_to_cell(&self, h: u64) -> u32 {
        let (blk, sub, lvl) =
            OsBmw5bInPlace::<G, K, false>::decompose_hash_64(h, self.n_blk, self.mode);
        debug_assert!(sub < 512 && lvl >= 1 && lvl <= 63);
        ((blk as u32) << 15) | ((sub as u32) << 6) | (lvl as u32)
    }
    #[inline(always)]
    fn add_cell_flat_slice(&self, slice: &mut [u8], cell: u32) {
        let lvl = (cell & 63) as u8;
        let sub = ((cell >> 6) & 511) as usize;
        let blk = (cell >> 15) as usize;
        OsBmw5bInPlace::<G, K, false>::add_level_flat_slice(slice, self.n_blk, blk, sub, lvl);
    }
    #[inline(always)]
    fn add_hash_64_flat_slice(&self, slice: &mut [u8], h: u64) {
        OsBmw5bInPlace::<G, K, false>::add_hash_64_flat_slice(slice, self.n_blk, self.mode, h);
    }
    #[inline(always)]
    fn merge_flat_slices(&self, dst: &mut [u8], src: &[u8]) {
        OsBmw5bInPlace::<G, K, false>::merge_flat_slices(dst, src, self.n_blk);
    }
    #[inline(always)]
    fn estimate_ml_flat_slice(&self, slice: &[u8], pmf: &PreimagePmf) -> f64 {
        OsBmw5bInPlace::<G, K, false>::estimate_ml_flat_slice(slice, self.n_blk, pmf).0
    }
    fn debug_label(&self) -> &'static str {
        "OsBmwFlatSlab"
    }
}

/// Symmetric Exact-Token (`(idx << 8) | k`, cf. Ertl ELL §4.3) + Flat-Slab kernel for `ELL(2,24)`.
#[derive(Clone, Copy)]
struct Ell24SlabOps {
    p: u32,
    m: usize,
    state_bytes: usize,
    expected_hdr: [u8; 8],
}

impl Ell24SlabOps {
    fn new(p: u32) -> Self {
        let m = 1usize << p;
        let state_bytes = Ell24Sketch::state_bytes_for(p);
        let mut hdr_probe = vec![0u8; state_bytes];
        Ell24Sketch::init_flat_slice(&mut hdr_probe, p);
        let expected_hdr: [u8; 8] = hdr_probe[0..8].try_into().unwrap();
        Self {
            p,
            m,
            state_bytes,
            expected_hdr,
        }
    }
}

impl FlatSlabSketchOps for Ell24SlabOps {
    #[inline(always)]
    fn state_bytes(&self) -> usize {
        self.state_bytes
    }
    #[inline(always)]
    fn expected_hdr(&self) -> [u8; 8] {
        self.expected_hdr
    }
    #[inline(always)]
    fn init_flat_slice(&self, slice: &mut [u8]) {
        Ell24Sketch::init_flat_slice(slice, self.p);
    }
    #[inline(always)]
    fn decompose_hash_to_cell(&self, h: u64) -> u32 {
        let (idx, k) = ell_hash_to_idx_k(h, self.p);
        debug_assert!(idx < self.m && k >= 1 && k <= 255);
        ((idx as u32) << 8) | k
    }
    #[inline(always)]
    fn add_cell_flat_slice(&self, slice: &mut [u8], cell: u32) {
        let k = cell & 0xFF;
        let idx = (cell >> 8) as usize;
        Ell24Sketch::add_level_flat_slice(slice, idx, k);
    }
    #[inline(always)]
    fn add_hash_64_flat_slice(&self, slice: &mut [u8], h: u64) {
        Ell24Sketch::add_hash_64_flat_slice(slice, self.p, h);
    }
    #[inline(always)]
    fn merge_flat_slices(&self, dst: &mut [u8], src: &[u8]) {
        Ell24Sketch::merge_flat_slices(dst, src, self.m);
    }
    #[inline(always)]
    fn estimate_ml_flat_slice(&self, slice: &[u8], pmf: &PreimagePmf) -> f64 {
        Ell24Sketch::estimate_ml_flat_slice(slice, self.m, pmf).0
    }
    fn debug_label(&self) -> &'static str {
        "Ell24FlatSlab"
    }
}

/// Symmetric Exact-Token (`(idx << 6) | lvl`) + Flat-Slab kernel for `ULL(p=12)` (MLE or FGRA).
#[derive(Clone, Copy)]
struct UllSlabOps {
    p: u32,
    m: usize,
    state_bytes: usize,
    expected_hdr: [u8; 8],
    use_fgra: bool,
}

impl UllSlabOps {
    fn new(p: u32) -> Self {
        Self::with_fgra(p, false)
    }

    fn new_fgra(p: u32) -> Self {
        Self::with_fgra(p, true)
    }

    fn with_fgra(p: u32, use_fgra: bool) -> Self {
        let m = 1usize << p;
        let state_bytes = UllSketch::state_bytes_for(p);
        let mut hdr_probe = vec![0u8; state_bytes];
        UllSketch::init_flat_slice(&mut hdr_probe, p);
        let expected_hdr: [u8; 8] = hdr_probe[0..8].try_into().unwrap();
        Self {
            p,
            m,
            state_bytes,
            expected_hdr,
            use_fgra,
        }
    }
}

impl FlatSlabSketchOps for UllSlabOps {
    #[inline(always)]
    fn state_bytes(&self) -> usize {
        self.state_bytes
    }
    #[inline(always)]
    fn expected_hdr(&self) -> [u8; 8] {
        self.expected_hdr
    }
    #[inline(always)]
    fn init_flat_slice(&self, slice: &mut [u8]) {
        UllSketch::init_flat_slice(slice, self.p);
    }
    #[inline(always)]
    fn decompose_hash_to_cell(&self, h: u64) -> u32 {
        let (idx, lvl) = UllSketch::decompose_hash_64(h, self.p);
        debug_assert!(idx < self.m && lvl >= 1 && lvl <= 63);
        ((idx as u32) << 6) | (lvl as u32)
    }
    #[inline(always)]
    fn add_cell_flat_slice(&self, slice: &mut [u8], cell: u32) {
        let lvl = (cell & 63) as u8;
        let idx = (cell >> 6) as usize;
        UllSketch::add_level_flat_slice(slice, idx, lvl);
    }
    #[inline(always)]
    fn add_hash_64_flat_slice(&self, slice: &mut [u8], h: u64) {
        UllSketch::add_hash_64_flat_slice(slice, self.p, h);
    }
    #[inline(always)]
    fn merge_flat_slices(&self, dst: &mut [u8], src: &[u8]) {
        UllSketch::merge_flat_slices(dst, src, self.m);
    }
    #[inline(always)]
    fn estimate_ml_flat_slice(&self, slice: &[u8], pmf: &PreimagePmf) -> f64 {
        if self.use_fgra {
            UllSketch::estimate_fgra_flat_slice(slice, self.p)
        } else {
            UllSketch::estimate_ml_flat_slice(slice, self.m, pmf).0
        }
    }
    fn debug_label(&self) -> &'static str {
        if self.use_fgra { "UllFgraFlatSlab" } else { "UllFlatSlab" }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SparseChunk {
    next: u32,
    used: u32,
    cells: [u32; SPARSE_CHUNK_CAP],
}

impl SparseChunk {
    #[inline(always)]
    const fn new(next: u32) -> Self {
        Self {
            next,
            used: 0,
            cells: [0u32; SPARSE_CHUNK_CAP],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlatGroupSlot {
    idx: u32,
    len_or_tag: u32,
}

impl FlatGroupSlot {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            idx: NO_CHUNK,
            len_or_tag: 0,
        }
    }
    #[inline(always)]
    fn is_dense(self) -> bool {
        self.len_or_tag == DENSE_TAG
    }
}

struct FlatSlabGroupsAcc<F: FlatSlabSketchOps> {
    ops: F,
    state_bytes: usize,
    expected_hdr: [u8; 8],
    sparse_promote_cells: u32,
    sparse_keep_cells: usize,
    meta: Vec<FlatGroupSlot>,
    sparse_pool: Vec<SparseChunk>,
    free_chunks: Vec<u32>,
    dense_slab: Vec<u8>,
    dense_slots_allocated: usize,
    free_dense: Vec<u32>,
    tmp_cells: Vec<u32>,
    tmp_bytes: Vec<u8>,
    pmf: Arc<PreimagePmf>,
    live_bytes_tracker: Arc<AtomicUsize>,
    peak_bytes_tracker: Arc<AtomicUsize>,
    promote_stats: PromoteStats,
    stage_timers: StageTimers,
    reported_bytes: usize,
}

impl<F: FlatSlabSketchOps> FlatSlabGroupsAcc<F> {
    fn new(
        ops: F,
        pmf: Arc<PreimagePmf>,
        live_bytes_tracker: Arc<AtomicUsize>,
        peak_bytes_tracker: Arc<AtomicUsize>,
        promote_stats: PromoteStats,
        stage_timers: StageTimers,
    ) -> Self {
        let state_bytes = ops.state_bytes();
        let expected_hdr = ops.expected_hdr();
        // Identical promotion rule for OS-BMW, ELL(2,24), and ULL, with optional E-07 fixed sparse cap override:
        let cap_cells = env::var("S1_FIXED_SPARSE_CAP_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|b| b / 4)
            .unwrap_or(state_bytes / 4);
        let max_chunks = (cap_cells.saturating_sub(SPARSE_CHUNK_CAP) / SPARSE_CHUNK_CAP).max(4);
        let sparse_promote_cells = (max_chunks * SPARSE_CHUNK_CAP) as u32;
        let sparse_keep_cells = ((max_chunks * 3) / 4) * SPARSE_CHUNK_CAP;
        Self {
            ops,
            state_bytes,
            expected_hdr,
            sparse_promote_cells,
            sparse_keep_cells,
            meta: Vec::new(),
            sparse_pool: Vec::new(),
            free_chunks: Vec::new(),
            dense_slab: Vec::new(),
            dense_slots_allocated: 0,
            free_dense: Vec::new(),
            tmp_cells: Vec::with_capacity(sparse_promote_cells as usize + 64),
            tmp_bytes: Vec::with_capacity(state_bytes),
            pmf,
            live_bytes_tracker,
            peak_bytes_tracker,
            promote_stats,
            stage_timers,
            reported_bytes: 0,
        }
    }

    #[inline(always)]
    fn current_logical_bytes(&self) -> usize {
        let meta_b = self.meta.len() * std::mem::size_of::<FlatGroupSlot>();
        let sparse_b =
            (self.sparse_pool.len() - self.free_chunks.len()) * std::mem::size_of::<SparseChunk>();
        let dense_b = (self.dense_slots_allocated - self.free_dense.len()) * self.state_bytes;
        meta_b + sparse_b + dense_b
    }

    #[inline]
    fn sync_tracked_bytes(&mut self) {
        let cur = self.current_logical_bytes();
        if cur > self.reported_bytes {
            let diff = cur - self.reported_bytes;
            self.reported_bytes = cur;
            let now = self.live_bytes_tracker.fetch_add(diff, Ordering::Relaxed) + diff;
            self.peak_bytes_tracker.fetch_max(now, Ordering::Relaxed);
        } else if cur < self.reported_bytes {
            let diff = self.reported_bytes - cur;
            self.reported_bytes = cur;
            self.live_bytes_tracker.fetch_sub(diff, Ordering::Relaxed);
        }
    }

    #[inline]
    fn ensure_groups(&mut self, total_num_groups: usize) {
        if total_num_groups > self.meta.len() {
            self.meta.resize(total_num_groups, FlatGroupSlot::empty());
            self.sync_tracked_bytes();
        }
    }

    #[inline(always)]
    fn alloc_chunk(&mut self, next: u32) -> u32 {
        if let Some(id) = self.free_chunks.pop() {
            let c = unsafe { self.sparse_pool.get_unchecked_mut(id as usize) };
            c.next = next;
            c.used = 0;
            id
        } else {
            let id = self.sparse_pool.len() as u32;
            self.sparse_pool.push(SparseChunk::new(next));
            id
        }
    }

    #[inline]
    fn alloc_dense_slot(&mut self, init_empty: bool) -> u32 {
        let d_idx = if let Some(id) = self.free_dense.pop() {
            id
        } else {
            let id = self.dense_slots_allocated as u32;
            self.dense_slots_allocated += 1;
            let new_len = self.dense_slots_allocated * self.state_bytes;
            self.dense_slab.resize(new_len, 0);
            id
        };
        if init_empty {
            let off = (d_idx as usize) * self.state_bytes;
            self.ops
                .init_flat_slice(&mut self.dense_slab[off..off + self.state_bytes]);
        }
        d_idx
    }

    #[inline]
    fn free_sparse_chain(&mut self, mut head: u32) {
        while head != NO_CHUNK {
            let next = unsafe { self.sparse_pool.get_unchecked(head as usize).next };
            self.free_chunks.push(head);
            head = next;
        }
    }

    #[inline(always)]
    fn push_cell_to_group(&mut self, gid: usize, cell: u32) {
        let slot = unsafe { *self.meta.get_unchecked(gid) };
        if slot.idx != NO_CHUNK {
            let c = unsafe { self.sparse_pool.get_unchecked_mut(slot.idx as usize) };
            let u = c.used as usize;
            if u < SPARSE_CHUNK_CAP {
                unsafe {
                    *c.cells.get_unchecked_mut(u) = cell;
                }
                c.used = (u + 1) as u32;
                let new_len = slot.len_or_tag + 1;
                unsafe {
                    self.meta.get_unchecked_mut(gid).len_or_tag = new_len;
                }
                if new_len >= self.sparse_promote_cells {
                    self.compact_or_promote(gid);
                }
                return;
            }
        }
        self.push_cell_alloc_chunk(gid, cell);
    }

    #[cold]
    #[inline(never)]
    fn push_cell_alloc_chunk(&mut self, gid: usize, cell: u32) {
        let old_head = self.meta[gid].idx;
        let new_head = self.alloc_chunk(old_head);
        let c = &mut self.sparse_pool[new_head as usize];
        c.cells[0] = cell;
        c.used = 1;
        self.meta[gid].idx = new_head;
        let new_len = self.meta[gid].len_or_tag + 1;
        self.meta[gid].len_or_tag = new_len;
        if new_len >= self.sparse_promote_cells {
            self.compact_or_promote(gid);
        }
    }

    #[cold]
    #[inline(never)]
    fn compact_or_promote(&mut self, gid: usize) {
        self.tmp_cells.clear();
        let mut head = self.meta[gid].idx;
        while head != NO_CHUNK {
            let c = self.sparse_pool[head as usize];
            self.tmp_cells.extend_from_slice(&c.cells[..c.used as usize]);
            self.free_chunks.push(head);
            head = c.next;
        }
        self.tmp_cells.sort_unstable();
        self.tmp_cells.dedup();
        let uniq = self.tmp_cells.len();
        if uniq <= self.sparse_keep_cells {
            let mut new_head = NO_CHUNK;
            let mut off = 0usize;
            while off < uniq {
                let take = (uniq - off).min(SPARSE_CHUNK_CAP);
                let cid = self.alloc_chunk(new_head);
                let c = &mut self.sparse_pool[cid as usize];
                c.cells[..take].copy_from_slice(&self.tmp_cells[off..off + take]);
                c.used = take as u32;
                new_head = cid;
                off += take;
            }
            self.meta[gid] = FlatGroupSlot {
                idx: new_head,
                len_or_tag: uniq as u32,
            };
        } else {
            let d_idx = self.alloc_dense_slot(true);
            let off = (d_idx as usize) * self.state_bytes;
            let slice = &mut self.dense_slab[off..off + self.state_bytes];
            for &cell in &self.tmp_cells {
                self.ops.add_cell_flat_slice(slice, cell);
            }
            self.meta[gid] = FlatGroupSlot {
                idx: d_idx,
                len_or_tag: DENSE_TAG,
            };
        }
    }

    #[inline(always)]
    fn add_hash_to_group(&mut self, gid: usize, h: u64) {
        let slot = unsafe { *self.meta.get_unchecked(gid) };
        if slot.is_dense() {
            let off = (slot.idx as usize) * self.state_bytes;
            let slice = unsafe { self.dense_slab.get_unchecked_mut(off..off + self.state_bytes) };
            self.ops.add_hash_64_flat_slice(slice, h);
        } else {
            let cell = self.ops.decompose_hash_to_cell(h);
            self.push_cell_to_group(gid, cell);
        }
    }

    fn promote_sparse_group_to_dense(&mut self, gid: usize) -> usize {
        let old_head = self.meta[gid].idx;
        let d_idx = self.alloc_dense_slot(true);
        let off = (d_idx as usize) * self.state_bytes;
        let mut head = old_head;
        while head != NO_CHUNK {
            let c = self.sparse_pool[head as usize];
            let slice = &mut self.dense_slab[off..off + self.state_bytes];
            for &cell in &c.cells[..c.used as usize] {
                self.ops.add_cell_flat_slice(slice, cell);
            }
            self.free_chunks.push(head);
            head = c.next;
        }
        self.meta[gid] = FlatGroupSlot {
            idx: d_idx,
            len_or_tag: DENSE_TAG,
        };
        off
    }

    #[cfg(test)]
    fn to_canonical_dense_bytes(&self, gid: usize) -> Vec<u8> {
        let slot = self.meta[gid];
        if slot.is_dense() {
            let off = (slot.idx as usize) * self.state_bytes;
            self.dense_slab[off..off + self.state_bytes].to_vec()
        } else {
            let mut out = vec![0u8; self.state_bytes];
            self.ops.init_flat_slice(&mut out);
            let mut head = slot.idx;
            while head != NO_CHUNK {
                let c = self.sparse_pool[head as usize];
                for &cell in &c.cells[..c.used as usize] {
                    self.ops.add_cell_flat_slice(&mut out, cell);
                }
                head = c.next;
            }
            out
        }
    }
}

impl<F: FlatSlabSketchOps> Drop for FlatSlabGroupsAcc<F> {
    fn drop(&mut self) {
        if self.reported_bytes > 0 {
            self.live_bytes_tracker
                .fetch_sub(self.reported_bytes, Ordering::Relaxed);
            self.reported_bytes = 0;
        }
    }
}

impl<F: FlatSlabSketchOps> GroupsAccumulator for FlatSlabGroupsAcc<F> {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> DfResult<()> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        self.ensure_groups(total_num_groups);
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        let vals = arr.values();
        match opt_filter {
            None => {
                if arr.null_count() == 0 {
                    for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                        let h = wyhash64(v, SQL_HASH_SEED);
                        self.add_hash_to_group(gid, h);
                    }
                } else {
                    for (row, &gid) in group_indices.iter().enumerate() {
                        if arr.is_valid(row) {
                            let h = wyhash64(vals[row], SQL_HASH_SEED);
                            self.add_hash_to_group(gid, h);
                        }
                    }
                }
            }
            Some(filter) => {
                for (row, &gid) in group_indices.iter().enumerate() {
                    if filter.is_valid(row) && filter.value(row) && arr.is_valid(row) {
                        let h = wyhash64(vals[row], SQL_HASH_SEED);
                        self.add_hash_to_group(gid, h);
                    }
                }
            }
        }
        self.sync_tracked_bytes();
        if let Some(t0) = t_stage {
            self.stage_timers.update_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> DfResult<()> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        self.ensure_groups(total_num_groups);
        let states = values[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| DataFusionError::Internal("expected BinaryArray".to_string()))?;
        for (row, &gid) in group_indices.iter().enumerate() {
            if !states.is_valid(row) {
                continue;
            }
            let bytes = states.value(row);
            if bytes.is_empty() {
                continue;
            }
            if bytes[0] == TAG_SINGLE_HASH && bytes.len() == 9 {
                let h = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                self.add_hash_to_group(gid, h);
            } else if bytes[0] == TAG_EXACT_CELL_SPARSE
                && bytes.len() >= 5
                && (bytes.len() - 1) % 4 == 0
            {
                let payload = &bytes[1..];
                let n_in = payload.len() / 4;
                let slot = self.meta[gid];
                if slot.is_dense() {
                    let off = (slot.idx as usize) * self.state_bytes;
                    let slice = &mut self.dense_slab[off..off + self.state_bytes];
                    for ch in payload.chunks_exact(4) {
                        let cell = u32::from_le_bytes(ch.try_into().unwrap());
                        self.ops.add_cell_flat_slice(slice, cell);
                    }
                } else if (slot.len_or_tag as usize) + n_in >= self.sparse_promote_cells as usize {
                    let off = self.promote_sparse_group_to_dense(gid);
                    let slice = &mut self.dense_slab[off..off + self.state_bytes];
                    for ch in payload.chunks_exact(4) {
                        let cell = u32::from_le_bytes(ch.try_into().unwrap());
                        self.ops.add_cell_flat_slice(slice, cell);
                    }
                } else {
                    for ch in payload.chunks_exact(4) {
                        let cell = u32::from_le_bytes(ch.try_into().unwrap());
                        self.push_cell_to_group(gid, cell);
                    }
                }
            } else if bytes.len() == self.state_bytes && bytes[0..8] == self.expected_hdr {
                let slot = self.meta[gid];
                if slot.is_dense() {
                    let off = (slot.idx as usize) * self.state_bytes;
                    self.ops.merge_flat_slices(
                        &mut self.dense_slab[off..off + self.state_bytes],
                        bytes,
                    );
                } else {
                    let old_head = slot.idx;
                    let d_idx = self.alloc_dense_slot(false);
                    let off = (d_idx as usize) * self.state_bytes;
                    self.dense_slab[off..off + self.state_bytes].copy_from_slice(bytes);
                    let mut head = old_head;
                    while head != NO_CHUNK {
                        let c = self.sparse_pool[head as usize];
                        let slice = &mut self.dense_slab[off..off + self.state_bytes];
                        for &cell in &c.cells[..c.used as usize] {
                            self.ops.add_cell_flat_slice(slice, cell);
                        }
                        self.free_chunks.push(head);
                        head = c.next;
                    }
                    self.meta[gid] = FlatGroupSlot {
                        idx: d_idx,
                        len_or_tag: DENSE_TAG,
                    };
                }
            } else {
                return Err(DataFusionError::Internal(format!(
                    "Unexpected {} serialized state tag=0x{:02x} len={}",
                    self.ops.debug_label(),
                    bytes[0],
                    bytes.len()
                )));
            }
        }
        self.sync_tracked_bytes();
        if let Some(t0) = t_stage {
            self.stage_timers.merge_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> DfResult<ArrayRef> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        let emitted = emit_to.take_needed(&mut self.meta);
        let mut dense_cnt = 0usize;
        let mut non_empty = 0usize;
        for slot in &emitted {
            if slot.is_dense() {
                dense_cnt += 1;
                non_empty += 1;
            } else if slot.idx != NO_CHUNK {
                non_empty += 1;
            }
        }
        self.promote_stats.final_dense.fetch_add(dense_cnt, Ordering::Relaxed);
        self.promote_stats.final_total.fetch_add(non_empty, Ordering::Relaxed);
        let mut scratch_dense = vec![0u8; self.state_bytes];
        let mut out_vals = Vec::with_capacity(emitted.len());
        for slot in &emitted {
            let est = if slot.is_dense() {
                let off = (slot.idx as usize) * self.state_bytes;
                self.ops
                    .estimate_ml_flat_slice(&self.dense_slab[off..off + self.state_bytes], &self.pmf)
            } else if slot.idx == NO_CHUNK {
                0.0
            } else {
                self.ops.init_flat_slice(&mut scratch_dense);
                let mut head = slot.idx;
                while head != NO_CHUNK {
                    let c = self.sparse_pool[head as usize];
                    for &cell in &c.cells[..c.used as usize] {
                        self.ops.add_cell_flat_slice(&mut scratch_dense, cell);
                    }
                    head = c.next;
                }
                self.ops.estimate_ml_flat_slice(&scratch_dense, &self.pmf)
            };
            out_vals.push(Some(est));
        }
        if self.meta.is_empty() {
            self.sparse_pool.clear();
            self.free_chunks.clear();
            self.dense_slab.clear();
            self.free_dense.clear();
            self.dense_slots_allocated = 0;
        } else {
            for slot in emitted {
                if slot.is_dense() {
                    self.free_dense.push(slot.idx);
                } else {
                    self.free_sparse_chain(slot.idx);
                }
            }
        }
        self.sync_tracked_bytes();
        let out: Float64Array = out_vals.into_iter().collect();
        if let Some(t0) = t_stage {
            self.stage_timers.eval_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(Arc::new(out))
    }

    fn state(&mut self, emit_to: EmitTo) -> DfResult<Vec<ArrayRef>> {
        let t_stage = if self.stage_timers.enabled { Some(Instant::now()) } else { None };
        let emitted = emit_to.take_needed(&mut self.meta);
        let mut total_bytes = 0usize;
        let mut dense_cnt = 0usize;
        let mut non_empty = 0usize;
        for slot in &emitted {
            if slot.is_dense() {
                total_bytes += self.state_bytes;
                dense_cnt += 1;
                non_empty += 1;
            } else if slot.len_or_tag > 0 {
                total_bytes += 1 + (slot.len_or_tag as usize) * 4;
                non_empty += 1;
            }
        }
        self.promote_stats.partial_dense.fetch_add(dense_cnt, Ordering::Relaxed);
        self.promote_stats.partial_total.fetch_add(non_empty, Ordering::Relaxed);
        let mut builder = BinaryBuilder::with_capacity(emitted.len(), total_bytes);
        for slot in &emitted {
            if slot.is_dense() {
                let off = (slot.idx as usize) * self.state_bytes;
                builder.append_value(&self.dense_slab[off..off + self.state_bytes]);
            } else if slot.len_or_tag == 0 {
                builder.append_value(&[]);
            } else {
                self.tmp_bytes.clear();
                self.tmp_bytes.push(TAG_EXACT_CELL_SPARSE);
                let mut head = slot.idx;
                while head != NO_CHUNK {
                    let c = &self.sparse_pool[head as usize];
                    let n_bytes = (c.used as usize) * 4;
                    let raw = unsafe {
                        std::slice::from_raw_parts(c.cells.as_ptr() as *const u8, n_bytes)
                    };
                    self.tmp_bytes.extend_from_slice(raw);
                    head = c.next;
                }
                builder.append_value(&self.tmp_bytes);
            }
        }
        if self.meta.is_empty() {
            self.sparse_pool.clear();
            self.free_chunks.clear();
            self.dense_slab.clear();
            self.free_dense.clear();
            self.dense_slots_allocated = 0;
        } else {
            for slot in emitted {
                if slot.is_dense() {
                    self.free_dense.push(slot.idx);
                } else {
                    self.free_sparse_chain(slot.idx);
                }
            }
        }
        self.sync_tracked_bytes();
        let res = Arc::new(builder.finish());
        if let Some(t0) = t_stage {
            self.stage_timers.state_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(vec![res])
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> DfResult<Vec<ArrayRef>> {
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        let mut builder = BinaryBuilder::with_capacity(arr.len(), arr.len() * 9);
        let mut buf = [0u8; 9];
        buf[0] = TAG_SINGLE_HASH;
        for row in 0..arr.len() {
            let ok = arr.is_valid(row)
                && opt_filter.map_or(true, |f| f.is_valid(row) && f.value(row));
            if ok {
                let h = wyhash64(arr.value(row), SQL_HASH_SEED);
                buf[1..9].copy_from_slice(&h.to_le_bytes());
                builder.append_value(&buf);
            } else {
                builder.append_value([]);
            }
        }
        Ok(vec![Arc::new(builder.finish())])
    }

    fn size(&self) -> usize {
        self.current_logical_bytes()
    }
}

struct SketchScalarAcc<S: SqlSketch> {
    sketch: S,
    pmf: Arc<PreimagePmf>,
}

impl<S: SqlSketch> std::fmt::Debug for SketchScalarAcc<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SketchScalarAcc")
    }
}

impl<S: SqlSketch> Accumulator for SketchScalarAcc<S> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        for row in 0..arr.len() {
            if arr.is_valid(row) {
                self.sketch.add_hash(wyhash64(arr.value(row), SQL_HASH_SEED));
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        Ok(ScalarValue::Float64(Some(self.sketch.estimate(&self.pmf))))
    }

    fn size(&self) -> usize {
        S::state_bytes()
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        let mut buf = Vec::with_capacity(S::state_bytes());
        self.sketch.serialize_into(&mut buf);
        Ok(vec![ScalarValue::Binary(Some(buf))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        let arr = states[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| DataFusionError::Internal("expected BinaryArray".to_string()))?;
        for row in 0..arr.len() {
            if arr.is_valid(row) {
                self.sketch.merge_serialized(arr.value(row));
            }
        }
        Ok(())
    }
}

/// Scalar Accumulator speaking the exact V3/V4 mutually-exclusive tagged wire format
/// (`TAG_SINGLE_HASH=0xE1` len=9, `TAG_EXACT_CELL_SPARSE=0xE5` len=1+4k, Dense `expected_hdr` len=state_bytes).
struct V3ScalarAcc<F: FlatSlabSketchOps> {
    ops: F,
    buf: Vec<u8>,
    expected_hdr: [u8; 8],
    pmf: Arc<PreimagePmf>,
}

impl<F: FlatSlabSketchOps> std::fmt::Debug for V3ScalarAcc<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V3ScalarAcc({})", self.ops.debug_label())
    }
}

impl<F: FlatSlabSketchOps> V3ScalarAcc<F> {
    fn new(ops: F, pmf: Arc<PreimagePmf>) -> Self {
        let mut buf = vec![0u8; ops.state_bytes()];
        ops.init_flat_slice(&mut buf);
        let expected_hdr = ops.expected_hdr();
        Self {
            ops,
            buf,
            expected_hdr,
            pmf,
        }
    }
}

impl<F: FlatSlabSketchOps> Accumulator for V3ScalarAcc<F> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        let arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("expected UInt64Array".to_string()))?;
        for row in 0..arr.len() {
            if arr.is_valid(row) {
                let h = wyhash64(arr.value(row), SQL_HASH_SEED);
                self.ops.add_hash_64_flat_slice(&mut self.buf, h);
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        Ok(ScalarValue::Float64(Some(
            self.ops.estimate_ml_flat_slice(&self.buf, &self.pmf),
        )))
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(self.buf.clone()))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        let arr = states[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| DataFusionError::Internal("expected BinaryArray".to_string()))?;
        let state_bytes = self.buf.len();
        for row in 0..arr.len() {
            if !arr.is_valid(row) {
                continue;
            }
            let bytes = arr.value(row);
            if bytes.is_empty() {
                continue;
            }
            if bytes[0] == TAG_SINGLE_HASH && bytes.len() == 9 {
                let h = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                self.ops.add_hash_64_flat_slice(&mut self.buf, h);
            } else if bytes[0] == TAG_EXACT_CELL_SPARSE
                && bytes.len() >= 5
                && (bytes.len() - 1) % 4 == 0
            {
                for ch in bytes[1..].chunks_exact(4) {
                    let cell = u32::from_le_bytes(ch.try_into().unwrap());
                    self.ops.add_cell_flat_slice(&mut self.buf, cell);
                }
            } else if bytes.len() == state_bytes && bytes[0..8] == self.expected_hdr {
                self.ops.merge_flat_slices(&mut self.buf, bytes);
            } else {
                return Err(DataFusionError::Internal(format!(
                    "Unexpected {} scalar serialized state tag=0x{:02x} len={}",
                    self.ops.debug_label(),
                    bytes[0],
                    bytes.len()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SketchKind {
    OsBmw5b,
    OsBmw5bV3,
    OsBmw5b16,
    OsBmw5b16V3,
    OsBmw5bMask,
    OsBmw256,
    OsBmw256V3,
    OsBmw256Ms,
    OsBmw256MsV3,
    Ell24,
    Ell24V3,
    Ell20,
    Ull,
    UllV3,
    UllFgraV3,
    Hll6,
    Avll,
}

#[derive(Debug)]
struct SketchUdaf {
    name: String,
    kind: SketchKind,
    signature: Signature,
    pmf: Arc<PreimagePmf>,
    live_bytes: Arc<AtomicUsize>,
    peak_bytes: Arc<AtomicUsize>,
    promote_stats: PromoteStats,
    stage_timers: StageTimers,
}

impl PartialEq for SketchUdaf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.kind == other.kind
    }
}
impl Eq for SketchUdaf {}
impl std::hash::Hash for SketchUdaf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.kind.hash(state);
    }
}

impl SketchUdaf {
    fn new(
        name: &str,
        kind: SketchKind,
        live_bytes: Arc<AtomicUsize>,
        peak_bytes: Arc<AtomicUsize>,
        promote_stats: PromoteStats,
        stage_timers: StageTimers,
    ) -> Self {
        let pmf = match kind {
            SketchKind::OsBmw5b | SketchKind::OsBmw5bV3 => Arc::new(OsBmw5b::build_pmf()),
            SketchKind::OsBmw5b16 | SketchKind::OsBmw5b16V3 => Arc::new(OsBmw5b16::build_pmf()),
            SketchKind::OsBmw5bMask => Arc::new(OsBmw5bMask::build_pmf()),
            SketchKind::OsBmw256 | SketchKind::OsBmw256V3 => Arc::new(OsBmw256::build_pmf()),
            SketchKind::OsBmw256Ms | SketchKind::OsBmw256MsV3 => Arc::new(OsBmw256Ms::build_pmf()),
            SketchKind::Ell24 | SketchKind::Ell24V3 => Arc::new(Ell24Sketch::build_pmf()),
            SketchKind::Ell20 => Arc::new(Ell20Sketch::build_pmf()),
            SketchKind::Ull | SketchKind::UllV3 | SketchKind::UllFgraV3 => Arc::new(UllSketch::build_pmf()),
            SketchKind::Hll6 => Arc::new(Hll6Sketch::build_pmf()),
            SketchKind::Avll => Arc::new(AvllSketch::build_pmf()),
        };
        Self {
            name: name.to_string(),
            kind,
            signature: Signature::exact(vec![DataType::UInt64], Volatility::Immutable),
            pmf,
            live_bytes,
            peak_bytes,
            promote_stats,
            stage_timers,
        }
    }
}

impl AggregateUDFImpl for SketchUdaf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Float64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> DfResult<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            format!("{}[state]", args.name),
            DataType::Binary,
            false,
        ))])
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> DfResult<Box<dyn Accumulator>> {
        let pmf = self.pmf.clone();
        Ok(match self.kind {
            SketchKind::OsBmw5b => Box::new(SketchScalarAcc { sketch: OsBmw5b::new_sketch(), pmf }),
            SketchKind::OsBmw5bV3 => Box::new(V3ScalarAcc::new(OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2), pmf)),
            SketchKind::OsBmw5b16 => Box::new(SketchScalarAcc { sketch: OsBmw5b16::new_sketch(), pmf }),
            SketchKind::OsBmw5b16V3 => Box::new(V3ScalarAcc::new(OsBmwSlabOps::<512, 24>::new(16, HashMode5b::Pow2), pmf)),
            SketchKind::OsBmw5bMask => Box::new(SketchScalarAcc { sketch: OsBmw5bMask::new_sketch(), pmf }),
            SketchKind::OsBmw256 => Box::new(SketchScalarAcc { sketch: OsBmw256::new_sketch(), pmf }),
            SketchKind::OsBmw256V3 => Box::new(V3ScalarAcc::new(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::Div3), pmf)),
            SketchKind::OsBmw256Ms => Box::new(SketchScalarAcc { sketch: OsBmw256Ms::new_sketch(), pmf }),
            SketchKind::OsBmw256MsV3 => Box::new(V3ScalarAcc::new(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::MulShift384), pmf)),
            SketchKind::Ell24 => Box::new(SketchScalarAcc { sketch: Ell24Sketch::new_sketch(), pmf }),
            SketchKind::Ell24V3 => Box::new(V3ScalarAcc::new(Ell24SlabOps::new(10), pmf)),
            SketchKind::Ell20 => Box::new(SketchScalarAcc { sketch: Ell20Sketch::new_sketch(), pmf }),
            SketchKind::Ull => Box::new(SketchScalarAcc { sketch: UllSketch::new_sketch(), pmf }),
            SketchKind::UllV3 => Box::new(V3ScalarAcc::new(UllSlabOps::new(12), pmf)),
            SketchKind::UllFgraV3 => Box::new(V3ScalarAcc::new(UllSlabOps::new_fgra(12), pmf)),
            SketchKind::Hll6 => Box::new(SketchScalarAcc { sketch: Hll6Sketch::new_sketch(), pmf }),
            SketchKind::Avll => Box::new(SketchScalarAcc { sketch: AvllSketch::new_sketch(), pmf }),
        })
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        _args: AccumulatorArgs,
    ) -> DfResult<Box<dyn GroupsAccumulator>> {
        let pmf = self.pmf.clone();
        let lb = self.live_bytes.clone();
        let pb = self.peak_bytes.clone();
        let ps = self.promote_stats.clone();
        let st = self.stage_timers.clone();
        Ok(match self.kind {
            SketchKind::OsBmw5b => Box::new(SketchGroupsAcc::<OsBmw5b>::new(pmf, lb, pb, ps, st)),
            SketchKind::OsBmw5bV3 => Box::new(FlatSlabGroupsAcc::new(OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2), pmf, lb, pb, ps, st)),
            SketchKind::OsBmw5b16 => Box::new(SketchGroupsAcc::<OsBmw5b16>::new(pmf, lb, pb, ps, st)),
            SketchKind::OsBmw5b16V3 => Box::new(FlatSlabGroupsAcc::new(OsBmwSlabOps::<512, 24>::new(16, HashMode5b::Pow2), pmf, lb, pb, ps, st)),
            SketchKind::OsBmw5bMask => Box::new(SketchGroupsAcc::<OsBmw5bMask>::new(pmf, lb, pb, ps, st)),
            SketchKind::OsBmw256 => Box::new(SketchGroupsAcc::<OsBmw256>::new(pmf, lb, pb, ps, st)),
            SketchKind::OsBmw256V3 => Box::new(FlatSlabGroupsAcc::new(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::Div3), pmf, lb, pb, ps, st)),
            SketchKind::OsBmw256Ms => Box::new(SketchGroupsAcc::<OsBmw256Ms>::new(pmf, lb, pb, ps, st)),
            SketchKind::OsBmw256MsV3 => Box::new(FlatSlabGroupsAcc::new(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::MulShift384), pmf, lb, pb, ps, st)),
            SketchKind::Ell24 => Box::new(SketchGroupsAcc::<Ell24Sketch>::new(pmf, lb, pb, ps, st)),
            SketchKind::Ell24V3 => Box::new(FlatSlabGroupsAcc::new(Ell24SlabOps::new(10), pmf, lb, pb, ps, st)),
            SketchKind::Ell20 => Box::new(SketchGroupsAcc::<Ell20Sketch>::new(pmf, lb, pb, ps, st)),
            SketchKind::Ull => Box::new(SketchGroupsAcc::<UllSketch>::new(pmf, lb, pb, ps, st)),
            SketchKind::UllV3 => Box::new(FlatSlabGroupsAcc::new(UllSlabOps::new(12), pmf, lb, pb, ps, st)),
            SketchKind::UllFgraV3 => Box::new(FlatSlabGroupsAcc::new(UllSlabOps::new_fgra(12), pmf, lb, pb, ps, st)),
            SketchKind::Hll6 => Box::new(SketchGroupsAcc::<Hll6Sketch>::new(pmf, lb, pb, ps, st)),
            SketchKind::Avll => Box::new(SketchGroupsAcc::<AvllSketch>::new(pmf, lb, pb, ps, st)),
        })
    }
}

// ---------------------------------------------------------------------------
// Deterministic High-Cardinality Multi-Group Arrow Dataset Generator
// ---------------------------------------------------------------------------

struct WorkloadSpec {
    name: &'static str,
    n_groups: usize,
    total_rows: usize,
    n_partitions: usize,
    zipf_skew: bool,
    repeat_passes: usize,
    external_bin: Option<PathBuf>,
}

/// Loads a real public dataset exported as binary:
/// `[u64 n_rows][u64 n_groups][u64 * n_groups true_distinct][u32 * n_rows gids][u64 * n_rows vals]`.
fn load_external_table(
    path: &std::path::Path,
    n_partitions: usize,
    repeat_passes: usize,
) -> (Arc<Schema>, Vec<Vec<RecordBatch>>, Vec<u64>) {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("failed to read external W9 {:?}: {}", path, e));
    assert!(raw.len() >= 16, "external W9 file too short");
    let n_rows = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let n_groups = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let off_td = 16usize;
    let off_gids = off_td + n_groups * 8;
    let off_vals = off_gids + n_rows * 4;
    let expected_len = off_vals + n_rows * 8;
    assert_eq!(raw.len(), expected_len, "external W9 size mismatch");

    let mut true_distinct = Vec::with_capacity(n_groups);
    for ch in raw[off_td..off_gids].chunks_exact(8) {
        true_distinct.push(u64::from_le_bytes(ch.try_into().unwrap()));
    }
    let mut flat_gids = Vec::with_capacity(n_rows);
    for ch in raw[off_gids..off_vals].chunks_exact(4) {
        flat_gids.push(u32::from_le_bytes(ch.try_into().unwrap()));
    }
    let mut flat_vals = Vec::with_capacity(n_rows);
    for ch in raw[off_vals..expected_len].chunks_exact(8) {
        flat_vals.push(u64::from_le_bytes(ch.try_into().unwrap()));
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("group_id", DataType::UInt32, false),
        Field::new("val", DataType::UInt64, false),
    ]));
    let batch_size = 65_536usize;
    let per_part = (n_rows + n_partitions - 1) / n_partitions;
    let mut partitions = Vec::with_capacity(n_partitions);
    for p in 0..n_partitions {
        let start = p * per_part;
        let end = (start + per_part).min(n_rows);
        let mut base_batches = Vec::new();
        let mut off = start;
        while off < end {
            let bend = (off + batch_size).min(end);
            let g_arr = Arc::new(UInt32Array::from(flat_gids[off..bend].to_vec())) as ArrayRef;
            let v_arr = Arc::new(UInt64Array::from(flat_vals[off..bend].to_vec())) as ArrayRef;
            base_batches.push(RecordBatch::try_new(schema.clone(), vec![g_arr, v_arr]).unwrap());
            off = bend;
        }
        let passes = repeat_passes.max(1);
        let mut batches = Vec::with_capacity(base_batches.len() * passes);
        for _ in 0..passes {
            batches.extend(base_batches.iter().cloned());
        }
        partitions.push(batches);
    }
    (schema, partitions, true_distinct)
}

/// Generates Arrow `MemTable` partitions with EXACT known ground-truth distinct count per group.
/// Every group `g` receives `rows_g >= distinct_g` rows whose values cycle over `0..distinct_g`,
/// interleaved across all partitions (so parallel DataFusion executes real cross-partition merges!).
fn build_workload_table(
    spec: &WorkloadSpec,
) -> (Arc<Schema>, Vec<Vec<RecordBatch>>, Vec<u64>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_id", DataType::UInt32, false),
        Field::new("val", DataType::UInt64, false),
    ]));

    let n_g = spec.n_groups;
    let mut rows_per_group = vec![0usize; n_g];
    let mut distinct_per_group = vec![0u64; n_g];

    if !spec.zipf_skew {
        let base_rows = spec.total_rows / n_g;
        for g in 0..n_g {
            rows_per_group[g] = base_rows;
            // Vary distinct count in [0.60, 0.90] * base_rows so duplicates exist & exact truth is known!
            let r = splitmix64(0xabcd_0000 ^ g as u64);
            let frac = 0.60 + 0.30 * ((r & 0xFFFF) as f64 / 65535.0);
            let d = ((base_rows as f64 * frac) as u64).max(400);
            distinct_per_group[g] = d.min(base_rows as u64);
        }
    } else {
        // Zipf-1.0 group popularity skew: top groups get >1M rows, tail groups get ~500 rows
        let mut weights = vec![0.0_f64; n_g];
        let mut sum_w = 0.0_f64;
        for g in 0..n_g {
            let w = 1.0 / ((g as f64) + 2.0);
            weights[g] = w;
            sum_w += w;
        }
        let mut assigned = 0usize;
        for g in 0..n_g {
            let rg = ((spec.total_rows as f64) * weights[g] / sum_w).round() as usize;
            let rg = rg.max(500);
            rows_per_group[g] = rg;
            assigned += rg;
        }
        // Adjust group 0 to match exact total_rows
        if assigned > spec.total_rows {
            rows_per_group[0] = rows_per_group[0].saturating_sub(assigned - spec.total_rows).max(500);
        } else {
            rows_per_group[0] += spec.total_rows - assigned;
        }
        for g in 0..n_g {
            let rg = rows_per_group[g] as u64;
            let d = ((rg as f64) * 0.75) as u64;
            distinct_per_group[g] = d.max(350).min(rg);
        }
    }

    let actual_total: usize = rows_per_group.iter().sum();
    let batch_size = 65_536usize;
    let n_part = spec.n_partitions;

    // Interleave rows across groups & partitions via deterministic permutation so every
    // batch touches ALL groups randomly (realistic OLAP scan without pre-sorted group locality!).
    let mut part_gids: Vec<Vec<u32>> = (0..n_part)
        .map(|_| Vec::with_capacity(actual_total / n_part + batch_size))
        .collect();
    let mut part_vals: Vec<Vec<u64>> = (0..n_part)
        .map(|_| Vec::with_capacity(actual_total / n_part + batch_size))
        .collect();

    // Generate all pairs then scatter-shuffle deterministically into partitions.
    let mut rng = 0x1357_9bdf_2468_ace0u64;
    // Use a fast strided pass over groups so total memory stays low and order is genuinely randomized!
    let mut flat_gids = Vec::with_capacity(actual_total);
    let mut flat_vals = Vec::with_capacity(actual_total);
    for g in 0..n_g {
        let d = distinct_per_group[g];
        let g_salt = (g as u64) << 36;
        for j in 0..rows_per_group[g] as u64 {
            flat_gids.push(g as u32);
            flat_vals.push(g_salt | (j % d));
        }
    }
    // Fisher-Yates shuffle so rows arrive in truly random group order (real hash-agg DRAM stress!)
    for i in (1..actual_total).rev() {
        rng = rng.wrapping_add(1);
        let j = (splitmix64(rng) as usize) % (i + 1);
        flat_gids.swap(i, j);
        flat_vals.swap(i, j);
    }

    let per_part = (actual_total + n_part - 1) / n_part;
    for p in 0..n_part {
        let start = p * per_part;
        let end = (start + per_part).min(actual_total);
        if start < end {
            part_gids[p].extend_from_slice(&flat_gids[start..end]);
            part_vals[p].extend_from_slice(&flat_vals[start..end]);
        }
    }
    drop(flat_gids);
    drop(flat_vals);

    let mut partitions = Vec::with_capacity(n_part);
    let passes = spec.repeat_passes.max(1);
    for p in 0..n_part {
        let mut base_batches = Vec::new();
        let len = part_gids[p].len();
        let mut off = 0;
        while off < len {
            let end = (off + batch_size).min(len);
            let g_arr = Arc::new(UInt32Array::from(part_gids[p][off..end].to_vec())) as ArrayRef;
            let v_arr = Arc::new(UInt64Array::from(part_vals[p][off..end].to_vec())) as ArrayRef;
            base_batches.push(RecordBatch::try_new(schema.clone(), vec![g_arr, v_arr]).unwrap());
            off = end;
        }
        let mut batches = Vec::with_capacity(base_batches.len() * passes);
        for _ in 0..passes {
            batches.extend(base_batches.iter().cloned());
        }
        partitions.push(batches);
    }
    (schema, partitions, distinct_per_group)
}

// ---------------------------------------------------------------------------
// Linux /proc/self RSS High-Water Mark & Stage-Consistency Helpers
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn trim_allocator() {
    #[cfg(target_env = "gnu")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        let _ = malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn trim_allocator() {}

/// Reads `(VmRSS_kib, VmHWM_kib)` from `/proc/self/status` on Linux.
#[cfg(target_os = "linux")]
fn read_proc_rss_hwm_kib() -> (u64, u64) {
    let status = fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let mut rss = None;
    let mut hwm = None;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok());
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok());
        }
    }
    (
        rss.expect("VmRSS missing in /proc/self/status"),
        hwm.expect("VmHWM missing in /proc/self/status"),
    )
}

#[cfg(not(target_os = "linux"))]
fn read_proc_rss_hwm_kib() -> (u64, u64) {
    (0, 0)
}

/// Resets kernel VmHWM to current VmRSS by writing `5` to `/proc/self/clear_refs`.
#[cfg(target_os = "linux")]
fn reset_vm_hwm() {
    let mut f = OpenOptions::new()
        .write(true)
        .open("/proc/self/clear_refs")
        .expect("open /proc/self/clear_refs");
    f.write_all(b"5\n")
        .expect("write 5 to /proc/self/clear_refs");
}

#[cfg(not(target_os = "linux"))]
fn reset_vm_hwm() {}

/// Zero-copy repartitioning of an existing Arrow `RecordBatch` table into `target_parts`
/// balanced partitions over the exact same underlying buffers.
fn repartition_batches(
    partitions: &[Vec<RecordBatch>],
    target_parts: usize,
) -> Vec<Vec<RecordBatch>> {
    assert!(target_parts >= 1);
    let all: Vec<RecordBatch> = partitions
        .iter()
        .flat_map(|p| p.iter().cloned())
        .collect();
    if target_parts == 1 {
        return vec![all];
    }
    let total_rows: usize = all.iter().map(|b| b.num_rows()).sum();
    let mut out = Vec::with_capacity(target_parts);
    let mut check_rows = 0usize;
    for p in 0..target_parts {
        let p_start = p * total_rows / target_parts;
        let p_end = (p + 1) * total_rows / target_parts;
        let mut part_batches = Vec::new();
        let mut cursor = 0usize;
        for b in &all {
            let b_len = b.num_rows();
            let b_end = cursor + b_len;
            if b_end > p_start && cursor < p_end {
                let lo = p_start.saturating_sub(cursor);
                let hi = p_end.min(b_end) - cursor;
                if hi > lo {
                    let sliced = b.slice(lo, hi - lo);
                    check_rows += sliced.num_rows();
                    part_batches.push(sliced);
                }
            }
            cursor = b_end;
        }
        out.push(part_batches);
    }
    assert_eq!(check_rows, total_rows, "repartition row count mismatch");
    out
}

/// Extracts per-group estimates indexed by `group_id` from SQL output batches.
fn extract_estimates(batches: &[RecordBatch], n_groups: usize, fn_name: &str) -> Vec<f64> {
    let mut ests = vec![f64::NAN; n_groups];
    let mut seen = vec![false; n_groups];
    let mut n_seen = 0usize;
    for b in batches {
        let g_col = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        if let Some(f_col) = b.column(1).as_any().downcast_ref::<Float64Array>() {
            for row in 0..b.num_rows() {
                let gid = g_col.value(row) as usize;
                assert!(!seen[gid], "Duplicate group_id {} for {}", gid, fn_name);
                seen[gid] = true;
                ests[gid] = f_col.value(row);
                n_seen += 1;
            }
        } else if let Some(u_col) = b.column(1).as_any().downcast_ref::<UInt64Array>() {
            for row in 0..b.num_rows() {
                let gid = g_col.value(row) as usize;
                assert!(!seen[gid], "Duplicate group_id {} for {}", gid, fn_name);
                seen[gid] = true;
                ests[gid] = u_col.value(row) as f64;
                n_seen += 1;
            }
        } else {
            panic!("Unexpected estimate column type for {}", fn_name);
        }
    }
    assert_eq!(n_seen, n_groups, "SQL output group count mismatch for {}", fn_name);
    ests
}

async fn print_physical_plan(ctx: &SessionContext, tag: &str) {
    let explain_df = ctx
        .sql("EXPLAIN SELECT group_id, osbmw5b_cd(val) AS cd FROM events GROUP BY group_id")
        .await
        .unwrap();
    let explain_batches = explain_df.collect().await.unwrap();
    println!("  [SQL Plan Verification - {}] Physical plan excerpt:", tag);
    for b in &explain_batches {
        if b.num_columns() >= 2 {
            let ptype = b
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>();
            let pbody = b
                .column(1)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>();
            if let (Some(pt), Some(pb)) = (ptype, pbody) {
                for r in 0..b.num_rows() {
                    if pt.value(r) == "physical_plan" {
                        for line in pb.value(r).lines().take(8) {
                            println!("    {}", line);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BenchRow {
    workload: String,
    n_threads: usize,
    n_groups: usize,
    total_rows: usize,
    method: String,
    sql_fn: String,
    bytes_per_group: usize,
    logical_state_mb: f64,
    logical_state_source: String,
    rss_pre_table_mb: f64,
    rss_post_table_mb: f64,
    rss_before_query_mb: f64,
    vm_hwm_peak_mb: f64,
    rss_peak_over_table_mb: f64,
    rss_hwm_delta_mb: f64,
    wall_median_s: f64,
    wall_iqr_s: f64,
    wall_ci95_lo_s: f64,
    wall_ci95_hi_s: f64,
    wall_runs_s: Vec<f64>,
    throughput_mrows_s: f64,
    rel_bias_pct: f64,
    rmse_pct: f64,
    mvp_ram: f64,
    final_dense_groups: usize,
    final_dense_frac: f64,
    partial_dense_frac: f64,
    stage_compare: String,
    stage_mismatch_groups: usize,
    upd_cpu_ms: f64,
    mrg_cpu_ms: f64,
    eval_cpu_ms: f64,
    state_cpu_ms: f64,
    freq_med_mhz: f64,
    freq_min_mhz: f64,
    freq_max_mhz: f64,
}

/// Computes `(median, iqr, boot_ci95_lo, boot_ci95_hi)` deterministically over timed repeats.
fn summarize_timings(runs: &[f64]) -> (f64, f64, f64, f64) {
    assert!(!runs.is_empty());
    let mut s = runs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let quantile = |p: f64| -> f64 {
        if n == 1 {
            return s[0];
        }
        let pos = p * ((n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - (lo as f64);
        s[lo] * (1.0 - frac) + s[hi] * frac
    };
    let med = quantile(0.5);
    let iqr = quantile(0.75) - quantile(0.25);
    if n < 3 {
        return (med, iqr, s[0], s[n - 1]);
    }
    // Deterministic 2,000-resample percentile bootstrap 95% CI of the median
    const B: usize = 2_000;
    let mut boot_meds = Vec::with_capacity(B);
    let mut sample = vec![0.0_f64; n];
    let mut rng = 0x51_c195_2027_0001u64;
    for _ in 0..B {
        for slot in &mut sample {
            rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let idx = (splitmix64(rng) as usize) % n;
            *slot = s[idx];
        }
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bm = if n % 2 == 1 {
            sample[n / 2]
        } else {
            0.5 * (sample[n / 2 - 1] + sample[n / 2])
        };
        boot_meds.push(bm);
    }
    boot_meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lo = boot_meds[(0.025 * (B as f64)).floor() as usize];
    let ci_hi = boot_meds[((0.975 * (B as f64)).ceil() as usize).min(B - 1)];
    (med, iqr, ci_lo, ci_hi)
}

async fn run_workload(spec: &WorkloadSpec, n_runs: usize, only_method: Option<&str>) -> Vec<BenchRow> {
    let (cmp_partitions, stage_compare) = if spec.n_partitions == 1 {
        (4usize, "p1_vs_p4".to_string())
    } else {
        (1usize, format!("p1_vs_p{}", spec.n_partitions))
    };

    trim_allocator();
    let rss_pre_table_kib = read_proc_rss_hwm_kib().0;
    let rss_pre_table_mb = rss_pre_table_kib as f64 / 1024.0;

    println!(
        "\n=== [M4 DataFusion SQL] Workload {}: {} groups, {:.3}M rows, {} partition(s), cmp={}, zipf={}, VmRSS_pre_table={:.1} MB ===",
        spec.name,
        spec.n_groups,
        spec.total_rows as f64 / 1e6,
        spec.n_partitions,
        stage_compare,
        spec.zipf_skew,
        rss_pre_table_mb
    );
    let t_gen = Instant::now();
    let (schema, partitions, true_distinct) = if let Some(ref ext_path) = spec.external_bin {
        load_external_table(ext_path, spec.n_partitions, spec.repeat_passes)
    } else {
        build_workload_table(spec)
    };
    let n_groups_actual = true_distinct.len();
    let actual_rows: usize = partitions
        .iter()
        .flat_map(|p| p.iter())
        .map(|b| b.num_rows())
        .sum();
    let cmp_partitions_batches = repartition_batches(&partitions, cmp_partitions);
    trim_allocator();
    let rss_post_table_kib = read_proc_rss_hwm_kib().0;
    let rss_post_table_mb = rss_post_table_kib as f64 / 1024.0;
    println!(
        "  Built Arrow MemTable ({} rows, {} groups across {} partitions; zero-copy compare table {} partitions) in {:.2} s | VmRSS_post_table={:.1} MB (table+={:.1} MB)",
        actual_rows,
        n_groups_actual,
        spec.n_partitions,
        cmp_partitions,
        t_gen.elapsed().as_secs_f64(),
        rss_post_table_mb,
        rss_post_table_mb - rss_pre_table_mb
    );

    let cfg = SessionConfig::new()
        .with_target_partitions(spec.n_partitions)
        .with_batch_size(65_536);
    let ctx = SessionContext::new_with_config(cfg);
    let mem_table = MemTable::try_new(schema.clone(), partitions).unwrap();
    ctx.register_table("events", Arc::new(mem_table)).unwrap();

    let cmp_cfg = SessionConfig::new()
        .with_target_partitions(cmp_partitions)
        .with_batch_size(65_536);
    let cmp_ctx = SessionContext::new_with_config(cmp_cfg);
    let cmp_mem_table = MemTable::try_new(schema, cmp_partitions_batches).unwrap();
    cmp_ctx
        .register_table("events", Arc::new(cmp_mem_table))
        .unwrap();

    // Register all custom UDAFs with dedicated live/peak state trackers on primary ctx,
    // and separate isolated trackers on cmp_ctx.
    let custom_defs: Vec<(&str, &str, SketchKind, usize)> = vec![
        ("OS-BMW-5b(k24)", "osbmw5b_cd", SketchKind::OsBmw5b, OsBmw5b::state_bytes()),
        ("OS-BMW-5b(k24,Slab+Sp)", "osbmw5b_v3_cd", SketchKind::OsBmw5bV3, OsBmw5b::state_bytes()),
        ("OS-BMW-5b(k24,16blk)", "osbmw5b16_cd", SketchKind::OsBmw5b16, OsBmw5b16::state_bytes()),
        ("OS-BMW-5b(16blk,Slab+Sp)", "osbmw5b16_v3_cd", SketchKind::OsBmw5b16V3, OsBmw5b16::state_bytes()),
        ("OS-BMW-5b(k20+m)", "osbmw5b_mask_cd", SketchKind::OsBmw5bMask, OsBmw5bMask::state_bytes()),
        ("OS-BMW-5b-256B", "osbmw256_cd", SketchKind::OsBmw256, OsBmw256::state_bytes()),
        ("OS-BMW-5b-256B(Slab+Sp)", "osbmw256_v3_cd", SketchKind::OsBmw256V3, OsBmw256::state_bytes()),
        ("OS-BMW-5b-256B(MS384)", "osbmw256_ms_cd", SketchKind::OsBmw256Ms, OsBmw256::state_bytes()),
        ("OS-BMW-5b-256B(MS384,Slab+Sp)", "osbmw256_ms_v3_cd", SketchKind::OsBmw256MsV3, OsBmw256::state_bytes()),
        ("ELL(2,24)", "ell24_cd", SketchKind::Ell24, Ell24Sketch::state_bytes()),
        ("ELL(2,24,Slab+Sp)", "ell24_v3_cd", SketchKind::Ell24V3, Ell24Sketch::state_bytes()),
        ("ELL(2,20)", "ell20_cd", SketchKind::Ell20, Ell20Sketch::state_bytes()),
        ("ULL", "ull_cd", SketchKind::Ull, UllSketch::state_bytes()),
        ("ULL(Slab+Sp)", "ull_v3_cd", SketchKind::UllV3, UllSketch::state_bytes()),
        ("ULL-FGRA(Slab+Sp)", "ull_fgra_v3_cd", SketchKind::UllFgraV3, UllSketch::state_bytes()),
        ("HLL++(6b)", "hll6_cd", SketchKind::Hll6, Hll6Sketch::state_bytes()),
        ("AVLL(base56)", "avll_cd", SketchKind::Avll, AvllSketch::state_bytes()),
    ];

    let stage_timers_on = env::var("S1_STAGE_TIMERS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let freq_log_on = env::var("S1_CPU_FREQ_LOG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut trackers: Vec<(String, String, usize, Option<(Arc<AtomicUsize>, Arc<AtomicUsize>, PromoteStats, StageTimers)>)> =
        Vec::new();
    for &(label, fn_name, kind, bpg) in &custom_defs {
        let lb = Arc::new(AtomicUsize::new(0));
        let pb = Arc::new(AtomicUsize::new(0));
        let ps = PromoteStats::new();
        let st = StageTimers::new(stage_timers_on);
        let udaf = AggregateUDF::from(SketchUdaf::new(fn_name, kind, lb.clone(), pb.clone(), ps.clone(), st.clone()));
        ctx.register_udaf(udaf);

        let cmp_lb = Arc::new(AtomicUsize::new(0));
        let cmp_pb = Arc::new(AtomicUsize::new(0));
        let cmp_ps = PromoteStats::new();
        let cmp_st = StageTimers::new(false);
        let cmp_udaf = AggregateUDF::from(SketchUdaf::new(fn_name, kind, cmp_lb, cmp_pb, cmp_ps, cmp_st));
        cmp_ctx.register_udaf(cmp_udaf);

        trackers.push((label.to_string(), fn_name.to_string(), bpg, Some((lb, pb, ps, st))));
    }
    // Also include DataFusion 55.3's native built-in `approx_distinct` (16 KiB HLL)
    trackers.push((
        "DF-BuiltIn-HLL(16KB)".to_string(),
        "approx_distinct".to_string(),
        16_384,
        None,
    ));

    // Confirm physical plans for BOTH primary and stage-compare contexts.
    print_physical_plan(&ctx, &format!("primary p={}", spec.n_partitions)).await;
    print_physical_plan(&cmp_ctx, &format!("compare p={}", cmp_partitions)).await;

    let mut out_rows = Vec::new();
    let freq_cpus = if freq_log_on { allowed_cpu_ids() } else { Vec::new() };
    for (label, fn_name, bpg, trk) in trackers {
        if let Some(filter_fn) = only_method {
            if fn_name != filter_fn && label != filter_fn {
                continue;
            }
        }
        let sql = format!(
            "SELECT group_id, {}(val) AS est_distinct FROM events GROUP BY group_id",
            fn_name
        );

        // n_warmup warmup runs + n_runs timed runs on primary ctx
        let n_warmup = if n_runs <= 2 || n_runs >= 10 { 1usize } else { 2usize };
        let mut run_times = Vec::with_capacity(n_runs);
        let mut last_batches: Vec<RecordBatch> = Vec::new();
        let mut peak_bytes_measured = 0usize;
        let mut max_rss_hwm_delta_mb = 0.0_f64;
        let mut max_vm_hwm_peak_mb = 0.0_f64;
        let mut rss_before_at_max_mb = 0.0_f64;
        let freq_samples = Arc::new(Mutex::new(Vec::<f64>::new()));
        let freq_stop = Arc::new(AtomicBool::new(false));
        let mut freq_handle: Option<thread::JoinHandle<()>> = None;

        for run_idx in 0..(n_runs + n_warmup) {
            last_batches.clear();
            if run_idx == n_warmup {
                if let Some((_, _, _, ref st)) = trk {
                    st.reset();
                }
                if freq_log_on && !freq_cpus.is_empty() && freq_handle.is_none() {
                    let cpus_c = freq_cpus.clone();
                    let s_c = Arc::clone(&freq_samples);
                    let stop_c = Arc::clone(&freq_stop);
                    freq_handle = Some(thread::spawn(move || {
                        while !stop_c.load(Ordering::Relaxed) {
                            if let Some(mhz) = read_mean_cpu_freq_mhz(&cpus_c) {
                                s_c.lock().unwrap().push(mhz);
                            }
                            thread::sleep(Duration::from_millis(25));
                        }
                    }));
                }
            }
            if let Some((ref lb, ref pb, ref ps, _)) = trk {
                lb.store(0, Ordering::SeqCst);
                pb.store(0, Ordering::SeqCst);
                ps.reset();
            }
            trim_allocator();
            reset_vm_hwm();
            let rss_before_kib = read_proc_rss_hwm_kib().0;
            let t0 = Instant::now();
            let df = ctx.sql(&sql).await.unwrap();
            let batches = df.collect().await.unwrap();
            let elapsed = t0.elapsed().as_secs_f64();
            let (rss_after_kib, hwm_after_kib) = read_proc_rss_hwm_kib();
            let hwm_mb = hwm_after_kib as f64 / 1024.0;
            let rss_b_mb = rss_before_kib as f64 / 1024.0;
            let delta_mb = hwm_after_kib.saturating_sub(rss_before_kib) as f64 / 1024.0;
            if env::var("S1_DF_TRACE_RSS").is_ok() {
                println!(
                    "    [RSS trace] {} run={} | VmRSS_pre={:.1} MB | VmHWM_peak={:.1} MB | VmRSS_post={:.1} MB | dHWM={:.1} MB | over_tbl={:.1} MB | wall={:.4}s",
                    label, run_idx, rss_b_mb, hwm_mb, rss_after_kib as f64 / 1024.0, delta_mb, hwm_mb - rss_post_table_mb, elapsed
                );
            }
            if hwm_mb > max_vm_hwm_peak_mb {
                max_vm_hwm_peak_mb = hwm_mb;
            }
            if run_idx >= n_warmup {
                if delta_mb >= max_rss_hwm_delta_mb {
                    max_rss_hwm_delta_mb = delta_mb;
                    rss_before_at_max_mb = rss_b_mb;
                }
                run_times.push(elapsed);
                if let Some((_, ref pb, _, _)) = trk {
                    peak_bytes_measured = peak_bytes_measured.max(pb.load(Ordering::SeqCst));
                }
            }
            last_batches = batches;
        }
        freq_stop.store(true, Ordering::Relaxed);
        if let Some(h) = freq_handle.take() {
            let _ = h.join();
        }
        let (freq_med_mhz, freq_min_mhz, freq_max_mhz) = {
            let mut fs = freq_samples.lock().unwrap().clone();
            if fs.is_empty() {
                (f64::NAN, f64::NAN, f64::NAN)
            } else {
                fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (fs[fs.len() / 2], fs[0], fs[fs.len() - 1])
            }
        };
        let (upd_cpu_ms, mrg_cpu_ms, eval_cpu_ms, state_cpu_ms) = if let Some((_, _, _, ref st)) = trk {
            if st.enabled {
                let d = (n_runs.max(1) as f64) * 1e6;
                (
                    st.update_ns.load(Ordering::SeqCst) as f64 / d,
                    st.merge_ns.load(Ordering::SeqCst) as f64 / d,
                    st.eval_ns.load(Ordering::SeqCst) as f64 / d,
                    st.state_ns.load(Ordering::SeqCst) as f64 / d,
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            }
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };
        let rss_peak_over_table_mb = (max_vm_hwm_peak_mb - rss_post_table_mb).max(0.0);

        let (final_dense_groups, final_dense_frac, partial_dense_frac) = if let Some((_, _, ref ps, _)) = trk {
            let fd = ps.final_dense.load(Ordering::SeqCst);
            let ft = ps.final_total.load(Ordering::SeqCst).max(1);
            let pd = ps.partial_dense.load(Ordering::SeqCst);
            let pt = ps.partial_total.load(Ordering::SeqCst);
            let f_frac = (fd as f64) / (ft as f64);
            let p_frac = if pt > 0 { (pd as f64) / (pt as f64) } else { f_frac };
            (fd, f_frac, p_frac)
        } else {
            // DataFusion built-in approx_distinct allocates 16 KiB dense HLL per group immediately
            (n_groups_actual, 1.0, 1.0)
        };

        // Exact accuracy verification across ALL groups from the primary SQL output batches
        let primary_ests = extract_estimates(&last_batches, n_groups_actual, &fn_name);
        drop(last_batches);

        let mut sum_rel_err = 0.0_f64;
        let mut sum_sq_err = 0.0_f64;
        for (gid, &est) in primary_ests.iter().enumerate() {
            let tr = true_distinct[gid] as f64;
            let re = (est - tr) / tr;
            sum_rel_err += re;
            sum_sq_err += re * re;
        }

        // Per-group stage-consistency verification on the exact same Arrow data with cmp_ctx
        let cmp_df = cmp_ctx.sql(&sql).await.unwrap();
        let cmp_batches = cmp_df.collect().await.unwrap();
        let cmp_ests = extract_estimates(&cmp_batches, n_groups_actual, &fn_name);
        drop(cmp_batches);

        let stage_mismatch_groups = primary_ests
            .iter()
            .zip(cmp_ests.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();

        let (median_s, iqr_s, ci95_lo_s, ci95_hi_s) = summarize_timings(&run_times);
        let n_seen = n_groups_actual as f64;
        let mean_bias_pct = 100.0 * sum_rel_err / n_seen;
        let mse = sum_sq_err / n_seen;
        let rmse_pct = 100.0 * mse.sqrt();
        let mvp_ram = (bpg as f64 * 8.0) * mse;
        let (logical_state_mb, logical_state_source) = if trk.is_some() {
            (
                peak_bytes_measured as f64 / (1024.0 * 1024.0),
                "tracked".to_string(),
            )
        } else {
            (
                (n_groups_actual * bpg * spec.n_partitions) as f64 / (1024.0 * 1024.0),
                "budget_formula".to_string(),
            )
        };
        let mrows_s = (actual_rows as f64 / 1e6) / median_s;

        println!(
            "  {:23} | B/grp={:5} | PeakOverTbl={:6.1} | Wall={:6.3}s (IQR={:.3}s CI=[{:.3},{:.3}]) ({:5.1} M/s) | Dense={:.1}%/{:.1}% | RMSE={:5.3}% | {}: mis={}",
            label,
            bpg,
            rss_peak_over_table_mb,
            median_s,
            iqr_s,
            ci95_lo_s,
            ci95_hi_s,
            mrows_s,
            partial_dense_frac * 100.0,
            final_dense_frac * 100.0,
            rmse_pct,
            stage_compare,
            stage_mismatch_groups
        );

        out_rows.push(BenchRow {
            workload: spec.name.to_string(),
            n_threads: spec.n_partitions,
            n_groups: n_groups_actual,
            total_rows: actual_rows,
            method: label,
            sql_fn: fn_name,
            bytes_per_group: bpg,
            logical_state_mb,
            logical_state_source,
            rss_pre_table_mb,
            rss_post_table_mb,
            rss_before_query_mb: rss_before_at_max_mb,
            vm_hwm_peak_mb: max_vm_hwm_peak_mb,
            rss_peak_over_table_mb,
            rss_hwm_delta_mb: max_rss_hwm_delta_mb,
            wall_median_s: median_s,
            wall_iqr_s: iqr_s,
            wall_ci95_lo_s: ci95_lo_s,
            wall_ci95_hi_s: ci95_hi_s,
            wall_runs_s: run_times,
            throughput_mrows_s: mrows_s,
            rel_bias_pct: mean_bias_pct,
            rmse_pct,
            mvp_ram,
            final_dense_groups,
            final_dense_frac,
            partial_dense_frac,
            stage_compare: stage_compare.clone(),
            stage_mismatch_groups,
            upd_cpu_ms,
            mrg_cpu_ms,
            eval_cpu_ms,
            state_cpu_ms,
            freq_med_mhz,
            freq_min_mhz,
            freq_max_mhz,
        });
    }
    out_rows
}

const ALL_SQL_FNS: &[&str] = &[
    "osbmw5b_cd",
    "osbmw5b16_cd",
    "osbmw5b_mask_cd",
    "osbmw256_cd",
    "ell24_cd",
    "ell20_cd",
    "ull_cd",
    "hll6_cd",
    "avll_cd",
    "approx_distinct",
];

const SCALING_SQL_FNS: &[&str] = &[
    "osbmw5b_cd",
    "osbmw256_cd",
    "ell24_cd",
    "ull_cd",
    "hll6_cd",
    "avll_cd",
    "approx_distinct",
];

const V3_SPARSE_SLAB_SQL_FNS: &[&str] = &[
    "osbmw5b_v3_cd",
    "osbmw256_v3_cd",
    "osbmw5b16_v3_cd",
    "osbmw5b_cd",
    "ell24_cd",
    "ull_cd",
    "hll6_cd",
    "approx_distinct",
];

const V4_SYMMETRIC_SQL_FNS: &[&str] = &[
    "osbmw5b_v3_cd",
    "osbmw256_v3_cd",
    "osbmw5b16_v3_cd",
    "ell24_v3_cd",
    "ull_v3_cd",
    "osbmw5b_cd",
    "osbmw256_cd",
    "ell24_cd",
    "ull_cd",
    "hll6_cd",
    "avll_cd",
    "approx_distinct",
];

const TSV_HEADER: &str = "workload\tn_threads\tn_groups\ttotal_rows\tmethod\tsql_fn\tbytes_per_group\tlogical_state_mb\tlogical_state_source\trss_pre_table_mb\trss_post_table_mb\trss_before_query_mb\tvm_hwm_peak_mb\trss_peak_over_table_mb\trss_hwm_delta_mb\twall_median_s\twall_iqr_s\twall_ci95_lo_s\twall_ci95_hi_s\tthroughput_mrows_s\trel_bias_pct\trmse_pct\tmvp_ram\tfinal_dense_groups\tfinal_dense_frac\tpartial_dense_frac\tstage_compare\tstage_mismatch_groups";

const SUPP_TSV_HEADER: &str = "workload\tn_threads\tn_groups\ttotal_rows\tmethod\tsql_fn\tbytes_per_group\tlogical_state_mb\tlogical_state_source\trss_pre_table_mb\trss_post_table_mb\trss_before_query_mb\tvm_hwm_peak_mb\trss_peak_over_table_mb\trss_hwm_delta_mb\twall_median_s\twall_iqr_s\twall_ci95_lo_s\twall_ci95_hi_s\tthroughput_mrows_s\trel_bias_pct\trmse_pct\tmvp_ram\tfinal_dense_groups\tfinal_dense_frac\tpartial_dense_frac\tstage_compare\tstage_mismatch_groups\tupd_cpu_ms\tmrg_cpu_ms\teval_cpu_ms\tstate_cpu_ms\tfreq_med_mhz\tfreq_min_mhz\tfreq_max_mhz\tbuild_tag";

fn format_supp_row_tsv(r: &BenchRow, build_tag: &str) -> String {
    format!(
        "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.1}\t{:.1}\t{}",
        format_row_tsv(r),
        r.upd_cpu_ms,
        r.mrg_cpu_ms,
        r.eval_cpu_ms,
        r.state_cpu_ms,
        r.freq_med_mhz,
        r.freq_min_mhz,
        r.freq_max_mhz,
        build_tag
    )
}

fn format_row_tsv(r: &BenchRow) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{}",
        r.workload,
        r.n_threads,
        r.n_groups,
        r.total_rows,
        r.method,
        r.sql_fn,
        r.bytes_per_group,
        r.logical_state_mb,
        r.logical_state_source,
        r.rss_pre_table_mb,
        r.rss_post_table_mb,
        r.rss_before_query_mb,
        r.vm_hwm_peak_mb,
        r.rss_peak_over_table_mb,
        r.rss_hwm_delta_mb,
        r.wall_median_s,
        r.wall_iqr_s,
        r.wall_ci95_lo_s,
        r.wall_ci95_hi_s,
        r.throughput_mrows_s,
        r.rel_bias_pct,
        r.rmse_pct,
        r.mvp_ram,
        r.final_dense_groups,
        r.final_dense_frac,
        r.partial_dense_frac,
        r.stage_compare,
        r.stage_mismatch_groups
    )
}

fn format_row_json(r: &BenchRow) -> String {
    let runs_str: Vec<String> = r.wall_runs_s.iter().map(|x| format!("{:.4}", x)).collect();
    format!(
        "{{\"workload\":\"{}\",\"n_threads\":{},\"n_groups\":{},\"total_rows\":{},\"method\":\"{}\",\"sql_fn\":\"{}\",\"bytes_per_group\":{},\"logical_state_mb\":{:.3},\"logical_state_source\":\"{}\",\"rss_pre_table_mb\":{:.3},\"rss_post_table_mb\":{:.3},\"rss_before_query_mb\":{:.3},\"vm_hwm_peak_mb\":{:.3},\"rss_peak_over_table_mb\":{:.3},\"rss_hwm_delta_mb\":{:.3},\"wall_median_s\":{:.4},\"wall_iqr_s\":{:.4},\"wall_ci95_lo_s\":{:.4},\"wall_ci95_hi_s\":{:.4},\"wall_runs_s\":[{}],\"throughput_mrows_s\":{:.3},\"rel_bias_pct\":{:.4},\"rmse_pct\":{:.4},\"mvp_ram\":{:.4},\"final_dense_groups\":{},\"final_dense_frac\":{:.4},\"partial_dense_frac\":{:.4},\"stage_compare\":\"{}\",\"stage_mismatch_groups\":{}}}",
        r.workload,
        r.n_threads,
        r.n_groups,
        r.total_rows,
        r.method,
        r.sql_fn,
        r.bytes_per_group,
        r.logical_state_mb,
        r.logical_state_source,
        r.rss_pre_table_mb,
        r.rss_post_table_mb,
        r.rss_before_query_mb,
        r.vm_hwm_peak_mb,
        r.rss_peak_over_table_mb,
        r.rss_hwm_delta_mb,
        r.wall_median_s,
        r.wall_iqr_s,
        r.wall_ci95_lo_s,
        r.wall_ci95_hi_s,
        runs_str.join(","),
        r.throughput_mrows_s,
        r.rel_bias_pct,
        r.rmse_pct,
        r.mvp_ram,
        r.final_dense_groups,
        r.final_dense_frac,
        r.partial_dense_frac,
        r.stage_compare,
        r.stage_mismatch_groups
    )
}

fn save_external_table(spec: &WorkloadSpec, path: &std::path::Path) {
    let (_schema, partitions, true_distinct) = build_workload_table(spec);
    let mut flat_gids: Vec<u32> = Vec::new();
    let mut flat_vals: Vec<u64> = Vec::new();
    for p in &partitions {
        for b in p {
            let g_col = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
            let v_col = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
            flat_gids.extend_from_slice(g_col.values());
            flat_vals.extend_from_slice(v_col.values());
        }
    }
    let mut f = File::create(path).expect("create external table bin");
    f.write_all(&(flat_gids.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&(true_distinct.len() as u64).to_le_bytes()).unwrap();
    for &td in &true_distinct {
        f.write_all(&td.to_le_bytes()).unwrap();
    }
    let g_bytes = unsafe { std::slice::from_raw_parts(flat_gids.as_ptr() as *const u8, flat_gids.len() * 4) };
    let v_bytes = unsafe { std::slice::from_raw_parts(flat_vals.as_ptr() as *const u8, flat_vals.len() * 8) };
    f.write_all(g_bytes).unwrap();
    f.write_all(v_bytes).unwrap();
    f.flush().unwrap();
}

fn run_w7_diag(out_dir: &std::path::Path) {
    println!("=== [E-02 W7 Operation Counter & Anchor Profile Diagnostic] ===");
    let w7 = WorkloadSpec {
        name: "W7_8T_128g_768M_Dense",
        n_groups: 128,
        total_rows: 64_000_000,
        n_partitions: 8,
        zipf_skew: false,
        repeat_passes: 12,
        external_bin: None,
    };
    let (_schema, partitions, _td) = build_workload_table(&w7);
    let part0 = &partitions[0];
    println!("  Built W7 table; profiling partition 0 ({}M rows across {} groups, {} passes)...",
        part0.iter().map(|b| b.num_rows()).sum::<usize>() as f64 / 1e6, w7.n_groups, w7.repeat_passes);

    fn profile_5b<const G: usize, const K: usize>(name: &str, n_blk: usize, mode: HashMode5b, part: &[RecordBatch]) -> String {
        let mut sks: Vec<OsBmw5bInPlace<G, K, false>> = (0..128).map(|_| OsBmw5bInPlace::new(n_blk, mode)).collect();
        let mut counts = [0u64; 5];
        for b in part {
            let g_col = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
            let v_col = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
            for row in 0..b.num_rows() {
                let gid = g_col.value(row) as usize;
                let h = wyhash64(v_col.value(row), SQL_HASH_SEED);
                let cat = sks[gid].classify_hash_64(h);
                if (cat as usize) < 5 { counts[cat as usize] += 1; }
                sks[gid].add_hash_64(h);
            }
        }
        let tot = counts.iter().sum::<u64>() as f64;
        let mut a_min = u8::MAX;
        let mut a_max = 0u8;
        let mut a_sum = 0u64;
        for sk in &sks {
            for blk in 0..n_blk {
                let a = sk.buf[HEADER_BYTES + blk];
                a_min = a_min.min(a);
                a_max = a_max.max(a);
                a_sum += a as u64;
            }
        }
        let a_mean = a_sum as f64 / ((sks.len() * n_blk) as f64);
        println!(
            "  {:20} | ops={:.1}M | EE={:.2}% | LUT={:.2}% | ExOV={:.3}% | OVIns={:.4}% | Reanch={:.5}% | A=[{}, {:.2}, {}]",
            name, tot / 1e6,
            counts[0] as f64 * 100.0 / tot,
            counts[1] as f64 * 100.0 / tot,
            counts[2] as f64 * 100.0 / tot,
            counts[3] as f64 * 100.0 / tot,
            counts[4] as f64 * 100.0 / tot,
            a_min, a_mean, a_max
        );
        format!(
            "{{\"method\":\"{}\",\"total_ops\":{},\"ee_hits\":{},\"normal_lut\":{},\"existing_ov\":{},\"ov_insert\":{},\"reanchor\":{},\"ee_pct\":{:.4},\"lut_pct\":{:.4},\"existing_ov_pct\":{:.4},\"ov_insert_pct\":{:.6},\"reanchor_pct\":{:.6},\"anchor_min\":{},\"anchor_mean\":{:.3},\"anchor_max\":{}}}",
            name, tot as u64, counts[0], counts[1], counts[2], counts[3], counts[4],
            counts[0] as f64 * 100.0 / tot,
            counts[1] as f64 * 100.0 / tot,
            counts[2] as f64 * 100.0 / tot,
            counts[3] as f64 * 100.0 / tot,
            counts[4] as f64 * 100.0 / tot,
            a_min, a_mean, a_max
        )
    }

    let entries = vec![
        profile_5b::<512, 24>("OS-BMW-5b(k24)", 8, HashMode5b::Pow2, part0),
        profile_5b::<512, 24>("OS-BMW-5b(k24,16blk)", 16, HashMode5b::Pow2, part0),
        profile_5b::<384, 16>("OS-BMW-5b-256B(Div3)", 16, HashMode5b::Div3, part0),
        profile_5b::<384, 16>("OS-BMW-5b-256B(MS384)", 16, HashMode5b::MulShift384, part0),
    ];
    let jpath = out_dir.join("m4_w7_diag.json");
    let mut jf = File::create(&jpath).unwrap();
    writeln!(jf, "{{\n  \"workload\": \"W7_8T_128g_768M_Dense_part0\",\n  \"profiles\": [\n    {}\n  ]\n}}", entries.join(",\n    ")).unwrap();
    println!("Wrote E-02 W7 diagnostics -> {}", jpath.display());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut mode = "all".to_string();
    let mut out_dir = PathBuf::from(".");
    let mut only_method: Option<String> = None;
    let mut methods_csv: Option<String> = None;
    let mut supp_tsv_name: Option<String> = env::var("S1_SUPP_TSV").ok();
    let mut supp_append = false;
    let mut build_tag = env::var("S1_BUILD_TAG").unwrap_or_else(|_| "avx512".to_string());
    let mut isolate = false;
    let mut child_prefix: Option<PathBuf> = None;
    let mut runs_override: Option<usize> = env::var("S1_DF_RUNS").ok().and_then(|v| v.parse().ok());
    let mut w9_file: PathBuf = env::var("S1_W9_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data/w9_clickbench.bin"));
    let mut smoke = env::var("S1_DF_SMOKE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                mode = args[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--only-method" => {
                only_method = Some(args[i + 1].clone());
                i += 2;
            }
            "--methods" => {
                methods_csv = Some(args[i + 1].clone());
                i += 2;
            }
            "--supp-tsv" => {
                supp_tsv_name = Some(args[i + 1].clone());
                i += 2;
            }
            "--supp-append" => {
                supp_append = true;
                i += 1;
            }
            "--build-tag" => {
                build_tag = args[i + 1].clone();
                i += 2;
            }
            "--isolate" => {
                isolate = true;
                i += 1;
            }
            "--child-prefix" => {
                child_prefix = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--runs" => {
                runs_override = Some(args[i + 1].parse().expect("invalid --runs"));
                i += 2;
            }
            "--w9-file" => {
                w9_file = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--smoke" => {
                smoke = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    fs::create_dir_all(&out_dir).expect("create out_dir");

    if mode == "w7_diag" {
        run_w7_diag(&out_dir);
        return;
    }

    if mode == "v5_hardened" && !isolate && only_method.is_none() {
        eprintln!("[info] --mode v5_hardened requires per-method process isolation; enabling --isolate automatically.");
        isolate = true;
    }

    // Isolated-process orchestrator: spawns a pristine child process per (workload, method)
    // so every UDAF starts from identical clean glibc arenas (zero cross-method RSS retention!).
    if isolate && only_method.is_none() {
        let exe = env::current_exe().expect("current_exe");
        let sub_modes: Vec<&str> = match mode.as_str() {
            "single" | "w1" => vec!["w1"],
            "parallel" => vec!["w2", "w3"],
            "scaling" => vec!["w4", "w5", "w6"],
            "v3_sparse_slab" => vec!["w1", "w2", "w3", "w4", "w5", "w6"],
            "v4_symmetric" => vec!["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"],
            "v5_hardened" => vec!["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"],
            "dense_hi" => vec!["w7", "w8"],
            "w2" => vec!["w2"],
            "w3" => vec!["w3"],
            "w4" => vec!["w4"],
            "w5" => vec!["w5"],
            "w5_64p" => vec!["w5_64p"],
            "w6" => vec!["w6"],
            "w7" => vec!["w7"],
            "w8" => vec!["w8"],
            "w9" => vec!["w9"],
            "w10" => vec!["w10"],
            _ => vec!["w1", "w2", "w3"],
        };
        let is_scaling = matches!(mode.as_str(), "scaling" | "w4" | "w5" | "w6");
        let is_v3 = mode == "v3_sparse_slab";
        let is_v5 = matches!(mode.as_str(), "v5_hardened" | "w9");
        let is_v4 = matches!(mode.as_str(), "v4_symmetric" | "dense_hi" | "w7" | "w8" | "w10" | "w5_64p");
        let child_runs = runs_override.unwrap_or(if is_v5 || mode == "w10" { 11 } else { 3 });
        let custom_methods: Vec<String> = methods_csv
            .as_ref()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
            .unwrap_or_default();
        let sql_fns: Vec<&str> = if !custom_methods.is_empty() {
            custom_methods.iter().map(|s| s.as_str()).collect()
        } else if is_v4 || is_v5 {
            V4_SYMMETRIC_SQL_FNS.to_vec()
        } else if is_v3 {
            V3_SPARSE_SLAB_SQL_FNS.to_vec()
        } else if is_scaling {
            SCALING_SQL_FNS.to_vec()
        } else {
            ALL_SQL_FNS.to_vec()
        };
        let stage_dir = out_dir.join(format!(".df_iso_{}", std::process::id()));
        fs::create_dir_all(&stage_dir).expect("create stage_dir");

        // Pre-materialize external table once for w10 so isolated children load from DRAM page cache in <0.5s!
        let mut temp_w10_bin: Option<PathBuf> = None;
        if sub_modes.contains(&"w10") && !smoke && env::var("S1_W10_BIN").is_err() {
            let p = stage_dir.join("w10_ooc_base.bin");
            println!("[Orchestrator] Pre-generating W10 table once to {:?} for fast isolated child loads...", p);
            let w10_base = WorkloadSpec {
                name: "W10_32T_2Kg_768M_DenseOOC",
                n_groups: 2_048,
                total_rows: 384_000_000,
                n_partitions: 32,
                zipf_skew: false,
                repeat_passes: 1,
                external_bin: None,
            };
            save_external_table(&w10_base, &p);
            temp_w10_bin = Some(p);
        }

        let mut tsv_lines: Vec<String> = Vec::new();
        let mut supp_lines: Vec<String> = Vec::new();
        let mut json_lines: Vec<String> = Vec::new();

        for &wm in &sub_modes {
            println!("\n=== [Isolated Process Orchestrator] Running {} across {} methods ({} timed runs/method) ===", wm, sql_fns.len(), child_runs);
            for &sql_fn in &sql_fns {
                let prefix = stage_dir.join(format!("{}_{}", wm, sql_fn));
                let mut cmd = Command::new(&exe);
                cmd.args(["--mode", wm, "--only-method", sql_fn, "--runs", &child_runs.to_string(), "--build-tag", &build_tag, "--out"])
                    .arg(&out_dir)
                    .arg("--w9-file")
                    .arg(&w9_file)
                    .arg("--child-prefix")
                    .arg(&prefix);
                if let Some(ref w10p) = temp_w10_bin {
                    cmd.env("S1_W10_BIN", w10p);
                }
                if smoke {
                    cmd.arg("--smoke");
                }
                let status = cmd.status().expect("spawn isolated child");
                assert!(status.success(), "Isolated child failed for {} / {}", wm, sql_fn);
                let t_line = fs::read_to_string(prefix.with_extension("tsv"))
                    .expect("read child tsv")
                    .trim()
                    .to_string();
                let s_line = fs::read_to_string(prefix.with_extension("supp"))
                    .expect("read child supp")
                    .trim()
                    .to_string();
                let j_line = fs::read_to_string(prefix.with_extension("json"))
                    .expect("read child json")
                    .trim()
                    .to_string();
                tsv_lines.push(t_line);
                supp_lines.push(s_line);
                json_lines.push(j_line);
            }
        }
        let _ = fs::remove_dir_all(&stage_dir);

        if let Some(ref stsv) = supp_tsv_name {
            let spath = out_dir.join(stsv);
            let mut sf = if supp_append && spath.exists() {
                OpenOptions::new().append(true).open(&spath).unwrap()
            } else {
                let mut nf = File::create(&spath).unwrap();
                writeln!(nf, "{}", SUPP_TSV_HEADER).unwrap();
                nf
            };
            for l in &supp_lines {
                writeln!(sf, "{}", l).unwrap();
            }
            println!("Wrote {} supplementary isolated rows to {:?}", supp_lines.len(), spath);
            if custom_methods.is_empty() && !matches!(mode.as_str(), "v5_hardened" | "v4_symmetric") {
                return;
            }
            if supp_append || !custom_methods.is_empty() || matches!(mode.as_str(), "w10" | "w5_64p") {
                return;
            }
        }

        let append_mode = mode == "parallel";
        let tsv_path = if mode == "w10" {
            out_dir.join("m4_w10_outofl3.tsv")
        } else if is_v5 {
            out_dir.join("m4_datafusion_v5_hardened.tsv")
        } else if is_v4 {
            out_dir.join("m4_datafusion_v4_symmetric.tsv")
        } else if is_v3 {
            out_dir.join("m4_datafusion_v3_sparse_slab.tsv")
        } else if is_scaling {
            out_dir.join("m4_datafusion_scaling.tsv")
        } else {
            out_dir.join("m4_datafusion_groupby.tsv")
        };
        let mut tsv = if append_mode && tsv_path.exists() {
            OpenOptions::new().append(true).open(&tsv_path).unwrap()
        } else {
            let mut f = File::create(&tsv_path).unwrap();
            writeln!(f, "{}", TSV_HEADER).unwrap();
            f
        };
        for l in &tsv_lines {
            writeln!(tsv, "{}", l).unwrap();
        }
        let json_name = match mode.as_str() {
            "single" | "w1" => "m4_datafusion_single.json",
            "parallel" => "m4_datafusion_parallel.json",
            "scaling" | "w4" | "w5" | "w6" => "m4_datafusion_scaling.json",
            "v3_sparse_slab" => "m4_datafusion_v3_sparse_slab.json",
            "v4_symmetric" | "dense_hi" | "w7" | "w8" => "m4_datafusion_v4_symmetric.json",
            "v5_hardened" | "w9" => "m4_datafusion_v5_hardened.json",
            "w10" => "m4_w10_outofl3.json",
            _ => "m4_datafusion_groupby.json",
        };
        let mut jf = File::create(out_dir.join(json_name)).unwrap();
        writeln!(jf, "{{").unwrap();
        writeln!(
            jf,
            "  \"engine\": \"Apache DataFusion 55.3.0\", \"mode\": \"{}\", \"isolated_per_method\": true, \"timed_runs\": {},",
            mode, child_runs
        )
        .unwrap();
        writeln!(jf, "  \"rows\": [").unwrap();
        for (idx, jl) in json_lines.iter().enumerate() {
            writeln!(jf, "    {}{}", jl, if idx + 1 < json_lines.len() { "," } else { "" }).unwrap();
        }
        writeln!(jf, "  ]").unwrap();
        writeln!(jf, "}}").unwrap();
        println!(
            "Wrote {} isolated rows to {:?} & {}",
            tsv_lines.len(),
            tsv_path,
            json_name
        );
        return;
    }

    let n_runs = if smoke { 2 } else { runs_override.unwrap_or(11) };
    if smoke {
        println!("[SMOKE MODE] Running tiny 64-group x 200k-row validation across selected mode={}", mode);
    }

    let mut all_rows: Vec<BenchRow> = Vec::new();
    let append_mode = mode == "parallel";

    if mode == "single" || mode == "w1" || mode == "all" {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let w1 = if smoke {
            WorkloadSpec {
                name: "W1_1T_16Kg_32M",
                n_groups: 64,
                total_rows: 200_000,
                n_partitions: 1,
                zipf_skew: false,
                repeat_passes: 1,
                external_bin: None,
            }
        } else {
            WorkloadSpec {
                name: "W1_1T_16Kg_32M",
                n_groups: 16_384,
                total_rows: 32_000_000,
                n_partitions: 1,
                zipf_skew: false,
                repeat_passes: 1,
                external_bin: None,
            }
        };
        let rows = rt.block_on(run_workload(&w1, n_runs, only_method.as_deref()));
        all_rows.extend(rows);
    }

    if mode == "parallel" || mode == "all" || mode == "w2" || mode == "w3" {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .enable_all()
            .build()
            .unwrap();
        let (w2, w3) = if smoke {
            (
                WorkloadSpec {
                    name: "W2_8T_8Kg_40M",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                },
                WorkloadSpec {
                    name: "W3_8T_Zipf_4Kg_40M",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 8,
                    zipf_skew: true,
                    repeat_passes: 1,
                    external_bin: None,
                },
            )
        } else {
            (
                WorkloadSpec {
                    name: "W2_8T_8Kg_40M",
                    n_groups: 8_192,
                    total_rows: 40_000_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                },
                WorkloadSpec {
                    name: "W3_8T_Zipf_4Kg_40M",
                    n_groups: 4_096,
                    total_rows: 40_000_000,
                    n_partitions: 8,
                    zipf_skew: true,
                    repeat_passes: 1,
                    external_bin: None,
                },
            )
        };
        if mode != "w3" {
            all_rows.extend(rt.block_on(run_workload(&w2, n_runs, only_method.as_deref())));
        }
        if mode != "w2" {
            all_rows.extend(rt.block_on(run_workload(&w3, n_runs, only_method.as_deref())));
        }
    }

    if matches!(mode.as_str(), "scaling" | "w4" | "w5" | "w5_64p" | "w6" | "dense_hi" | "w7" | "w8" | "w9" | "w10") {
        let mut specs: Vec<WorkloadSpec> = Vec::new();
        if mode == "scaling" || mode == "w4" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W4_16T_16Kg_64M",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 16,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W4_16T_16Kg_64M",
                    n_groups: 16_384,
                    total_rows: 64_000_000,
                    n_partitions: 16,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            });
        }
        if mode == "scaling" || mode == "w5" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W5_32T_16Kg_64M",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 32,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W5_32T_16Kg_64M",
                    n_groups: 16_384,
                    total_rows: 64_000_000,
                    n_partitions: 32,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            });
        }
        if mode == "scaling" || mode == "w6" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W6_8T_65Kg_64M",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W6_8T_65Kg_64M",
                    n_groups: 65_536,
                    total_rows: 64_000_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            });
        }
        if mode == "dense_hi" || mode == "w7" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W7_8T_128g_768M_Dense",
                    n_groups: 4,
                    total_rows: 1_200_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W7_8T_128g_768M_Dense",
                    n_groups: 128,
                    total_rows: 64_000_000,
                    n_partitions: 8,
                    zipf_skew: false,
                    repeat_passes: 12,
                    external_bin: None,
                }
            });
        }
        if mode == "dense_hi" || mode == "w8" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W8_16T_256g_768M_Dense",
                    n_groups: 4,
                    total_rows: 1_200_000,
                    n_partitions: 16,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W8_16T_256g_768M_Dense",
                    n_groups: 256,
                    total_rows: 64_000_000,
                    n_partitions: 16,
                    zipf_skew: false,
                    repeat_passes: 12,
                    external_bin: None,
                }
            });
        }
        if mode == "w9" {
            specs.push(WorkloadSpec {
                name: "W9_8T_ClickBench_32M",
                n_groups: 0,
                total_rows: 0,
                n_partitions: 8,
                zipf_skew: false,
                repeat_passes: 1,
                external_bin: Some(w9_file.clone()),
            });
        }
        if mode == "w10" {
            let ext_w10 = env::var("S1_W10_BIN").ok().map(PathBuf::from);
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W10_32T_2Kg_768M_DenseOOC",
                    n_groups: 64,
                    total_rows: 1_200_000,
                    n_partitions: 32,
                    zipf_skew: false,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W10_32T_2Kg_768M_DenseOOC",
                    n_groups: 2_048,
                    total_rows: 384_000_000,
                    n_partitions: 32,
                    zipf_skew: false,
                    repeat_passes: 2,
                    external_bin: ext_w10,
                }
            });
        }
        if mode == "w5_64p" {
            specs.push(if smoke {
                WorkloadSpec {
                    name: "W5_64P_16Kg_64M_Skew",
                    n_groups: 64,
                    total_rows: 200_000,
                    n_partitions: 64,
                    zipf_skew: true,
                    repeat_passes: 1,
                    external_bin: None,
                }
            } else {
                WorkloadSpec {
                    name: "W5_64P_16Kg_64M_Skew",
                    n_groups: 16_384,
                    total_rows: 64_000_000,
                    n_partitions: 64,
                    zipf_skew: true,
                    repeat_passes: 1,
                    external_bin: None,
                }
            });
        }
        for spec in &specs {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(spec.n_partitions)
                .enable_all()
                .build()
                .unwrap();
            all_rows.extend(rt.block_on(run_workload(spec, n_runs, only_method.as_deref())));
        }
    }

    if let Some(prefix) = child_prefix {
        assert_eq!(all_rows.len(), 1, "expected exactly 1 row in child mode");
        fs::write(prefix.with_extension("tsv"), format!("{}\n", format_row_tsv(&all_rows[0]))).unwrap();
        fs::write(prefix.with_extension("supp"), format!("{}\n", format_supp_row_tsv(&all_rows[0], &build_tag))).unwrap();
        fs::write(prefix.with_extension("json"), format!("{}\n", format_row_json(&all_rows[0]))).unwrap();
        return;
    }

    if let Some(ref stsv) = supp_tsv_name {
        let spath = out_dir.join(stsv);
        let mut sf = if supp_append && spath.exists() {
            OpenOptions::new().append(true).open(&spath).unwrap()
        } else {
            let mut nf = File::create(&spath).unwrap();
            writeln!(nf, "{}", SUPP_TSV_HEADER).unwrap();
            nf
        };
        for r in &all_rows {
            writeln!(sf, "{}", format_supp_row_tsv(r, &build_tag)).unwrap();
        }
        println!("Wrote {} supplementary rows to {:?}", all_rows.len(), spath);
    }

    // Write TSV (create on single/all/scaling, append on parallel)
    let is_scaling = matches!(mode.as_str(), "scaling" | "w4" | "w5" | "w6");
    let tsv_path = if is_scaling {
        out_dir.join("m4_datafusion_scaling.tsv")
    } else {
        out_dir.join("m4_datafusion_groupby.tsv")
    };
    let mut tsv = if append_mode && tsv_path.exists() {
        OpenOptions::new().append(true).open(&tsv_path).unwrap()
    } else {
        let mut f = File::create(&tsv_path).unwrap();
        writeln!(f, "{}", TSV_HEADER).unwrap();
        f
    };
    for r in &all_rows {
        writeln!(tsv, "{}", format_row_tsv(r)).unwrap();
    }

    // Write mode-specific JSON for clean parsing
    let json_name = match mode.as_str() {
        "single" | "w1" => "m4_datafusion_single.json",
        "parallel" => "m4_datafusion_parallel.json",
        "scaling" | "w4" | "w5" | "w6" => "m4_datafusion_scaling.json",
        _ => "m4_datafusion_groupby.json",
    };
    let mut jf = File::create(out_dir.join(json_name)).unwrap();
    writeln!(jf, "{{").unwrap();
    writeln!(jf, "  \"engine\": \"Apache DataFusion 55.3.0\", \"mode\": \"{}\", \"isolated_per_method\": false,", mode).unwrap();
    writeln!(jf, "  \"rows\": [").unwrap();
    for (idx, r) in all_rows.iter().enumerate() {
        writeln!(
            jf,
            "    {}{}",
            format_row_json(r),
            if idx + 1 < all_rows.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(jf, "  ]").unwrap();
    writeln!(jf, "}}").unwrap();
    println!("Wrote {} rows to {:?} & {}", all_rows.len(), tsv_path, json_name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use s1_v2_rust::osbmw_inplace::ANCHOR_INIT;

    /// Collects 256 distinct `u64` keys whose `wyhash64(k, SQL_HASH_SEED)` covers every
    /// possible first byte `0x00..=0xFF` (specifically including `0xE5`, `0xE1`, `0x53`, `0x00`).
    fn collect_all_256_first_byte_keys() -> Vec<u64> {
        let mut found = [None::<u64>; 256];
        let mut rem = 256usize;
        let mut x = 1u64;
        while rem > 0 {
            let h = wyhash64(x, SQL_HASH_SEED);
            let b0 = (h & 0xFF) as usize;
            if found[b0].is_none() {
                found[b0] = Some(x);
                rem -= 1;
            }
            x += 1;
        }
        found.into_iter().map(|v| v.unwrap()).collect()
    }

    fn verify_v3_ops<F: FlatSlabSketchOps>(ops: F, pmf: Arc<PreimagePmf>, keys: &[u64], expect_dense: bool) {
        let make_acc = || {
            FlatSlabGroupsAcc::new(
                ops.clone(),
                pmf.clone(),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                PromoteStats::new(),
                StageTimers::new(false),
            )
        };

        // Reference direct dense flat-slice sketch
        let mut ref_bytes = vec![0u8; ops.state_bytes()];
        ops.init_flat_slice(&mut ref_bytes);
        for &k in keys {
            let h = wyhash64(k, SQL_HASH_SEED);
            ops.add_hash_64_flat_slice(&mut ref_bytes, h);
        }
        let ref_est = ops.estimate_ml_flat_slice(&ref_bytes, &pmf);

        let arr: ArrayRef = Arc::new(UInt64Array::from(keys.to_vec()));
        let gids = vec![0usize; keys.len()];

        // Path A: direct update_batch
        let mut acc_direct = make_acc();
        acc_direct.update_batch(&[arr.clone()], &gids, None, 1).unwrap();
        assert_eq!(acc_direct.meta[0].is_dense(), expect_dense, "{} regime check failed", ops.debug_label());
        assert_eq!(acc_direct.to_canonical_dense_bytes(0), ref_bytes, "{} Path A canonical dense bytes mismatch", ops.debug_label());
        let est_direct = acc_direct
            .evaluate(EmitTo::All)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(est_direct.to_bits(), ref_est.to_bits(), "{} Path A estimate mismatch", ops.debug_label());

        // Path B: convert_to_state -> merge_batch (every 0x00..=0xFF first hash byte through wire!)
        let mut acc_cts = make_acc();
        let single_states = acc_cts.convert_to_state(&[arr.clone()], None).unwrap();
        let bin_states = single_states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
        for row in 0..bin_states.len() {
            let b = bin_states.value(row);
            assert_eq!(b.len(), 9, "convert_to_state must emit 9B tagged frame");
            assert_eq!(b[0], TAG_SINGLE_HASH, "convert_to_state tag must be 0xE1");
        }
        acc_cts.merge_batch(&single_states, &gids, 1).unwrap();
        assert_eq!(acc_cts.to_canonical_dense_bytes(0), ref_bytes, "{} Path B canonical bytes mismatch", ops.debug_label());
        let est_cts = acc_cts
            .evaluate(EmitTo::All)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(est_cts.to_bits(), ref_est.to_bits(), "{} Path B estimate mismatch", ops.debug_label());

        // Path C: multi-shard state() -> merge_batch in both forward & reversed shard orders
        let n_shards = 4usize;
        let mut shard_states: Vec<Vec<u8>> = Vec::with_capacity(n_shards);
        for s in 0..n_shards {
            let sub_keys: Vec<u64> = keys.iter().skip(s).step_by(n_shards).copied().collect();
            let sub_arr: ArrayRef = Arc::new(UInt64Array::from(sub_keys.clone()));
            let sub_gids = vec![0usize; sub_keys.len()];
            let mut sa = make_acc();
            sa.update_batch(&[sub_arr], &sub_gids, None, 1).unwrap();
            let st_arr = sa.state(EmitTo::All).unwrap();
            let st_bin = st_arr[0].as_any().downcast_ref::<BinaryArray>().unwrap();
            shard_states.push(st_bin.value(0).to_vec());
        }
        for rev in [false, true] {
            let mut bld = BinaryBuilder::new();
            if rev {
                for s in shard_states.iter().rev() {
                    bld.append_value(s);
                }
            } else {
                for s in &shard_states {
                    bld.append_value(s);
                }
            }
            let st_arr: ArrayRef = Arc::new(bld.finish());
            let m_gids = vec![0usize; n_shards];
            let mut acc_m = make_acc();
            acc_m.merge_batch(&[st_arr], &m_gids, 1).unwrap();
            assert_eq!(acc_m.to_canonical_dense_bytes(0), ref_bytes, "{} Path C (rev={}) canonical bytes mismatch", ops.debug_label(), rev);
            let est_m = acc_m
                .evaluate(EmitTo::All)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0);
            assert_eq!(est_m.to_bits(), ref_est.to_bits(), "{} Path C (rev={}) estimate mismatch", ops.debug_label(), rev);
        }

        // Path D: mixed dense + sparse merge in both orders (dense-into-sparse AND sparse-into-dense)
        if expect_dense {
            let split = 128usize;
            let mut sa_sparse = make_acc();
            let arr_sp: ArrayRef = Arc::new(UInt64Array::from(keys[..split].to_vec()));
            sa_sparse.update_batch(&[arr_sp], &vec![0usize; split], None, 1).unwrap();
            assert!(!sa_sparse.meta[0].is_dense());
            let wire_sp = sa_sparse.state(EmitTo::All).unwrap()[0]
                .as_any().downcast_ref::<BinaryArray>().unwrap().value(0).to_vec();
            assert_eq!(wire_sp[0], TAG_EXACT_CELL_SPARSE);

            let mut sa_dense = make_acc();
            let arr_dn: ArrayRef = Arc::new(UInt64Array::from(keys[split..].to_vec()));
            sa_dense.update_batch(&[arr_dn], &vec![0usize; keys.len() - split], None, 1).unwrap();
            assert!(sa_dense.meta[0].is_dense());
            let wire_dn = sa_dense.state(EmitTo::All).unwrap()[0]
                .as_any().downcast_ref::<BinaryArray>().unwrap().value(0).to_vec();
            assert_eq!(&wire_dn[0..8], &sa_dense.expected_hdr);

            for (first, second) in [(&wire_sp, &wire_dn), (&wire_dn, &wire_sp)] {
                let mut bld = BinaryBuilder::new();
                bld.append_value(first);
                bld.append_value(second);
                let mut acc_mix = make_acc();
                acc_mix.merge_batch(&[Arc::new(bld.finish())], &[0, 0], 1).unwrap();
                assert_eq!(acc_mix.to_canonical_dense_bytes(0), ref_bytes, "{} Path D mixed dense/sparse mismatch", ops.debug_label());
            }
        }

        // Path E: scalar Accumulator consumes both convert_to_state (0xE1) and shard state() (0xE5 / dense)
        let mut sc_cts = V3ScalarAcc::new(ops.clone(), pmf.clone());
        sc_cts.merge_batch(&single_states).unwrap();
        assert_eq!(sc_cts.buf, ref_bytes, "{} Path E scalar convert_to_state mismatch", ops.debug_label());

        let mut bld_sc = BinaryBuilder::new();
        for s in &shard_states {
            bld_sc.append_value(s);
        }
        let mut sc_sh = V3ScalarAcc::new(ops.clone(), pmf.clone());
        sc_sh.merge_batch(&[Arc::new(bld_sc.finish())]).unwrap();
        assert_eq!(sc_sh.buf, ref_bytes, "{} Path E scalar shard state mismatch", ops.debug_label());

        // Path F: explicit sparse append (5B -> 9B transient frame) vs canonical dense idempotence
        let mut sa_one = make_acc();
        let one_arr: ArrayRef = Arc::new(UInt64Array::from(vec![keys[0]]));
        sa_one.update_batch(&[one_arr], &[0], None, 1).unwrap();
        let canonical_one = sa_one.to_canonical_dense_bytes(0);
        let wire_one = sa_one.state(EmitTo::All).unwrap()[0]
            .as_any().downcast_ref::<BinaryArray>().unwrap().value(0).to_vec();
        assert_eq!(wire_one.len(), 5, "single-token sparse wire frame must be 1+4 = 5B");
        assert_eq!(wire_one[0], TAG_EXACT_CELL_SPARSE);
        let mut bld_dup = BinaryBuilder::new();
        bld_dup.append_value(&wire_one);
        bld_dup.append_value(&wire_one);
        let mut acc_dup = make_acc();
        acc_dup.merge_batch(&[Arc::new(bld_dup.finish())], &[0, 0], 1).unwrap();
        assert_eq!(acc_dup.to_canonical_dense_bytes(0), canonical_one, "canonical dense projection MUST be bit-idempotent");
        let wire_dup = acc_dup.state(EmitTo::All).unwrap()[0]
            .as_any().downcast_ref::<BinaryArray>().unwrap().value(0).to_vec();
        assert_eq!(wire_dup.len(), 9, "appended sparse wire frame grows 5B -> 9B before compaction/promotion");
    }

    #[test]
    fn test_v3_wire_framing_all_256_first_bytes_and_canonical_dense_eq() {
        let base_256 = collect_all_256_first_byte_keys();
        assert_eq!(base_256.len(), 256);

        // Build dense-forcing key set (2,048 distinct keys containing all 256 first-byte keys)
        let mut dense_keys = base_256.clone();
        let max_base = *base_256.iter().max().unwrap();
        for i in 1..=1_792u64 {
            dense_keys.push(max_base + i);
        }

        // Test ALL 5 symmetric Slab+Sp production configs in BOTH sparse-only and dense-promoted regimes:
        // 1) OS-BMW-5b(k24, 8blk, 2760B, Pow2)
        let pmf_5b = Arc::new(OsBmw5b::build_pmf());
        verify_v3_ops(OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2), pmf_5b.clone(), &base_256, false);
        verify_v3_ops(OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2), pmf_5b, &dense_keys, true);
        // 2) OS-BMW-5b-256B(k16, 16blk, 4104B, Div3)
        let pmf_256 = Arc::new(OsBmw256::build_pmf());
        verify_v3_ops(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::Div3), pmf_256.clone(), &base_256, false);
        verify_v3_ops(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::Div3), pmf_256, &dense_keys, true);
        // 3) OS-BMW-5b(k24, 16blk, 5512B, Pow2)
        let pmf_16 = Arc::new(OsBmw5b16::build_pmf());
        verify_v3_ops(OsBmwSlabOps::<512, 24>::new(16, HashMode5b::Pow2), pmf_16.clone(), &base_256, false);
        verify_v3_ops(OsBmwSlabOps::<512, 24>::new(16, HashMode5b::Pow2), pmf_16, &dense_keys, true);
        // 4) Symmetric ELL(2,24)(p=10, 4104B)
        let pmf_ell = Arc::new(Ell24Sketch::build_pmf());
        verify_v3_ops(Ell24SlabOps::new(10), pmf_ell.clone(), &base_256, false);
        verify_v3_ops(Ell24SlabOps::new(10), pmf_ell, &dense_keys, true);
        // 5) Symmetric ULL(p=12, 4104B)
        let pmf_ull = Arc::new(UllSketch::build_pmf());
        verify_v3_ops(UllSlabOps::new(12), pmf_ull.clone(), &base_256, false);
        verify_v3_ops(UllSlabOps::new(12), pmf_ull, &dense_keys, true);
        // 6) OS-BMW-5b-256B(k16, 16blk, 4104B, MulShift384)
        let pmf_ms = Arc::new(OsBmw256Ms::build_pmf());
        verify_v3_ops(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::MulShift384), pmf_ms.clone(), &base_256, false);
        verify_v3_ops(OsBmwSlabOps::<384, 16>::new(16, HashMode5b::MulShift384), pmf_ms, &dense_keys, true);
    }

    #[test]
    fn test_v3_wire_framing_rejects_malformed_states() {
        let bad_payloads: Vec<Vec<u8>> = vec![
            vec![0xE5, 1, 2, 3, 4, 5, 6, 7], // legacy untagged 8B hash starting with 0xE5
            vec![0x00, 1, 2, 3, 4, 5, 6, 7], // legacy untagged 8B hash
            vec![TAG_SINGLE_HASH, 1, 2, 3],  // truncated 0xE1 single hash
            vec![TAG_SINGLE_HASH; 10],       // overlong 0xE1 single hash
            vec![TAG_EXACT_CELL_SPARSE],     // empty 0xE5 sparse payload
            vec![TAG_EXACT_CELL_SPARSE, 1, 2, 3], // misaligned 0xE5 sparse payload
            vec![0x53; 16],                  // truncated dense header
            vec![0xFF; 9],                   // unknown tag
        ];

        fn check_reject<F: FlatSlabSketchOps>(ops: F, pmf: Arc<PreimagePmf>, bad_payloads: &[Vec<u8>]) {
            for bad in bad_payloads {
                let mut bld = BinaryBuilder::new();
                bld.append_value(bad);
                let arr: ArrayRef = Arc::new(bld.finish());

                let mut acc = FlatSlabGroupsAcc::new(
                    ops.clone(),
                    pmf.clone(),
                    Arc::new(AtomicUsize::new(0)),
                    Arc::new(AtomicUsize::new(0)),
                    PromoteStats::new(),
                    StageTimers::new(false),
                );
                assert!(
                    acc.merge_batch(&[arr.clone()], &[0], 1).is_err(),
                    "{} GroupsAccumulator must reject malformed payload {:?}",
                    ops.debug_label(),
                    bad
                );

                let mut sc = V3ScalarAcc::new(ops.clone(), pmf.clone());
                assert!(
                    sc.merge_batch(&[arr]).is_err(),
                    "{} ScalarAcc must reject malformed payload {:?}",
                    ops.debug_label(),
                    bad
                );
            }

            // Empty slice (&[]) represents filtered/null rows and MUST succeed as a no-op
            let mut bld_empty = BinaryBuilder::new();
            bld_empty.append_value(&[]);
            let arr_empty: ArrayRef = Arc::new(bld_empty.finish());
            let mut acc = FlatSlabGroupsAcc::new(
                ops.clone(),
                pmf.clone(),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                PromoteStats::new(),
                StageTimers::new(false),
            );
            acc.merge_batch(&[arr_empty.clone()], &[0], 1).unwrap();
            assert_eq!(acc.meta[0].len_or_tag, 0);
            let mut sc = V3ScalarAcc::new(ops, pmf);
            sc.merge_batch(&[arr_empty]).unwrap();
        }

        check_reject(OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2), Arc::new(OsBmw5b::build_pmf()), &bad_payloads);
        check_reject(Ell24SlabOps::new(10), Arc::new(Ell24Sketch::build_pmf()), &bad_payloads);
        check_reject(UllSlabOps::new(12), Arc::new(UllSketch::build_pmf()), &bad_payloads);
    }

    #[test]
    fn test_w7_w8_dense_high_cardinality_triggers_truncation_and_diverges_from_ull() {
        // Verify that W7 (300K-450K distinct/group) and W8 (150K-225K distinct/group) trigger real
        // M_(24) > 8 window truncation (anchor > ANCHOR_INIT=8) in OS-BMW-5b(k24) and yield genuinely
        // different estimates from ULL(p=12) (eliminating the W1-W6 zero-truncation isomorphism).
        let pmf_5b = OsBmw5b::build_pmf();
        let pmf_ull = UllSketch::build_pmf();
        for &(label, distinct_cnt) in &[("W8_low_150K", 150_000u64), ("W7_low_300K", 300_000u64)] {
            let mut osb = OsBmw5b::new(8, HashMode5b::Pow2);
            let mut ull = UllSketch::new(12);
            let mut ee_hits = 0u64;
            for j in 0..distinct_cnt {
                let h = wyhash64(j, SQL_HASH_SEED ^ distinct_cnt);
                if osb.add_hash_64(h) {
                    ee_hits += 1;
                }
                ull.add_hash_64(h);
            }
            let min_anchor = (0..8).map(|b| osb.buf[HEADER_BYTES + b]).min().unwrap();
            let max_anchor = (0..8).map(|b| osb.buf[HEADER_BYTES + b]).max().unwrap();
            assert!(
                min_anchor > ANCHOR_INIT,
                "{}: expected ALL 8 blocks to promote anchor > 8 (got min={}, max={})",
                label,
                min_anchor,
                max_anchor
            );
            assert!(ee_hits > 0, "{}: expected real Anchor-First early exits", label);
            let est_osb = osb.estimate_ml(&pmf_5b).0;
            let est_ull = ull.estimate_ml(&pmf_ull).0;
            assert_ne!(
                est_osb.to_bits(),
                est_ull.to_bits(),
                "{}: OS-BMW-5b(k24) and ULL(p=12) must diverge once M_(24) > 8 triggers truncation",
                label
            );
        }
    }

    #[test]
    fn test_promote_stats_and_ci95_summary() {
        let pmf = Arc::new(OsBmw5b::build_pmf());
        let ps = PromoteStats::new();
        let mut acc = FlatSlabGroupsAcc::new(
            OsBmwSlabOps::<512, 24>::new(8, HashMode5b::Pow2),
            pmf,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            ps.clone(),
            StageTimers::new(false),
        );
        // Group 0 gets 20 keys (stays sparse); Group 1 gets 2,000 distinct keys (promotes to dense)
        let mut vals = Vec::with_capacity(2_020);
        let mut gids = Vec::with_capacity(2_020);
        for i in 0..20u64 {
            vals.push(i);
            gids.push(0usize);
        }
        for i in 0..2_000u64 {
            vals.push(10_000 + i);
            gids.push(1usize);
        }
        let arr: ArrayRef = Arc::new(UInt64Array::from(vals));
        acc.update_batch(&[arr], &gids, None, 2).unwrap();
        let _ = acc.evaluate(EmitTo::All).unwrap();
        assert_eq!(ps.final_dense.load(Ordering::SeqCst), 1);
        assert_eq!(ps.final_total.load(Ordering::SeqCst), 2);

        let samples = vec![1.01, 1.05, 1.02, 1.04, 1.03, 1.06, 1.02, 1.04, 1.03, 1.05, 1.04];
        let (med, iqr, lo, hi) = summarize_timings(&samples);
        assert!((med - 1.04).abs() < 1e-9);
        assert!(iqr >= 0.0 && lo <= med && med <= hi);
    }
}
