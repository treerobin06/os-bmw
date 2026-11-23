#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${RUST_DIR}/.." && pwd)"
OUT_JSON="${1:-${ROOT_DIR}/out/env_info.json}"
RELEASE_DIR="${RUST_DIR}/target/release"

mkdir -p "$(dirname "${OUT_JSON}")"

RUSTC_VER="$(rustc -V 2>/dev/null || echo unknown)"
CARGO_VER="$(cargo -V 2>/dev/null || echo unknown)"
HOST_NAME="$(hostname 2>/dev/null || echo unknown)"
UNAME_ALL="$(uname -a 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"

CPU_MODEL="unknown"
LSCPU_AVX512=""
if command -v lscpu >/dev/null 2>&1; then
    CPU_MODEL="$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' || echo unknown)"
    FLAGS_LINE="$(lscpu | awk -F: '/Flags/ {print $2; exit}' || true)"
    if [ -z "${FLAGS_LINE}" ] && [ -r /proc/cpuinfo ]; then
        FLAGS_LINE="$(awk -F: '/^flags/ {print $2; exit}' /proc/cpuinfo || true)"
    fi
    LSCPU_AVX512="$(printf '%s\n' "${FLAGS_LINE}" | tr ' ' '\n' | grep '^avx512' | sort -u | tr '\n' ' ' | sed 's/ *$//' || true)"
elif command -v sysctl >/dev/null 2>&1; then
    CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
fi

NATIVE_FEATURES="$(rustc -C target-cpu=native --print cfg 2>/dev/null | sed -n 's/^target_feature="\([^"]*\)"$/\1/p' | sort -u | tr '\n' ' ' | sed 's/ *$//' || true)"

python3 - "${OUT_JSON}" "${RELEASE_DIR}" "${RUSTC_VER}" "${CARGO_VER}" "${HOST_NAME}" "${UNAME_ALL}" "${ARCH_NAME}" "${CPU_MODEL}" "${LSCPU_AVX512}" "${NATIVE_FEATURES}" "${RUSTFLAGS:-}" <<'PY'
import json
import os
import shutil
import subprocess
import sys

(
    out_json,
    release_dir,
    rustc_ver,
    cargo_ver,
    host_name,
    uname_all,
    arch_name,
    cpu_model,
    lscpu_avx512_str,
    native_feats_str,
    rustflags,
) = sys.argv[1:12]

avx512_flags = [x for x in lscpu_avx512_str.split() if x]
native_features = [x for x in native_feats_str.split() if x]
native_avx512_features = [x for x in native_features if x.startswith("avx512")]

objdump_bin = shutil.which("objdump")
bins = ["s1_v2_reconcile", "s1_v2_bench_sys", "s1_v2_accuracy"]
zmm_counts = {}
for name in bins:
    path = os.path.join(release_dir, name)
    entry = {"path": path, "exists": os.path.isfile(path), "zmm_instruction_count": None}
    if entry["exists"] and objdump_bin:
        p1 = subprocess.Popen([objdump_bin, "-d", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        cnt = 0
        assert p1.stdout is not None
        for line in p1.stdout:
            if "zmm" in line:
                cnt += 1
        p1.wait()
        entry["zmm_instruction_count"] = cnt
    zmm_counts[name] = entry

payload = {
    "rustc_version": rustc_ver,
    "cargo_version": cargo_ver,
    "rustflags": rustflags,
    "hostname": host_name,
    "uname": uname_all,
    "arch": arch_name,
    "cpu_model": cpu_model,
    "lscpu_avx512_flags": avx512_flags,
    "rustc_native_avx512_features": native_avx512_features,
    "rustc_native_target_features": native_features,
    "avx512_5b_kernel_requirements_met": all(
        f in native_avx512_features for f in ("avx512bw", "avx512vbmi", "avx512vbmi2")
    ),
    "release_binaries_zmm": zmm_counts,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=False)
    f.write("\n")

print(f"Wrote {out_json}")
PY
