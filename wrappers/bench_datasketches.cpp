#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpc_sketch.hpp"
#include "cpc_union.hpp"
#include "hll.hpp"

// Exact heap-tracking C++ allocator for DataSketches RAM working-set measurement
struct HeapTracker {
    size_t live_bytes = 0;
    size_t peak_bytes = 0;
};

template <typename T>
class TrackedAlloc {
public:
    using value_type = T;
    HeapTracker* tracker = nullptr;

    TrackedAlloc() noexcept = default;
    explicit TrackedAlloc(HeapTracker* tr) noexcept : tracker(tr) {}
    template <typename U>
    TrackedAlloc(const TrackedAlloc<U>& other) noexcept : tracker(other.tracker) {}

    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        T* p = static_cast<T*>(::operator new(bytes));
        if (tracker) {
            tracker->live_bytes += bytes;
            if (tracker->live_bytes > tracker->peak_bytes) {
                tracker->peak_bytes = tracker->live_bytes;
            }
        }
        return p;
    }

    void deallocate(T* p, size_t n) noexcept {
        if (tracker) {
            size_t bytes = n * sizeof(T);
            tracker->live_bytes = (tracker->live_bytes >= bytes) ? (tracker->live_bytes - bytes) : 0;
        }
        ::operator delete(p);
    }

    template <typename U>
    bool operator==(const TrackedAlloc<U>& o) const noexcept { return tracker == o.tracker; }
    template <typename U>
    bool operator!=(const TrackedAlloc<U>& o) const noexcept { return tracker != o.tracker; }
};

using TrackedCpc = datasketches::cpc_sketch_alloc<TrackedAlloc<uint8_t>>;
using TrackedHll = datasketches::hll_sketch_alloc<TrackedAlloc<uint8_t>>;
using TrackedHllUnion = datasketches::hll_union_alloc<TrackedAlloc<uint8_t>>;
using TrackedCpcUnion = datasketches::cpc_union_alloc<TrackedAlloc<uint8_t>>;

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t wyhash64_seed(uint64_t key, uint64_t seed) {
    return splitmix64(key ^ seed);
}

struct CellStats {
    std::string method;
    uint8_t lg_k;
    double n_val;
    int trials;
    double ram_live_mean;
    double ram_peak_mean;
    double wire_compact_mean;
    double wire_updatable_mean;
    double rel_bias_pct;
    double rmse_pct;
    double mvp_ram_live;
    double mvp_wire_compact;
};

template <typename Fn>
static CellStats run_mc_cell(const std::string& method, uint8_t lg_k, double n_val, int trials, Fn&& run_trial) {
    std::vector<double> errs(trials);
    std::vector<double> ram_live(trials);
    std::vector<double> ram_peak(trials);
    std::vector<double> wire_cmp(trials);
    std::vector<double> wire_upd(trials);

#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < trials; ++t) {
        HeapTracker ht;
        auto res = run_trial(t, &ht);
        errs[t] = (res.est - n_val) / n_val;
        ram_live[t] = static_cast<double>(res.ram_live);
        ram_peak[t] = static_cast<double>(res.ram_peak);
        wire_cmp[t] = static_cast<double>(res.wire_compact);
        wire_upd[t] = static_cast<double>(res.wire_updatable);
    }

    double sum_e = 0, sum_e2 = 0, sum_rl = 0, sum_rp = 0, sum_wc = 0, sum_wu = 0;
    for (int t = 0; t < trials; ++t) {
        sum_e += errs[t];
        sum_e2 += errs[t] * errs[t];
        sum_rl += ram_live[t];
        sum_rp += ram_peak[t];
        sum_wc += wire_cmp[t];
        sum_wu += wire_upd[t];
    }
    double rbias = sum_e / trials;
    double rmse = std::sqrt(sum_e2 / trials);
    double mean_rl = sum_rl / trials;
    double mean_wc = sum_wc / trials;
    double se_var = rmse * std::sqrt(1.0 / (2.0 * trials));

    CellStats s;
    s.method = method;
    s.lg_k = lg_k;
    s.n_val = n_val;
    s.trials = trials;
    s.ram_live_mean = mean_rl;
    s.ram_peak_mean = sum_rp / trials;
    s.wire_compact_mean = mean_wc;
    s.wire_updatable_mean = sum_wu / trials;
    s.rel_bias_pct = rbias * 100.0;
    s.rmse_pct = rmse * 100.0;
    // Standard MVP = (mean_bytes * 8 bits) * RMSE^2
    s.mvp_ram_live = (mean_rl * 8.0) * (rmse * rmse);
    s.mvp_wire_compact = (mean_wc * 8.0) * (rmse * rmse);
    return s;
}

struct TrialOut {
    double est;
    size_t ram_live;
    size_t ram_peak;
    size_t wire_compact;
    size_t wire_updatable;
};

int main(int argc, char** argv) {
    std::string out_dir = "../out";
    if (argc > 1) out_dir = argv[1];

    std::ofstream pareto_f(out_dir + "/official_datasketches_pareto.tsv");
    pareto_f << "method\tlg_k\tnominal_k\tn\ttrials\tram_live_bytes\tram_peak_bytes\twire_compact_bytes\twire_updatable_bytes\trel_bias_pct\trmse_pct\tmvp_ram_live\tmvp_wire_compact\n";

    std::vector<uint8_t> lg_ks = {10, 11, 12, 13};
    std::vector<std::pair<double, int>> n_grid = {
        {1e3, 200}, {1e4, 200}, {1e5, 200}, {1e6, 200}, {1e7, 32}
    };

    for (uint8_t lg_k : lg_ks) {
        size_t nom_k = size_t(1) << lg_k;
        for (auto [n_val, trials] : n_grid) {
            uint64_t n_u = static_cast<uint64_t>(n_val);

            // 1. Official Apache CPC_C++
            auto s_cpc = run_mc_cell("Apache_CPC", lg_k, n_val, trials, [&](int t, HeapTracker* ht) -> TrialOut {
                uint64_t seed = 0xCAFEBABE11110000ULL ^ (static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL) ^ (static_cast<uint64_t>(lg_k) << 32) ^ (n_u * 0xbf58476d1ce4e5b9ULL);
                TrackedAlloc<uint8_t> alloc(ht);
                TrackedCpc sk(lg_k, datasketches::DEFAULT_SEED, alloc);
                for (uint64_t i = 0; i < n_u; ++i) {
                    sk.update(wyhash64_seed(i, seed));
                }
                size_t rl = sizeof(sk) + ht->live_bytes;
                size_t rp = sizeof(sk) + ht->peak_bytes;
                auto bytes = sk.serialize();
                return TrialOut{sk.get_estimate(), rl, rp, bytes.size(), bytes.size()};
            });
            pareto_f << s_cpc.method << "\t" << int(lg_k) << "\t" << nom_k << "\t" << std::scientific << n_val << "\t" << s_cpc.trials
                     << "\t" << std::fixed << std::setprecision(1) << s_cpc.ram_live_mean << "\t" << s_cpc.ram_peak_mean
                     << "\t" << s_cpc.wire_compact_mean << "\t" << s_cpc.wire_updatable_mean
                     << "\t" << std::setprecision(4) << s_cpc.rel_bias_pct << "\t" << s_cpc.rmse_pct
                     << "\t" << s_cpc.mvp_ram_live << "\t" << s_cpc.mvp_wire_compact << "\n";

            // 2. Official Apache HLL_4 (start_full_size = false & true)
            for (bool full_sz : {false, true}) {
                std::string mname = full_sz ? "Apache_HLL4_Full" : "Apache_HLL4_Dyn";
                auto s_hll4 = run_mc_cell(mname, lg_k, n_val, trials, [&](int t, HeapTracker* ht) -> TrialOut {
                    uint64_t seed = 0xCAFEBABE22220000ULL ^ (static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL) ^ (static_cast<uint64_t>(lg_k) << 32) ^ (n_u * 0xbf58476d1ce4e5b9ULL);
                    TrackedAlloc<uint8_t> alloc(ht);
                    TrackedHll sk(lg_k, datasketches::HLL_4, full_sz, alloc);
                    for (uint64_t i = 0; i < n_u; ++i) {
                        sk.update(wyhash64_seed(i, seed));
                    }
                    size_t rl = sizeof(sk) + ht->live_bytes;
                    size_t rp = sizeof(sk) + ht->peak_bytes;
                    size_t wc = sk.get_compact_serialization_bytes();
                    size_t wu = sk.get_updatable_serialization_bytes();
                    return TrialOut{sk.get_estimate(), rl, rp, wc, wu};
                });
                pareto_f << s_hll4.method << "\t" << int(lg_k) << "\t" << nom_k << "\t" << std::scientific << n_val << "\t" << s_hll4.trials
                         << "\t" << std::fixed << std::setprecision(1) << s_hll4.ram_live_mean << "\t" << s_hll4.ram_peak_mean
                         << "\t" << s_hll4.wire_compact_mean << "\t" << s_hll4.wire_updatable_mean
                         << "\t" << std::setprecision(4) << s_hll4.rel_bias_pct << "\t" << s_hll4.rmse_pct
                         << "\t" << s_hll4.mvp_ram_live << "\t" << s_hll4.mvp_wire_compact << "\n";
            }

            // 3. Official Apache HLL_8 (start_full_size = false & true)
            for (bool full_sz : {false, true}) {
                std::string mname = full_sz ? "Apache_HLL8_Full" : "Apache_HLL8_Dyn";
                auto s_hll8 = run_mc_cell(mname, lg_k, n_val, trials, [&](int t, HeapTracker* ht) -> TrialOut {
                    uint64_t seed = 0xCAFEBABE33330000ULL ^ (static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL) ^ (static_cast<uint64_t>(lg_k) << 32) ^ (n_u * 0xbf58476d1ce4e5b9ULL);
                    TrackedAlloc<uint8_t> alloc(ht);
                    TrackedHll sk(lg_k, datasketches::HLL_8, full_sz, alloc);
                    for (uint64_t i = 0; i < n_u; ++i) {
                        sk.update(wyhash64_seed(i, seed));
                    }
                    size_t rl = sizeof(sk) + ht->live_bytes;
                    size_t rp = sizeof(sk) + ht->peak_bytes;
                    size_t wc = sk.get_compact_serialization_bytes();
                    size_t wu = sk.get_updatable_serialization_bytes();
                    return TrialOut{sk.get_estimate(), rl, rp, wc, wu};
                });
                pareto_f << s_hll8.method << "\t" << int(lg_k) << "\t" << nom_k << "\t" << std::scientific << n_val << "\t" << s_hll8.trials
                         << "\t" << std::fixed << std::setprecision(1) << s_hll8.ram_live_mean << "\t" << s_hll8.ram_peak_mean
                         << "\t" << s_hll8.wire_compact_mean << "\t" << s_hll8.wire_updatable_mean
                         << "\t" << std::setprecision(4) << s_hll8.rel_bias_pct << "\t" << s_hll8.rmse_pct
                         << "\t" << s_hll8.mvp_ram_live << "\t" << s_hll8.mvp_wire_compact << "\n";
            }
            pareto_f.flush();
            std::printf("[Pareto] lg_k=%2d n=%.0e done (CPC: rl=%.0fB wc=%.0fB mvp_ram=%.3f mvp_wire=%.3f)\n",
                        int(lg_k), n_val, s_cpc.ram_live_mean, s_cpc.wire_compact_mean, s_cpc.mvp_ram_live, s_cpc.mvp_wire_compact);
            std::fflush(stdout);
        }
    }

    // Single-core throughput latency comparison at nominal ~4 KB budget:
    // HLL_8(lg_k=12 -> 4096 regs = 4KB array), HLL_4(lg_k=13 -> 8192 nibbles = 4KB array), CPC(lg_k=11, 12)
    std::ofstream sys_f(out_dir + "/official_datasketches_sys.tsv");
    sys_f << "method\tlg_k\tupdate_1e4_ns\tupdate_1e7_ns\tmerge_equal_us\tmerge_skew_us\testimate_us\tserialize_ns\tram_live_1e7\twire_compact_1e7\n";

    auto bench_sys_cpc = [&](uint8_t lg_k) {
        const size_t n_cold = 10000;
        const int reps_cold = 200;
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0.0;
        for (int r = 0; r < reps_cold; ++r) {
            datasketches::cpc_sketch sk(lg_k);
            for (size_t i = 0; i < n_cold; ++i) sk.update(wyhash64_seed(i, r));
            sink += sk.get_estimate();
        }
        auto t1 = std::chrono::steady_clock::now();
        double upd_1e4 = std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps_cold * n_cold);

        HeapTracker ht;
        TrackedAlloc<uint8_t> alloc(&ht);
        TrackedCpc sk1e7(lg_k, datasketches::DEFAULT_SEED, alloc);
        const size_t n_1e7 = 10000000;
        t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n_1e7; ++i) sk1e7.update(wyhash64_seed(i, 123));
        t1 = std::chrono::steady_clock::now();
        double upd_1e7 = std::chrono::duration<double, std::nano>(t1 - t0).count() / n_1e7;
        size_t rl = sizeof(sk1e7) + ht.live_bytes;
        size_t wc = sk1e7.serialize().size();

        datasketches::cpc_sketch sk_a(lg_k), sk_b(lg_k), sk_s(lg_k);
        for (size_t i = 0; i < 100000; ++i) { sk_a.update(wyhash64_seed(i, 1)); sk_b.update(wyhash64_seed(i, 2)); }
        for (size_t i = 0; i < 1000; ++i) sk_s.update(wyhash64_seed(i, 3));
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            datasketches::cpc_union u(lg_k);
            u.update(sk_a); u.update(sk_b);
            sink += u.get_result().get_estimate();
        }
        t1 = std::chrono::steady_clock::now();
        double m_eq = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            datasketches::cpc_union u(lg_k);
            u.update(sk_a); u.update(sk_s);
            sink += u.get_result().get_estimate();
        }
        t1 = std::chrono::steady_clock::now();
        double m_sk = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 2000; ++r) sink += sk_a.get_estimate();
        t1 = std::chrono::steady_clock::now();
        double est_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 2000.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 500; ++r) { auto bv = sk_a.serialize(); sink += bv.size(); }
        t1 = std::chrono::steady_clock::now();
        double ser_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / 500.0;

        sys_f << "Apache_CPC\t" << int(lg_k) << "\t" << upd_1e4 << "\t" << upd_1e7 << "\t" << m_eq << "\t" << m_sk << "\t" << est_us << "\t" << ser_ns << "\t" << rl << "\t" << wc << "\n";
    };

    auto bench_sys_hll = [&](const std::string& label, uint8_t lg_k, datasketches::target_hll_type tp, bool full_sz) {
        const size_t n_cold = 10000;
        const int reps_cold = 200;
        volatile double sink = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps_cold; ++r) {
            datasketches::hll_sketch sk(lg_k, tp, full_sz);
            for (size_t i = 0; i < n_cold; ++i) sk.update(wyhash64_seed(i, r));
            sink += sk.get_estimate();
        }
        auto t1 = std::chrono::steady_clock::now();
        double upd_1e4 = std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps_cold * n_cold);

        HeapTracker ht;
        TrackedAlloc<uint8_t> alloc(&ht);
        TrackedHll sk1e7(lg_k, tp, full_sz, alloc);
        const size_t n_1e7 = 10000000;
        t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n_1e7; ++i) sk1e7.update(wyhash64_seed(i, 123));
        t1 = std::chrono::steady_clock::now();
        double upd_1e7 = std::chrono::duration<double, std::nano>(t1 - t0).count() / n_1e7;
        size_t rl = sizeof(sk1e7) + ht.live_bytes;
        size_t wc = sk1e7.get_compact_serialization_bytes();

        datasketches::hll_sketch sk_a(lg_k, tp, full_sz), sk_b(lg_k, tp, full_sz), sk_s(lg_k, tp, full_sz);
        for (size_t i = 0; i < 100000; ++i) { sk_a.update(wyhash64_seed(i, 1)); sk_b.update(wyhash64_seed(i, 2)); }
        for (size_t i = 0; i < 1000; ++i) sk_s.update(wyhash64_seed(i, 3));
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            datasketches::hll_union u(lg_k);
            u.update(sk_a); u.update(sk_b);
            sink += u.get_estimate();
        }
        t1 = std::chrono::steady_clock::now();
        double m_eq = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            datasketches::hll_union u(lg_k);
            u.update(sk_a); u.update(sk_s);
            sink += u.get_estimate();
        }
        t1 = std::chrono::steady_clock::now();
        double m_sk = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 2000; ++r) sink += sk_a.get_estimate();
        t1 = std::chrono::steady_clock::now();
        double est_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 2000.0;

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 500; ++r) { auto bv = sk_a.serialize_compact(); sink += bv.size(); }
        t1 = std::chrono::steady_clock::now();
        double ser_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / 500.0;

        sys_f << label << "\t" << int(lg_k) << "\t" << upd_1e4 << "\t" << upd_1e7 << "\t" << m_eq << "\t" << m_sk << "\t" << est_us << "\t" << ser_ns << "\t" << rl << "\t" << wc << "\n";
    };

    bench_sys_cpc(11);
    bench_sys_cpc(12);
    bench_sys_hll("Apache_HLL4_Dyn", 13, datasketches::HLL_4, false);
    bench_sys_hll("Apache_HLL4_Full", 13, datasketches::HLL_4, true);
    bench_sys_hll("Apache_HLL8_Dyn", 12, datasketches::HLL_8, false);
    bench_sys_hll("Apache_HLL8_Full", 12, datasketches::HLL_8, true);

    std::cout << "Completed official_datasketches_pareto.tsv and official_datasketches_sys.tsv in " << out_dir << std::endl;
    return 0;
}
