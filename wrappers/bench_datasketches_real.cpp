#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpc_sketch.hpp"
#include "cpc_union.hpp"
#include "hll.hpp"

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
    TrackedAlloc(const TrackedAlloc<U>& o) noexcept : tracker(o.tracker) {}
    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        T* p = static_cast<T*>(::operator new(bytes));
        if (tracker) {
            tracker->live_bytes += bytes;
            if (tracker->live_bytes > tracker->peak_bytes) tracker->peak_bytes = tracker->live_bytes;
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
    template <typename U> bool operator==(const TrackedAlloc<U>& o) const noexcept { return tracker == o.tracker; }
    template <typename U> bool operator!=(const TrackedAlloc<U>& o) const noexcept { return tracker != o.tracker; }
};

using TrackedCpc = datasketches::cpc_sketch_alloc<TrackedAlloc<uint8_t>>;
using TrackedHll = datasketches::hll_sketch_alloc<TrackedAlloc<uint8_t>>;
using TrackedCpcUnion = datasketches::cpc_union_alloc<TrackedAlloc<uint8_t>>;
using TrackedHllUnion = datasketches::hll_union_alloc<TrackedAlloc<uint8_t>>;

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t hash_item(uint64_t key, uint64_t seed) {
    return splitmix64(key ^ seed);
}

struct Workload {
    std::string name;
    std::vector<uint64_t> events;
    size_t true_distinct = 0;
};

static size_t count_distinct(std::vector<uint64_t> copy) {
    if (copy.empty()) return 0;
    std::sort(copy.begin(), copy.end());
    return static_cast<size_t>(std::unique(copy.begin(), copy.end()) - copy.begin());
}

static Workload load_edge_gz(const std::string& name, const std::string& path) {
    Workload w;
    w.name = name;
    std::string cmd = "gzip -dc \"" + path + "\"";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) { std::cerr << "Failed popen " << path << "\n"; std::exit(1); }
    char buf[512];
    while (fgets(buf, sizeof(buf), fp)) {
        if (buf[0] == '#' || buf[0] == '\n' || buf[0] == '\r') continue;
        char* end1 = nullptr;
        uint64_t u = strtoull(buf, &end1, 10);
        if (!end1 || end1 == buf) continue;
        uint64_t v = strtoull(end1, nullptr, 10);
        w.events.push_back((u << 32) ^ v);
    }
    pclose(fp);
    w.true_distinct = count_distinct(w.events);
    std::printf("[Loaded %s] events=%zu true_distinct=%zu\n", name.c_str(), w.events.size(), w.true_distinct);
    std::fflush(stdout);
    return w;
}

static Workload load_ecoli_31mer_gz(const std::string& path) {
    Workload w;
    w.name = "NCBI_Ecoli_31mers";
    std::string cmd = "gzip -dc \"" + path + "\"";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) { std::cerr << "Failed popen " << path << "\n"; std::exit(1); }
    char buf[4096];
    uint64_t kmer = 0;
    int len = 0;
    const uint64_t mask62 = (1ULL << 62) - 1;
    while (fgets(buf, sizeof(buf), fp)) {
        if (buf[0] == '>') { len = 0; kmer = 0; continue; }
        for (char* p = buf; *p && *p != '\n' && *p != '\r'; ++p) {
            uint64_t c = 99;
            switch (*p) {
                case 'A': case 'a': c = 0; break;
                case 'C': case 'c': c = 1; break;
                case 'G': case 'g': c = 2; break;
                case 'T': case 't': c = 3; break;
                default: len = 0; kmer = 0; continue;
            }
            kmer = ((kmer << 2) | c) & mask62;
            if (++len >= 31) {
                w.events.push_back(kmer);
            }
        }
    }
    pclose(fp);
    w.true_distinct = count_distinct(w.events);
    std::printf("[Loaded %s] events=%zu true_distinct=%zu\n", w.name.c_str(), w.events.size(), w.true_distinct);
    std::fflush(stdout);
    return w;
}

static Workload make_zipf(double alpha, size_t dom, size_t n_events, uint64_t rng_seed) {
    Workload w;
    std::ostringstream oss;
    oss << "Zipf_a" << std::fixed << std::setprecision(1) << alpha;
    w.name = oss.str();
    std::vector<double> cdf(dom);
    double acc = 0.0;
    for (size_t i = 0; i < dom; ++i) {
        acc += std::pow(static_cast<double>(i + 1), -alpha);
        cdf[i] = acc;
    }
    for (size_t i = 0; i < dom; ++i) cdf[i] /= acc;
    w.events.resize(n_events);
    uint64_t st = rng_seed;
    for (size_t i = 0; i < n_events; ++i) {
        st = splitmix64(st);
        double u = (st >> 11) * 0x1.0p-53;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        size_t idx = static_cast<size_t>(it - cdf.begin());
        if (idx >= dom) idx = dom - 1;
        w.events[i] = idx + 1;
    }
    w.true_distinct = count_distinct(w.events);
    std::printf("[Loaded %s] events=%zu true_distinct=%zu\n", w.name.c_str(), w.events.size(), w.true_distinct);
    std::fflush(stdout);
    return w;
}

static Workload make_highdup(size_t n_events, size_t dom) {
    Workload w;
    w.name = "HighDup_20M_10k";
    w.events.resize(n_events);
    for (size_t i = 0; i < n_events; ++i) {
        w.events[i] = (i * 2654435761ULL) % dom;
    }
    w.true_distinct = dom;
    return w;
}

static Workload make_uniform(size_t n_events) {
    Workload w;
    w.name = "Uniform_1e7";
    w.events.resize(n_events);
    for (size_t i = 0; i < n_events; ++i) w.events[i] = i;
    w.true_distinct = n_events;
    return w;
}

int main(int argc, char** argv) {
    bool multishard_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--multishard-only") {
            multishard_only = true;
        }
    }
    std::string ds_dir = "../datasets";
    std::string out_dir = "../out";

    if (!multishard_only) {
        std::vector<Workload> workloads;
        workloads.push_back(make_uniform(10000000));
        workloads.push_back(make_highdup(20000000, 10000));
        workloads.push_back(make_zipf(0.8, 200000, 3000000, 0x111));
        workloads.push_back(make_zipf(1.2, 200000, 3000000, 0x222));
        workloads.push_back(make_zipf(1.6, 200000, 3000000, 0x333));
        workloads.push_back(load_edge_gz("SNAP_web_Google", ds_dir + "/web-Google.txt.gz"));
        workloads.push_back(load_edge_gz("SNAP_cit_Patents", ds_dir + "/cit-Patents.txt.gz"));
        workloads.push_back(load_ecoli_31mer_gz(ds_dir + "/ecoli_k12.fna.gz"));

        std::ofstream rf(out_dir + "/official_datasketches_real_streams.tsv");
        rf << "workload\tevents\ttrue_distinct\tbudget\tmethod\tlg_k\tseeds\tram_live_bytes\twire_compact_bytes\trel_bias_pct\trmse_pct\teff_mvp_ram\teff_mvp_wire\n";

        struct BudgetCfg { std::string bname; uint8_t cpc_k; uint8_t hll4_k; uint8_t hll8_k; };
        std::vector<BudgetCfg> cfgs = {
            {"512B", 9, 10, 9},
            {"1KB", 10, 11, 10},
            {"4KB", 12, 13, 12},
        };
        const int n_seeds = 32;

        for (const auto& w : workloads) {
            double td = static_cast<double>(w.true_distinct);
            for (const auto& c : cfgs) {
                for (int m_id = 0; m_id < 5; ++m_id) {
                    std::string mname;
                    uint8_t lk = 10;
                    if (m_id == 0) { mname = "Apache_CPC"; lk = c.cpc_k; }
                    else if (m_id == 1) { mname = "Apache_HLL4_Dyn"; lk = c.hll4_k; }
                    else if (m_id == 2) { mname = "Apache_HLL4_Full"; lk = c.hll4_k; }
                    else if (m_id == 3) { mname = "Apache_HLL8_Dyn"; lk = c.hll8_k; }
                    else { mname = "Apache_HLL8_Full"; lk = c.hll8_k; }

                    std::vector<double> errs(n_seeds), rl(n_seeds), wc(n_seeds);
#pragma omp parallel for schedule(dynamic, 1)
                    for (int s = 0; s < n_seeds; ++s) {
                        uint64_t seed = 0xD00DBEEF00000000ULL ^ (static_cast<uint64_t>(s) * 0x9e3779b97f4a7c15ULL);
                        HeapTracker ht;
                        TrackedAlloc<uint8_t> alloc(&ht);
                        double est = 0.0;
                        size_t live = 0, wire = 0;
                        if (m_id == 0) {
                            TrackedCpc sk(lk, datasketches::DEFAULT_SEED, alloc);
                            for (uint64_t x : w.events) sk.update(hash_item(x, seed));
                            live = sizeof(sk) + ht.live_bytes;
                            wire = sk.serialize().size();
                            est = sk.get_estimate();
                        } else {
                            auto tp = (m_id <= 2) ? datasketches::HLL_4 : datasketches::HLL_8;
                            bool full = (m_id == 2 || m_id == 4);
                            TrackedHll sk(lk, tp, full, alloc);
                            for (uint64_t x : w.events) sk.update(hash_item(x, seed));
                            live = sizeof(sk) + ht.live_bytes;
                            wire = sk.get_compact_serialization_bytes();
                            est = sk.get_estimate();
                        }
                        errs[s] = (est - td) / td;
                        rl[s] = static_cast<double>(live);
                        wc[s] = static_cast<double>(wire);
                    }
                    double se = 0, se2 = 0, srl = 0, swc = 0;
                    for (int s = 0; s < n_seeds; ++s) {
                        se += errs[s]; se2 += errs[s] * errs[s]; srl += rl[s]; swc += wc[s];
                    }
                    double rbias = (se / n_seeds) * 100.0;
                    double rmse = std::sqrt(se2 / n_seeds);
                    double mrl = srl / n_seeds, mwc = swc / n_seeds;
                    rf << w.name << "\t" << w.events.size() << "\t" << w.true_distinct << "\t" << c.bname
                       << "\t" << mname << "\t" << int(lk) << "\t" << n_seeds
                       << "\t" << std::fixed << std::setprecision(1) << mrl << "\t" << mwc
                       << "\t" << std::setprecision(4) << rbias << "\t" << rmse * 100.0
                       << "\t" << (mrl * 8.0) * (rmse * rmse) << "\t" << (mwc * 8.0) * (rmse * rmse) << "\n";
                    rf.flush();
                }
            }
            std::printf("[Real Stream Done] %s\n", w.name.c_str());
            std::fflush(stdout);
        }
    }

    // Part 2: Multi-shard union drift check (K in {1, 4, 16, 64}, n=1e6, 100 trials, ~4 KB)
    std::ofstream mf(out_dir + "/official_datasketches_multishard.tsv");
    mf << "method\tlg_k\tn_shards\tn\ttrials\tram_live_bytes\twire_compact_bytes\trel_bias_pct\trmse_pct\tmvp_ram_live\tmvp_wire_compact\n";
    const size_t n_ms = 1000000;
    const int ms_trials = 100;
    for (int K : {1, 4, 16, 64}) {
        for (int mid = 0; mid < 3; ++mid) {
            std::string mname = (mid == 0) ? "Apache_CPC" : (mid == 1 ? "Apache_HLL4_Dyn" : "Apache_HLL8_Dyn");
            uint8_t lk = (mid == 0) ? 12 : (mid == 1 ? 13 : 12);
            std::vector<double> errs(ms_trials), rl(ms_trials), wc(ms_trials);
#pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < ms_trials; ++t) {
                uint64_t seed = 0x5555AAAA00000000ULL ^ (static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL);
                double est = 0.0;
                size_t live = 0, wire = 0;
                if (mid == 0) {
                    HeapTracker uht;
                    TrackedAlloc<uint8_t> ualloc(&uht);
                    auto do_union = [&]() -> TrackedCpc {
                        TrackedCpcUnion u(lk, datasketches::DEFAULT_SEED, ualloc);
                        for (int k = 0; k < K; ++k) {
                            TrackedCpc sk(lk);
                            for (size_t i = static_cast<size_t>(k); i < n_ms; i += K) sk.update(hash_item(i, seed));
                            u.update(sk);
                        }
                        return u.get_result();
                    };
                    TrackedCpc res = do_union();
                    if (uht.live_bytes == 0) {
                        std::cerr << "FATAL: Apache_CPC union result uht.live_bytes == 0!\n";
                        std::exit(1);
                    }
                    live = sizeof(res) + uht.live_bytes;
                    wire = res.serialize().size();
                    est = res.get_estimate();
                } else {
                    auto tp = (mid == 1) ? datasketches::HLL_4 : datasketches::HLL_8;
                    datasketches::hll_union u(lk);
                    for (int k = 0; k < K; ++k) {
                        datasketches::hll_sketch sk(lk, tp, false);
                        for (size_t i = static_cast<size_t>(k); i < n_ms; i += K) sk.update(hash_item(i, seed));
                        u.update(sk);
                    }
                    datasketches::hll_sketch res = u.get_result(tp);
                    live = res.get_updatable_serialization_bytes();
                    wire = res.get_compact_serialization_bytes();
                    est = res.get_estimate();
                }
                errs[t] = (est - static_cast<double>(n_ms)) / static_cast<double>(n_ms);
                rl[t] = static_cast<double>(live);
                wc[t] = static_cast<double>(wire);
            }
            double se = 0, se2 = 0, srl = 0, swc = 0;
            for (int t = 0; t < ms_trials; ++t) { se += errs[t]; se2 += errs[t] * errs[t]; srl += rl[t]; swc += wc[t]; }
            double rbias = (se / ms_trials) * 100.0;
            double rmse = std::sqrt(se2 / ms_trials);
            double mrl = srl / ms_trials, mwc = swc / ms_trials;
            mf << mname << "\t" << int(lk) << "\t" << K << "\t" << n_ms << "\t" << ms_trials
               << "\t" << std::fixed << std::setprecision(1) << mrl << "\t" << mwc
               << "\t" << std::setprecision(4) << rbias << "\t" << rmse * 100.0
               << "\t" << (mrl * 8.0) * (rmse * rmse) << "\t" << (mwc * 8.0) * (rmse * rmse) << "\n";
            mf.flush();
        }
    }
    std::cout << "DONE official_datasketches_real_streams & multishard!\n";
    return 0;
}
