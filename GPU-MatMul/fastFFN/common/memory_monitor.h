#pragma once

#include <chrono>
#include <string>
#include <vector>

#ifdef USE_HE_GPU
#include <cuda_runtime.h>
#endif

class FFNMemoryMonitor
{
private:
    struct MemorySnapshot
    {
        size_t free_memory;
        size_t total_memory;
        size_t used_memory;
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string stage_name;
    };

    std::vector<MemorySnapshot> snapshots;
    std::string current_stage;
    bool enabled_;

public:
    FFNMemoryMonitor(bool enabled = true);

    void setEnabled(bool enabled)
    {
        enabled_ = enabled;
    }
    bool is_enabled() const
    {
        return enabled_;
    }

    void start_stage(const std::string &stage_name);
    void take_snapshot(const std::string &stage_name);
    void end_stage();

    // Memory usage analysis
    size_t get_peak_mem_usage() const;
    size_t get_memory_increment(const std::string &stage) const;
    void print_memory_report() const;

    // Memory estimation calculation
    static size_t estimate_ctxt_mem(size_t poly_modulus_degree, size_t coeff_modulus_size);
    static size_t estimate_ptxt_mem(size_t poly_modulus_degree, size_t coeff_modulus_size);
    static size_t estimate_bsgs_mem(
        size_t baby_step, size_t giant_step, size_t num_cols, size_t ciphertext_size, size_t plaintext_size);

    // Command line argument parsing
    static bool parse_mem_monitor_args(int argc, char *argv[]);
    static bool is_memory_monitoring_enabled();
};
