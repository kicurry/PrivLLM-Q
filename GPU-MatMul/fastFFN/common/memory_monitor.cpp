#include "memory_monitor.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "utils.h"

using namespace std;

// Global variable to control memory monitoring
static bool g_memory_monitoring_enabled = false;

FFNMemoryMonitor::FFNMemoryMonitor(bool enabled) : enabled_(enabled)
{}

void FFNMemoryMonitor::start_stage(const std::string &stage_name)
{
    if (!enabled_)
        return;

    current_stage = stage_name;
    take_snapshot(stage_name);
}

void FFNMemoryMonitor::take_snapshot(const std::string &stage_name)
{
    if (!enabled_)
        return;

#ifdef USE_HE_GPU
    // Get current GPU memory status
    size_t free_memory;
    size_t total_memory;
    cudaError_t result = cudaMemGetInfo(&free_memory, &total_memory);

    if (result == cudaSuccess)
    {
        MemorySnapshot snapshot;
        snapshot.free_memory = free_memory;
        snapshot.total_memory = total_memory;
        snapshot.used_memory = total_memory - free_memory;
        snapshot.timestamp = chrono::high_resolution_clock::now();
        snapshot.stage_name = stage_name;
        snapshots.push_back(snapshot);
    }
#endif
}

void FFNMemoryMonitor::end_stage()
{
    if (!enabled_)
        return;

    if (!current_stage.empty())
    {
        take_snapshot("End of " + current_stage);
        current_stage.clear();
    }
}

size_t FFNMemoryMonitor::get_peak_mem_usage() const
{
    if (snapshots.empty())
        return 0;

    auto max_it = max_element(snapshots.begin(), snapshots.end(), [](const MemorySnapshot &a, const MemorySnapshot &b) {
        return a.used_memory < b.used_memory;
    });

    return max_it->used_memory;
}

size_t FFNMemoryMonitor::get_memory_increment(const std::string &stage) const
{
    if (snapshots.size() < 2)
        return 0;

    auto it = find_if(snapshots.begin(), snapshots.end(), [&stage](const MemorySnapshot &snapshot) {
        return snapshot.stage_name == stage;
    });

    if (it == snapshots.end() || it == snapshots.begin())
        return 0;

    auto prev_it = it - 1;
    return (it->used_memory > prev_it->used_memory) ? (it->used_memory - prev_it->used_memory) : 0;
}

void FFNMemoryMonitor::print_memory_report() const
{
    if (!enabled_ || snapshots.empty())
        return;

    using namespace Colors;

    cout << "\n" << string(60, '=') << endl;
    cout << BOLD << CYAN << "GPU MEMORY USAGE MONITORING" << RESET << endl;
    cout << string(60, '=') << endl;

    // Table header with proper alignment
    cout << BOLD << setw(30) << left << "Stage" << setw(18) << right << "Free Memory" << setw(18) << right
         << "Used Memory" << setw(18) << right << "Increment" << RESET << endl;
    cout << string(84, '-') << endl;

    for (size_t i = 0; i < snapshots.size(); ++i)
    {
        const auto &snapshot = snapshots[i];
        double free_gb = static_cast<double>(snapshot.free_memory) / (1024.0 * 1024.0 * 1024.0);
        double used_gb = static_cast<double>(snapshot.used_memory) / (1024.0 * 1024.0 * 1024.0);

        // Stage name (left aligned)
        cout << setw(30) << left << snapshot.stage_name;

        // Free memory (right aligned)
        cout << setw(18) << right << fixed << setprecision(2) << free_gb << " GB";

        // Used memory (right aligned)
        cout << setw(18) << right << fixed << setprecision(2) << used_gb << " GB";

        // Increment (right aligned)
        if (i > 0)
        {
            size_t increment = snapshot.used_memory - snapshots[i - 1].used_memory;
            if (increment > 0)
            {
                double inc_gb = static_cast<double>(increment) / (1024.0 * 1024.0 * 1024.0);
                cout << setw(18) << right << fixed << setprecision(2) << GREEN << "+" << inc_gb << " GB" << RESET;
            }
            else if (increment < 0)
            {
                double inc_gb = static_cast<double>(-increment) / (1024.0 * 1024.0 * 1024.0);
                cout << setw(18) << right << fixed << setprecision(2) << YELLOW << "-" << inc_gb << " GB" << RESET;
            }
            else
            {
                cout << setw(18) << right << "0 GB";
            }
        }
        else
        {
            cout << setw(18) << right << "-";
        }
        cout << endl;
    }

    cout << string(84, '-') << endl;

    // Summary statistics
    size_t peak_usage = get_peak_mem_usage();
    double peak_gb = static_cast<double>(peak_usage) / (1024.0 * 1024.0 * 1024.0);

    cout << "\n" << BOLD << CYAN << "MEMORY USAGE SUMMARY:" << RESET << endl;
    cout << string(30, '-') << endl;
    cout << BOLD << WHITE << "Peak Memory Usage: " << RESET << fixed << setprecision(2) << peak_gb << " GB" << endl;

    if (snapshots.size() > 1)
    {
        size_t total_fluctuation = 0;
        for (size_t i = 1; i < snapshots.size(); ++i)
        {
            total_fluctuation +=
                abs(static_cast<long long>(snapshots[i].used_memory) -
                    static_cast<long long>(snapshots[i - 1].used_memory));
        }
        double fluct_gb = static_cast<double>(total_fluctuation) / (1024.0 * 1024.0 * 1024.0);
        cout << BOLD << WHITE << "Total Memory Fluctuation: " << RESET << fixed << setprecision(2) << fluct_gb << " GB"
             << endl;
    }

    cout << string(60, '=') << endl << endl;
}

size_t FFNMemoryMonitor::estimate_ctxt_mem(size_t poly_modulus_degree, size_t coeff_modulus_size)
{
    // Ciphertext: 2 * poly_modulus_degree * coeff_modulus_size * sizeof(uint64_t)
    return 2 * poly_modulus_degree * coeff_modulus_size * sizeof(uint64_t);
}

size_t FFNMemoryMonitor::estimate_ptxt_mem(size_t poly_modulus_degree, size_t coeff_modulus_size)
{
    // Plaintext: poly_modulus_degree * coeff_modulus_size * sizeof(uint64_t)
    return poly_modulus_degree * coeff_modulus_size * sizeof(uint64_t);
}

size_t FFNMemoryMonitor::estimate_bsgs_mem(
    size_t baby_step, size_t giant_step, size_t num_cols, size_t ciphertext_size, size_t plaintext_size)
{
    // BSGS algorithm memory estimation
    size_t baby_ctxts_memory = baby_step * ciphertext_size; // baby-step ciphertexts
    size_t giant_ctxts_memory = num_cols * giant_step * ciphertext_size; // giant-step ciphertexts
    size_t result_ctxts_memory = num_cols * ciphertext_size; // result ciphertexts

    return baby_ctxts_memory + giant_ctxts_memory + result_ctxts_memory;
}

bool FFNMemoryMonitor::parse_mem_monitor_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--memory-monitor" || arg == "-m")
        {
            g_memory_monitoring_enabled = true;
            cout << "GPU Memory monitoring enabled via command line argument" << endl;
            return true;
        }
    }
    cout << "GPU Memory monitoring disabled (default)" << endl;
    return false;
}

bool FFNMemoryMonitor::is_memory_monitoring_enabled()
{
    return g_memory_monitoring_enabled;
}
