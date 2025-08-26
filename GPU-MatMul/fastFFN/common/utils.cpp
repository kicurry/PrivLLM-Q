
#include "utils.h"
#include <iomanip>
#include <random>
#ifdef USE_HE_GPU
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace seal;

#define FIX_RAND_SEED

#ifdef FIX_RAND_SEED
mt19937 gen(0);
#else
random_device rd;
mt19937 gen(rd);
#endif

void print_parameters(const SEALContext &context)
{
    using namespace Colors;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << BOLD << CYAN << "ENCRYPTION PARAMETERS" << RESET << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    auto &context_data = *context.key_context_data();
    const auto &parms = context_data.parms();

    /*
    Which scheme are we using?
    */
    string scheme_name;
    switch (parms.scheme())
    {
    case scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw invalid_argument("unsupported scheme");
    }
    std::cout << BOLD << BLUE << "Encryption Scheme: " << RESET << WHITE << scheme_name << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    std::cout << BOLD << GREEN << "Polynomial Modulus Degree: " << RESET << WHITE << parms.poly_modulus_degree()
              << RESET << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << BOLD << YELLOW << "Coefficient Modulus Size: " << RESET;
    std::cout << WHITE << context_data.total_coeff_modulus_bit_count() << " bits" << RESET << std::endl;

    std::cout << BOLD << MAGENTA << "   Breakdown: " << RESET << WHITE << "(";
    auto coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    for (size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ")" << RESET << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (parms.scheme() == scheme_type::bfv)
    {
        std::cout << BOLD << CYAN << "Plain Modulus: " << RESET << WHITE << parms.plain_modulus().value() << " ("
                  << parms.plain_modulus().bit_count() << " bits)" << RESET << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;
}

void print_cuda_device_info(bool disable_gpu)
{
    using namespace Colors;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << BOLD << CYAN << "ENVIRONMENT INFORMATION" << RESET << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Build type with color coding
#ifdef NDEBUG
    std::cout << BOLD << GREEN << "BUILD TYPE: RELEASE MODE" << RESET << std::endl;
#else
    std::cout << BOLD << YELLOW << "BUILD TYPE: DEBUG MODE" << RESET << std::endl;
#endif
    std::cout << std::string(40, '-') << std::endl;

    // CPU basic information
    std::cout << BOLD << MAGENTA << "CPU INFORMATION:" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    // Get number of CPU cores
    unsigned int num_cpus = std::thread::hardware_concurrency();
    std::cout << BOLD << WHITE << "   CPU Cores: " << RESET << num_cpus << std::endl;

// Get CPU model (simplified version, platform-specific code may be needed for details)
#ifdef _WIN32
    // Windows system CPU information
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << BOLD << WHITE << "   Architecture: " << RESET << sysInfo.dwNumberOfProcessors << " processors"
              << std::endl;
#elif defined(__linux__)
    // Linux system can get more details by reading /proc/cpuinfo
    std::cout << BOLD << WHITE << "   System: " << RESET << "Linux" << std::endl;
#elif defined(__APPLE__)
    // macOS system
    std::cout << BOLD << WHITE << "   System: " << RESET << "macOS" << std::endl;
#endif
    std::cout << std::string(40, '-') << std::endl;

#ifdef USE_HE_GPU
    if (!disable_gpu)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0)
        {
            std::cout << BOLD << RED << "No CUDA-capable devices found" << RESET << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            return;
        }

        // Get current device
        int currentDevice;
        cudaGetDevice(&currentDevice);

        // Get current device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, currentDevice);

        std::cout << BOLD << MAGENTA << "CURRENT DEVICE: " << WHITE << "Device " << currentDevice << " ("
                  << deviceProp.name << ")" << RESET << std::endl;
        std::cout << std::string(40, '-') << std::endl;

        // Get memory info
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);

        double freeMemoryGB = static_cast<double>(freeMemory) / (1024.0 * 1024.0 * 1024.0);
        double totalMemoryGB = static_cast<double>(totalMemory) / (1024.0 * 1024.0 * 1024.0);
        double usedMemoryGB = totalMemoryGB - freeMemoryGB;

        std::cout << BOLD << CYAN << "GPU MEMORY STATUS:" << RESET << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << BOLD << WHITE << "   Total Memory: " << RESET << std::fixed << std::setprecision(2)
                  << totalMemoryGB << " GB" << std::endl;
        std::cout << BOLD << GREEN << "   Free Memory:  " << RESET << std::fixed << std::setprecision(2) << freeMemoryGB
                  << " GB" << std::endl;
        std::cout << BOLD << YELLOW << "   Used Memory:  " << RESET << std::fixed << std::setprecision(2)
                  << usedMemoryGB << " GB" << std::endl;

        // Memory usage bar
        double usagePercent = (usedMemoryGB / totalMemoryGB) * 100.0;
        std::cout << "\n" << BOLD << BLUE << "GPU Memory Usage: " << RESET;
        std::cout << std::fixed << std::setprecision(1) << usagePercent << "%" << std::endl;

        // Visual memory bar
        int barWidth = 30;
        int filledWidth = static_cast<int>((usagePercent / 100.0) * barWidth);
        std::cout << "   [";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < filledWidth)
            {
                std::cout << GREEN << "=";
            }
            else
            {
                std::cout << WHITE << "-";
            }
        }
        std::cout << RESET << "]" << std::endl;

        std::cout << std::string(60, '=') << std::endl;
#else
    std::cout << BOLD << RED << "CUDA support not enabled (USE_HE_GPU not defined)" << RESET << std::endl;
    std::cout << std::string(60, '=') << std::endl;
#endif
    }
}

void fill_random_vector(vector<uint64_t> &vec, uint64_t plainWidth)
{
    uniform_int_distribution<uint64_t> dis(0, (1ULL << plainWidth) - 1);
    for (auto &v : vec)
    {
        v = dis(gen);
    }
}

void fill_random_weight(vector<uint64_t> &vec, size_t copy_count, uint64_t plainWidth)
{
    uniform_int_distribution<uint64_t> dis(0, (1ULL << plainWidth) - 1);

    auto len = vec.size() / copy_count;
    for (size_t i = 0; i < copy_count; i++)
    {
        auto random_val = dis(gen);
        for (size_t j = 0; j < len; j++)
        {
            vec[i * len + j] = random_val;
        }
    }
}

size_t get_baby_step(size_t M)
{
    size_t minval = M, maxk = 0;
    for (size_t k = 1; k <= 3 * sqrt(M); k++)
    {
        auto currval = ceil((M + 0.0) / (k + 0.0)) + k - 1;
        if (currval <= minval)
        {
            minval = currval;
            maxk = max(maxk, k);
        }
    }
    return maxk;
}