#include "ffn_utils.h"
#include <seal/keygenerator.h>
#include "common/memory_monitor.h"
#include "common/utils.h"

#define BFV_BATCH_NUM 2UL

using namespace std;
using namespace seal;
using namespace HE::unified;

// Global variable to control noise budget monitoring
static bool g_noise_budget_monitoring_enabled = true;

std::vector<int> generate_bsgs_steps(size_t seq_len, int bs, int gs)
{
    std::vector<int> steps;
    for (int i = 1; i < bs; i++)
    {
        steps.push_back(i * seq_len);
    }
    for (int i = 1; i < gs; i++)
    {
        steps.push_back(i * seq_len * bs);
    }
    return steps;
}

ffnInitizer::ffnInitizer(
    const seal::EncryptionParameters &params, const FFNConfig &ffn_config, Datatype::LOCATION backend,
    optional<BSGSConfig> custom_bsgs_config)
    : backend(backend)
{
    // Initialize Activation config (Plaintext)
    activation_config.activation_rows = ffn_config.seq_len;
    activation_config.activation_cols = ffn_config.activation_cols;

    // Initialize Weight config (Plaintext)
    weight_config.weight_rows = ffn_config.activation_cols;
    weight_config.weight_cols = ffn_config.weight_cols;

    context = new UnifiedContext(params, backend);
    print_parameters(*context);

    // Initialize keys
    secretKeys = new SecretKey();
    publicKeys = new PublicKey();
    galoisKeys = new UnifiedGaloisKeys(HOST);

    KeyGenerator keygen(*context);
    *secretKeys = keygen.secret_key();
    keygen.create_public_key(*publicKeys);

    // Initialize runtime objects
    encryptor = new Encryptor(*context, *publicKeys);
    decryptor = new Decryptor(*context, *secretKeys);
    encoder = new UnifiedBatchEncoder(*context);
    evaluator = new UnifiedEvaluator(*context);

    // Initialize Activation config (Ciphertext)
    size_t slot_count = encoder->slot_count();
    this->bfv_row_size = slot_count / BFV_BATCH_NUM;
    activation_config.num_activation_ctxt =
        activation_config.activation_rows * activation_config.activation_cols / bfv_row_size;
    activation_config.num_col_per_act_ctxt = bfv_row_size / activation_config.activation_rows;
    activation_config.activation_bits = ffn_config.activation_bits;

    // Initialize Weight config (Ciphertext)
    weight_config.tile_size = activation_config.num_col_per_act_ctxt;
    weight_config.num_tiled_weight_rows = weight_config.weight_rows / weight_config.tile_size;
    weight_config.num_tiled_weight_cols = weight_config.weight_cols / weight_config.tile_size / BFV_BATCH_NUM;
    weight_config.weight_bits = ffn_config.weight_bits;

    // Generate or validate BSGS steps
    if (custom_bsgs_config.has_value())
    {
        // Use custom BSGS configuration
        bsgs_config = custom_bsgs_config.value();

        // Validate BSGS configuration
        // TODO: <
        if (bsgs_config.bs * bsgs_config.gs != activation_config.num_col_per_act_ctxt)
        {
            throw std::invalid_argument(
                "Invalid BSGS configuration: bs * gs must equal tile_size (" +
                std::to_string(activation_config.num_col_per_act_ctxt) +
                "), but got bs=" + std::to_string(bsgs_config.bs) + ", gs=" + std::to_string(bsgs_config.gs));
        }

        cout << "Using custom BSGS configuration: bs=" << bsgs_config.bs << ", gs=" << bsgs_config.gs << endl;
    }
    else
    {
        // Generate BSGS steps using default strategy
        bsgs_config.bs = get_baby_step(activation_config.num_col_per_act_ctxt);
        bsgs_config.gs = activation_config.num_col_per_act_ctxt / bsgs_config.bs;
        cout << "Generated BSGS configuration: bs=" << bsgs_config.bs << ", gs=" << bsgs_config.gs << endl;
    }

    auto bsgs_steps = generate_bsgs_steps(ffn_config.seq_len, bsgs_config.bs, bsgs_config.gs);
    // keygen.create_galois_keys(*galoisKeys);
    keygen.create_galois_keys(bsgs_steps, *galoisKeys); // TODO
    if (backend == Datatype::DEVICE)
    {
        galoisKeys->to_device(*context);
    }
}

bool ffnInitizer::parse_ffn_backend(int argc, char *argv[], bool &disable_gpu)
{
    bool params_modified = false;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--disable-gpu" || arg == "-cpu")
        {
#ifdef USE_HE_GPU
            disable_gpu = true;
            params_modified = true;
            cout << "Disable GPU via command line" << endl;
#endif
        }
    }

    if (!params_modified)
    {
        cout << "Using default encryption parameters:" << endl;
        cout << "  Backend: " << (disable_gpu ? "CPU" : "GPU") << endl;
    }

    return true;
}

bool ffnInitizer::parse_ffn_config_args(
    int argc, char *argv[], int &activation_bits, int &weight_bits, size_t &seq_len, size_t &activation_cols,
    size_t &weight_cols)
{
    bool modified = false;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--activation-bits" || arg == "-abw")
        {
            if (i + 1 < argc)
            {
                activation_bits = stoi(argv[++i]);
                if (activation_bits <= 0)
                {
                    cerr << "Error: activation bits must be positive" << endl;
                    return false;
                }
                modified = true;
                cout << "Activation bit width set to: " << activation_bits << endl;
            }
            else
            {
                cerr << "Error: --activation-bits requires a value" << endl;
                return false;
            }
        }
        else if (arg == "--weight-bits" || arg == "-wbw")
        {
            if (i + 1 < argc)
            {
                weight_bits = stoi(argv[++i]);
                if (weight_bits <= 0)
                {
                    cerr << "Error: weight bits must be positive" << endl;
                    return false;
                }
                modified = true;
                cout << "Weight bit width set to: " << weight_bits << endl;
            }
            else
            {
                cerr << "Error: --weight-bits requires a value" << endl;
                return false;
            }
        }
        else if (arg == "--ffn-config" || arg == "-ffn")
        {
            if (i + 1 < argc)
            {
                string ffn_str = argv[++i];
                vector<size_t> ffn_dims;

                // Parse comma-separated values like "128,4096,12288"
                size_t pos = 0;
                while ((pos = ffn_str.find(',')) != string::npos)
                {
                    ffn_dims.push_back(stoi(ffn_str.substr(0, pos)));
                    ffn_str.erase(0, pos + 1);
                }
                if (!ffn_str.empty())
                {
                    ffn_dims.push_back(stoi(ffn_str));
                }
                if (ffn_dims.size() != 3)
                {
                    cerr << "Error: --ffn-config requires exactly 3 values (seq_len, activation_cols, weight_cols)"
                         << endl;
                    return false;
                }
                seq_len = ffn_dims[0];
                activation_cols = ffn_dims[1];
                weight_cols = ffn_dims[2];

                if (seq_len <= 0 || activation_cols <= 0 || weight_cols <= 0)
                {
                    cerr << "Error: FFN dimensions must be positive" << endl;
                    return false;
                }

                modified = true;
                cout << "FFN configuration set to: [";
                for (size_t j = 0; j < ffn_dims.size(); ++j)
                {
                    if (j > 0)
                    {
                        cout << ", ";
                    }
                    cout << ffn_dims[j];
                }
                cout << "]" << endl;
            }
            else
            {
                cerr << "Error: --ffn-config requires a value (comma-separated)" << endl;
                return false;
            }
        }
    }

    if (!modified)
    {
        cout << "Using default bit widths:" << endl;
        cout << "  Activation bits: " << activation_bits << endl;
        cout << "  Weight bits: " << weight_bits << endl;
        cout << "  FFN configuration set to: [" << seq_len << ", " << activation_cols << ", " << weight_cols << "]"
             << endl;
    }
    return true;
}

bool ffnInitizer::parse_noise_budget_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--no-noise-monitor" || arg == "-nn")
        {
            g_noise_budget_monitoring_enabled = false;
            cout << "Noise budget monitoring disabled via command line argument" << endl;
            return true;
        }
    }
    cout << "Noise budget monitoring enabled (default)" << endl;
    return false;
}

bool ffnInitizer::parse_encryption_params_args(
    int argc, char *argv[], uint64_t &polyModulusDegree, uint64_t &plainWidth, vector<int> &rnsBitSizes)
{
    bool params_modified = false;

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];

        if (arg == "--poly-degree" || arg == "-d")
        {
            if (i + 1 < argc)
            {
                polyModulusDegree = stoull(argv[++i]);
                params_modified = true;
                cout << "Poly modulus degree set to: " << polyModulusDegree << endl;
            }
            else
            {
                cerr << "Error: --poly-degree requires a value" << endl;
                return false;
            }
        }
        else if (arg == "--plain-width" || arg == "-w")
        {
            if (i + 1 < argc)
            {
                plainWidth = stoull(argv[++i]);
                params_modified = true;
                cout << "Plain width set to: " << plainWidth << endl;
            }
            else
            {
                cerr << "Error: --plain-width requires a value" << endl;
                return false;
            }
        }
        else if (arg == "--rns-moduli" || arg == "-r")
        {
            if (i + 1 < argc)
            {
                string moduli_str = argv[++i];
                rnsBitSizes.clear();

                // Parse comma-separated values like "49,49,49"
                size_t pos = 0;
                while ((pos = moduli_str.find(',')) != string::npos)
                {
                    rnsBitSizes.push_back(stoi(moduli_str.substr(0, pos)));
                    moduli_str.erase(0, pos + 1);
                }
                if (!moduli_str.empty())
                {
                    rnsBitSizes.push_back(stoi(moduli_str));
                }

                params_modified = true;
                cout << "RNS bit sizes set to: [";
                for (size_t j = 0; j < rnsBitSizes.size(); ++j)
                {
                    if (j > 0)
                        cout << ", ";
                    cout << rnsBitSizes[j];
                }
                cout << "]" << endl;
            }
            else
            {
                cerr << "Error: --rns-moduli requires a value (comma-separated)" << endl;
                return false;
            }
        }
    }

    if (!params_modified)
    {
        cout << "Using default encryption parameters:" << endl;
        cout << "  Poly modulus degree: " << polyModulusDegree << endl;
        cout << "  Plain width: " << plainWidth << endl;
        cout << "  RNS bit sizes: [";
        for (size_t j = 0; j < rnsBitSizes.size(); ++j)
        {
            if (j > 0)
                cout << ", ";
            cout << rnsBitSizes[j];
        }
        cout << "]" << endl;
    }

    return true;
}

bool ffnInitizer::is_noise_budget_monitoring_enabled()
{
    return g_noise_budget_monitoring_enabled;
}

ffnInitizer::~ffnInitizer()
{
    delete secretKeys;
    delete publicKeys;
    delete galoisKeys;
    delete encryptor;
    delete decryptor;
    delete encoder;
    delete evaluator;
}

void ffnInitizer::init_matrices(ThreadPool &pool, vector<future<void>> &futures, bool preload_weights)
{
    size_t slot_count = encoder->slot_count();
    // Generate activation matrix
    size_t activation_rows = activation_config.activation_rows;
    size_t activation_cols = activation_config.activation_cols;
    activation_matrix.resize(activation_rows * activation_cols, 0ULL);
    fill_random_vector(activation_matrix, activation_config.activation_bits);

    // Generate weight matrix
    size_t weight_rows = weight_config.weight_rows;
    size_t weight_cols = weight_config.weight_cols;
    weight_matrix.resize(weight_rows * weight_cols, 1ULL);
    if (preload_weights)
    {
        fill_random_vector(weight_matrix, weight_config.weight_bits);
    }

    // Encrypt packed activation matrix
    auto start_time = chrono::high_resolution_clock::now();

    size_t num_activation_ctxt = activation_config.num_activation_ctxt;
    size_t num_col_per_act_ctxt = activation_config.num_col_per_act_ctxt;
    encrypted_activation.resize(num_activation_ctxt, HOST);
    cout << num_activation_ctxt << " * [" << BFV_BATCH_NUM << ", " << activation_rows << ", " << num_col_per_act_ctxt
         << "]" << endl;

    // Column-wise packing
    for (size_t i = 0; i < num_activation_ctxt; i++)
    {
        vector<uint64_t> packed_activation(slot_count, 0ULL);
        for (size_t j = 0; j < num_col_per_act_ctxt; j++)
        {
            for (size_t k = 0; k < activation_rows; k++)
            {
                for (size_t bfv_batch_idx = 0; bfv_batch_idx < BFV_BATCH_NUM; bfv_batch_idx++)
                {
                    packed_activation[bfv_batch_idx * bfv_row_size + j * activation_rows + k] =
                        activation_matrix[k * activation_cols + i * num_col_per_act_ctxt + j];
                }
            }
        }
        UnifiedPlaintext plain_activation(HOST);
        encoder->encode(packed_activation, plain_activation);
        encryptor->encrypt(plain_activation, encrypted_activation[i]);
        if (backend == Datatype::DEVICE)
        {
            encrypted_activation[i].to_device(*context);
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Encrypted packed activation matrix - Ready. Time: " << duration.count() << " ms" << endl;

    // Encode packed weight matrix
    size_t tile_size = weight_config.tile_size;
    size_t num_tiled_weight_rows = weight_config.num_tiled_weight_rows;
    size_t num_tiled_weight_cols = weight_config.num_tiled_weight_cols;
    size_t copy_count = bfv_row_size / tile_size;
    if (preload_weights)
    {
        cout << num_tiled_weight_rows << " * " << num_tiled_weight_cols << " * " << tile_size << " * [" << BFV_BATCH_NUM
             << ", " << copy_count << ", " << tile_size << "]" << "("
             << num_tiled_weight_cols * num_tiled_weight_rows * tile_size * slot_count * sizeof(uint64_t) /
                    static_cast<double>(1024 * 1024 * 1024)
             << "GB)" << endl;

        encoded_weight.resize(
            num_tiled_weight_rows * num_tiled_weight_cols, vector<UnifiedPlaintext>(tile_size, backend));

        pre_encode_weights(pool, futures);
    }
    else
    {
        cout << num_tiled_weight_rows << " * " << num_tiled_weight_cols << " * " << tile_size << " * [" << BFV_BATCH_NUM
             << ", " << copy_count << ", " << tile_size << "]" << " (Skip)" << endl;
    }
}

const vector<uint64_t> &ffnInitizer::get_activation_matrix() const
{
    return activation_matrix;
}

const vector<uint64_t> &ffnInitizer::get_weight_matrix() const
{
    return weight_matrix;
}

const vector<HE::unified::UnifiedCiphertext> &ffnInitizer::get_encrypted_activation() const
{
    return encrypted_activation;
}

const vector<vector<HE::unified::UnifiedPlaintext>> &ffnInitizer::get_encoded_weight() const
{
    return encoded_weight;
}

size_t ffnInitizer::get_num_activation_ctxt() const
{
    return activation_config.num_activation_ctxt;
}

size_t ffnInitizer::get_num_col_per_act_ctxt() const
{
    return activation_config.num_col_per_act_ctxt;
}

size_t ffnInitizer::get_tile_size() const
{
    return weight_config.tile_size;
}

size_t ffnInitizer::get_num_tiled_weight_rows() const
{
    return weight_config.num_tiled_weight_rows;
}

size_t ffnInitizer::get_num_tiled_weight_cols() const
{
    return weight_config.num_tiled_weight_cols;
}

void ffnInitizer::pre_encode_weights(ThreadPool &pool, vector<future<void>> &futures)
{
    size_t slot_count = encoder->slot_count();
    size_t num_tiled_weight_rows = weight_config.num_tiled_weight_rows;
    size_t num_tiled_weight_cols = weight_config.num_tiled_weight_cols;
    size_t tile_size = weight_config.tile_size;
    size_t copy_count = bfv_row_size / tile_size;

    auto start_time = chrono::high_resolution_clock::now();

    futures.clear();
    for (size_t i = 0; i < num_tiled_weight_rows; i++)
    {
        for (size_t j = 0; j < num_tiled_weight_cols; j++) // (i, j)-th tile
        {
            // FIXME: multi-stream
            // futures.push_back(pool.enqueue([&, i, j]() {
            size_t base_row_idx = i * tile_size;
            size_t base_col_idx = j * tile_size * BFV_BATCH_NUM;

            for (size_t di = 0; di < tile_size; di++) // (i, j, di)-th diagonal [down]
            {
                auto &plain = encoded_weight[i * num_tiled_weight_cols + j][di];

                auto expected_rot = di % bsgs_config.bs;
                auto right_rot = di - expected_rot;

                vector<uint64_t> packed_weight(slot_count, 0ULL);
                for (size_t batch_idx = 0; batch_idx < BFV_BATCH_NUM; batch_idx++)
                {
                    size_t base_batch_col_idx = base_col_idx + batch_idx * tile_size;
                    for (size_t dj = 0; dj < tile_size; dj++) // (i, j, di, dj)-th element
                    {
                        int djj = (dj - right_rot + tile_size) % tile_size;
                        size_t col_idx = base_batch_col_idx + djj;
                        size_t row_idx = base_row_idx + (di + djj) % tile_size;
                        for (size_t copy_idx = 0; copy_idx < copy_count; copy_idx++)
                        {
                            packed_weight[batch_idx * bfv_row_size + dj * copy_count + copy_idx] =
                                weight_matrix[row_idx * weight_config.weight_cols + col_idx];
                        }
                    }
                }
                encoder->encode(packed_weight, plain);
#ifdef USE_HE_GPU
                if (backend == Datatype::DEVICE)
                {
                    evaluator->transform_to_ntt_inplace(plain, encrypted_activation.front().dcipher().chain_index());
                }
                else
                {
                    evaluator->transform_to_ntt_inplace(plain, encrypted_activation.front().hcipher().parms_id());
                }
#else
                evaluator->transform_to_ntt_inplace(plain, encrypted_activation.front().hcipher().parms_id());
#endif
            }
            // }));
        }

        // Wait for all weight encoding tasks to complete
        for (auto &future : futures)
        {
            future.wait();
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Parallel weight encoding completed in " << duration.count() << " ms" << endl;
}

FFNMemFootprintCalculator::MemoryBreakdown FFNMemFootprintCalculator::calculate_ffn_memory(
    size_t tile_size, size_t num_tiled_weight_rows, size_t num_tiled_weight_cols, size_t poly_modulus_degree,
    size_t coeff_modulus_size, const BSGSConfig &bsgs_config)
{
    MemoryBreakdown breakdown;

    size_t ciphertext_size = FFNMemoryMonitor::estimate_ctxt_mem(poly_modulus_degree, coeff_modulus_size);
    size_t plaintext_size = FFNMemoryMonitor::estimate_ptxt_mem(poly_modulus_degree, coeff_modulus_size);

    breakdown.input_data = num_tiled_weight_rows * ciphertext_size;
    breakdown.weight_matrix = num_tiled_weight_rows * num_tiled_weight_cols * tile_size * plaintext_size;
    breakdown.intermediate_results = FFNMemoryMonitor::estimate_bsgs_mem(
        bsgs_config.bs, bsgs_config.gs, num_tiled_weight_cols, ciphertext_size, plaintext_size);
    breakdown.temporary_variables = ciphertext_size;
    breakdown.total_estimated =
        breakdown.input_data + breakdown.weight_matrix + breakdown.intermediate_results + breakdown.temporary_variables;

    return breakdown;
}

void FFNMemFootprintCalculator::print_mem_breakdown(const MemoryBreakdown &breakdown)
{
    using namespace Colors;

    cout << "\n" << string(60, '=') << endl;
    cout << BOLD << CYAN << "GPU MEMORY FOOTPRINT ANALYSIS" << RESET << endl;
    cout << string(60, '=') << endl;
    cout << BOLD << "Estimated Memory Usage:" << RESET << endl;

    auto print_memory_size = [](const string &name, size_t bytes) {
        double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
        cout << "  " << setw(25) << left << name << ": " << setw(12) << right << fixed << setprecision(2) << mb << " MB"
             << endl;
    };

    print_memory_size("Input Data", breakdown.input_data);
    print_memory_size("Weight Matrix", breakdown.weight_matrix);
    print_memory_size("Intermediate Results", breakdown.intermediate_results);
    print_memory_size("Temporary Variables", breakdown.temporary_variables);
    cout << string(40, '-') << endl;
    print_memory_size("Total Estimated", breakdown.total_estimated);
    cout << string(60, '=') << endl << endl;
}