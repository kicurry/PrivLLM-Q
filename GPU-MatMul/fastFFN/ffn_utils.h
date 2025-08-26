#ifndef FFN_UTILS_H
#define FFN_UTILS_H

#include <optional>
#include <seal/decryptor.h>
#include <seal/encryptor.h>
#include "HE/unified/UnifiedContext.h"
#include "HE/unified/UnifiedEncoder.h"
#include "HE/unified/UnifiedEvaluator.h"
#include "common/thread_pool.h"

#define BFV_BATCH_NUM 2UL

struct FFNConfig
{
    size_t seq_len;
    size_t activation_cols;
    size_t weight_cols;

    int activation_bits;
    int weight_bits;
};

struct BSGSConfig
{
    size_t bs;
    size_t gs;
};

struct ActivationConfig
{
    // Plaintext related members
    int activation_bits = 4;
    size_t activation_rows;
    size_t activation_cols;

    // Ciphertext related members
    size_t num_activation_ctxt;
    size_t num_col_per_act_ctxt;
};

struct WeightConfig
{
    // Plaintext related members
    int weight_bits = 4;
    size_t weight_rows;
    size_t weight_cols;

    // Ciphertext related members
    size_t tile_size;
    size_t num_tiled_weight_rows;
    size_t num_tiled_weight_cols;
};

class ffnInitizer
{
public:
    HE::unified::UnifiedContext *context;
    seal::SecretKey *secretKeys;
    seal::PublicKey *publicKeys;
    HE::unified::UnifiedGaloisKeys *galoisKeys;
    seal::Encryptor *encryptor;
    seal::Decryptor *decryptor;
    HE::unified::UnifiedBatchEncoder *encoder;
    HE::unified::UnifiedEvaluator *evaluator;

    ffnInitizer(
        const seal::EncryptionParameters &params, const FFNConfig &ffn_config,
        Datatype::LOCATION backend = Datatype::HOST, std::optional<BSGSConfig> custom_bsgs_config = std::nullopt);
    ~ffnInitizer();

    // Getter methods
    inline auto &getSecretKey()
    {
        return *secretKeys;
    }

    inline auto &getPublicKey()
    {
        return *publicKeys;
    }

    inline auto &get_galoisKeys()
    {
        return *galoisKeys;
    }

    inline auto &get_encryptor()
    {
        return *encryptor;
    }

    inline auto &get_decryptor()
    {
        return *decryptor;
    }

    inline auto &get_he_context()
    {
        return *context;
    }

    inline auto &get_encoder()
    {
        return *encoder;
    }

    inline auto &get_evaluator()
    {
        return *evaluator;
    }

    inline auto &get_bsgs_config()
    {
        return bsgs_config;
    }

    size_t get_num_activation_ctxt() const;
    size_t get_num_col_per_act_ctxt() const;
    size_t get_tile_size() const;
    size_t get_num_tiled_weight_rows() const;
    size_t get_num_tiled_weight_cols() const;

    const vector<uint64_t> &get_activation_matrix() const;
    const vector<uint64_t> &get_weight_matrix() const;
    const vector<HE::unified::UnifiedCiphertext> &get_encrypted_activation() const;
    const vector<vector<HE::unified::UnifiedPlaintext>> &get_encoded_weight() const;

    template <typename PrintHanlder>
    inline void decrypt_and_print(const HE::unified::UnifiedCiphertext &ctxt, PrintHanlder print_hanlder)
    {
        auto tmp = ctxt;

        if (backend == Datatype::DEVICE)
        {
#ifdef USE_HE_GPU
            if (tmp.dcipher().is_ntt_form())
            {
                evaluator->transform_from_ntt_inplace(tmp);
            }
            tmp.to_host(*context);
#else
            throw std::runtime_error("USE_HE_GPU=OFF");
#endif
        }
        else
        {
            if (tmp.hcipher().is_ntt_form())
            {
                evaluator->transform_from_ntt_inplace(tmp);
            }
        }
        HE::unified::UnifiedPlaintext ptxt(HOST);
        decryptor->decrypt(tmp, ptxt);
        vector<uint64_t> raw_vec(encoder->slot_count(), 0ULL);
        encoder->decode(ptxt, raw_vec);
        print_hanlder(raw_vec);
    }

    // Print noise budget after a specific operation
    inline void print_noise_budget(const HE::unified::UnifiedCiphertext &ctxt, const std::string &after_tag)
    {
        // Check if noise budget monitoring is enabled
        if (!is_noise_budget_monitoring_enabled())
            return;

        auto tmp = ctxt;

        if (ctxt.on_device())
        {
#ifdef USE_HE_GPU
            if (tmp.dcipher().is_ntt_form())
            {
                evaluator->transform_from_ntt_inplace(tmp);
            }
            tmp.to_host(*context);
#else
            throw std::runtime_error("USE_HE_GPU=OFF");
#endif
        }
        else
        {
            if (tmp.hcipher().is_ntt_form())
            {
                evaluator->transform_from_ntt_inplace(tmp);
            }
        }

        cout << "    + Noise budget " << after_tag << ": " << decryptor->invariant_noise_budget(tmp) << " bits" << endl;
    }

    // Static methods for backend selection
    static bool parse_ffn_backend(int argc, char *argv[], bool &disable_gpu);

    // Static methods for noise budget monitoring control
    static bool parse_noise_budget_args(int argc, char *argv[]);
    static bool is_noise_budget_monitoring_enabled();

    // Static methods for encryption parameters
    static bool parse_encryption_params_args(
        int argc, char *argv[], uint64_t &polyModulusDegree, uint64_t &plainWidth, vector<int> &rnsBitSizes);

    // Static methods for bit width parameters
    static bool parse_ffn_config_args(
        int argc, char *argv[], int &activation_bits, int &weight_bits, size_t &seq_len, size_t &activation_cols,
        size_t &weight_cols);

    // Matrix initialization methods
    void init_matrices(ThreadPool &pool, vector<future<void>> &futures, bool preload_weights = false);

private:
    // Weight encoding methods
    void pre_encode_weights(ThreadPool &pool, vector<future<void>> &futures);

    // Matrix related members
    vector<uint64_t> activation_matrix;
    vector<uint64_t> weight_matrix;
    vector<HE::unified::UnifiedCiphertext> encrypted_activation;
    vector<vector<HE::unified::UnifiedPlaintext>> encoded_weight;

    size_t bfv_row_size;
    Datatype::LOCATION backend;
    ActivationConfig activation_config;
    WeightConfig weight_config;
    BSGSConfig bsgs_config;
};

class FFNMemFootprintCalculator
{
public:
    struct MemoryBreakdown
    {
        size_t input_data;
        size_t weight_matrix;
        size_t intermediate_results;
        size_t temporary_variables;
        size_t total_estimated;
    };

    static MemoryBreakdown calculate_ffn_memory(
        size_t tile_size, size_t num_tiled_weight_rows, size_t num_tiled_weight_cols, size_t poly_modulus_degree,
        size_t coeff_modulus_size, const BSGSConfig &bsgs_config);

    static void print_mem_breakdown(const MemoryBreakdown &breakdown);
};

#endif // FFN_UTILS_H
