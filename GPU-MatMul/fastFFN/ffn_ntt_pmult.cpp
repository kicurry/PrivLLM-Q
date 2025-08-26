#include "common/memory_monitor.h"
#include "common/profile_helper.h"
#include "common/thread_pool.h"
#include "common/utils.h"
#include "ffn_utils.h"

using namespace std;
using namespace seal;
using namespace HE::unified;

// #define PRE_ENCODING
#if defined ENABLE_MATRIX_VALIDATION && defined PRE_ENCODING
#include "common/matrix_validator.h"
#endif

#ifdef USE_HE_GPU
size_t thread_num = 48;
bool disable_gpu = false;
#else
size_t thread_num = std::thread::hardware_concurrency();
bool disable_gpu = true;
#endif

bool multi_thread = false;

int main(int argc, char *argv[])
{
    // Set backend
    ffnInitizer::parse_ffn_backend(argc, argv, disable_gpu);
    auto backend = disable_gpu ? Datatype::HOST : Datatype::DEVICE;
    thread_num = !disable_gpu ? 1 : thread_num;

    // Parse command line arguments
    FFNMemoryMonitor::parse_mem_monitor_args(argc, argv);
    ffnInitizer::parse_noise_budget_args(argc, argv);

    print_cuda_device_info(disable_gpu);

    size_t seq_len = 128;

    // Initialize using ffnInitizer with command line configurable parameters
    uint64_t polyModulusDegree = 4096;
    uint64_t plainWidth = 17;
    vector<int> rnsBitSizes = { 31, 31, 31 };
    // vector<int> rnsBitSizes = { 57, 57 };

    // Parse encryption parameters from command line
    if (!ffnInitizer::parse_encryption_params_args(argc, argv, polyModulusDegree, plainWidth, rnsBitSizes))
    {
        cerr << "Error parsing encryption parameters. Exiting." << endl;
        return 1;
    }

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(polyModulusDegree);
    parms.set_plain_modulus(PlainModulus::Batching(polyModulusDegree, plainWidth));
    parms.set_coeff_modulus(CoeffModulus::Create(polyModulusDegree, rnsBitSizes));

    // Create thread pool first
    ThreadPool pool(thread_num);
    vector<future<void>> futures;

    FFNConfig ffn_config{
        .seq_len = seq_len, .activation_cols = 4096, .weight_cols = 12288, .activation_bits = 4, .weight_bits = 4
    };
    ffnInitizer::parse_ffn_config_args(
        argc, argv, ffn_config.activation_bits, ffn_config.weight_bits, ffn_config.seq_len, ffn_config.activation_cols,
        ffn_config.weight_cols);

    // Option 1: Use default BSGS configuration (current behavior)
    ffnInitizer init(parms, ffn_config, backend);

    // Option 2: Use custom BSGS configuration (uncomment to use)
    // BSGSConfig custom_bsgs{ .bs = 8, .gs = 2 };  // Example: bs=8, gs=2
    // ffnInitizer init(parms, ffn_config, backend, custom_bsgs);

    auto &context = init.get_he_context();
    auto &encoder = init.get_encoder();
    auto &encryptor = init.get_encryptor();
    auto &decryptor = init.get_decryptor();
    auto &evaluator = init.get_evaluator();
    auto &galoisKeys = init.get_galoisKeys();
#ifdef FP64_MM_ARITH
    if (context.max_data_modulus_bit() >= 52)
    {
        throw std::runtime_error("FP64_MM_ARITH mode REQUIRES 51-bit data modulus");
    }
#endif

#ifdef PRE_ENCODING
    init.init_matrices(pool, futures, true);
#else
    init.init_matrices(pool, futures);
#endif

    auto &encrypted_activation = init.get_encrypted_activation();
    auto &encoded_weight = init.get_encoded_weight();
    size_t num_activation_ctxt = init.get_num_activation_ctxt();
    size_t num_col_per_act_ctxt = init.get_num_col_per_act_ctxt();
    size_t tile_size = init.get_tile_size();

    size_t slot_count = encoder.slot_count();
    size_t row_size = slot_count / BFV_BATCH_NUM;
    size_t num_tiled_weight_rows = init.get_num_tiled_weight_rows();
    size_t num_tiled_weight_cols = init.get_num_tiled_weight_cols();
    size_t activation_rows = seq_len;

    // Calculate and display estimated memory usage
    FFNMemFootprintCalculator::MemoryBreakdown mem_breakdown = FFNMemFootprintCalculator::calculate_ffn_memory(
        tile_size, num_tiled_weight_rows, num_tiled_weight_cols, polyModulusDegree,
        context.hcontext().first_context_data()->parms().coeff_modulus().size(), init.get_bsgs_config());
    FFNMemFootprintCalculator::print_mem_breakdown(mem_breakdown);

#ifdef PRE_ENCODING
    if (rnsBitSizes.size() > 2 && mem_breakdown.total_estimated > 75 * 1024 * 1024 * 1024UL)
    {
        throw std::runtime_error(
            "PRE_ENCODING mode requires no more than 2 RNS components to avoid excessive memory usage");
    }
#endif

#ifndef PRE_ENCODING
    vector<uint64_t> rand_raw(slot_count, 1ULL);
    fill_random_weight(rand_raw, tile_size * BFV_BATCH_NUM, ffn_config.weight_bits);
    print_matrix(rand_raw, row_size);
    UnifiedPlaintext random_pt(backend);
    encoder.encode(rand_raw, random_pt);
#ifdef USE_HE_GPU
    if (random_pt.on_device())
    {
        evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().dcipher().chain_index());
    }
    else
    {
        evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().hcipher().parms_id());
    }
#else
    evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().hcipher().parms_id());
#endif // USE_HE_GPU
#endif // PRE_ENCODING

    // Create memory monitor
    FFNMemoryMonitor memory_monitor(FFNMemoryMonitor::is_memory_monitoring_enabled());

    // Start FFN algorithm memory monitoring
    if (memory_monitor.is_enabled())
    {
        memory_monitor.start_stage("FFN Algorithm Execution");
    }

    // == Multiply activation matrix and weight matrix
    // == BSGS block matrix multiplication
    vector<UnifiedCiphertext> result_ctxts(num_tiled_weight_cols, backend);

    auto start_time = chrono::high_resolution_clock::now();
    if (!multi_thread)
    {
        evaluator.sync();
        for (size_t group_k_idx = 0; group_k_idx < init.get_num_activation_ctxt(); group_k_idx++)
        {
            size_t baby_step = init.get_bsgs_config().bs;
            size_t giant_step = init.get_bsgs_config().gs;

            if (group_k_idx % 64 == 0)
            {
                cout << group_k_idx << "-th group-k:" << baby_step - 1 << " baby-step pre-rotations, "
                     << num_tiled_weight_cols << " * (" << num_col_per_act_ctxt << " multiplications, "
                     << giant_step - 1 << " giant-step post-rotations)" << endl;
            }

            // 1. Baby-step Pre-Rotation
            const auto &baby_input_ctxt = encrypted_activation[group_k_idx];
            vector<UnifiedCiphertext> baby_ctxts(baby_step, backend);

            nvtxPush("BS-Rot", backend);
            for (size_t i = 0; i < baby_step; i++)
            {
                evaluator.rotate_rows(baby_input_ctxt, i * activation_rows, galoisKeys, baby_ctxts[i]);
                evaluator.transform_to_ntt_inplace(baby_ctxts[i]);
            }
            nvtxPop("BS-Rot", backend);

            for (size_t tiled_col_idx = 0; tiled_col_idx < result_ctxts.size(); tiled_col_idx++)
            {
                auto &result_ctxt = result_ctxts[tiled_col_idx];

#ifdef PRE_ENCODING
                auto &weight_ptxt = encoded_weight[group_k_idx * num_tiled_weight_cols + tiled_col_idx];
#endif

                for (size_t i = 0; i < giant_step; i++)
                {
                    // 2. Baby-step Plaintext Weight Multiplication and Accumulation

                    UnifiedCiphertext giant_ctxt(backend);

                    nvtxPush("MAC", backend);
#ifdef PRE_ENCODING
                    evaluator.multiply_plain_ntt(baby_ctxts[0], weight_ptxt[i * baby_step], giant_ctxt);
#else
                    evaluator.multiply_plain_ntt(baby_ctxts[0], random_pt, giant_ctxt);
#endif

                    for (size_t baby_idx = 1; baby_idx < baby_step; baby_idx++)
                    {
                        UnifiedCiphertext temp_ctxt(backend);
#ifdef PRE_ENCODING
                        evaluator.multiply_plain_ntt(
                            baby_ctxts[baby_idx], weight_ptxt[i * baby_step + baby_idx], temp_ctxt);
#else
                        evaluator.multiply_plain_ntt(baby_ctxts[baby_idx], random_pt, temp_ctxt);
#endif
                        evaluator.add_inplace(giant_ctxt, temp_ctxt);
                    }
                    nvtxPop("MAC", backend);

                    if (group_k_idx == 0 && i == 0)
                    {
                        nvtxPush("GS-Rot", backend);
                        evaluator.transform_from_ntt(giant_ctxt, result_ctxt);
                        nvtxPop("GS-Rot", backend);
                    }
                    else
                    {
                        nvtxPush("GS-Rot", backend);
                        evaluator.transform_from_ntt_inplace(giant_ctxt);
                        evaluator.rotate_rows_inplace(giant_ctxt, i * activation_rows * baby_step, galoisKeys);
                        nvtxPop("GS-Rot", backend);
                        nvtxPush("GS-Add", backend);
                        evaluator.add_inplace(result_ctxt, giant_ctxt);
                        nvtxPop("GS-Add", backend);
                    }
                }
            }
        }
        evaluator.sync();
        printNVTXStats();
    }
    else
    {
        std::vector<std::mutex> result_mutexes(num_tiled_weight_cols);
        // Parallelize matrix multiplication by processing groups in parallel
        futures.clear();
        for (size_t group_k_idx = 0; group_k_idx < num_tiled_weight_rows; group_k_idx++)
        {
            futures.push_back(pool.enqueue([&, group_k_idx]() {
                size_t baby_step = init.get_bsgs_config().bs;
                size_t giant_step = init.get_bsgs_config().gs;

                if (group_k_idx % 64 == 0)
                {
                    cout << group_k_idx << "-th group-k:" << baby_step - 1 << " baby-step pre-rotations, "
                         << num_tiled_weight_cols << " * (" << num_col_per_act_ctxt << " multiplications, "
                         << giant_step - 1 << " giant-step post-rotations)" << endl;
                }

                // 1. Baby-step Pre-Rotation
                const auto &baby_input_ctxt = encrypted_activation[group_k_idx];
                vector<UnifiedCiphertext> baby_ctxts(baby_step, backend);
                for (size_t i = 0; i < baby_step; i++)
                {
                    evaluator.rotate_rows(baby_input_ctxt, i * activation_rows, galoisKeys, baby_ctxts[i]);
                    evaluator.transform_to_ntt_inplace(baby_ctxts[i]);
                }

                for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
                {
                    auto &result_ctxt = result_ctxts[tiled_col_idx];

#ifdef PRE_ENCODING
                    auto &weight_ptxt = encoded_weight[group_k_idx * num_tiled_weight_cols + tiled_col_idx];
#endif
                    for (size_t i = 0; i < giant_step; i++)
                    {
                        // 2. Baby-step Plaintext Weight Multiplication and Accumulation

                        UnifiedCiphertext giant_ctxt(backend);

#ifdef PRE_ENCODING
                        evaluator.multiply_plain_ntt(baby_ctxts[0], weight_ptxt[i * baby_step], giant_ctxt);
#else
                        evaluator.multiply_plain_ntt(baby_ctxts[0], random_pt, giant_ctxt);
#endif

                        for (size_t baby_idx = 1; baby_idx < baby_step; baby_idx++)
                        {
                            UnifiedCiphertext temp_ctxt(backend);
#ifdef PRE_ENCODING
                            evaluator.multiply_plain_ntt(
                                baby_ctxts[baby_idx], weight_ptxt[i * baby_step + baby_idx], temp_ctxt);
#else
                            evaluator.multiply_plain_ntt(baby_ctxts[baby_idx], random_pt, temp_ctxt);
#endif
                            evaluator.add_inplace(giant_ctxt, temp_ctxt);
                        }

                        if (group_k_idx == 0 && i == 0)
                        {
                            {
                                std::lock_guard<std::mutex> lock(result_mutexes[tiled_col_idx]);
                                evaluator.transform_from_ntt(giant_ctxt, result_ctxt);
                            }
                        }
                        else
                        {
                            evaluator.transform_from_ntt_inplace(giant_ctxt);
                            evaluator.rotate_rows_inplace(giant_ctxt, i * activation_rows * baby_step, galoisKeys);
                            {
                                std::lock_guard<std::mutex> lock(result_mutexes[tiled_col_idx]);
                                evaluator.add_inplace(result_ctxt, giant_ctxt);
                            }
                        }
                    }
                }
            }));
        }

        // Wait for all matrix multiplication tasks to complete
        for (auto &future : futures)
        {
            future.wait();
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Ciphertext activation-Plaintext weight matrix multiplication - Complete. Time: " << duration.count()
         << " ms" << endl;

    // End FFN algorithm memory monitoring and print report
    if (memory_monitor.is_enabled())
    {
        memory_monitor.end_stage();
        memory_monitor.print_memory_report();
    }

    // Decrypt and decode
    if (result_ctxts[0].on_device())
    {
        result_ctxts[0].to_host(context);
    }
    cout << "    + Noise budget after PCMM: " << decryptor.invariant_noise_budget(result_ctxts[0]) << " bits" << endl;

#if defined ENABLE_MATRIX_VALIDATION && defined PRE_ENCODING
    UnifiedPlaintext plain_result(HOST);
    decryptor.decrypt(result_ctxts[0], plain_result);
    vector<uint64_t> pod_result(slot_count, 0ULL);
    encoder.decode(plain_result, pod_result);
    print_matrix(pod_result, row_size);

    bool match = MatrixValidator::validate(
        init.get_activation_matrix(), init.get_weight_matrix(), pod_result, 0, ffn_config.seq_len,
        ffn_config.activation_cols, ffn_config.weight_cols,
        context.hcontext().first_context_data()->parms().plain_modulus().value());

    if (match)
    {
        cout << "Verification successful: Plaintext and encrypted results match for first " << ffn_config.seq_len << "x"
             << 2 * tile_size << " block" << endl;
    }
    else
    {
        cout << "Verification failed: Plaintext and encrypted results do not match" << endl;
    }
#endif

    return 0;
}
