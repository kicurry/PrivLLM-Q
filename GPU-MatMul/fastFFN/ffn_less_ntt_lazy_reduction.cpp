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

size_t thread_num = 4;
bool disable_gpu = false;

int main(int argc, char *argv[])
{
    thread_num = !disable_gpu ? 1 : thread_num;
    auto backend = disable_gpu ? Datatype::HOST : Datatype::DEVICE;
    if (backend == Datatype::HOST)
    {
        throw std::runtime_error("unsupport");
    }

    // Parse command line arguments
    FFNMemoryMonitor::parse_mem_monitor_args(argc, argv);
    ffnInitizer::parse_noise_budget_args(argc, argv);

    print_cuda_device_info(disable_gpu);

    size_t seq_len = 128;

    // Initialize using ffnInitizer with command line configurable parameters
    uint64_t polyModulusDegree = 4096;
    uint64_t plainWidth = 17;
    vector<int> rnsBitSizes = { 31, 31, 31 };

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

    // Ciphertext activation-Plaintext weight matrix multiplication
#ifndef PRE_ENCODING
    vector<uint64_t> rand_raw(slot_count, 0ULL);
    fill_random_weight(rand_raw, tile_size * BFV_BATCH_NUM, 4);
    UnifiedPlaintext random_pt(backend);
    encoder.encode(rand_raw, random_pt);
    evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().dcipher().chain_index());
#endif

    // Create memory monitor
    FFNMemoryMonitor memory_monitor(FFNMemoryMonitor::is_memory_monitoring_enabled());

    // Start FFN algorithm memory monitoring
    if (memory_monitor.is_enabled())
    {
        memory_monitor.start_stage("FFN Algorithm Execution");
    }

    // 2. Multiply activation matrix and weight matrix
    auto prepare_fma_pointers = [&](const std::vector<UnifiedCiphertext> &baby_ctxts,
                                    const std::vector<std::vector<UnifiedPlaintext>> &encoded_weight,
                                    std::vector<UnifiedCiphertext> &batch_giant_ctxts, size_t group_k_idx,
                                    size_t num_batch, std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
                                    std::vector<const uint64_t *> &h_weight_ptxt_rs) {
        size_t baby_step = baby_ctxts.size();
        size_t tot_num_giant = batch_giant_ctxts.size();
        size_t giant_step = tot_num_giant / num_batch;
        // This calculation assumes encoded_weight is structured as [group][col]
        size_t k_total = baby_step;

        // 1. Prepare baby_ctxts pointers
        h_baby_ctxt_ptrs.resize(baby_step);
        for (size_t i = 0; i < baby_step; ++i)
        {
            h_baby_ctxt_ptrs[i] = baby_ctxts[i].dcipher().data();
        }

        // 2. Prepare weight_ptxt pointers (this is the most complex part)
        // The kernel expects a flat layout indexed by [giant_idx * k_total + k_idx]
        h_weight_ptxt_rs.resize(k_total * tot_num_giant);
        for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
        {
            // This logic must exactly match how you access weights in the original code
#ifdef PRE_ENCODING
            const auto &weight_ptxt_slice = encoded_weight[group_k_idx * num_tiled_weight_cols + tiled_col_idx];
#endif
            for (size_t giant_idx = 0; giant_idx < giant_step; giant_idx++)
            {
                for (size_t baby_idx = 0; baby_idx < baby_step; baby_idx++)
                {
                    size_t ptxt_col = tiled_col_idx * giant_step + giant_idx;
                    size_t flat_idx = baby_idx * tot_num_giant + ptxt_col;

                    // The pointer to the specific plaintext polynomial
#ifdef PRE_ENCODING
                    h_weight_ptxt_rs[flat_idx] = weight_ptxt_slice[giant_idx * baby_step + baby_idx].dplain().data();
#else
                    h_weight_ptxt_rs[flat_idx] = random_pt.dplain().data();
#endif
                }
            }
        }
    };

    std::vector<const uint64_t *> h_baby_ctxt_ptrs;
    std::vector<const uint64_t *> h_weight_ptxt_rs;
    std::vector<uint64_t *> h_giant_ctxt_ptrs;

    // TODO: remove zeros_ct
    UnifiedPlaintext zeros_pt(HOST);
    UnifiedCiphertext zeros_ct(HOST);
    std::vector<uint64_t> zeros(polyModulusDegree, 0);
    encoder.encode(zeros, zeros_pt);
    encryptor.encrypt(zeros_pt, zeros_ct);
    evaluator.transform_to_ntt_inplace(zeros_ct);
    if (backend == LOCATION::DEVICE)
    {
        zeros_ct.to_device(context);
    }

    // == Multiply activation matrix and weight matrix
    // == BSGS block matrix multiplication
    size_t baby_step = init.get_bsgs_config().bs;
    size_t giant_step = init.get_bsgs_config().gs;
    vector<UnifiedCiphertext> result_ctxts(num_tiled_weight_cols, backend);
    vector<UnifiedCiphertext> giant_ctxts(num_tiled_weight_cols * giant_step, zeros_ct);

    // Prepare giant_ctxts pointers buffer on DEVICE
    h_giant_ctxt_ptrs.resize(giant_step * num_tiled_weight_cols);
    for (size_t i = 0; i < h_giant_ctxt_ptrs.size(); ++i)
    {
        h_giant_ctxt_ptrs[i] = giant_ctxts[i].dcipher().data();
    }

    auto start_time = chrono::high_resolution_clock::now();
    evaluator.sync();
    for (size_t group_k_idx = 0; group_k_idx < num_tiled_weight_rows; group_k_idx++)
    {
        if (group_k_idx % 64 == 0)
        {
            cout << group_k_idx << "-th group-k:" << baby_step - 1 << " baby-step pre-rotations, "
                 << num_tiled_weight_cols << " * (" << num_col_per_act_ctxt << " multiplications, " << giant_step - 1
                 << " giant-step post-rotations)" << endl;
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

        nvtxPush("MAC", backend);
        prepare_fma_pointers(
            baby_ctxts, encoded_weight, giant_ctxts, group_k_idx, num_tiled_weight_cols, h_baby_ctxt_ptrs,
            h_weight_ptxt_rs);
        if (group_k_idx == 0)
        {
            evaluator.fused_bsgs_fma_fast(
                context.max_data_modulus_bit(), baby_ctxts.front().dcipher().chain_index(), h_baby_ctxt_ptrs,
                h_weight_ptxt_rs, h_giant_ctxt_ptrs, baby_step, giant_step, num_tiled_weight_cols, false);
        }
        else
        {
            evaluator.fused_bsgs_fma_fast(
                context.max_data_modulus_bit(), baby_ctxts.front().dcipher().chain_index(), h_baby_ctxt_ptrs,
                h_weight_ptxt_rs, h_giant_ctxt_ptrs, baby_step, giant_step, num_tiled_weight_cols);
        }
        nvtxPop("MAC", backend);
    }
    init.print_noise_budget(giant_ctxts[0], "after MAC");
    for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
    {
        auto &result_ctxt = result_ctxts[tiled_col_idx];
        nvtxPush("GS-Rot", backend);
        evaluator.transform_from_ntt(giant_ctxts[tiled_col_idx * giant_step], result_ctxt);
        nvtxPop("GS-Rot", backend);

        for (size_t i = 1; i < giant_step; i++)
        {
            auto &giant_ctxt = giant_ctxts[tiled_col_idx * giant_step + i];

            nvtxPush("GS-Rot", backend);
            evaluator.transform_from_ntt_inplace(giant_ctxt);
            evaluator.rotate_rows_inplace(giant_ctxt, i * activation_rows * baby_step, galoisKeys);
            nvtxPop("GS-Rot", backend);
            if (i == 1 && tiled_col_idx == 1)
            {
                init.print_noise_budget(giant_ctxt, "after GS-Rot");
            }

            nvtxPush("GS-Add", backend);
            evaluator.add_inplace(result_ctxt, giant_ctxt);
            nvtxPop("GS-Add", backend);
        }
    }
    evaluator.sync();
    printNVTXStats();

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
