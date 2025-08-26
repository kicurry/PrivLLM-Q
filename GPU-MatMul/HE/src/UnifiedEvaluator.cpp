#include "HE/unified/UnifiedEvaluator.h"

using namespace HE::unified;

#ifdef USE_HE_GPU

void UnifiedEvaluator::negate_inplace(UnifiedCiphertext &encrypted) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        seal_eval_->negate_inplace(encrypted);
    }
    else
    {
        phantom_eval_->negate_inplace(encrypted);
    }
}

void UnifiedEvaluator::add_inplace(UnifiedCiphertext &encrypted1, const UnifiedCiphertext &encrypted2) const
{
    backend_check(encrypted1, encrypted2);
    if (encrypted1.on_host() && encrypted2.on_host())
    {
        seal_eval_->add_inplace(encrypted1, encrypted2);
    }
    else
    {
        phantom_eval_->add_inplace(encrypted1, encrypted2);
    }
}

void UnifiedEvaluator::sub_inplace(UnifiedCiphertext &encrypted1, const UnifiedCiphertext &encrypted2) const
{
    backend_check(encrypted1, encrypted2);
    if (encrypted1.on_host() && encrypted2.on_host())
    {
        seal_eval_->sub_inplace(encrypted1, encrypted2);
    }
    else
    {
        phantom_eval_->sub_inplace(encrypted1, encrypted2);
    }
}

void UnifiedEvaluator::multiply_inplace(UnifiedCiphertext &encrypted1, const UnifiedCiphertext &encrypted2) const
{
    backend_check(encrypted1, encrypted2);
    if (encrypted1.on_host() && encrypted2.on_host())
    {
        seal_eval_->multiply_inplace(encrypted1, encrypted2);
    }
    else
    {
        phantom_eval_->multiply_inplace(encrypted1, encrypted2);
    }
}

void UnifiedEvaluator::fused_bsgs_fma(
    size_t max_bit_count, size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols, bool acc) const
{
#ifdef FP64_MM_ARITH
    auto threshold = 104;
#else
    auto threshold = 128;
#endif
    if (2 * max_bit_count >= threshold)
    {
        throw std::invalid_argument("fused_bsgs_fma: max_bit_count must be less than " + std::to_string(threshold / 2));
    }
    size_t lazy_reduction_interval =
        threshold - 2 * max_bit_count >= 64 ? 0xFFFFFFFFFFFFFFFF : (1ULL << (threshold - 2 * max_bit_count)) - 1;
    phantom_eval_->fused_bsgs_fma(
        chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step, tiled_weight_cols,
        lazy_reduction_interval, acc);
}

void UnifiedEvaluator::fused_bsgs_fma_fast(
    size_t max_bit_count, size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols, bool acc) const
{
#ifdef FP64_MM_ARITH
    auto threshold = 104;
#else
    auto threshold = 128;
#endif
    if (2 * max_bit_count >= threshold || std::pow(2.0, threshold - 2 * max_bit_count) <= double(baby_step))
    {
        throw std::invalid_argument("fused_bsgs_fma_fast: max_bit_count exceeded");
    }
    phantom_eval_->fused_bsgs_fma_fast(
        chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step, tiled_weight_cols, acc);
}

void UnifiedEvaluator::relinearize_inplace(UnifiedCiphertext &encrypted, const UnifiedRelinKeys &relin_keys) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        seal_eval_->relinearize_inplace(encrypted, relin_keys);
    }
    else
    {
        phantom_eval_->relinearize_inplace(encrypted, relin_keys);
    }
}

void UnifiedEvaluator::mod_switch_to_next(const UnifiedCiphertext &encrypted, UnifiedCiphertext &destination) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        seal_eval_->mod_switch_to_next(encrypted, destination);
    }
    else
    {
        phantom_eval_->mod_switch_to_next(encrypted, destination);
    }
}

void UnifiedEvaluator::rescale_to_next(const UnifiedCiphertext &encrypted, UnifiedCiphertext &destination) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        seal_eval_->rescale_to_next(encrypted, destination);
    }
    else
    {
        phantom_eval_->rescale_to_next(encrypted, destination);
    }
}

void UnifiedEvaluator::add_plain_inplace(UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const
{
    backend_check(encrypted, plain);
    if (encrypted.on_host() && plain.on_host())
    {
        // std::cout << "add_plain_inplace on host" << std::endl;
        seal_eval_->add_plain_inplace(encrypted, plain);
    }
    else
    {
        phantom_eval_->add_plain_inplace(encrypted, plain);
    }
}

void UnifiedEvaluator::sub_plain_inplace(UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const
{
    backend_check(encrypted, plain);
    if (encrypted.on_host() && plain.on_host())
    {
        seal_eval_->sub_plain_inplace(encrypted, plain);
    }
    else
    {
        phantom_eval_->sub_plain_inplace(encrypted, plain);
    }
}

void UnifiedEvaluator::multiply_plain_inplace(UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const
{
    backend_check(encrypted, plain);
    if (encrypted.on_host() && plain.on_host())
    {
        seal_eval_->multiply_plain_inplace(encrypted, plain);
    }
    else
    {
        phantom_eval_->multiply_plain_inplace(encrypted, plain);
    }
}

void UnifiedEvaluator::multiply_plain_ntt_inplace(UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const
{
    backend_check(encrypted, plain);
    if (encrypted.on_host() && plain.on_host())
    {
        if (!plain.hplain().is_ntt_form() || !encrypted.hcipher().is_ntt_form())
        {
            throw std::invalid_argument("multiply_plain_ntt_inplace: plaintext and ciphertext must be in NTT form");
        }
        seal_eval_->multiply_plain_inplace(encrypted, plain);
    }
    else
    {
        if (!encrypted.dcipher().is_ntt_form())
        {
            throw std::invalid_argument("multiply_plain_ntt_inplace: plaintext and ciphertext must be in NTT form");
        }
        phantom_eval_->multiply_plain_ntt_inplace(encrypted, plain);
    }
}

void UnifiedEvaluator::rotate_vector_inplace(
    UnifiedCiphertext &encrypted, int step, const UnifiedGaloisKeys &galois_key) const
{
    backend_check(encrypted, galois_key);
    if (encrypted.on_host() && galois_key.on_host())
    {
        seal_eval_->rotate_vector_inplace(encrypted, step, galois_key);
    }
    else
    {
        phantom_eval_->rotate_vector_inplace(encrypted, step, galois_key);
    }
}

void UnifiedEvaluator::complex_conjugate_inplace(
    UnifiedCiphertext &encrypted, const UnifiedGaloisKeys &galois_key) const
{
    backend_check(encrypted, galois_key);
    if (encrypted.on_host() && galois_key.on_host())
    {
        seal_eval_->complex_conjugate_inplace(encrypted, galois_key);
    }
    else
    {
        phantom_eval_->complex_conjugate_inplace(encrypted, galois_key);
    }
}

void UnifiedEvaluator::rotate_rows_inplace(
    UnifiedCiphertext &encrypted, int step, const UnifiedGaloisKeys &galois_key) const
{
    backend_check(encrypted, galois_key);
    if (encrypted.on_host() && galois_key.on_host())
    {
        seal_eval_->rotate_rows_inplace(encrypted, step, galois_key);
    }
    else
    {
        phantom_eval_->rotate_vector_inplace(encrypted, step, galois_key);
    }
}

void UnifiedEvaluator::rotate_columns_inplace(UnifiedCiphertext &encrypted, const UnifiedGaloisKeys &galois_key) const
{
    backend_check(encrypted, galois_key);
    if (encrypted.on_host() && galois_key.on_host())
    {
        seal_eval_->rotate_columns_inplace(encrypted, galois_key);
    }
    else
    {
        phantom_eval_->complex_conjugate_inplace(encrypted, galois_key);
    }
}

void UnifiedEvaluator::transform_to_ntt_inplace(UnifiedPlaintext &plain, const seal::parms_id_type &parms_id) const
{
    backend_check(plain);
    if (plain.on_host())
    {
        seal_eval_->transform_to_ntt_inplace(plain, parms_id);
    }
    else
    {
        throw std::invalid_argument("transform_to_ntt_inplace: plaintext must be on host when using parms_id");
    }
}

void UnifiedEvaluator::transform_to_ntt_inplace(UnifiedPlaintext &plain, size_t chain_index) const
{
    backend_check(plain);
    if (plain.on_host())
    {
        throw std::invalid_argument("transform_to_ntt_inplace: plaintext must be on device when using chain_index");
    }
    else
    {
        phantom_eval_->transform_to_ntt_inplace(plain, chain_index);
    }
}

void UnifiedEvaluator::transform_to_ntt_inplace(UnifiedCiphertext &encrypted) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        if (encrypted.hcipher().is_ntt_form())
        {
            throw std::invalid_argument("transform_to_ntt_inplace: ciphertext must be NOT in NTT form");
        }
        seal_eval_->transform_to_ntt_inplace(encrypted);
    }
    else
    {
        if (encrypted.dcipher().is_ntt_form())
        {
            throw std::invalid_argument("transform_to_ntt_inplace: ciphertext must be NOT in NTT form");
        }
        phantom_eval_->transform_to_ntt_inplace(encrypted);
    }
}

void UnifiedEvaluator::transform_from_ntt_inplace(UnifiedCiphertext &encrypted) const
{
    backend_check(encrypted);
    if (encrypted.on_host())
    {
        if (!encrypted.hcipher().is_ntt_form())
        {
            throw std::invalid_argument("transform_from_ntt_inplace: ciphertext must be in NTT form");
        }
        seal_eval_->transform_from_ntt_inplace(encrypted);
    }
    else
    {
        if (!encrypted.dcipher().is_ntt_form())
        {
            throw std::invalid_argument("transform_from_ntt_inplace: ciphertext must be in NTT form");
        }
        phantom_eval_->transform_from_ntt_inplace(encrypted);
    }
}

#else

void UnifiedEvaluator::multiply_plain_ntt_inplace(UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const
{
    if (encrypted.on_host() && plain.on_host())
    {
        if (!plain.hplain().is_ntt_form() || !encrypted.hcipher().is_ntt_form())
        {
            throw std::invalid_argument("multiply_plain_ntt_inplace: plaintext and ciphertext must be in NTT form");
        }
        seal_eval_->multiply_plain_inplace(encrypted, plain);
    }
    else
    {
        throw std::runtime_error("USE_HE_GPU=OFF");
    }
}

void UnifiedEvaluator::fused_bsgs_fma(
    size_t max_bit_count, size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols) const
{
    throw std::invalid_argument("fused_bsgs_fma: not implemented on host");
}

void UnifiedEvaluator::fused_bsgs_fma_fast(
    size_t max_bit_count, size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols) const
{
    throw std::invalid_argument("fused_bsgs_fma_fast: not implemented on host");
}

#endif