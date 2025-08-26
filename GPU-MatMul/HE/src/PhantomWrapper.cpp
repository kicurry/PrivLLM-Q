#include "HE/unified/PhantomWrapper.h"
#include <phantom/evaluate.cuh>

using namespace HE;

void PhantomEvaluator::negate_inplace(PhantomCiphertext &encrypted) const
{
    phantom::negate_inplace(context_, encrypted);
}

void PhantomEvaluator::add_inplace(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const
{
    phantom::add_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::sub_inplace(
    PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, bool negate) const
{
    phantom::sub_inplace(context_, encrypted1, encrypted2, negate);
}

void PhantomEvaluator::add_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const
{
    phantom::add_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::sub_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const
{
    phantom::sub_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const
{
    phantom::multiply_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_plain_ntt_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const
{
    phantom::multiply_plain_ntt_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::fused_bsgs_fma(
    size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols, size_t lazy_reduction_interval, bool acc) const
{
    if (acc)
    {
        phantom::fused_bsgs_fma<true>(
            context_, chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step,
            tiled_weight_cols, lazy_reduction_interval);
    }
    else
    {
        phantom::fused_bsgs_fma<false>(
            context_, chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step,
            tiled_weight_cols, lazy_reduction_interval);
    }
}

void PhantomEvaluator::fused_bsgs_fma_fast(
    size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
    const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
    size_t baby_step, size_t giant_step, size_t tiled_weight_cols, bool acc) const
{
    if (acc)
    {
        phantom::fused_bsgs_fma_fast<true>(
            context_, chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step,
            tiled_weight_cols);
    }
    else
    {
        phantom::fused_bsgs_fma_fast<false>(
            context_, chain_index, h_baby_ctxt_ptrs, h_weight_ptxt_ptrs, h_giant_ptrs, baby_step, giant_step,
            tiled_weight_cols);
    }
}

void PhantomEvaluator::multiply_inplace(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const
{
    phantom::multiply_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::relinearize_inplace(PhantomCiphertext &encrypted, const PhantomRelinKey &relin_keys) const
{
    phantom::relinearize_inplace(context_, encrypted, relin_keys);
}

void PhantomEvaluator::rescale_to_next(const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const
{
    destination = std::move(phantom::rescale_to_next(context_, encrypted));
}

void PhantomEvaluator::mod_switch_to_next(const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const
{
    destination = std::move(phantom::mod_switch_to_next(context_, encrypted));
}

void PhantomEvaluator::rotate_vector_inplace(
    PhantomCiphertext &encrypted, int step, const PhantomGaloisKey &galois_key) const
{
    const auto n_full_slots = context_.key_context_data().parms().poly_modulus_degree() >> 1;
    step %= n_full_slots;
    if (((step + n_full_slots) % n_full_slots) == 0)
    {
        return;
    }
    phantom::rotate_inplace(context_, encrypted, step, galois_key);
}

void PhantomEvaluator::complex_conjugate_inplace(PhantomCiphertext &encrypted, const PhantomGaloisKey &galois_key) const
{
    phantom::rotate_inplace(context_, encrypted, 0, galois_key);
}

void PhantomEvaluator::transform_to_ntt_inplace(PhantomPlaintext &plain, size_t chain_index) const
{
    phantom::transform_to_ntt_inplace(context_, plain, chain_index);
}

void PhantomEvaluator::transform_to_ntt_inplace(PhantomCiphertext &encrypted) const
{
    phantom::transform_to_ntt_inplace(context_, encrypted);
}

void PhantomEvaluator::transform_from_ntt_inplace(PhantomCiphertext &encrypted) const
{
    phantom::transform_from_ntt_inplace(context_, encrypted);
}