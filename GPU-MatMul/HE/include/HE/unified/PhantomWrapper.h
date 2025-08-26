#pragma once

#include <phantom/batchencoder.h>
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/plaintext.h>
#include <phantom/secretkey.h>

namespace HE
{

    // ************************ PhantomEvaluator ***********************

    class UnifiedEvaluator;

    class PhantomEvaluator
    {
    public:
        friend class UnifiedEvaluator;

        PhantomEvaluator() = delete;

        PhantomEvaluator(const PhantomContext &context) : context_(context)
        {}

        ~PhantomEvaluator() = default;

        // encrypted = -encrypted
        void negate_inplace(PhantomCiphertext &encrypted) const;

        inline void negate(const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const
        {
            destination = encrypted;
            negate_inplace(destination);
        }

        // encrypted1 += encrypted2
        void add_inplace(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const;

        inline void add(
            const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
            PhantomCiphertext &destination) const
        {
            destination = encrypted1;
            add_inplace(destination, encrypted2);
        }

        // if negate = false (default): encrypted1 -= encrypted2
        // if negate = true: encrypted1 = encrypted2 - encrypted1
        void sub_inplace(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, bool negate = false) const;

        inline void sub(
            const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, PhantomCiphertext &destination,
            bool negate = false) const
        {
            destination = encrypted1;
            sub_inplace(destination, encrypted2, negate);
        }

        // encrypted += plain
        void add_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const;

        // encrypted -= plain
        void sub_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const;

        // encrypted *= plain
        void multiply_plain_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const;

        void multiply_plain_ntt_inplace(PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const;

        void fused_bsgs_fma(
            size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
            const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
            size_t baby_step, size_t giant_step, size_t tiled_weight_cols, size_t lazy_reduction_interval, bool acc = true) const;

        void fused_bsgs_fma_fast(
            size_t chain_index, const std::vector<const uint64_t *> &h_baby_ctxt_ptrs,
            const std::vector<const uint64_t *> &h_weight_ptxt_ptrs, const std::vector<uint64_t *> &h_giant_ptrs,
            size_t baby_step, size_t giant_step, size_t tiled_weight_cols, bool acc = true) const;

        // encrypted1 *= encrypted2
        void multiply_inplace(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const;

        inline void square_inplace(PhantomCiphertext &encrypted) const
        {
            multiply_inplace(encrypted, encrypted);
        }

        void relinearize_inplace(PhantomCiphertext &encrypted, const PhantomRelinKey &relin_keys) const;

        void rescale_to_next(const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const;

        void mod_switch_to_next(const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const;

        void rotate_vector_inplace(PhantomCiphertext &encrypted, int step, const PhantomGaloisKey &galois_key) const;

        void complex_conjugate_inplace(PhantomCiphertext &encrypted, const PhantomGaloisKey &galois_key) const;

        inline void rotate_rows_inplace(
            PhantomCiphertext &encrypted, int step, const PhantomGaloisKey &galois_key) const
        {
            rotate_vector_inplace(encrypted, step, galois_key);
        }

        inline void rotate_columns_inplace(PhantomCiphertext &encrypted, const PhantomGaloisKey &galois_key) const
        {
            complex_conjugate_inplace(encrypted, galois_key);
        }

        void transform_to_ntt_inplace(PhantomPlaintext &plain, size_t chain_index) const;

        void transform_to_ntt_inplace(PhantomCiphertext &encrypted) const;

        void transform_from_ntt_inplace(PhantomCiphertext &encrypted) const;

    private:
        const PhantomContext &context_;
    };

    // ********************** PhantomBatchEncoder **********************

    // For BFV/BGV
    class PhantomIntegerEncoder : PhantomBatchEncoder
    {
    public:
        PhantomIntegerEncoder(const PhantomContext &context) : PhantomBatchEncoder(context), context_(context)
        {}

        inline void encode(const std::vector<uint64_t> &values_matrix, PhantomPlaintext &destination) const
        {
            PhantomBatchEncoder::encode(context_, values_matrix, destination);
        }

        inline void decode(const PhantomPlaintext &plain, std::vector<uint64_t> &destination) const
        {
            PhantomBatchEncoder::decode(context_, plain, destination);
        }

    private:
        const PhantomContext &context_;
    };

    // *********************** PhantomCKKSEncoder **********************

    // TODO:

} // namespace HE