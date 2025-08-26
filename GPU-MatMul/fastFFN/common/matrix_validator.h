#pragma once

#ifdef ENABLE_MATRIX_VALIDATION
#include <cstdint>
#include <vector>

class MatrixValidator
{
public:
    /**
     * @brief Validate encrypted matrix multiplication result
     * @param activation_matrix Input matrix A
     * @param weight_matrix Input matrix B
     * @param encrypted_result Encrypted computation result
     * @param encrypted_result_id ID of the encrypted result
     * @param activation_rows Number of rows in matrix A
     * @param activation_cols Number of columns in matrix A / rows in matrix B
     * @param weight_cols Number of columns in matrix B
     * @param plain_modulus BFV plaintext modulus
     * @param bfv_batch_num BFV batch number (default=2)
     * @return Whether the validation result matches
     */
    static bool validate(
        const std::vector<uint64_t> &activation_matrix, const std::vector<uint64_t> &weight_matrix,
        const std::vector<uint64_t> &encrypted_result, std::size_t encrypted_result_id, std::size_t activation_rows,
        std::size_t activation_cols, std::size_t weight_cols, std::size_t plain_modulus, std::size_t bfv_batch_num = 2);
};

#endif // ENABLE_MATRIX_VALIDATION