#include "matrix_validator.h"
#ifdef ENABLE_MATRIX_VALIDATION

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

bool MatrixValidator::validate(
    const vector<uint64_t> &activation_matrix, const vector<uint64_t> &weight_matrix,
    const vector<uint64_t> &encrypted_result, size_t encrypted_result_id, size_t activation_rows,
    size_t activation_cols, size_t weight_cols, size_t plain_modulus, size_t bfv_batch_num)
{
    // Convert input to Eigen matrices
    Map<const Matrix<uint64_t, Dynamic, Dynamic, RowMajor>> A(
        activation_matrix.data(), activation_rows, activation_cols);

    Map<const Matrix<uint64_t, Dynamic, Dynamic, RowMajor>> B(weight_matrix.data(), activation_cols, weight_cols);

    // Perform matrix multiplication
    Matrix<uint64_t, Dynamic, Dynamic, RowMajor> expected = A * B;

    // Compare results - access data according to BFV encoding method
    bool match = true;
    size_t row_size = encrypted_result.size() / bfv_batch_num;
    size_t num_col_per_act_ctxt = row_size / activation_rows;

    // Count overflow and mismatch occurrences
    size_t overflow_count = 0;

    for (size_t i = 0; i < activation_rows; ++i)
    {
        for (size_t j = encrypted_result_id * num_col_per_act_ctxt * bfv_batch_num;
             j < (encrypted_result_id + 1) * num_col_per_act_ctxt * bfv_batch_num; ++j)
        {
            auto actual = encrypted_result[j * activation_rows + i];
            if (expected(i, j) != actual)
            {
                if ((expected(i, j) % plain_modulus) == actual)
                {
                    // Count overflow but don't output details
                    overflow_count++;
                }
                else
                {
                    // Count mismatch and output details
                    cout << "Mismatch at (" << i << "," << j << "): "
                         << "expected=" << expected(i, j) << ", actual=" << actual << endl;
                    return false;
                }
            }
        }
    }

    // Output summary counts
    if (overflow_count > 0)
    {
        cout << "Plaintext Overflow occurrences: " << overflow_count << endl;
    }

    return match;
}
#endif // ENABLE_MATRIX_VALIDATION
