#pragma once

#include "seal/context.h"
#include <iomanip>
#include <iostream>
#include <vector>

// Global ANSI color codes for better visibility
namespace Colors
{
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
} // namespace Colors

inline void print_line(int line_number)
{
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

inline void print_matrix_mod(
    uint64_t mod, const uint64_t *matrix, std::size_t row_size, std::size_t num_row = 2, std::size_t print_size = 5)
{
    std::cout << std::endl;
    for (std::size_t r = 0; r < num_row; r++)
    {
        std::cout << "    [";
        for (std::size_t i = r * row_size; i < r * row_size + print_size; i++)
        {
            std::cout << std::setw(3) << std::right << (matrix[i] % mod) << ",";
        }
        std::cout << std::setw(3) << " ...,";
        for (std::size_t i = (r + 1) * row_size - print_size; i < (r + 1) * row_size; i++)
        {
            std::cout << std::setw(3) << (matrix[i] % mod) << ((i != (r + 1) * row_size - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;
}

inline void print_matrix_mod(
    uint64_t mod, const std::vector<uint64_t> matrixx, std::size_t row_size, std::size_t num_row = 2,
    std::size_t print_size = 5);

template <typename T>
inline void print_matrix(const T *matrix, std::size_t row_size, std::size_t num_row = 2, std::size_t print_size = 5)
{
    if (print_size * 2 > row_size)
    {
        throw std::invalid_argument("print_size * 2 > row_size");
    }
    std::cout << std::endl;
    for (std::size_t r = 0; r < num_row; r++)
    {
        std::cout << "    [";
        for (std::size_t i = r * row_size; i < r * row_size + print_size; i++)
        {
            std::cout << std::setw(3) << std::right << matrix[i] << ",";
        }
        if (print_size * 2 < row_size)
        {
            std::cout << std::setw(3) << " ...,";
        }
        for (std::size_t i = (r + 1) * row_size - print_size; i < (r + 1) * row_size; i++)
        {
            std::cout << std::setw(3) << matrix[i] << ((i != (r + 1) * row_size - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_matrix(
    const std::vector<T> matrix, std::size_t row_size, std::size_t num_row = 2, std::size_t print_size = 5)
{
    print_matrix(matrix.data(), row_size, num_row, print_size);
}

template <typename T>
inline void print_matrix_block(
    const T *matrix, std::size_t total_size, std::size_t total_row_size, std::size_t block_rows, std::size_t block_cols,
    std::size_t block_row_idx, std::size_t block_col_idx)
{
    std::size_t start_row = block_row_idx * block_rows;
    std::size_t start_col = block_col_idx * block_cols;
    std::size_t total_col_size = total_size / total_row_size;
    std::size_t end_row = std::min(start_row + block_rows, total_col_size);
    std::size_t end_col = std::min(start_col + block_cols, total_row_size);

    std::cout << std::endl
              << block_rows << "x" << block_cols << " Block [" << block_row_idx << "," << block_col_idx
              << "]:" << std::endl;
    for (std::size_t r = start_row; r < end_row; r++)
    {
        std::cout << "    [";
        for (std::size_t c = start_col; c < end_col; c++)
        {
            std::cout << std::setw(3) << std::right << matrix[r * total_row_size + c]
                      << ((c != end_col - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_matrix_block(
    const std::vector<T> &matrix, std::size_t total_row_size, std::size_t block_rows, std::size_t block_cols,
    std::size_t block_row_idx, std::size_t block_col_idx)
{
    print_matrix_block(
        matrix.data(), matrix.size(), total_row_size, block_rows, block_cols, block_row_idx, block_col_idx);
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const seal::SEALContext &context);

void print_cuda_device_info(bool disable_gpu);

void fill_random_vector(std::vector<uint64_t> &vec, uint64_t plainWidth);

void fill_random_weight(std::vector<uint64_t> &vec, size_t copy_count, uint64_t plainWidth);

size_t get_baby_step(size_t M);