#pragma once

#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>
#include "Datatype/UnifiedType.h"

namespace Datatype
{
    enum OT_TYPE
    {
        IKNP = 0,
        VOLE
    };
    enum CONV_TYPE
    {
        Cheetah = 0,
        Nest
    };
    enum PARTY
    {
        SERVER = 1,
        CLIENT = 2
    };

    template <typename T>
    class Tensor
    {
    public:
        // for fixed point number
        int32_t bitwidth = 0;
        int32_t scale = 0;
        Tensor() = default;

        explicit Tensor(const std::vector<size_t> &shape) : shape_(shape)
        {
            computeStrides();
            data_.resize(totalSize(), T(0));
            computeStrides();
        }

        explicit Tensor(const std::vector<size_t> &shape, LOCATION loc) : shape_(shape)
        {
            computeStrides();
            data_.resize(totalSize(), T(loc));
            computeStrides();
        }

        explicit Tensor(const std::vector<size_t> &shape, int32_t bitwidth, int32_t scale)
            : shape_(shape), bitwidth(bitwidth), scale(scale)
        {
            computeStrides();
            data_.resize(totalSize(), T(0));
            computeStrides();
        }

        explicit Tensor(const std::vector<size_t> &shape, LOCATION loc, int32_t bitwidth, int32_t scale)
            : shape_(shape), bitwidth(bitwidth), scale(scale)
        {
            computeStrides();
            data_.resize(totalSize(), T(loc));
            computeStrides();
        }

        Tensor(const std::vector<size_t> &shape, T zeros_ct) : shape_(shape)
        {
            computeStrides();
            data_.resize(totalSize(), zeros_ct);
            computeStrides();
        }

        Tensor(const std::vector<size_t> &shape, const std::initializer_list<T> &values) : shape_(shape), data_(values)
        {
            // static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
            computeStrides();
            assert(data_.size() == totalSize() && "Data size does not match shape");
        }

        // Randomize a tensor mod Q, must be integer type. Can not be applied in secure application!
        void randomize(uint64_t Q)
        {
            // change seed randomly
            srand(time(0));
            if constexpr (std::is_signed_v<T>)
            {
                for (size_t i = 0; i < data_.size(); ++i)
                {
                    data_[i] = static_cast<T>(rand()) % Q;
                    data_[i] -= static_cast<T>(Q / 2);
                }
            }
            else if constexpr (std::is_unsigned_v<T>)
            {
                for (size_t i = 0; i < data_.size(); ++i)
                {
                    data_[i] = static_cast<T>(rand()) % Q;
                }
            }
            else
            {
                std::cerr << "Randomize for non-integer type under modulous is not supported" << std::endl;
            }
        }

        // Randomize a tensor without mod, can be any type like float
        void randomize()
        {
            for (size_t i = 0; i < data_.size(); ++i)
            {
                data_[i] = static_cast<T>(rand() / double(RAND_MAX));
            }
        }

        // Flatten
        void flatten()
        {
            shape_ = { data_.size() };
            strides_ = { 1 };
        }
        // Reshape
        void reshape(const std::vector<size_t> &new_shape)
        {
            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
            assert(new_size == data_.size() && "Total size must remain unchanged in reshape");
            shape_ = new_shape;
            computeStrides();
        }

        const T &operator()(const size_t &indice) const
        {
            return data_[indice];
        }

        T &operator()(const size_t &indice)
        {
            return data_[indice];
        }

        T &operator()(const std::vector<size_t> &indices)
        {
            size_t idx = computeIndex(indices);
            return data_[idx];
        }

        const T &operator()(const std::vector<size_t> &indices) const
        {
            size_t idx = computeIndex(indices);
            return data_[idx];
        }

        T &operator()(std::initializer_list<size_t> indices)
        {
            return operator_(std::vector<size_t>(indices));
        }

        const T &operator()(std::initializer_list<size_t> indices) const
        {
            return operator_(std::vector<size_t>(indices));
        }

        const std::vector<size_t> &shape() const
        {
            return shape_;
        }

        const size_t size() const
        {
            return totalSize();
        }

        const std::vector<T> &data() const
        {
            return data_;
        }

        std::vector<T> &data()
        {
            return data_;
        }

        // Apply Element-wise function
        template <typename Func>
        void apply(Func func)
        {
            for (auto &element : data_)
            {
                func(element);
            }
        }

        Tensor<T> operator+(const Tensor<T> &other) const
        {
            static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
            assert(shape_ == other.shape_ && "Shapes must match for addition");
            // TODO only support arithmetic types
            Tensor<T> result(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                result.data_[i] = data_[i] + other.data_[i];
            }
            return result;
        }

        Tensor<T> operator-(const Tensor<T> &other) const
        {
            static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
            assert(shape_ == other.shape_ && "Shapes must match for subtraction");
            Tensor<T> result(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                result.data_[i] = data_[i] - other.data_[i];
            }
            return result;
        }

        // multiply a scalar
        Tensor<T> operator*(const T &scalar) const
        {
            Tensor<T> result(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                result.data_[i] = data_[i] * scalar;
            }
            return result;
        }

        void print_shape() const
        {
            std::cout << "Shape: [";
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                std::cout << shape_[i];
                if (i != shape_.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        void print(size_t num_print = 1e7) const
        {
            std::cout << "Shape: [";
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                std::cout << shape_[i];
                if (i != shape_.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]\nData: [";
            size_t size = std::min(this->size(), num_print);
            for (size_t i = 0; i < size; ++i)
            {
                if (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value)
                {
                    std::cout << (int)data_[i];
                }
                else
                {
                    std::cout << data_[i];
                }
                if (i != size - 1)
                    std::cout << ", ";
            }
            if (this->size() > num_print)
            {
                std::cout << "...";
            }
            std::cout << "]\n";
        }

        const T &operator_(const std::vector<size_t> &indices) const
        {
            return operator()(indices);
        }

        Tensor(const Tensor &other)
        {
            *this = other;
        }

        Tensor &operator=(const Tensor &other)
        {
            // std::cout << "[warning] deep copy" << std::endl;
            shape_ = other.shape_;
            strides_ = other.strides_;
            data_ = other.data_;
            bitwidth = other.bitwidth;
            scale = other.scale;
            return *this;
        }

        Tensor(Tensor &&) = default;

        Tensor &operator=(Tensor &&other) = default;

        uint64_t get_mask() const
        {
            return (1ULL << bitwidth) - 1;
        }

    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        std::vector<T> data_;

        void computeStrides()
        {
            strides_.resize(shape_.size());
            if (shape_.empty())
                return;
            strides_[shape_.size() - 1] = 1;
            for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
            {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }

        size_t totalSize() const
        {
            if (shape_.empty())
                return 0;
            return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
        }

        size_t computeIndex(const std::vector<size_t> &indices) const
        {
            assert(indices.size() == shape_.size() && "Number of indices must match number of dimensions");
            size_t idx = 0;
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                assert(indices[i] < shape_[i] && "Index out of bounds");
                idx += indices[i] * strides_[i];
            }
            return idx;
        }

        T &operator_(const std::vector<size_t> &indices)
        {
            return const_cast<T &>(static_cast<const Tensor<T> &>(*this).operator()(indices));
        }
    };

} // namespace Datatype