#pragma once

#include <complex>
#include <fstream>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace tnpy {

class Npy {
public:
  using dtype_t =
      std::variant<bool, _Float16, float, double, uint8_t, int8_t, uint16_t,
                   int16_t, uint32_t, int32_t, uint64_t, int64_t,
                   std::complex<float>, std::complex<double>>;
  using shape_t = std::vector<uint32_t>;
  using buffer_t = std::unique_ptr<std::complex<double>[]>; // for max alignment

  // https://numpy.org/doc/stable/reference/generated/numpy.array.html?highlight=array#numpy-array
  enum class DataOrder : char { // NOLINT
    C = 'C',
    Fortran = 'F',
  };

  explicit Npy(std::istream &Stream);

  template <typename Type>
  Npy(shape_t const &Shape, std::vector<Type> const &Data,
      DataOrder Order = DataOrder::C)
      : DType(dtype_t(Type())), Shape(Shape), Order(Order) {
    allocateBuffer();
    std::copy(begin(Data), end(Data), reinterpret_cast<Type *>(Buffer.get()));
  }

  DataOrder order() const;

  dtype_t dtype() const;

  template <typename T> T const *data() const {
    if (not std::holds_alternative<T>(DType))
      throw std::runtime_error("Requested type do not match to holded type");

    return reinterpret_cast<T const *>(ptr());
  }

  void const *ptr() const;

  size_t bytes() const;

  shape_t const &shape() const;

  bool operator==(const Npy &Rhs) const;
  bool operator!=(const Npy &Rhs) const;

private:
  void populateArrayData(std::istream &Stream);
  size_t allocateBuffer();

private:
  dtype_t DType;
  shape_t Shape;
  DataOrder Order;
  buffer_t Buffer;
  size_t NBytes;
};

std::ostream &operator<<(std::ostream &, tnpy::Npy const &);
} // namespace tnpy
