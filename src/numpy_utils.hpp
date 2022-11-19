#pragma once
#include "npy.hpp"

#include <numeric>
#include <regex>

namespace tnpy::utils {

struct Header {
  // https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0
  static constexpr inline size_t Alignment = 64;
};

struct DataOrder {
  static std::string toString(Npy::DataOrder const &Order) {
    return std::string(
        1, static_cast<std::underlying_type_t<Npy::DataOrder>>(Order));
  }

  static Npy::DataOrder fromString(std::string const &Value) {
    if (Value == "True")
      return Npy::DataOrder::Fortran;
    if (Value == "False")
      return Npy::DataOrder::C;

    throw std::runtime_error("Failed to parse fortran_order");
  }
};

struct MagicAndVersion {
  // https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0
  static constexpr inline uint64_t ExpectedMagic = 0x59504d554e93;
  static constexpr inline uint64_t ExpectedVersionMajor = 1;
  static constexpr inline uint64_t ExpectedVersionMinor = 0;

  MagicAndVersion()
      : Magic(ExpectedMagic), VersionMajor(ExpectedVersionMajor),
        VersionMinor(ExpectedVersionMinor) {}

  void validate() const {
    if (Magic != ExpectedMagic)
      throw std::runtime_error("Invalid magic value");

    if (VersionMajor != ExpectedVersionMajor)
      throw std::runtime_error("Only version 1.x supported");
  }

  uint64_t Magic : 48;
  uint8_t VersionMajor;
  uint8_t VersionMinor;
};

struct Shape {
  Npy::shape_t operator()(std::string const &ShapeStr) {
    Npy::shape_t Shape;
    std::regex const NumberRegex("[0-9]+");

    std::transform(
        std::sregex_iterator(begin(ShapeStr), end(ShapeStr), NumberRegex),
        std::sregex_iterator(), std::back_inserter(Shape),
        [](auto &&Number) { return std::stoul(Number.str()); });

    return Shape.empty() ? Npy::shape_t{1} : Shape;
  }

  std::string operator()(Npy::shape_t const &Shape) {
    return Shape == Npy::shape_t{1}
               ? ""
               : std::accumulate(begin(Shape), end(Shape), std::string(),
                                 [](std::string Acc, auto Value) {
                                   return Acc + std::to_string(Value) + ", ";
                                 });
  }
};

struct DType {
  static inline constexpr auto SupportedDTypes = {
      std::pair("b1", Npy::dtype_t(bool())),
      std::pair("f4", Npy::dtype_t(float())),
      std::pair("f8", Npy::dtype_t(double())),
      std::pair("i1", Npy::dtype_t(int8_t())),
      std::pair("i2", Npy::dtype_t(int16_t())),
      std::pair("i4", Npy::dtype_t(int32_t())),
      std::pair("i8", Npy::dtype_t(int64_t())),
      std::pair("u1", Npy::dtype_t(uint8_t())),
      std::pair("u2", Npy::dtype_t(uint16_t())),
      std::pair("u4", Npy::dtype_t(uint32_t())),
      std::pair("u8", Npy::dtype_t(uint64_t())),
  };

  static std::string toString(Npy::dtype_t const &In) {
    for (auto const &[DTypeStr, DType] : SupportedDTypes)
      if (DType == In)
        return DTypeStr;
    throw std::runtime_error("Failed to convert dtype");
  }

  static Npy::dtype_t fromString(std::string const &In) {
    for (auto const &[DTypeStr, DType] : SupportedDTypes)
      if (DTypeStr == In)
        return DType;
    throw std::runtime_error("Failed to convert dtype");
  }

  static size_t elementSize(Npy::dtype_t const &DType) {
    return std::visit([](auto &&Arg) -> size_t { return sizeof(Arg); }, DType);
  }

  // https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html#numpy-dtype-byteorder
  enum class ByteOrder : char {
    Little = '<',
    Big = '>',
    NotApplicable = '|',
    Native = '='
  };

  ByteOrder getByteOrder(Npy::dtype_t const &DType) {
    return elementSize(DType) == 1 ? ByteOrder::NotApplicable
                                   : ByteOrder::Little;
  }
};
} // namespace tnpy::utils
