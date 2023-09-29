#pragma once
#include "npy.hpp"

#include <complex>
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

// clang-format off
template <typename> struct CType;
template <> struct CType<                 bool > { static inline constexpr auto Py =  "b1"; };
template <> struct CType<             _Float16 > { static inline constexpr auto Py =  "f2"; };
template <> struct CType<                float > { static inline constexpr auto Py =  "f4"; };
template <> struct CType<               double > { static inline constexpr auto Py =  "f8"; };
template <> struct CType<               int8_t > { static inline constexpr auto Py =  "i1"; };
template <> struct CType<              int16_t > { static inline constexpr auto Py =  "i2"; };
template <> struct CType<              int32_t > { static inline constexpr auto Py =  "i4"; };
template <> struct CType<              int64_t > { static inline constexpr auto Py =  "i8"; };
template <> struct CType<              uint8_t > { static inline constexpr auto Py =  "u1"; };
template <> struct CType<             uint16_t > { static inline constexpr auto Py =  "u2"; };
template <> struct CType<             uint32_t > { static inline constexpr auto Py =  "u4"; };
template <> struct CType<             uint64_t > { static inline constexpr auto Py =  "u8"; };
template <> struct CType<  std::complex<float> > { static inline constexpr auto Py =  "c8"; };
template <> struct CType< std::complex<double> > { static inline constexpr auto Py = "c16"; };
// clang-format on

struct DType {
  template <typename Func, typename... Types>
  static void iterTypes(Func F, std::variant<Types...>) {
    std::apply([&F](Types... Args) { return ((F(Args)), ...); },
               std::tuple<Types...>());
  }

  static std::string from(Npy::dtype_t const &In) {
    return std::visit([](auto Arg) { return CType<decltype(Arg)>::Py; }, In);
  }

  static Npy::dtype_t from(std::string const &In) {
    Npy::dtype_t Result;
    iterTypes(
        [&In, &Result](auto Arg) {
          if (In == CType<decltype(Arg)>::Py)
            Result = Npy::dtype_t(decltype(Arg)());
        },
        Npy::dtype_t());
    return Result;
  }

  static std::string generatePyTypesRegex() {
    std::vector<std::string> PyTypes;
    iterTypes(
        [&PyTypes](auto Arg) { PyTypes.push_back(CType<decltype(Arg)>::Py); },
        Npy::dtype_t());
    return std::accumulate(begin(PyTypes), end(PyTypes), std::string(),
                           [](auto &Acc, auto const &V) {
                             return Acc.empty() ? V : (Acc + "|" + V);
                           });
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
