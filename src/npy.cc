#include "npy.hpp"

#include <numeric>
#include <regex>

using namespace tnpy;

namespace {
std::string parseHeader(std::istream &Stream) {
  struct {
    uint64_t Magic : 48;
    uint8_t Major;
    uint8_t Minor;
  } MagicAndVersion;

  Stream.read(reinterpret_cast<char *>(&MagicAndVersion),
              sizeof(MagicAndVersion));

  if (MagicAndVersion.Magic != 0x59504d554e93)
    throw std::runtime_error("Invalid magic value");

  if (MagicAndVersion.Major != 1)
    throw std::runtime_error("Only version 1.x supported");

  uint16_t HeaderLen;
  Stream.read(reinterpret_cast<char *>(&HeaderLen), sizeof(HeaderLen));
  if ((sizeof(MagicAndVersion) + sizeof(HeaderLen) + HeaderLen) & 63)
    throw std::runtime_error("Invalid padding");

  std::string HeaderData(HeaderLen, '0');
  Stream.read(HeaderData.data(), HeaderLen);

  return HeaderData;
}

Npy::dtype_t parseDType(std::string const &Value) {
  for (auto const &[Str, Result] : {
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
       })
    if (Str == Value)
      return Result;

  throw std::runtime_error("data type " + Value + " not understood");
}

Npy::shape_t parseShape(std::string const &Value) {
  Npy::shape_t Shape;
  std::regex const NumberRegex("[0-9]+");

  std::transform(std::sregex_iterator(begin(Value), end(Value), NumberRegex),
                 std::sregex_iterator(), std::back_inserter(Shape),
                 [](auto &&Number) { return std::stoul(Number.str()); });

  return Shape.empty() ? Npy::shape_t{1} : Shape;
}

bool parseFortranOrder(std::string_view const &Value) {
  for (auto const &[StrValue, BoolValue] :
       {std::pair{"True", true}, std::pair{"False", false}})
    if (Value == StrValue)
      return BoolValue;

  throw std::runtime_error("Failed to parse fortran_order");
}

void parseHeaderData(std::string const &HeaderData, Npy::dtype_t &DataType,
                     bool &FortranOrder, Npy::shape_t &Shape) {
  std::regex const BaseRegex(
      R"(\{'descr':\s'[|<]([bfiu][1248])',\s'fortran_order':\s(True|False),\s'shape':\s([()0-9,\s]+),\s}\s*)");

  std::smatch BaseMatch;
  if (!std::regex_match(HeaderData, BaseMatch, BaseRegex))
    throw std::runtime_error("Failed to parse header data");

  DataType = parseDType(BaseMatch[1].str());
  FortranOrder = parseFortranOrder(BaseMatch[2].str());
  Shape = parseShape(BaseMatch[3].str());
}

bool isLittleEndian() {
  uint32_t const Value{1};
  static_assert(sizeof(Value) == 4);
  return *(reinterpret_cast<const std::byte *>(&Value)) == std::byte(1);
}

} // namespace

Npy::Npy(std::istream &Stream) {
  if (not isLittleEndian())
    throw std::runtime_error("Only little endian is supported atm");

  parseHeaderData(parseHeader(Stream), DType, FortranOrder, Shape);
  populateArrayData(Stream);
}

bool Npy::isFortranOrder() const { return FortranOrder; }

Npy::dtype_t Npy::dtype() const { return DType; }

void const *Npy::ptr() const { return reinterpret_cast<void *>(Buffer.get()); }

Npy::shape_t const &Npy::shape() const { return Shape; }

void Npy::populateArrayData(std::istream &Stream) {
  auto const ElementSize =
      std::visit([](auto &&Arg) -> size_t { return sizeof(Arg); }, DType);
  auto const ElementsCount = std::accumulate(
      begin(Shape), end(Shape), 1, std::multiplies<shape_t::value_type>());
  auto const SizeBytes = ElementsCount * ElementSize;
  Buffer = buffer_t(
      reinterpret_cast<std::byte *>(std::aligned_alloc(ElementSize, SizeBytes)),
      std::free);
  Stream.read(reinterpret_cast<char *>(Buffer.get()), SizeBytes);
}
