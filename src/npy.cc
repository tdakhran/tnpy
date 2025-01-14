#include "npy.hpp"
#include "numpy_utils.hpp"

#include <cstring>
#include <regex>

using namespace tnpy;

namespace {

std::string parseHeader(std::istream &Stream) {
  utils::MagicAndVersion MagicAndVersion;

  Stream.read(reinterpret_cast<char *>(&MagicAndVersion),
              sizeof(MagicAndVersion));

  MagicAndVersion.validate();

  uint16_t HeaderLen;
  Stream.read(reinterpret_cast<char *>(&HeaderLen), sizeof(HeaderLen));
  if ((sizeof(utils::MagicAndVersion) + sizeof(HeaderLen) + HeaderLen) %
      utils::Header::Alignment)
    throw std::runtime_error("Invalid padding");

  std::string HeaderData(HeaderLen, '0');
  Stream.read(HeaderData.data(), HeaderLen);

  return HeaderData;
}

void parseHeaderData(std::string const &HeaderData, Npy::dtype_t &DataType,
                     Npy::DataOrder &Order, Npy::shape_t &Shape) {
  std::regex const BaseRegex(
      R"(\{'descr':\s'[|<]()" + utils::DType::generatePyTypesRegex() +
      R"()',\s'fortran_order':\s(True|False),\s'shape':\s([()0-9,\s]+),\s\}\s*)");

  std::smatch BaseMatch;
  if (!std::regex_match(HeaderData, BaseMatch, BaseRegex))
    throw std::runtime_error("Failed to parse header data");

  DataType = utils::DType::from(BaseMatch[1].str());
  Order = utils::DataOrder::fromString(BaseMatch[2].str());
  Shape = utils::Shape()(BaseMatch[3].str());
}

bool isLittleEndian() {
  uint32_t const Value{1};
  static_assert(sizeof(Value) == 4);
  return *(reinterpret_cast<const std::byte *>(&Value)) == std::byte(1);
}

std::pair<size_t, size_t> calculateSizes(Npy::shape_t const &Shape,
                                         Npy::dtype_t const &DType) {
  auto const ElementsCount = std::accumulate(
      begin(Shape), end(Shape), 1, std::multiplies<Npy::shape_t::value_type>());
  auto const ElementSize =
      std::visit([](auto &&Arg) -> size_t { return sizeof(Arg); }, DType);
  return {ElementsCount, ElementSize};
}

} // namespace

Npy::Npy(std::istream &Stream) {
  if (not isLittleEndian())
    throw std::runtime_error("Only little endian is supported atm");

  parseHeaderData(parseHeader(Stream), DType, Order, Shape);
  populateArrayData(Stream);
}

Npy::DataOrder Npy::order() const { return Order; }

Npy::dtype_t Npy::dtype() const { return DType; }

void const *Npy::ptr() const { return reinterpret_cast<void *>(Buffer.get()); }

Npy::shape_t const &Npy::shape() const { return Shape; }

void Npy::populateArrayData(std::istream &Stream) {
  size_t SizeBytes = allocateBuffer();
  Stream.read(reinterpret_cast<char *>(Buffer.get()), SizeBytes);
}

size_t Npy::allocateBuffer() {
  auto const [ElementsCount, ElementSize] = calculateSizes(Shape, DType);
  NBytes = ElementsCount * ElementSize;
  Buffer = Npy::buffer_t(
      reinterpret_cast<std::byte *>(std::aligned_alloc(ElementSize, NBytes)),
      std::free);
  return NBytes;
}

size_t Npy::bytes() const { return NBytes; }

bool Npy::operator==(const Npy &Rhs) const {
  auto const [ElementsCount, ElementSize] = calculateSizes(Shape, DType);
  return std::tie(DType, Order, Shape) ==
             std::tie(Rhs.DType, Rhs.Order, Rhs.Shape) &&
         std::memcmp(ptr(), Rhs.ptr(), ElementsCount * ElementSize) == 0;
}

bool Npy::operator!=(const Npy &Rhs) const { return not operator==(Rhs); }

std::ostream &tnpy::operator<<(std::ostream &Stream, tnpy::Npy const &Object) {
  utils::MagicAndVersion MagicAndVersion;
  Stream.write(reinterpret_cast<const char *>(&MagicAndVersion),
               sizeof(MagicAndVersion));
  auto const [ElementsCount, ElementSize] =
      calculateSizes(Object.shape(), Object.dtype());
  auto const EndianChar =
      static_cast<std::underlying_type_t<utils::DType::ByteOrder>>(
          utils::DType().getByteOrder(Object.dtype()));

  // clang-format off
  auto DictStr = std::string("{")
        + "'descr': '" + EndianChar + utils::DType::from(Object.dtype()) + "', "
        + "'fortran_order': " + (Object.order() == Npy::DataOrder::Fortran ? "True" : "False") + ", "
        + "'shape': (" + utils::Shape()(Object.shape()) + "), "
        + "}\n";
  // clang-format on

  while ((sizeof(MagicAndVersion) + sizeof(uint16_t) + DictStr.size()) %
         utils::Header::Alignment)
    DictStr.push_back(' ');

  uint16_t const PaddedDictLength = DictStr.size();

  Stream.write(reinterpret_cast<const char *>(&PaddedDictLength),
               sizeof(PaddedDictLength));
  Stream.write(DictStr.data(), PaddedDictLength);
  Stream.write(reinterpret_cast<const char *>(Object.ptr()),
               ElementSize * ElementsCount);

  return Stream;
}
