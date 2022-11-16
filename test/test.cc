#include <npy.hpp>

#include <filesystem>
#include <iostream>

using namespace tnpy;

template <typename Type> void failure() {
  std::cerr << __PRETTY_FUNCTION__ << std::endl;
}
template <typename Type> void success() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

namespace {

template <typename... T> constexpr bool is_always_false_v = false; // NOLINT

template <typename T> struct TestConfiguration {
  using type = T;

  std::string const PythonDType;
  std::string const PythonData;
  Npy::shape_t const Shape;
  std::vector<T> const Data;
};

std::filesystem::path generateNpyFile(std::string const &PythonData,
                                      std::string const &PythonDtype) {
  auto const FileName = std::filesystem::temp_directory_path() /
                        ("tnpy.test." + std::to_string(rand()) + ".npy");
  if (std::filesystem::exists(FileName))
    std::filesystem::remove(FileName);
  std::string const PythonCode =
      "import numpy; numpy.save('" + FileName.string() + "',numpy.array(" +
      PythonData + ",numpy.dtype('" + PythonDtype + "')))";
  std::string const Command = "python3 -c \"" + PythonCode + "\"";
  if (system(Command.c_str()) or not std::filesystem::exists(FileName))
    throw std::runtime_error("Test file was not created");

  return FileName;
}

template <typename T>
bool compare(Npy const &Instance, Npy::shape_t const &Shape,
             std::vector<T> const &Data) {
  if (Shape != Instance.shape())
    return false;
  if (not std::equal(begin(Data), end(Data), Instance.data<T>()))
    return false;
  return true;
}

template <typename T> constexpr std::string_view cppTypeToDtype() {
  if constexpr (std::is_same_v<T, bool>)
    return "b1";
  else if constexpr (std::is_same_v<T, float>)
    return "f4";
  else if constexpr (std::is_same_v<T, double>)
    return "f8";
  else if constexpr (std::is_same_v<T, int8_t>)
    return "i1";
  else if constexpr (std::is_same_v<T, int16_t>)
    return "i2";
  else if constexpr (std::is_same_v<T, int32_t>)
    return "i4";
  else if constexpr (std::is_same_v<T, int64_t>)
    return "i8";
  else if constexpr (std::is_same_v<T, uint8_t>)
    return "u1";
  else if constexpr (std::is_same_v<T, uint16_t>)
    return "u2";
  else if constexpr (std::is_same_v<T, uint32_t>)
    return "u4";
  else if constexpr (std::is_same_v<T, uint64_t>)
    return "u8";
  else
    static_assert(is_always_false_v<T>);
}

template <typename Type> bool isFailed() {
  auto const DtypeStr = std::string(cppTypeToDtype<Type>());
  for (auto const &Config : {
           TestConfiguration<Type>{DtypeStr, "1", {1}, {1}},
           TestConfiguration<Type>{DtypeStr, "[]", {0}, {}},
           TestConfiguration<Type>{DtypeStr, "[[]]", {1, 0}, {}},
           TestConfiguration<Type>{DtypeStr, "[[1], [1]]", {2, 1}, {1, 1}},
       }) {

    auto FileName = generateNpyFile(Config.PythonData, Config.PythonDType);
    std::ifstream Stream(FileName, std::ios::binary);
    auto NpyInstance = Npy(Stream);
    Stream.close();
    std::filesystem::remove(FileName);

    if (not compare(NpyInstance, Config.Shape, Config.Data)) {
      failure<Type>();
      return true;
    }
  }
  success<Type>();
  return false;
}

template <typename... Types>
size_t runTests(std::variant<Types...> &&, std::tuple<Types...> T = {}) {
  return std::apply(
      [](auto &&...Args) {
        return ((isFailed<std::decay_t<decltype(Args)>>()) + ... + 0);
      },
      T);
}
} // namespace

int main() {

  if (auto const NumFailed = runTests(Npy::dtype_t()); NumFailed) {
    std::cerr << NumFailed << " tests failed" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}