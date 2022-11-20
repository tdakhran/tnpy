#include "numpy_utils.hpp"

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
template <typename T> struct TestConfiguration {
  using type = T;

  std::string const PythonDType;
  std::string const PythonData;
  Npy::shape_t const Shape;
  std::vector<T> const Data;
  Npy::DataOrder const Order;
};

std::filesystem::path generateNpyFile(std::string const &PythonData,
                                      std::string const &PythonDtype,
                                      Npy::DataOrder const Order) {
  auto const FileName = std::filesystem::temp_directory_path() /
                        ("tnpy.test." + std::to_string(rand()) + ".npy");
  std::array<char, 256> Command;
  if (auto Result = std::snprintf(
          Command.data(), Command.size(),
          "python3 -c \"import numpy; "
          "numpy.save('%s',numpy.array(%s,numpy.dtype('%s'),order='%s'))\"",
          FileName.c_str(), PythonData.c_str(), PythonDtype.c_str(),
          utils::DataOrder::toString(Order).c_str());
      Result < 0 or Result >= static_cast<decltype(Result)>(Command.size())) {
    throw std::runtime_error("Failed to assemble command");
  }
  if (system(Command.data()) or not std::filesystem::exists(FileName))
    throw std::runtime_error("Test file was not created");

  return FileName;
}

template <typename Type> bool isFailed() {
  auto const PyType = utils::DType::from(Npy::dtype_t(Type()));
  for (auto const &Config : {
           TestConfiguration<Type>{PyType, "1", {1}, {1}, Npy::DataOrder::C},
           TestConfiguration<Type>{PyType, "[]", {0}, {}, Npy::DataOrder::C},
           TestConfiguration<Type>{
               PyType, "[[]]", {1, 0}, {}, Npy::DataOrder::C},
           TestConfiguration<Type>{
               PyType, "[[1], [1]]", {2, 1}, {1, 1}, Npy::DataOrder::C},
           TestConfiguration<Type>{PyType,
                                   "[[1, 1], [0, 1]]",
                                   {2, 2},
                                   {1, 0, 1, 1},
                                   Npy::DataOrder::Fortran},
       }) {

    auto const FileName =
        generateNpyFile(Config.PythonData, Config.PythonDType, Config.Order);
    std::ifstream InFile(FileName, std::ios::binary);
    auto const NpyFromPy = Npy(InFile);
    InFile.close();
    std::filesystem::remove(FileName);

    auto const NpyConstructed = Npy(Config.Shape, Config.Data, Config.Order);
    if (NpyFromPy != NpyConstructed) {
      failure<Type>();
      return true;
    }

    std::stringstream Stream;
    Stream << NpyConstructed;
    auto const NpyFromStream = Npy(Stream);
    if (NpyFromPy != NpyFromStream) {
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
