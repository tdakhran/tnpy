# TNPY
tnpy is a C++17 library for reading [numpy .npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) files.

## Compilation
### Prerequisites
The library requires C++17 compiler and [cmake](https://cmake.org/) 3.21+.

```
host$ docker run -it --rm -v $PWD:/mnt ubuntu:22.04
docker# apt update && apt install cmake g++
```
### Compile tnpy
```bash
docker# mkdir /tmp/build && cd /tmp/build && cmake /mnt && make -j4
```

## Usage
```cpp
#include "npy.hpp"
#include <fstream>

int main() {
  std::ifstream File("example.npy");
  auto Npy = tnpy::Npy(File);

  std::vector<uint32_t> const Shape =          Npy.shape();
  bool const                  isFortranOrder = Npy.isFortranOrder();
  tnpy::Npy::dtype_t          DType =          Npy.dtype();
}
```

## Known issues
* Only subset of python data types is supported
* big endian npy files are not supported
