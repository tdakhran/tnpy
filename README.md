# TNPY
tnpy is a C++17 library for reading and writing [numpy .npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) files.

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
  {
    // read
    std::ifstream File("input.npy");
    tnpy::Npy InNpy(File);
    std::vector<uint32_t> const Shape = InNpy.shape();
    tnpy::Npy::DataOrder const Order  = InNpy.order();
    tnpy::Npy::dtype_t const DType    = InNpy.dtype();
  }

  {
    // write
    std::vector<uint32_t> const Shape{2, 1};
    std::vector<float> const    Data{1., 1.};
    tnpy::Npy OutNpy(Shape, Data);
    std::ofstream("output.npy") << OutNpy;
  }
}
```

## Known issues
* Only subset of python data types is supported
* big endian is not supported
