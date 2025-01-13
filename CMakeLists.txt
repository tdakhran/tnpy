cmake_minimum_required(VERSION 3.21)

project(tnpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror")

add_library(tnpy ${CMAKE_CURRENT_SOURCE_DIR}/src/npy.cc)
target_include_directories(tnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)

include(CMakeDependentOption)
cmake_dependent_option(BUILD_TESTS "Build tests" ON "PROJECT_IS_TOP_LEVEL" OFF)
if(BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif(BUILD_TESTS)
