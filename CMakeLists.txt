cmake_minimum_required(VERSION 3.10)
project(ANN VERSION 1.1.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# OpenMP support for parallel tree construction
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found - tree construction will be single-threaded")
endif()

# Include directories
include_directories(include)
include_directories(src)

# Optional AddressSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
set(ASAN_FLAGS "")
if(ENABLE_ASAN)
    if(MSVC)
        set(ASAN_FLAGS "/fsanitize=address")
    else()
        set(ASAN_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer")
    endif()
endif()

# ANN Library
file(GLOB ANN_SOURCES "src/*.cpp")
add_library(ANN STATIC ${ANN_SOURCES})
if(ASAN_FLAGS)
    target_compile_options(ANN PUBLIC ${ASAN_FLAGS})
    target_link_options(ANN PUBLIC ${ASAN_FLAGS})
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(ANN PUBLIC OpenMP::OpenMP_CXX)
endif()

# Testing dependencies
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest googlebenchmark)

enable_testing()

add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE ANN GTest::gtest_main)
if(ASAN_FLAGS)
    target_compile_options(unit_tests PRIVATE ${ASAN_FLAGS})
    target_link_options(unit_tests PRIVATE ${ASAN_FLAGS})
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)

# Benchmarking
add_executable(ann_benchmarks benchmarks/ann_benchmarks.cpp)
target_link_libraries(ann_benchmarks PRIVATE ANN benchmark::benchmark_main)
if(ASAN_FLAGS)
    target_compile_options(ann_benchmarks PRIVATE ${ASAN_FLAGS})
    target_link_options(ann_benchmarks PRIVATE ${ASAN_FLAGS})
endif()

# Leak Testing
add_executable(leak_test tests/leak_test.cpp)
target_link_libraries(leak_test PRIVATE ANN)
if(ASAN_FLAGS)
    target_compile_options(leak_test PRIVATE ${ASAN_FLAGS})
    target_link_options(leak_test PRIVATE ${ASAN_FLAGS})
endif()

# Executables (Legacy tests/samples)
add_executable(ann_test tests/ann_test.cpp tests/rand.cpp)
target_link_libraries(ann_test PRIVATE ANN)

add_executable(ann_sample sample/ann_sample.cpp)
target_link_libraries(ann_sample PRIVATE ANN)

add_executable(high_level_sample sample/high_level_sample.cpp)
target_link_libraries(high_level_sample PRIVATE ANN)

# Installation
install(TARGETS ANN ann_test ann_sample
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/ANN DESTINATION include)
