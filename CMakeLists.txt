cmake_minimum_required(VERSION 3.16)
project(ANN VERSION 1.1.2)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Version configuration
set(ANN_VERSION_MAJOR ${ANN_PROJECT_VERSION_MAJOR})
set(ANN_VERSION_MINOR ${ANN_PROJECT_VERSION_MINOR})
set(ANN_VERSION_PATCH ${ANN_PROJECT_VERSION_PATCH})
set(ANN_VERSION ${ANN_PROJECT_VERSION})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ANN/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/ANN/Config.h
)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenMP support for parallel tree construction
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found - tree construction will be single-threaded")
endif()

# Assimp for mesh loading
find_package(assimp QUIET)
if(assimp_FOUND)
    message(STATUS "Assimp found: ${assimp_VERSION}")
else()
    message(STATUS "Assimp not found - mesh loading disabled")
endif()

# ANN Library
set(ANN_HEADERS
    include/ANN/ANN.h
    include/ANN/ANNperf.h
    include/ANN/ANNx.h
    include/ANN/NearestNeighborSearch.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/ANN/Config.h
)

set(ANN_SOURCES
    src/ANN.cpp
    src/bd_fix_rad_search.cpp
    src/bd_pr_search.cpp
    src/bd_search.cpp
    src/bd_tree.cpp
    src/brute.cpp
    src/kd_dump.cpp
    src/kd_fix_rad_search.cpp
    src/kd_pr_search.cpp
    src/kd_search.cpp
    src/kd_split.cpp
    src/kd_tree.cpp
    src/kd_util.cpp
    src/perf.cpp
)

set(ANN_INTERNAL_HEADERS
    src/bd_tree.h
    src/kd_fix_rad_search.h
    src/kd_pr_search.h
    src/kd_search.h
    src/kd_split.h
    src/kd_tree.h
    src/kd_util.h
    src/pr_queue_k.h
    src/pr_queue.h
)

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

add_library(ANN STATIC ${ANN_SOURCES} ${ANN_HEADERS} ${ANN_INTERNAL_HEADERS})

target_include_directories(ANN 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        src
)

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

add_executable(min_dist_stats sample/min_dist_stats.cpp)
target_link_libraries(min_dist_stats PRIVATE ANN)
if(assimp_FOUND)
    target_link_libraries(min_dist_stats PRIVATE assimp::assimp)
    target_compile_definitions(min_dist_stats PRIVATE ASSIMP_FOUND)
endif()

# Installation
install(TARGETS ANN
        EXPORT ANNTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(EXPORT ANNTargets
        FILE ANNTargets.cmake
        NAMESPACE ANN::
        DESTINATION lib/cmake/ANN)

install(DIRECTORY include/ANN DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ANN/Config.h DESTINATION include/ANN)

