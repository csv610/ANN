# ANN: Approximate Nearest Neighbor Library (Modernized)

A high-performance C++ library for approximate nearest neighbor searching in multi-dimensional spaces.

---

## 📜 Attribution & Credits

**This project is a modernized port of the original ANN library.** 

Full credit for the underlying algorithms, data structures, and the core implementation belongs to the original researchers and authors:
*   **Sunil Arya** (Hong Kong University of Science and Technology)
*   **David Mount** (University of Maryland)

The original version of ANN was released under the GNU Lesser General Public License (LGPL). This port maintains all original copyright notices and licenses.

---

### Modernization & Contributions

This fork updates the classic ANN library (version 1.1.2) to meet modern software engineering standards.

### Memory Safety & Stability (Critical Fixes)
* **Double-Free Resolved**: Fixed a major crash in parallel tree construction where bounding box copies were shallow-copying raw pointers.
* **Leak-Free**: Verified with AddressSanitizer (ASan) through millions of allocations—ensuring the wrapper and core library are completely memory-safe.
* **Hardened Deallocation**: Added defensive null-checks to internal deallocation routines to prevent undefined behavior during cleanup.

### High-Level C++ Wrapper (New!)
* **RAII Management**: Automated memory management for point arrays and search trees—no more manual `annAllocPts` or `delete`.
* **Type Safety**: Uses `std::array<double, Dim>` for points, making dimensionality errors a compile-time check.
* **Modern Results**: Returns results in a clean `std::vector<Result>` with Euclidean distances already computed.
* **Exact Matching**: Dedicated `findExactMatches` method that returns all indices within a tolerance.

### Build System

* Replaced legacy Makefiles and `Make-config` with a robust, cross-platform **CMake** build system.
* Integrated **Google Test (GTest)** for automated unit testing.
* Integrated **Google Benchmark** for performance profiling.
* Added **OpenMP** for parallel kd-tree construction on large datasets.

### C++20 Modernization
* **Type Safety**: Replaced legacy `ANNbool` enum with `bool`, added `[[nodiscard]]` attributes to API functions.
* **Exception Safety**: Added `noexcept` specifiers to const methods and trivial operations.
* **Const Correctness**: Added `const` to all non-mutating accessor methods.
* **Modern Types**: Replaced C-style `#define` constants with `constexpr` and `std::string_view`.
* **Distance Functions**: Replaced macros (`ANN_POW`, `ANN_ROOT`, etc.) with `[[nodiscard]] constexpr inline` functions for zero-overhead abstraction.
* **Header Hygiene**: Removed `using namespace std;` from headers, use explicit `std::` prefix.
* **Legacy Fixes**: Removed deprecated `register` storage class, resolved stream extraction vulnerabilities, fixed implicit return types.

---

## 🧠 The Landscape of Nearest Neighbor Search

### 1. Modern Nearest Neighbor Algorithms
Since the original release of ANN, the field has evolved significantly, especially for high-dimensional data (e.g., AI embeddings):

*   **HNSW (Hierarchical Navigable Small World)**: A graph-based approach that is currently the industry standard for high-speed, in-memory approximate search.
*   **IVF (Inverted File Index)**: Divides the vector space into Voronoi cells to narrow the search scope.
*   **Product Quantization (PQ)**: Compresses vectors to allow searching billions of points with minimal memory.
*   **DiskANN**: Optimized for datasets that are too large for RAM, utilizing SSD-resident indices.
*   **ScaNN**: Developed by Google, it uses anisotropic quantization to achieve state-of-the-art throughput.

### 2. Should we still use ANN?
**Yes, but it depends on your data.**

The ANN library implements **kd-trees** and **bd-trees**, which are "space-partitioning" structures.
*   **Use ANN if**: You are working with **low-dimensional data** (e.g., 2D/3D points in GIS, robotics, physics simulations, or CAD). In these domains, tree-based methods are often faster and more memory-efficient than modern graph-based methods.
*   **Avoid ANN if**: You are working with **high-dimensional embeddings** (e.g., 128D to 1536D vectors from LLMs or image models). In high dimensions, tree-based methods suffer from the "curse of dimensionality" and degrade to linear search ($O(n)$). For these cases, use libraries like [FAISS](https://github.com/facebookresearch/faiss) or [USearch](https://github.com/unum-cloud/usearch).

### 3. Pros and Cons of ANN

| **Pros** | **Cons** |
| :--- | :--- |
| **Deterministic**: Offers exact nearest neighbor search (if `eps=0`). | **Dimension Limit**: Performance collapses above ~20 dimensions. |
| **Low Memory**: No heavy graph edges or quantization tables to store. | **Static Data**: Better for static sets; expensive to re-balance after many inserts. |
| **Simplicity**: No complex hyperparameters like `M` or `efConstruction`. | **Single Core**: Not natively optimized for massive GPU/SIMD acceleration. |
| **Precision**: Excellent for geometric algorithms requiring high accuracy. | **Age**: Original codebase is legacy (hence the need for this modernized port). |

---

## 🚀 Getting Started

### Prerequisites

*   CMake (3.10 or higher)
*   A C++20 compatible compiler (GCC 10+, Clang 10+, or MSVC 19.29+)
*   Git (for fetching test dependencies)

### Installation & Build

```bash
# Clone the repository
git clone git@github.com:csv610/ANN.git
cd ANN

# Create a build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

The build will produce the following targets:
*   `libANN.a`: The core static library.
*   `high_level_sample`: **(Recommended)** Demonstrates the new modern C++ wrapper.
*   `ann_sample`: A simple example program using the raw C API.
*   `unit_tests`: The new GTest-based test suite.
*   `ann_benchmarks`: The new performance benchmarking tool.

---

## 🧪 Testing and Benchmarking

### Running Unit Tests
We use Google Test to ensure the integrity of the search algorithms:
```bash
./unit_tests
```

### Memory Leak Testing
A dedicated stress test to verify stability under high frequency allocation/destruction:
```bash
# Enable ASan for the most rigorous check
cmake -DENABLE_ASAN=ON ..
make leak_test
./leak_test
```

### Running Benchmarks
Measure the performance of tree construction and search on your specific hardware:
```bash
./ann_benchmarks
```

---

## 📖 Usage Example

### Modern C++ Wrapper (Recommended)

The new `NearestNeighborSearch` class hides all the manual pointer management:

```cpp
#include <ANN/NearestNeighborSearch.h>
#include <vector>
#include <array>
#include <iostream>

int main() {
    // 1. Prepare points using standard C++ containers
    std::vector<std::array<double, 3>> data = {
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}
    };

    // 2. Build the index (RAII handles memory)
    ANN::NearestNeighborSearch<3> nns(data);

    // 3. Search for 2 nearest neighbors
    std::array<double, 3> query = {1.1, 1.1, 1.1};
    auto results = nns.search(query, 2);

    for (const auto& res : results) {
        std::cout << "Index: " << res.index << ", Distance: " << res.distance << "\n";
    }

    // 4. Find exact matches (returns a vector of indices)
    auto matches = nns.findExactMatches({1.0, 1.0, 1.0});
    for (int idx : matches) {
        std::cout << "Exact match found at index: " << idx << "\n";
    }

    return 0;
}
```

### Raw ANN API (Legacy)

For low-level control, the original C-style API is still available:

```cpp
#include <ANN/ANN.h>

int main() {
    int dim = 2;
    int max_pts = 100;
    
    ANNpointArray data_pts = annAllocPts(max_pts, dim);
    // ... fill data_pts ...

    ANNkd_tree* kdTree = new ANNkd_tree(data_pts, max_pts, dim);

    ANNpoint query_pt = annAllocPt(dim);
    ANNidxArray nn_idx = new ANNidx[1];
    ANNdistArray dists = new ANNdist[1];

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    // ... use results ...

    delete kdTree;
    annClose();
    return 0;
}
```

---

## 📄 License
This library is provided under the **GNU Lesser General Public License (LGPL)**. See `License.txt` and `Copyright.txt` for the full text.
