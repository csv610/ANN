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

## 🛠 Modernization & Contributions

This fork updates the classic ANN library (version 1.1.2) to meet modern software engineering standards. Key improvements include:

*   **Build System**: Replaced legacy Makefiles and `Make-config` with a robust, cross-platform **CMake** build system.
*   **C++ Standards**: Upgraded the codebase to **C++20**.
    *   Removed deprecated `register` storage class specifiers.
    *   Resolved `char*` stream extraction vulnerabilities in compliance with C++20 standards.
    *   Fixed implicit return type issues and modernized function signatures.
*   **Testing Infrastructure**: Integrated **Google Test (GTest)** for automated unit testing.
*   **Performance Profiling**: Integrated **Google Benchmark** to provide empirical performance data for kd-tree and bd-tree structures.
*   **Code Quality**: Cleaned up legacy artifacts (redundant `MS_Win32` directories) and resolved long-standing compiler warnings.

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
*   `ann_test`: The original evaluation and validation tool.
*   `ann_sample`: A simple example program.
*   `ann2fig`: A utility to visualize the search structures.
*   `unit_tests`: The new GTest-based test suite.
*   `ann_benchmarks`: The new performance benchmarking tool.

---

## 🧪 Testing and Benchmarking

### Running Unit Tests
We use Google Test to ensure the integrity of the search algorithms:
```bash
./unit_tests
```

### Running Benchmarks
Measure the performance of tree construction and search on your specific hardware:
```bash
./ann_benchmarks
```

---

## 📖 Usage Example

To use ANN in your own C++ project:

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
