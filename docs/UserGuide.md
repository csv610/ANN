# User Guide: Approximate Nearest Neighbor (ANN) Library

Welcome to the world of computational geometry and high-speed searching! This guide is designed for undergraduate students and researchers who need to solve the **Nearest Neighbor Search (NNS)** problem efficiently.

---

## 📖 Table of Contents
1. [Introduction to Nearest Neighbor Search](#1-introduction)
2. [The Geometry of Space and Distance](#2-geometry)
3. [Understanding KD-Trees](#3-kd-trees)
4. [Balanced Box-Decomposition (BD) Trees](#4-bd-trees)
5. [Standard vs. Priority Search](#5-search-algorithms)
6. [The Power of Approximation (Epsilon)](#6-approximation)
7. [Programming with ANN](#7-programming)
8. [Performance and Limitations](#8-performance)

---

## 1. Introduction
Imagine you have a map with thousands of cities. If I give you a new location, how do you find the closest city? In 2D, this is simple. But what if you are working with 10-dimensional physics data or 3D robotic sensor points? 

As the number of points ($N$) and the number of dimensions ($d$) grow, the "brute-force" method (checking every single point) becomes painfully slow ($O(N \cdot d)$). The **ANN Library** provides specialized data structures that partition space to find these neighbors in $O(d \cdot \log N)$ time.

---

## 2. Geometry: Points and Metrics
In ANN, a **Point** is an array of coordinates. A 3D point might look like `[1.2, 3.4, -0.5]`.

### Distance Metrics
How do we define "closest"? ANN supports the **Minkowski $L_p$ distance**:
$$Dist(P, Q) = \left( \sum_{i=1}^d |P_i - Q_i|^p \right)^{1/p}$$

*   **$L_2$ (Euclidean)**: The straight-line distance ($p=2$). This is the default.
*   **$L_1$ (Manhattan)**: The "taxicab" distance ($p=1$).
*   **$L_\infty$ (Max)**: The maximum difference along any single axis.

> **Note:** For efficiency, ANN often works with **squared distances** to avoid expensive square root operations during internal calculations.

---

## 3. KD-Trees: Dividing the World
The **kd-tree** (k-dimensional tree) is the workhorse of this library. It is a binary tree where every node represents a rectangular region of space.

### How it works:
1.  **Split**: Choose an axis (e.g., $x$) and a split point.
2.  **Partition**: Points to the left of the split go to the left child; points to the right go to the right child.
3.  **Repeat**: Alternate axes ($x, y, z, x \dots$) until each leaf contains only a few points (the "bucket").

ANN offers several splitting rules, such as `ANN_KD_SUGGEST` (the default optimized rule) and `ANN_KD_MIDPT`.

---

## 4. BD-Trees: Handling Clusters
Standard kd-trees can become "skinny" and inefficient if points are very tightly clustered. The **BD-tree** (Balanced Box-Decomposition) adds a new trick: **Shrinking**.

Instead of just splitting a box in half, a BD-tree can "scoop out" a smaller box from inside a larger one. This ensures that the regions (cells) remain well-shaped even with highly non-uniform data.

---

## 5. Search Algorithms
Once your tree is built, how do you find the neighbor?

*   **Standard Search**: A recursive traversal. It visits the "closer" child first and uses a pruning technique to skip branches that are too far away.
*   **Priority Search**: Uses a priority queue to always visit the cell that is closest to the query point. This is often faster for approximate searches.

---

## 6. The Power of Approximation (Epsilon)
The "A" in ANN stands for **Approximate**. Why would we want an approximate answer?

In high dimensions, finding the *exact* nearest neighbor is very hard. However, if you are willing to accept a neighbor that is at most $(1 + \epsilon)$ times further than the true nearest neighbor, the search becomes significantly faster.

*   $\epsilon = 0.0$: Exact nearest neighbor.
*   $\epsilon = 0.1$: The result is within 10% of the true shortest distance.

---

## 7. Programming with ANN
Here is the basic workflow for using the library in C++:

### Step 1: Initialization and Allocation
```cpp
#include <ANN/ANN.h>

int dim = 3;      // 3D Space
int maxPts = 1000; 

// Allocate memory for points
ANNpointArray dataPts = annAllocPts(maxPts, dim);
ANNpoint queryPt = annAllocPt(dim);
```

### Step 2: Build the Tree
```cpp
// Assume dataPts is filled with your data...
ANNkd_tree* tree = new ANNkd_tree(dataPts, maxPts, dim);
```

### Step 3: Search
```cpp
ANNidxArray nnIdx = new ANNidx[1];    // To store result index
ANNdistArray dists = new ANNdist[1];  // To store result distance

tree->annkSearch(queryPt, 1, nnIdx, dists, 0.0);

std::cout << "Closest point is at index: " << nnIdx[0] << std::endl;
```

### Step 4: Cleanup
```cpp
delete tree;
delete[] nnIdx;
delete[] dists;
annClose(); // Clean up internal ANN resources
```

---

## 8. Performance and Limitations

### The Curse of Dimensionality
Tree-based structures like kd-trees are incredibly fast for low dimensions (2D to ~15D). However, as dimensions increase (e.g., 100D), the tree becomes less effective, and performance may degrade toward a simple linear scan.

### Memory Management
ANN does **not** copy your input points. It stores pointers to them. Therefore, you must ensure your `dataPts` array remains valid for as long as the tree exists.

### Static vs. Dynamic
ANN is optimized for **static** datasets. You build the tree once and query it many times. If your points are constantly moving or being deleted, you may need to rebuild the tree.

---
*Happy Coding! For more details, refer to the original research papers by Sunil Arya and David Mount.*
