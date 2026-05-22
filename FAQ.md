# Technical Frequently Asked Questions (FAQ)

### 1. What are the algorithmic complexities for tree construction and searching?
*   **Construction**: $O(d \cdot N \log N)$ for $N$ points in $d$ dimensions.
*   **Search**: $O(2^d + \log N)$ expected time. However, as $d$ increases, the constant factor grows exponentially, eventually leading to $O(N)$ behavior (the "Curse of Dimensionality").

### 2. How does the library implement epsilon-approximate searches?
The library uses an error bound $\epsilon \ge 0$. For a query point $q$, the algorithm returns a point $p$ such that $dist(q, p) \le (1 + \epsilon) dist(q, p^*)$, where $p^*$ is the true nearest neighbor. This allows the search to prune branches of the tree earlier, significantly speeding up queries in exchange for a small loss in precision.

### 3. What is the "Sliding Midpoint" splitting rule?
Unlike the standard "Midpoint" rule which splits a cell exactly in half along its longest side, the **Sliding Midpoint** rule ensures that no empty cells are created. If a split would result in all points being on one side, the splitting plane is "slid" until it encounters the first point. This prevents the creation of deep, unbalanced trees with many empty leaf nodes.

### 4. How does the library handle different Minkowski distance metrics?
ANN supports any $L_p$ norm ($p \ge 1$), including:
*   **$L_1$ (Manhattan)**
*   **$L_2$ (Euclidean)** - Default
*   **$L_\infty$ (Max)**
The metric is parameterized via macros/constexpr functions in `ANN.h`. To optimize search, the library works with "squared" distances (or the $p$-th power) to avoid expensive `pow()` or `sqrt()` calls during internal tree traversals.

### 5. What are "Shrinking Nodes" in a bd-tree?
A **bd-tree** (Balanced Box-Decomposition tree) introduces shrinking nodes, which allow for a cell to be partitioned into an "inner" box and an "outer" region. This is a generalization of the kd-tree split that helps maintain a bounded aspect ratio for cells even in the presence of highly clustered data distributions, ensuring logarithmic search depth.

### 6. How is thread safety achieved for concurrent queries?
The search structures (`ANNkd_tree`, `ANNbd_tree`) are immutable after construction. Internal search state—such as the query point, the priority queue, and the distance buffers—is managed using **thread-local storage** or stack-allocated objects. This allows multiple threads to query the same tree instance simultaneously without locks.

### 7. How does the "Partial Distance" optimization work?
During tree traversal, the algorithm maintains a "box distance"—the minimum distance from the query point to the current cell's bounding box. If this partial distance already exceeds the distance to the $k$-th nearest neighbor found so far (adjusted by $1+\epsilon$), the entire subtree is pruned immediately without inspecting individual points.

### 8. Why is the library sensitive to the "Curse of Dimensionality"?
Space-partitioning trees rely on the ability to prune large volumes of space. In high dimensions, the volume of a $d$-dimensional sphere (the query ball) becomes a negligible fraction of the $d$-dimensional cube (the search space), and almost all points end up at roughly the same distance from the origin. Consequently, the pruning logic fails, and the search must visit nearly every node.

### 9. What is the impact of "Bucket Size" on performance?
The `bkt_size` parameter determines the maximum number of points in a leaf node. 
*   **Small buckets** (e.g., 1) lead to deeper trees and more pruning opportunities.
*   **Large buckets** reduce tree depth and recursion overhead but increase the number of point-to-point distance calculations in the leaf. For most 3D applications, a bucket size of 1 to 5 is optimal.

### 10. How are internal nodes optimized for cache locality?
The library uses a compact representation for internal nodes (`ANNkd_split`). By separating coordinate data from tree topology and using a contiguous array for point indices (`pidx`), the library minimizes pointer chasing. Leaf nodes are shared via a "trivial leaf" singleton (`KD_TRIVIAL`) where possible to reduce the memory footprint and improve cache hits during broad traversals.
