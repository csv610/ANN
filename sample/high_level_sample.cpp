#include <iostream>
#include <vector>
#include <array>
#include <ANN/NearestNeighborSearch.h>

int main() {
    // 1. Define the dimension of our points
    constexpr size_t Dim = 3;

    // 2. Prepare some data points using standard C++ containers
    std::vector<std::array<double, Dim>> data = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {3.0, 3.0, 3.0},
        {1.1, 1.1, 1.1}
    };

    try {
        // 3. Create the high-level search object
        // This builds the internal kd-tree and handles all ANN memory
        ANN::NearestNeighborSearch<Dim> nns(data);

        std::cout << "Index built with " << nns.size() << " points.\n";

        // 4. Perform a search
        std::array<double, Dim> query = {1.05, 1.05, 1.05};
        int k = 2;
        
        std::cout << "Searching for " << k << " nearest neighbors to (1.05, 1.05, 1.05)...\n";
        
        auto results = nns.search(query, k);

        // 5. Display results
        std::cout << "Found " << results.size() << " neighbors:\n";
        for (const auto& res : results) {
            std::cout << " - Index: " << res.index 
                      << ", Point: (" << data[res.index][0] << ", " << data[res.index][1] << ", " << data[res.index][2] << ")"
                      << ", Distance: " << res.distance << "\n";
        }

        // 6. Find exact matches
        auto matches = nns.findExactMatches(query, 0.1);
        if (!matches.empty()) {
            std::cout << "Exact (or near) matches found: ";
            for (int idx : matches) std::cout << idx << " ";
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Optional: annClose() can be called if this is the end of the program
    annClose();

    return 0;
}
