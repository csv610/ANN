#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <ANN/NearestNeighborSearch.h>

void run_iteration(int n_pts) {
    constexpr int dim = 3;
    std::vector<std::array<double, dim>> points(n_pts);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 100);
    for (int i = 0; i < n_pts; i++) {
        for (int d = 0; d < dim; d++) {
            points[i][d] = dis(gen);
        }
    }

    // Constructor allocates data_pts and kd_tree
    ANN::NearestNeighborSearch<dim> nns(points);
    
    // Perform a dummy search
    std::array<double, dim> query = {50.0, 50.0, 50.0};
    auto results = nns.search(query, 5);
    auto matches = nns.findExactMatches(query, 0.1);

    // Destructor should free everything
}

int main() {
    constexpr int iterations = 500;
    constexpr int points_per_iter = 10000;

    std::cout << "Starting leak test: " << iterations << " iterations of " << points_per_iter << " points...\n";

    for (int i = 0; i < iterations; ++i) {
        run_iteration(points_per_iter);
        if ((i + 1) % 100 == 0) {
            std::cout << "Completed " << (i + 1) << " iterations...\n";
        }
    }

    std::cout << "Leak test completed successfully.\n";
    // annClose() cleans up the trivial leaf node shared across all trees
    annClose();
    
    return 0;
}
