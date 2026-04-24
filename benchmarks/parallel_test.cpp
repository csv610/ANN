#include <iostream>
#include <chrono>
#include <ANN/ANN.h>
#include <random>
#include <cstdlib>

int main(int argc, char** argv) {
    int dim = 3;
    int sizes[] = {1000, 10000, 100000, 1000000};
    
    for (int n_pts : sizes) {
        std::cout << "\n=== " << n_pts << " points ===" << std::endl;
        
        ANNpointArray data_pts = annAllocPts(n_pts, dim);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 100);
        for (int i = 0; i < n_pts; i++) {
            for (int d = 0; d < dim; d++) {
                data_pts[i][d] = dis(gen);
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        ANNkd_tree* kdTree = new ANNkd_tree(data_pts, n_pts, dim);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Construction time: " << duration.count() << " µs" << std::endl;
        
        delete kdTree;
        annDeallocPts(data_pts);
    }
    
    annClose();
    return 0;
}