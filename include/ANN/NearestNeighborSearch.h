#ifndef ANN_NEAREST_NEIGHBOR_SEARCH_H
#define ANN_NEAREST_NEIGHBOR_SEARCH_H

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <ANN/ANN.h>

namespace ANN {

/**
 * @brief A high-level wrapper for the ANN library.
 * 
 * This class provides a clean C++ interface for performing nearest neighbor searches
 * using std::array for points and std::vector for results, hiding the raw pointer
 * management and internal ANN data structures.
 * 
 * @tparam Dim The dimension of the space.
 */
template <size_t Dim>
class NearestNeighborSearch {
public:
    using Point = std::array<double, Dim>;

    /**
     * @brief Structure to hold search results.
     */
    struct Result {
        int index;       ///< Index of the point in the original collection.
        double distance; ///< Euclidean distance to the query point.
    };

    /**
     * @brief Construct a new Nearest Neighbor Search object.
     * 
     * Builds a kd-tree from the provided points. The points are copied into
     * an internal ANN-compatible format.
     * 
     * @param points A collection of points (e.g., std::vector<std::array<double, Dim>>).
     * @throws std::invalid_argument If the point set is empty.
     */
    template <typename Container>
    explicit NearestNeighborSearch(const Container& points) 
        : n_pts(static_cast<int>(points.size())) {
        
        if (n_pts == 0) {
            throw std::invalid_argument("Point set cannot be empty.");
        }

        // Allocate ANN-style point array
        data_pts = annAllocPts(n_pts, static_cast<int>(Dim));
        
        int i = 0;
        for (const auto& pt : points) {
            for (size_t d = 0; d < Dim; ++d) {
                data_pts[i][d] = pt[d];
            }
            i++;
        }
        
        // Build the kd-tree search structure
        kd_tree = new ANNkd_tree(data_pts, n_pts, static_cast<int>(Dim));
    }

    // Move-only semantics to manage raw ANN pointers
    NearestNeighborSearch(const NearestNeighborSearch&) = delete;
    NearestNeighborSearch& operator=(const NearestNeighborSearch&) = delete;

    NearestNeighborSearch(NearestNeighborSearch&& other) noexcept
        : n_pts(other.n_pts), data_pts(other.data_pts), kd_tree(other.kd_tree) {
        other.data_pts = nullptr;
        other.kd_tree = nullptr;
        other.n_pts = 0;
    }

    NearestNeighborSearch& operator=(NearestNeighborSearch&& other) noexcept {
        if (this != &other) {
            cleanup();
            n_pts = other.n_pts;
            data_pts = other.data_pts;
            kd_tree = other.kd_tree;
            other.data_pts = nullptr;
            other.kd_tree = nullptr;
            other.n_pts = 0;
        }
        return *this;
    }

    /**
     * @brief Destroy the Nearest Neighbor Search object and free ANN memory.
     */
    ~NearestNeighborSearch() {
        cleanup();
    }

    /**
     * @brief Search for the k nearest neighbors.
     * 
     * @param query The query point.
     * @param k The number of nearest neighbors to find.
     * @param eps The error bound (0.0 for exact search).
     * @return std::vector<Result> A vector of results containing indices and distances.
     */
    [[nodiscard]] std::vector<Result> search(const Point& query, int k, double eps = 0.0) const {
        if (k <= 0 || !kd_tree) return {};
        
        // Ensure we don't request more points than available
        int actual_k = std::min(k, n_pts);
        
        std::vector<ANNidx> nn_idx(actual_k);
        std::vector<ANNdist> dists(actual_k);

        // ANN uses non-const pointers internally but does not modify query point
        ANNpoint q = const_cast<double*>(query.data());

        kd_tree->annkSearch(q, actual_k, nn_idx.data(), dists.data(), eps);

        std::vector<Result> results;
        results.reserve(actual_k);
        for (int i = 0; i < actual_k; ++i) {
            if (nn_idx[i] != ANN_NULL_IDX) {
                results.push_back({nn_idx[i], std::sqrt(dists[i])});
            }
        }
        return results;
    }

    /**
     * @brief Returns the number of points in the index.
     */
    [[nodiscard]] int size() const noexcept { return n_pts; }

private:
    void cleanup() {
        if (kd_tree) {
            delete kd_tree;
            kd_tree = nullptr;
        }
        if (data_pts) {
            annDeallocPts(data_pts);
            data_pts = nullptr;
        }
    }

    int n_pts = 0;
    ANNpointArray data_pts = nullptr;
    ANNkd_tree* kd_tree = nullptr;
};

} // namespace ANN

#endif // ANN_NEAREST_NEIGHBOR_SEARCH_H
