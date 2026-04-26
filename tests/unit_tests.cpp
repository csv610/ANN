#include <gtest/gtest.h>
#include <ANN/ANN.h>
#include <ANN/NearestNeighborSearch.h>
#include <memory>
#include <vector>
#include <array>

class ANNTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 2;
        max_pts = 10;
        data_pts = annAllocPts(max_pts, dim);
        query_pt = annAllocPt(dim);
        nn_idx = new ANNidx[1];
        dists = new ANNdist[1];

        for (int i = 0; i < max_pts; i++) {
            data_pts[i][0] = (ANNcoord)i;
            data_pts[i][1] = (ANNcoord)i;
        }
    }

    void TearDown() override {
        annDeallocPts(data_pts);
        annDeallocPt(query_pt);
        delete[] nn_idx;
        delete[] dists;
        annClose();
    }

    int dim;
    int max_pts;
    ANNpointArray data_pts;
    ANNpoint query_pt;
    ANNidxArray nn_idx;
    ANNdistArray dists;
};

TEST_F(ANNTest, HighLevelWrapperBasicSearch) {
    std::vector<std::array<double, 2>> points;
    for (int i = 0; i < max_pts; ++i) {
        points.push_back({static_cast<double>(i), static_cast<double>(i)});
    }

    ANN::NearestNeighborSearch<2> nns(points);
    EXPECT_EQ(nns.size(), max_pts);

    std::array<double, 2> query = {5.1, 5.1};
    auto results = nns.search(query, 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, 5);
    EXPECT_NEAR(results[0].distance, std::sqrt(0.02), 1e-6);
}

TEST_F(ANNTest, HighLevelWrapperKSearch) {
    std::vector<std::array<double, 2>> points;
    for (int i = 0; i < max_pts; ++i) {
        points.push_back({static_cast<double>(i), static_cast<double>(i)});
    }

    ANN::NearestNeighborSearch<2> nns(points);
    
    std::array<double, 2> query = {5.5, 5.5};
    auto results = nns.search(query, 5);

    ASSERT_EQ(results.size(), 5);
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].distance, results[i-1].distance);
    }
}

TEST_F(ANNTest, HighLevelWrapperMoveSemantics) {
    std::vector<std::array<double, 2>> points = {{0.0, 0.0}, {1.0, 1.0}};
    ANN::NearestNeighborSearch<2> nns1(points);
    
    ANN::NearestNeighborSearch<2> nns2 = std::move(nns1);
    
    EXPECT_EQ(nns1.size(), 0);
    EXPECT_EQ(nns2.size(), 2);
    
    auto results = nns2.search({0.1, 0.1}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, 0);
}

TEST_F(ANNTest, HighLevelWrapperEmptyThrows) {
    std::vector<std::array<double, 2>> points;
    EXPECT_THROW(ANN::NearestNeighborSearch<2> nns(points), std::invalid_argument);
}

TEST_F(ANNTest, HighLevelWrapperExactMatch) {
    std::vector<std::array<double, 2>> points = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    ANN::NearestNeighborSearch<2> nns(points);

    // Exact match exists
    auto match = nns.findExactMatch({2.0, 2.0});
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(*match, 1);

    // Exact match does not exist
    auto no_match = nns.findExactMatch({2.1, 2.1});
    EXPECT_FALSE(no_match.has_value());

    // Match within tolerance
    auto tolerance_match = nns.findExactMatch({2.1, 2.1}, 0.2);
    ASSERT_TRUE(tolerance_match.has_value());
    EXPECT_EQ(*tolerance_match, 1);
}

TEST_F(ANNTest, KdTreeBasicSearch) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 5.1;
    query_pt[1] = 5.1;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 5);
    EXPECT_NEAR(dists[0], 0.02, 1e-6);
}

TEST_F(ANNTest, BdTreeBasicSearch) {
    std::unique_ptr<ANNbd_tree> bdTree(new ANNbd_tree(data_pts, max_pts, dim));

    query_pt[0] = 1.9;
    query_pt[1] = 1.9;

    bdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 2);
    EXPECT_NEAR(dists[0], 0.02, 1e-6);
}

TEST_F(ANNTest, ExactMatch) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 7.0;
    query_pt[1] = 7.0;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 7);
    EXPECT_EQ(dists[0], 0.0);
}

TEST_F(ANNTest, KdTreeQueryOutOfBounds) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 100.0;
    query_pt[1] = 100.0;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 9);
    EXPECT_GE(dists[0], 0.0);
}

TEST_F(ANNTest, KdTreeQueryAtOrigin) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 0.0;
    query_pt[1] = 0.0;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 0);
    EXPECT_EQ(dists[0], 0.0);
}

TEST_F(ANNTest, KdTreeKNearestNeighbors) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 5.5;
    query_pt[1] = 5.5;

    delete[] nn_idx;
    delete[] dists;
    nn_idx = new ANNidx[max_pts];
    dists = new ANNdist[max_pts];

    kdTree->annkSearch(query_pt, max_pts, nn_idx, dists);

    EXPECT_GE(dists[0], 0.0);
    for (int i = 1; i < max_pts; i++) {
        EXPECT_GE(dists[i], dists[i-1]);
    }

    delete[] nn_idx;
    delete[] dists;
    nn_idx = nullptr;
    dists = nullptr;
}

TEST_F(ANNTest, KdTreePrioritySearch) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 5.1;
    query_pt[1] = 5.1;

    nn_idx = new ANNidx[1];
    dists = new ANNdist[1];

    kdTree->annkPriSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 5);
    EXPECT_NEAR(dists[0], 0.02, 1e-6);
}

TEST_F(ANNTest, KdTreeEpsilonSearch) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 5.1;
    query_pt[1] = 5.1;

    double eps = 0.5;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists, eps);

    EXPECT_GE(nn_idx[0], 0);
    EXPECT_LT(nn_idx[0], max_pts);
}

TEST_F(ANNTest, KdTreeEmptyDataset) {
    ANNpointArray empty_pts = annAllocPts(0, dim);
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(empty_pts, 0, dim));

    EXPECT_EQ(kdTree->nPoints(), 0);

    annDeallocPts(empty_pts);
}

TEST_F(ANNTest, KdTreeSinglePoint) {
    ANNpointArray single_pt = annAllocPts(1, dim);
    single_pt[0][0] = 5.0;
    single_pt[0][1] = 5.0;

    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(single_pt, 1, dim));

    query_pt[0] = 5.0;
    query_pt[1] = 5.0;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 0);
    EXPECT_EQ(dists[0], 0.0);

    annDeallocPts(single_pt);
}

TEST_F(ANNTest, KdTreeDimensionAccessor) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    EXPECT_EQ(kdTree->theDim(), dim);
    EXPECT_EQ(kdTree->nPoints(), max_pts);
}

TEST_F(ANNTest, BruteForceSearch) {
    std::unique_ptr<ANNbruteForce> brute(new ANNbruteForce(data_pts, max_pts, dim));

    query_pt[0] = 5.1;
    query_pt[1] = 5.1;

    brute->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 5);
    EXPECT_NEAR(dists[0], 0.02, 1e-6);
}

TEST_F(ANNTest, FixedRadiusSearch) {
    std::unique_ptr<ANNkd_tree> kdTree(new ANNkd_tree(data_pts, max_pts, dim));

    query_pt[0] = 5.0;
    query_pt[1] = 5.0;

    ANNdist sqRad = 2.0;

    nn_idx = new ANNidx[max_pts];
    dists = new ANNdist[max_pts];

    int cnt = kdTree->annkFRSearch(query_pt, sqRad, max_pts, nn_idx, dists);

    EXPECT_GE(cnt, 0);
    for (int i = 0; i < cnt; i++) {
        EXPECT_LE(dists[i], sqRad);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}