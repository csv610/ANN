#include <gtest/gtest.h>
#include <ANN/ANN.h>

class ANNTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 2;
        max_pts = 10;
        data_pts = annAllocPts(max_pts, dim);
        query_pt = annAllocPt(dim);
        nn_idx = new ANNidx[1];
        dists = new ANNdist[1];

        // Fill with simple data points: (0,0), (1,1), ..., (9,9)
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
        annClose(); // Clean up internal ANN memory
    }

    int dim;
    int max_pts;
    ANNpointArray data_pts;
    ANNpoint query_pt;
    ANNidxArray nn_idx;
    ANNdistArray dists;
};

TEST_F(ANNTest, KdTreeBasicSearch) {
    ANNkd_tree* kdTree = new ANNkd_tree(data_pts, max_pts, dim);

    // Query point near (5,5)
    query_pt[0] = 5.1;
    query_pt[1] = 5.1;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 5);
    EXPECT_NEAR(dists[0], 0.02, 1e-6); // (5.1-5)^2 + (5.1-5)^2 = 0.01 + 0.01 = 0.02

    delete kdTree;
}

TEST_F(ANNTest, BdTreeBasicSearch) {
    ANNbd_tree* bdTree = new ANNbd_tree(data_pts, max_pts, dim);

    // Query point near (2,2)
    query_pt[0] = 1.9;
    query_pt[1] = 1.9;

    bdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 2);
    EXPECT_NEAR(dists[0], 0.02, 1e-6);

    delete bdTree;
}

TEST_F(ANNTest, ExactMatch) {
    ANNkd_tree* kdTree = new ANNkd_tree(data_pts, max_pts, dim);

    // Query point exactly at (7,7)
    query_pt[0] = 7.0;
    query_pt[1] = 7.0;

    kdTree->annkSearch(query_pt, 1, nn_idx, dists);

    EXPECT_EQ(nn_idx[0], 7);
    EXPECT_EQ(dists[0], 0.0);

    delete kdTree;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
