#include <benchmark/benchmark.h>
#include <ANN/ANN.h>
#include <vector>
#include <random>

static void BM_KdTreeConstruction(benchmark::State& state) {
    int dim = 3;
    int n_pts = state.range(0);
    ANNpointArray data_pts = annAllocPts(n_pts, dim);
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 100);
    for (int i = 0; i < n_pts; i++) {
        for (int d = 0; d < dim; d++) {
            data_pts[i][d] = dis(gen);
        }
    }

    for (auto _ : state) {
        ANNkd_tree* kdTree = new ANNkd_tree(data_pts, n_pts, dim);
        benchmark::DoNotOptimize(kdTree);
        delete kdTree;
    }

    annDeallocPts(data_pts);
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_KdTreeConstruction)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_KdTreeSearch(benchmark::State& state) {
    int dim = 3;
    int n_pts = state.range(0);
    ANNpointArray data_pts = annAllocPts(n_pts, dim);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 100);
    for (int i = 0; i < n_pts; i++) {
        for (int d = 0; d < dim; d++) {
            data_pts[i][d] = dis(gen);
        }
    }

    ANNkd_tree* kdTree = new ANNkd_tree(data_pts, n_pts, dim);
    ANNpoint query_pt = annAllocPt(dim);
    ANNidx nn_idx[1];
    ANNdist dists[1];

    for (auto _ : state) {
        query_pt[0] = dis(gen);
        query_pt[1] = dis(gen);
        query_pt[2] = dis(gen);
        kdTree->annkSearch(query_pt, 1, nn_idx, dists);
        benchmark::DoNotOptimize(nn_idx);
        benchmark::DoNotOptimize(dists);
    }

    delete kdTree;
    annDeallocPt(query_pt);
    annDeallocPts(data_pts);
}
BENCHMARK(BM_KdTreeSearch)->RangeMultiplier(10)->Range(100, 100000);

static void BM_BdTreeSearch(benchmark::State& state) {
    int dim = 3;
    int n_pts = state.range(0);
    ANNpointArray data_pts = annAllocPts(n_pts, dim);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 100);
    for (int i = 0; i < n_pts; i++) {
        for (int d = 0; d < dim; d++) {
            data_pts[i][d] = dis(gen);
        }
    }

    ANNbd_tree* bdTree = new ANNbd_tree(data_pts, n_pts, dim);
    ANNpoint query_pt = annAllocPt(dim);
    ANNidx nn_idx[1];
    ANNdist dists[1];

    for (auto _ : state) {
        query_pt[0] = dis(gen);
        query_pt[1] = dis(gen);
        query_pt[2] = dis(gen);
        bdTree->annkSearch(query_pt, 1, nn_idx, dists);
        benchmark::DoNotOptimize(nn_idx);
        benchmark::DoNotOptimize(dists);
    }

    delete bdTree;
    annDeallocPt(query_pt);
    annDeallocPts(data_pts);
}
BENCHMARK(BM_BdTreeSearch)->RangeMultiplier(10)->Range(100, 100000);

BENCHMARK_MAIN();
