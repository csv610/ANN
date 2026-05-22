#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>

#ifdef ASSIMP_FOUND
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#endif

#include <ANN/NearestNeighborSearch.h>
using namespace ANN;

#ifdef ASSIMP_FOUND

template <size_t Dim>
std::vector<std::array<double, Dim>> loadMesh(const std::string& filename) {
    std::vector<std::array<double, Dim>> points;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename,
        aiProcess_PreTransformVertices | aiProcess_Triangulate | aiProcess_SortByPType);

    if (!scene) {
        throw std::runtime_error("Failed to load mesh: " + std::string(importer.GetErrorString()));
    }

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* mesh = scene->mMeshes[m];
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            aiVector3D v = mesh->mVertices[i];
            if constexpr (Dim == 3) {
                points.push_back({static_cast<double>(v.x),
                               static_cast<double>(v.y),
                               static_cast<double>(v.z)});
            } else if constexpr (Dim == 2) {
                points.push_back({static_cast<double>(v.x),
                               static_cast<double>(v.y)});
            }
        }
    }

    return points;
}

#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file> [k=100] [eps=1e-9]\n";
        std::cerr << "Supported formats: obj, fbx, 3ds, ply, stl, etc.\n";
        return 1;
    }

    std::string filename = argv[1];
    int k = 200;
    double eps = 1e-9;

    if (argc >= 3) {
        k = std::stoi(argv[2]) + 1;
    }
    if (argc >= 4) {
        eps = std::stod(argv[3]);
    }

    constexpr size_t Dim = 3;

#ifndef ASSIMP_FOUND
    std::cerr << "Error: Assimp not found\n";
    return 1;
#else
    try {
        auto points = loadMesh<Dim>(filename);
        std::cout << "Loaded " << points.size() << " vertices from " << filename << "\n";

        if (points.size() < 2) {
            std::cerr << "Need at least 2 points\n";
            return 1;
        }

        k = std::min(k, static_cast<int>(points.size()));

        ANN::NearestNeighborSearch<Dim> nns(points);

        std::vector<double> minDistances;
        minDistances.reserve(points.size());

        std::set<double> uniqueDists;

        for (size_t idx = 0; idx < points.size(); ++idx) {
            const auto& query = points[idx];
            auto results = nns.search(query, k);

            double minDist = 0;
            for (const auto& r : results) {
                if (r.distance > eps) {
                    minDist = r.distance;
                    uniqueDists.insert(r.distance);
                    break;
                }
            }
            minDistances.push_back(minDist);
        }

        double min = *std::min_element(minDistances.begin(), minDistances.end());
        double max = *std::max_element(minDistances.begin(), minDistances.end());
        double mean = std::accumulate(minDistances.begin(), minDistances.end(), 0.0) / minDistances.size();

        double sqSum = 0.0;
        for (double d : minDistances) {
            sqSum += (d - mean) * (d - mean);
        }
        double stddev = std::sqrt(sqSum / minDistances.size());

        std::cout << "\n=== Minimum Distance Statistics (using ANN) ===\n";
        std::cout << "Min:    " << min << "\n";
        std::cout << "Max:    " << max << "\n";
        std::cout << "Mean:   " << mean << "\n";
        std::cout << "StdDev: " << stddev << "\n";

        annClose();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
#endif
}