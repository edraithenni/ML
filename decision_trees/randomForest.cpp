#include "randomForest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <algorithm>
#include <set>
#include <random>
#include <numeric>
#include <variant>
#include <iomanip>

randomForest::randomForest(int nTrees, int maxDepth, int featuresPerSplit)
    : n_trees(nTrees),
    max_depth(maxDepth),
    features_per_split(featuresPerSplit) {}

void randomForest::fit(const std::vector<Row>& data, const std::vector<int>& train_indices) {
    std::mt19937 gen(666);
    std::uniform_int_distribution<size_t> dist(0, train_indices.size() - 1);

    trees.clear();
    roots.clear();

    for (int i = 0; i < n_trees; ++i) {
        std::vector<int> bootstrap_indices;
        bootstrap_indices.reserve(train_indices.size());

        for (size_t j = 0; j < train_indices.size(); ++j) {
            bootstrap_indices.push_back(
            train_indices[dist(gen)]
            );
        }

        decisionTree tree;

        NodeTree* root = tree.fit(data,bootstrap_indices, max_depth,0,features_per_split);

        trees.push_back(tree);
        roots.push_back(root);
    }
}

int randomForest::predict(const std::vector<double>& features) {
    return predict_proba(features) >= 0.5 ? 1 : 0;
}

// double randomForest::predict_proba(const std::vector<double>& features) {
//     double sum = 0.0;
//
//     for (size_t i = 0; i < trees.size(); ++i) {
//         auto prediction = trees[i].predict(roots[i], features);
//
//         double value = std::visit(
//         [](auto x) {
//             return static_cast<double>(x);
//         },
//         prediction
//         );
//
//         sum += value;
//     }
//
//     return sum / trees.size();
// }

//voting mechanism
double randomForest::predict_proba(const std::vector<double>& features) {
    int votes_for_1 = 0;

    for (size_t i = 0; i < trees.size(); ++i) {

        auto prediction =
            trees[i].predict(roots[i], features);

        double value = std::visit(
            [](auto x) {
                return static_cast<double>(x);
            },
            prediction
        );

        // Each tree gives one vote
        if (value >= 0.5) {
            ++votes_for_1;
        }
    }

    return static_cast<double>(votes_for_1) / trees.size();
}
