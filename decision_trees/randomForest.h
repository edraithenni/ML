#pragma once

#include "decisionTree.h"
#include <vector>

class randomForest {
public:
    randomForest(
        int nTrees,
        int maxDepth,
        int featuresPerSplit
    );

    void fit(
        const std::vector<Row>& data,
        const std::vector<int>& train_indices
    );

    int predict(
        const std::vector<double>& features
    );

    double predict_proba(
        const std::vector<double>& features
    );

private:
    std::vector<decisionTree> trees;
    std::vector<NodeTree*> roots;

    int n_trees;
    int max_depth;
    int features_per_split;
};