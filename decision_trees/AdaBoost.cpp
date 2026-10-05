#include "decisionTree.h"
#include "AdaBoost.h"
#include <vector>
#include <queue>
#include <utility>
#include <random>
#include <cmath>

void AdaBoost::fit( const std::vector<Row>& data, std::vector<int>& train_indices, int sample_sz)
{
    std::vector<double> weights( train_indices.size(),1.0 / train_indices.size());

    std::mt19937 gen(666);

    trees.clear();
    roots.clear();
    alphas.clear();

    for (int i = 0; i < n_estimators; ++i) {
        // Weighted sample
        std::discrete_distribution<int> dist(weights.begin(),weights.end());

        std::vector<int> sample;
        sample.reserve(sample_sz);

        for (int j = 0; j < sample_sz; ++j) {
            sample.push_back(train_indices[dist(gen)]);
        }
        decisionTree tree;

        NodeTree* root = tree.fit(data,sample,1,0, FEATURES_COUNT);

        std::vector<int> predictions(train_indices.size());

        double error = 0.0;
        for (size_t j = 0; j < train_indices.size(); ++j) {

            auto result = tree.predict(root,data[train_indices[j]].features);

            double value = std::visit(
                [](auto x) {
                    return static_cast<double>(x);
                },
                result
            );

            predictions[j] = value >= 0.5 ? 1 : 0;

            if (predictions[j] != data[train_indices[j]].label) {
                error += weights[j];
            }
        }
        if (error >= 0.5) {
            continue;
        }
        error = std::max(error, 1e-10);

        double alpha = 0.5 * std::log((1.0 - error) / error);

        double weight_sum = 0.0;

        for (size_t j = 0; j < train_indices.size(); ++j) {

            if (predictions[j] == data[train_indices[j]].label)
            {
                weights[j] *= std::exp(-alpha);
            }
            else {
                weights[j] *= std::exp(alpha);
            }

            weight_sum += weights[j];
        }

        // normalize weights
        for (double& w : weights) {
            w /= weight_sum;
        }

        trees.push_back(tree);
        roots.push_back(root);
        alphas.push_back(alpha);
    }
}

int AdaBoost::predict(const std::vector<double>& features) {
    double score = 0.0;

    for (size_t i = 0; i < trees.size(); ++i) {

        auto result = trees[i].predict(roots[i], features);
        double value = std::visit(
            [](auto x) {
                return static_cast<double>(x);
            },
            result
        );

        int prediction = value >= 0.5 ? 1 : 0;
        if (prediction == 1) {
            score += alphas[i];
        } else {
            score -= alphas[i];
        }
    }

    return score >= 0.0 ? 1 : 0;
}