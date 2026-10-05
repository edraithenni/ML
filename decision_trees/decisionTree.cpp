#include "decisionTree.h"
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

int FEATURES_COUNT = 22;

//===========================data preparation utils=======================
std::vector<Row> load_data_file(const std::string& filename) {
    std::vector<Row> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        Row row;

        if (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                row.label = (token[0] == 'p') ? 1 : 0;
             //  row.label = static_cast<int>(token[0]);
            }
        }

        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                row.features.push_back(static_cast<double>(token[0]));
            }
        }

        if (!row.features.empty()) {
            data.push_back(row);
        }
    }
    return data;
}
//==========================================================================

decisionTree::decisionTree() : root(nullptr) {}

NodeTree* decisionTree::getRoot() {
    return root;
}

NodeTree* decisionTree::fit(const std::vector<Row>& data, std::vector<int>& indices,  int maxDepth, int curDepth, int features_count) {

    NodeTree* current = new NodeTree();

    current->gini_score = calculate_gini(data, indices);

    if (current->gini_score == 0.0 || curDepth >= maxDepth || indices.size() < 2) {
        current->leaf_feature = get_most_frequent_label(data, indices);
        return current;
    }

    double best_gini = 1.0;
    int best_feature_idx = -1;
    double best_threshold = 0.0;

    std::vector<int> features(FEATURES_COUNT);
    std::iota(features.begin(), features.end(), 0);

    static std::mt19937 gen(666);
    std::shuffle(features.begin(), features.end(), gen);

    features.resize(std::min(features_count, FEATURES_COUNT));

    for (int i : features) {
        std::set<double> unique_thresholds;
        for (int idx : indices) {
            if (data[idx].features[i] != 63) {
                unique_thresholds.insert(data[idx].features[i]);
            }
        }
        for (double thresh : unique_thresholds) {
            std::vector<int> left_indices, right_indices;

            std::copy_if(indices.begin(), indices.end(), std::back_inserter(left_indices),
                         [&](int idx) { return data[idx].features[i] <= thresh; });

            std::copy_if(indices.begin(), indices.end(), std::back_inserter(right_indices),
                         [&](int idx) { return data[idx].features[i] > thresh; });

            if (left_indices.empty() || right_indices.empty()) continue;

            double left_gini = calculate_gini(data, left_indices);
            double right_gini = calculate_gini(data, right_indices);

            size_t left_size = left_indices.size();
            size_t right_size = right_indices.size();

            double split_gini = (left_size / static_cast<double>(indices.size())) * left_gini + (right_size / static_cast<double>(indices.size())) * right_gini;

            if (split_gini < best_gini) {
                best_gini = split_gini;
                best_feature_idx = i;
                best_threshold = thresh;
            }
        }
    }

    //критерий остановки разбиения вершины на поддеревья
    if (best_feature_idx == -1 || best_gini >= current->gini_score) {
        current->leaf_feature = get_most_frequent_label(data, indices);
        return current;
    }

    current->split_feature = best_feature_idx;
    current->split_threshold = best_threshold;

    std::vector<int> left_indices;
    std::vector<int> right_indices;
    std::vector<int> missing_indices;

    for (int idx : indices) {
        double value = data[idx].features[best_feature_idx];

        if (value == 63) {
            missing_indices.push_back(idx);
        } else if (value <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Send indices with skipped features to all branches
    left_indices.insert(
        left_indices.end(),
        missing_indices.begin(),
        missing_indices.end()
    );

    right_indices.insert(
        right_indices.end(),
        missing_indices.begin(),
        missing_indices.end()
    );

    current->left_size = left_indices.size();
    current->right_size = right_indices.size();

    current->left = fit(data, left_indices, maxDepth, curDepth + 1, features_count);
    current->right = fit(data, right_indices, maxDepth, curDepth + 1, features_count);

    return current;

}


// int    -> no missing values were encountered
// double -> at least one missing value was encountered
std::variant<int, double> decisionTree::predict(NodeTree* current, const std::vector<double>& features) {
    if (!current) return -1.0;

    if (current->left == nullptr && current->right == nullptr) {
        return static_cast<int>(current->leaf_feature);
    }

    double value = features[current->split_feature];

    // feature is skipped
    if (value == 63) {

        std::variant<int, double> left_prediction = predict(current->left, features);

        std::variant<int, double> right_prediction = predict(current->right, features);

        double left_value = std::visit(
        [](auto value) {
                return static_cast<double>(value);
            },
            left_prediction
        );

        double right_value = std::visit(
            [](auto value) {
                return static_cast<double>(value);
            },
            right_prediction
        );

        double total_size = static_cast<double>(current->left_size + current->right_size);

        double left_weight = current->left_size / total_size;

        double right_weight = current->right_size / total_size;

        return left_weight * left_value + right_weight * right_value;
    }

    // feature is OK
    if (value <= current->split_threshold) {
        return predict(current->left, features);
    } else {
        return predict(current->right, features);
    }
}

//критерий разбиения
double decisionTree::calculate_gini(const std::vector<Row>& data, std::vector<int>& indices) {
    std::map<int, int> label_counts;
    for (const auto& idx : indices) {
        label_counts[data[idx].label]++;
    }

    double sum_squares = 0.0;

    for (const auto& pair : label_counts) {
        double p = pair.second / static_cast<double>(indices.size()); // p = probability
        sum_squares += p * p;
    }

    return 1.0 - sum_squares;
}

int decisionTree::get_most_frequent_label(const std::vector<Row>& data, const std::vector<int>& indices) {
    std::map<int, int> label_counts;
    for (const auto& idx : indices) {
        label_counts[data[idx].label]++;
    }

    int most_frequent_label = -1;
    int max_count = -1;

    for (const auto& [label, count] : label_counts) {
        if (count > max_count) {
            max_count = count;
            most_frequent_label = label;
        }
    }

    return most_frequent_label;
}

void decisionTree::visualizeTree(NodeTree* root, const std::string& filename) {
    std::ofstream out(filename);

    auto dump = [&](auto&& self, NodeTree* node, const std::string& path, int depth) -> void {
        if (!node) return;

        out << std::string(depth * 4, ' ')
            << path << ": "
            << "split_feature=" << node->split_feature
            << ", split_threshold=" << node->split_threshold
            << ", gini=" << std::fixed << std::setprecision(4) << node->gini_score
            << ", leaf_feature=" << node->leaf_feature
            << ", left_size=" << node->left_size
            << ", right_size=" << node->right_size
            << '\n';

        self(self, node->left,  path + "L", depth + 1);
        self(self, node->right, path + "R", depth + 1);
    };

    dump(dump, root, "root", 0);
}

