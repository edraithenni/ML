#pragma once

#include <vector>
#include <string>
#include <variant>

extern int FEATURES_COUNT;

struct Row {
    std::vector<double> features;
    int label;
};

struct NodeTree {
    int split_feature = -1;
    double split_threshold = 0.0;
    double gini_score = 0.0;
    int leaf_feature = -1;

    size_t left_size = 0;
    size_t right_size = 0;

    NodeTree* left = nullptr;
    NodeTree* right = nullptr;
};

std::vector<Row> load_data_file(const std::string& filename);

class decisionTree {
public:
    decisionTree();

    NodeTree* getRoot();

    NodeTree* fit(
        const std::vector<Row>& data,
        std::vector<int>& indices,
        int maxDepth,
        int curDepth,
        int features_count
    );

    std::variant<int, double> predict(
        NodeTree* current,
        const std::vector<double>& features
    );

    double calculate_gini(
        const std::vector<Row>& data,
        std::vector<int>& indices
    );

    int get_most_frequent_label(
        const std::vector<Row>& data,
        const std::vector<int>& indices
    );

    void visualizeTree(
        NodeTree* root,
        const std::string& filename
    );

private:
    NodeTree* root;
};