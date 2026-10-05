#include "randomForest.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[]) {

    // Number of trees from command line
    int n_trees = 50;

    if (argc > 1) {
        n_trees = std::stoi(argv[1]);
    }

    std::vector<Row> dataset =
        load_data_file("mushroom/agaricus-lepiota.data");

    if (dataset.empty()) {
        std::cerr << "Dataset is empty\n";
        return 1;
    }

    // Same train/test split as for the single Decision Tree
    std::vector<int> train_indices;
    std::vector<int> test_indices;

    std::ifstream split_file("mushroom_split.csv");

    std::string line;
    std::getline(split_file, line); // skip header

    while (std::getline(split_file, line)) {
        std::stringstream ss(line);

        std::string index_str;
        std::string split;
        std::string label;

        std::getline(ss, index_str, ',');
        std::getline(ss, split, ',');
        std::getline(ss, label, ',');

        int idx = std::stoi(index_str);

        if (split == "train") {
            train_indices.push_back(idx);
        } else if (split == "test") {
            test_indices.push_back(idx);
        }
    }

    // sqrt(22) ~= 5 random features per split
    randomForest forest(n_trees, 5, 5);
    forest.fit(dataset, train_indices);

    size_t tp = 0, tn = 0, fp = 0, fn = 0;

    // Save probabilities for visualization in Python
    std::ofstream file("random_forest_predictions.csv");
    file << "true_label,predicted_label,probability\n";

    for (int idx : test_indices) {

        double probability =
            forest.predict_proba(dataset[idx].features);

        int predicted =
            probability >= 0.5 ? 1 : 0;

        int actual = dataset[idx].label;

        file << actual << ","<< predicted << ","<< probability << "\n";

        if (actual == 1 && predicted == 1) ++tp;
        else if (actual == 0 && predicted == 0) ++tn;
        else if (actual == 0 && predicted == 1) ++fp;
        else ++fn;
    }

    file.close();

    double accuracy = static_cast<double>(tp + tn) / test_indices.size();

    double error_rate = 1.0 - accuracy;

    double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;

    double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;

    double specificity = (tn + fp > 0) ? static_cast<double>(tn) / (tn + fp) : 0.0;

    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    double balanced_accuracy = (recall + specificity) / 2.0;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Trees:              " << n_trees << '\n';
    std::cout << "Test size:          " << test_indices.size() << '\n';

    std::cout << "\nConfusion matrix:\n";
    std::cout << "TN: " << tn << "  FP: " << fp << '\n';
    std::cout << "FN: " << fn << "  TP: " << tp << '\n';

    std::cout << "\nMetrics:\n";
    std::cout << "Accuracy:           " << accuracy << '\n';
    std::cout << "Error rate:         " << error_rate << '\n';
    std::cout << "Precision:          " << precision << '\n';
    std::cout << "Recall:             " << recall << '\n';
    std::cout << "F1:                 " << f1 << '\n';
    std::cout << "Specificity:        " << specificity << '\n';
    std::cout << "Balanced accuracy:  " << balanced_accuracy << '\n';

    return 0;
}