#include "decisionTree.h"

#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <fstream>
int main() {
    std::vector<Row> dataset = load_data_file("mushroom/agaricus-lepiota.data");

    if (dataset.empty()) {
        std::cerr << "Dataset is empty\n";
        return 1;
    }

    // Train / test split: 80% / 20%
    std::vector<int> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Fixed seed -> reproducible experiment
    std::mt19937 gen(666);
    std::shuffle(indices.begin(), indices.end(), gen);

    const double test_ratio = 0.2;

    size_t test_size = static_cast<size_t>(dataset.size() * test_ratio);

    size_t train_size = dataset.size() - test_size;

    std::vector<int> train_indices( indices.begin(), indices.begin() + train_size);
    std::vector<int> test_indices(indices.begin() + train_size, indices.end());

    std::ofstream split_file("mushroom_split.csv");
    split_file << "index,split,label\n";

    for (int idx : train_indices) {
        split_file << idx << ",train," << dataset[idx].label << '\n';
    }

    for (int idx : test_indices) {
        split_file << idx << ",test," << dataset[idx].label << '\n';
    }

    split_file.close();

    std::cout << "Dataset size: " << dataset.size() << '\n';
    std::cout << "Train size:   " << train_indices.size() << '\n';
    std::cout << "Test size:    " << test_indices.size() << '\n';

    // Training
    decisionTree tree;

    NodeTree* root = tree.fit(dataset, train_indices, 5, 0, FEATURES_COUNT);

    tree.visualizeTree(root, "decision_tree.txt");

    // Testing
    size_t correct = 0;
    size_t errors = 0;

    // Confusion matrix:
    //             predicted
    //             0       1
    // actual 0   TN      FP
    // actual 1   FN      TP

    size_t tp = 0;
    size_t tn = 0;
    size_t fp = 0;
    size_t fn = 0;

    // Save prediction results for Python
    std::ofstream prediction_file("mushroom_test_predictions.csv");

    prediction_file << "index,true_label,predicted_label,probability\n";

    for (int idx : test_indices) {

        auto prediction = tree.predict(root, dataset[idx].features);

        // Convert variant to numerical score
        double probability;

        if (std::holds_alternative<int>(prediction)) {
            probability = static_cast<double>(std::get<int>(prediction));
        } else {
            probability = std::get<double>(prediction);
        }

        // Binary classification
        int predicted_label = probability >= 0.5 ? 1 : 0;

        int true_label = dataset[idx].label;

        // Save prediction
        prediction_file<< idx << ','<< true_label << ','<< predicted_label << ','<< probability << '\n';

        // Accuracy
        if (predicted_label == true_label) {
            ++correct;
        } else {
            ++errors;
        }

        // Confusion matrix
        if (true_label == 1 && predicted_label == 1) {
            ++tp;
        }
        else if (true_label == 0 && predicted_label == 0) {
            ++tn;
        }
        else if (true_label == 0 && predicted_label == 1) {
            ++fp;
        }
        else if (true_label == 1 && predicted_label == 0) {
            ++fn;
        }
    }

    prediction_file.close();

    // Metrics

    double accuracy = static_cast<double>(correct) / test_indices.size();

    double error_rate = static_cast<double>(errors) / test_indices.size();

    double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;

    double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;

    double specificity = (tn + fp > 0) ? static_cast<double>(tn) / (tn + fp) : 0.0;

    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    double balanced_accuracy = (recall + specificity) / 2.0;

    // Results

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\nresults on test subset (train / test split: 80% / 20%), random sample\n";

    std::cout << "Test size:            "<< test_indices.size() << '\n';

    std::cout << "Correct:              "<< correct << '\n';

    std::cout << "Errors:               "<< errors << '\n';

    std::cout << "\nConfusion matrix:\n";

    std::cout << "TN: " << tn<< "  FP: " << fp << '\n';

    std::cout << "FN: " << fn<< "  TP: " << tp << '\n';

    std::cout << "\nMetrics:\n";

    std::cout << "Accuracy:             "<< accuracy << '\n';

    std::cout << "Error rate:           "<< error_rate << '\n';

    std::cout << "Precision:            "<< precision << '\n';

    std::cout << "Recall:               "<< recall << '\n';

    std::cout << "F1:                   "<< f1 << '\n';

    std::cout << "Specificity:          "<< specificity << '\n';

    std::cout << "Balanced accuracy:    "<< balanced_accuracy << '\n';

    std::cout << "\nSaved:\n";
    std::cout << "mushroom_test_predictions.csv\n";
    std::cout << "mushroom_split.csv\n";

    return 0;
}