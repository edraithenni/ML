#include "AdaBoost.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]) {

    int n_estimators = 50;

    if (argc > 1) {
        n_estimators = std::stoi(argv[1]);
    }
    std::vector<Row> dataset =
        load_data_file("mushroom/agaricus-lepiota.data");

    if (dataset.empty()) {
        std::cerr << "Dataset is empty\n";
        return 1;
    }

    // the same train/test split
    std::vector<int> train_indices;
    std::vector<int> test_indices;

    std::ifstream split_file("mushroom_split.csv");

    if (!split_file.is_open()) {
        std::cerr << "Cannot open mushroom_split.csv\n";
        return 1;
    }

    std::string line;
    std::getline(split_file, line);

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

    // Train AdaBoost
    AdaBoost model(n_estimators);

    model.fit(
        dataset,
        train_indices,
        train_indices.size()
    );

    // Test
    size_t tp = 0, tn = 0, fp = 0, fn = 0;

    for (int idx : test_indices) {

        int predicted =
            model.predict(dataset[idx].features);

        int actual =
            dataset[idx].label;

        if (actual == 1 && predicted == 1) ++tp;
        else if (actual == 0 && predicted == 0) ++tn;
        else if (actual == 0 && predicted == 1) ++fp;
        else ++fn;
    }

    // Metrics
    double accuracy =
        static_cast<double>(tp + tn) / test_indices.size();

    double error_rate = 1.0 - accuracy;

    double precision =
        (tp + fp > 0)
        ? static_cast<double>(tp) / (tp + fp)
        : 0.0;

    double recall =
        (tp + fn > 0)
        ? static_cast<double>(tp) / (tp + fn)
        : 0.0;

    double specificity =
        (tn + fp > 0)
        ? static_cast<double>(tn) / (tn + fp)
        : 0.0;

    double f1 =
        (precision + recall > 0)
        ? 2.0 * precision * recall / (precision + recall)
        : 0.0;

    double balanced_accuracy =
        (recall + specificity) / 2.0;

    // Results
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Estimators:          " << n_estimators << '\n';

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