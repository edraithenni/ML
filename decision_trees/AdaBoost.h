#include "decisionTree.h"
#include <vector>

class AdaBoost {
public:
    AdaBoost(int nEstimators)
        : n_estimators(nEstimators) {}

    void fit(const std::vector<Row>& data, std::vector<int>& train_indices, int sample_sz);

    int predict(const std::vector<double>& features);

private:
    std::vector<decisionTree> trees;
    std::vector<NodeTree*> roots;
    std::vector<double> alphas;

    int n_estimators;
};
