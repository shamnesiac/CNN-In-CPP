#ifndef EVALUATE_MLP_HPP
#define EVALUATE_MLP_HPP

#include "mlp.hpp"
#include <vector>
#include <string>

class MLPEvaluator
{
private:
    MLP &mlp;
    int num_threads;
    std::vector<std::vector<int>> confusion_matrix;

    void compute_confusion_matrix(const std::vector<std::vector<float>> &test_images,
                                  const std::vector<int> &test_labels);
    void print_confusion_matrix() const;
    float calculate_accuracy() const;
    std::vector<float> calculate_per_class_accuracy() const;

public:
    explicit MLPEvaluator(MLP &mlp_model);

    void evaluate_model(const std::string &test_images_path,
                        const std::string &test_labels_path,
                        int test_size);

    float get_accuracy() const { return calculate_accuracy(); }
    std::vector<float> get_class_accuracies() const { return calculate_per_class_accuracy(); }
};

#endif // EVALUATE_MLP_HPP