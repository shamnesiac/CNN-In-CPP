#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <random>
#include <string>
#include <fstream>

class MLP
{
private:
    // Network architecture
    int input_size;
    std::vector<int> hidden_sizes;
    int output_size;
    float learning_rate;
    std::mt19937 rng;

    // Network weights and biases
    std::vector<std::vector<float>> fc1_weights; // 784 x 400
    std::vector<float> fc1_bias;                 // 400
    std::vector<std::vector<float>> fc2_weights; // 400 x 120
    std::vector<float> fc2_bias;                 // 120
    std::vector<std::vector<float>> fc3_weights; // 120 x 84
    std::vector<float> fc3_bias;                 // 84
    std::vector<std::vector<float>> fc4_weights; // 84 x 10
    std::vector<float> fc4_bias;                 // 10

    // Best model weights and biases
    std::vector<std::vector<float>> best_fc1_weights;
    std::vector<float> best_fc1_bias;
    std::vector<std::vector<float>> best_fc2_weights;
    std::vector<float> best_fc2_bias;
    std::vector<std::vector<float>> best_fc3_weights;
    std::vector<float> best_fc3_bias;
    std::vector<std::vector<float>> best_fc4_weights;
    std::vector<float> best_fc4_bias;

    // Layer activations (for backpropagation)
    std::vector<float> fc1_output;   // 400
    std::vector<float> fc2_output;   // 120
    std::vector<float> fc3_output;   // 84
    std::vector<float> final_output; // 10

    // Helper functions
    void initialize_weights();
    void save_best_model();
    void restore_best_model();
    float relu(float x);
    float relu_derivative(float x);
    std::vector<float> softmax(const std::vector<float> &x);
    float cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target);

public:
    // Constructor for the network architecture
    MLP(int input_size, const std::vector<int> &hidden_sizes, int output_size, float learning_rate = 0.001);

    // Training and evaluation functions
    void train(const std::vector<std::vector<float>> &training_data,
               const std::vector<std::vector<float>> &training_labels,
               const std::vector<std::vector<float>> &test_data,
               const std::vector<std::vector<float>> &test_labels,
               int epochs,
               int batch_size = 32);
    std::vector<float> forward(const std::vector<float> &input);
    void backward(const std::vector<float> &input, const std::vector<float> &target);
    float evaluate(const std::vector<std::vector<float>> &test_data,
                   const std::vector<std::vector<float>> &test_labels,
                   bool print_predictions = false);

    // Load and save weights
    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename) const;

    // Static function to load MNIST data
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    load_mnist(const std::string &images_file, const std::string &labels_file, int num_images);
};

#endif // MLP_HPP