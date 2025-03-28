#include "../include/mlp.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 2)
        {
            std::cerr << "Usage: " << argv[0] << " <test_size>" << std::endl;
            return 1;
        }

        int test_size = std::stoi(argv[1]);
        if (test_size <= 0)
        {
            std::cerr << "Error: test_size must be positive" << std::endl;
            return 1;
        }

        // File paths
        std::string test_images_file = "../data/t10k-images.idx3-ubyte";
        std::string test_labels_file = "../data/t10k-labels.idx1-ubyte";
        std::string weights_file = "../mlp_weights.bin";

        // Check if MNIST test files exist
        if (!std::filesystem::exists(test_images_file) || !std::filesystem::exists(test_labels_file))
        {
            std::cerr << "Error: MNIST test files not found" << std::endl;
            return 1;
        }

        // Initialize MLP
        std::cout << "Initializing MLP..." << std::endl;
        std::vector<int> hidden_sizes = {400, 120, 84};
        MLP mlp(784, hidden_sizes, 10);

        // Try to load weights if they exist
        if (std::filesystem::exists(weights_file))
        {
            std::cout << "Loading weights from " << weights_file << "..." << std::endl;
            try
            {
                mlp.load_weights(weights_file);
                std::cout << "Successfully loaded weights." << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cout << "Warning: Failed to load weights (" << e.what() << "). Using random initialization." << std::endl;
            }
        }
        else
        {
            std::cout << "Warning: Weights file not found. Using random initialization." << std::endl;
        }

        // Load test data
        std::cout << "Loading test data..." << std::endl;
        auto [test_images, test_labels] = MLP::load_mnist(test_images_file, test_labels_file, test_size);
        std::cout << "Loaded " << test_images.size() << " test images." << std::endl;

        // Initialize confusion matrix
        std::vector<std::vector<int>> confusion_matrix(10, std::vector<int>(10, 0));
        int total_correct = 0;

        // Evaluate model
        std::cout << "\nEvaluating MLP model..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_images.size(); ++i)
        {
            std::vector<float> predicted = mlp.forward(test_images[i]);
            int predicted_label = std::max_element(predicted.begin(), predicted.end()) - predicted.begin();
            int true_label = std::max_element(test_labels[i].begin(), test_labels[i].end()) - test_labels[i].begin();

            // Update confusion matrix
            confusion_matrix[true_label][predicted_label]++;
            if (predicted_label == true_label)
                total_correct++;

            // Print progress every 100 samples
            if ((i + 1) % 100 == 0)
            {
                std::cout << "\rProcessed " << i + 1 << "/" << test_images.size() << " samples" << std::flush;
            }
        }
        std::cout << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;

        std::cout << "\nEvaluation completed in " << seconds << " seconds" << std::endl;
        std::cout << "Average time per sample: " << std::fixed << std::setprecision(2)
                  << (seconds * 1000.0 / test_images.size()) << " ms" << std::endl;

        // Print confusion matrix with better formatting
        std::cout << "\nConfusion Matrix:\n";
        std::cout << "X-Axis: Predicted labels (0-9)\n";
        std::cout << "Y-Axis: Actual labels (0-9)\n\n";

        // Print column headers
        std::cout << "     ";
        for (int i = 0; i < 10; i++)
        {
            std::cout << std::setw(6) << i;
        }
        std::cout << "  │ Class Acc." << std::endl;

        // Print separator line
        std::cout << "   ┌";
        for (int i = 0; i < 62; i++)
            std::cout << "─";
        std::cout << "┐" << std::endl;

        // Print matrix rows with class accuracies
        for (int i = 0; i < 10; i++)
        {
            std::cout << std::setw(2) << i << " │";
            int row_total = 0;
            for (int j = 0; j < 10; j++)
            {
                std::cout << std::setw(6) << confusion_matrix[i][j];
                row_total += confusion_matrix[i][j];
            }
            // Print row accuracy
            float row_accuracy = row_total > 0 ? (float)confusion_matrix[i][i] / row_total * 100 : 0.0f;
            std::cout << "  │" << std::setw(7) << std::fixed << std::setprecision(1) << row_accuracy << "%" << std::endl;
        }

        // Print bottom border
        std::cout << "   └";
        for (int i = 0; i < 62; i++)
            std::cout << "─";
        std::cout << "┘" << std::endl;

        // Print column accuracies
        std::cout << "Acc│";
        for (int j = 0; j < 10; j++)
        {
            int col_total = 0;
            for (int i = 0; i < 10; i++)
            {
                col_total += confusion_matrix[i][j];
            }
            float col_accuracy = col_total > 0 ? (float)confusion_matrix[j][j] / col_total * 100 : 0.0f;
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << col_accuracy;
        }
        std::cout << "% │" << std::endl;

        // Calculate and print overall accuracy
        float accuracy = static_cast<float>(total_correct) / test_images.size() * 100;
        std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "Total samples: " << test_images.size() << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}