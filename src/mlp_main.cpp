#include "../include/mlp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

void print_usage()
{
    std::cout << "Usage: ./mlp_train [options]\n"
              << "Options:\n"
              << "  --train-size <n>   Number of training images (1-60000, default: 60000)\n"
              << "  --test-size <n>    Number of test images (1-10000, default: 10000)\n"
              << "  --epochs <n>       Number of training epochs (default: 10)\n"
              << "  --batch-size <n>   Mini-batch size (default: 32)\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char *argv[])
{
    try
    {
        // Default parameters
        int train_size = 60000;
        int test_size = 10000;
        int epochs = 10;
        int batch_size = 32;

        // Parse command line arguments
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--help")
            {
                print_usage();
                return 0;
            }
            else if (arg == "--train-size" && i + 1 < argc)
            {
                train_size = std::atoi(argv[++i]);
                if (train_size <= 0 || train_size > 60000)
                {
                    std::cerr << "Error: training size must be between 1 and 60000\n";
                    return 1;
                }
            }
            else if (arg == "--test-size" && i + 1 < argc)
            {
                test_size = std::atoi(argv[++i]);
                if (test_size <= 0 || test_size > 10000)
                {
                    std::cerr << "Error: test size must be between 1 and 10000\n";
                    return 1;
                }
            }
            else if (arg == "--epochs" && i + 1 < argc)
            {
                epochs = std::atoi(argv[++i]);
                if (epochs <= 0)
                {
                    std::cerr << "Error: epochs must be positive\n";
                    return 1;
                }
            }
            else if (arg == "--batch-size" && i + 1 < argc)
            {
                batch_size = std::atoi(argv[++i]);
                if (batch_size <= 0)
                {
                    std::cerr << "Error: batch size must be positive\n";
                    return 1;
                }
            }
        }

        // Check if data directory exists
        if (!std::filesystem::exists("../data"))
        {
            std::cerr << "Error: data directory not found at ../data\n";
            return 1;
        }

        // Check if training files exist
        std::string train_images_path = "../data/train-images.idx3-ubyte";
        std::string train_labels_path = "../data/train-labels.idx1-ubyte";
        if (!std::filesystem::exists(train_images_path) || !std::filesystem::exists(train_labels_path))
        {
            std::cerr << "Error: MNIST training files not found in ../data directory\n";
            return 1;
        }

        // Initialize MLP
        const int input_size = 784;                           // 28x28 images
        const std::vector<int> hidden_sizes = {400, 120, 84}; // LeNet-5 inspired architecture
        const int output_size = 10;                           // 10 digits
        MLP mlp(input_size, hidden_sizes, output_size);

        // Load training data
        std::cout << "\nLoading training data..." << std::endl;
        auto [train_images, train_labels] = MLP::load_mnist(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            train_size);

        std::cout << "Successfully loaded training data." << std::endl;
        std::cout << "Number of training images: " << train_images.size() << std::endl;

        // Load test data
        std::cout << "\nLoading test data..." << std::endl;
        auto [test_images, test_labels] = MLP::load_mnist(
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            test_size);

        std::cout << "Successfully loaded test data." << std::endl;
        std::cout << "Number of test images: " << test_images.size() << std::endl;

        // Train the network
        std::cout << "\nTraining MLP..." << std::endl;
        std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        mlp.train(train_images, train_labels, test_images, test_labels, epochs, batch_size);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "\nTraining completed in " << duration.count() << " seconds" << std::endl;

        // Save the best weights
        std::cout << "\nSaving best weights to mlp_weights.bin..." << std::endl;
        try
        {
            mlp.save_weights("../mlp_weights.bin");
            std::cout << "Weights saved successfully." << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving weights: " << e.what() << std::endl;
            return 1;
        }

        // Final evaluation
        std::cout << "\nPerforming final evaluation..." << std::endl;
        float accuracy = mlp.evaluate(test_images, test_labels, true);
        std::cout << "\nFinal test accuracy: " << std::fixed << std::setprecision(2)
                  << (accuracy * 100) << "%" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}