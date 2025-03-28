#include "../include/evaluate.hpp"
#include <iostream>
#include <filesystem>
#include <iomanip>

void print_usage()
{
    std::cout << "Usage: ./evaluate [options]\n"
              << "Options:\n"
              << "  <test_size>        Number of test images to use (1-10000, default: 10000)\n"
              << "  <weights_file>     Path to weights file (default: ../weights.bin)\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char *argv[])
{
    try
    {
        // Parse command line arguments
        if (argc > 1 && std::string(argv[1]) == "--help")
        {
            print_usage();
            return 0;
        }

        int test_size = 10000;                       // Default test size
        std::string weights_file = "../weights.bin"; // Default weights file

        // Override defaults with command line arguments if provided
        if (argc > 1)
            test_size = std::stoi(argv[1]);
        if (argc > 2)
            weights_file = argv[2];

        // Validate test size
        if (test_size < 1 || test_size > 10000)
        {
            std::cerr << "Error: test size must be between 1 and 10000\n";
            return 1;
        }

        // Check if data directory exists
        if (!std::filesystem::exists("../data"))
        {
            std::cerr << "Error: data directory not found" << std::endl;
            return 1;
        }

        // Check if weights file exists
        if (!std::filesystem::exists(weights_file))
        {
            std::cerr << "Error: weights file not found: " << weights_file << std::endl;
            return 1;
        }

        // Check if MNIST test files exist
        std::string test_images_file = "../data/t10k-images.idx3-ubyte";
        std::string test_labels_file = "../data/t10k-labels.idx1-ubyte";
        if (!std::filesystem::exists(test_images_file) || !std::filesystem::exists(test_labels_file))
        {
            std::cerr << "Error: MNIST test files not found" << std::endl;
            return 1;
        }

        // Initialize CNN and load weights
        std::cout << "Initializing CNN..." << std::endl;
        CNN cnn(0.01f); // Learning rate doesn't matter for evaluation

        std::cout << "Loading weights from " << weights_file << "..." << std::endl;
        cnn.load_weights_from_file(weights_file);

        // Create evaluator and run evaluation
        Evaluator evaluator(cnn);
        auto start = std::chrono::high_resolution_clock::now();
        evaluator.evaluate_model(test_images_file, test_labels_file, test_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Successfully loaded " << test_size << " test images." << std::endl;
        std::cout << "Evaluation completed in " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}