#include "../include/mlp.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <sstream>

MLP::MLP(int input_size, const std::vector<int> &hidden_sizes, int output_size, float learning_rate)
    : input_size(input_size), hidden_sizes(hidden_sizes), output_size(output_size), learning_rate(learning_rate)
{

    // Initialize weights and biases with the correct sizes
    fc1_weights = std::vector<std::vector<float>>(input_size, std::vector<float>(hidden_sizes[0]));
    fc1_bias = std::vector<float>(hidden_sizes[0]);

    fc2_weights = std::vector<std::vector<float>>(hidden_sizes[0], std::vector<float>(hidden_sizes[1]));
    fc2_bias = std::vector<float>(hidden_sizes[1]);

    fc3_weights = std::vector<std::vector<float>>(hidden_sizes[1], std::vector<float>(hidden_sizes[2]));
    fc3_bias = std::vector<float>(hidden_sizes[2]);

    fc4_weights = std::vector<std::vector<float>>(hidden_sizes[2], std::vector<float>(output_size));
    fc4_bias = std::vector<float>(output_size);

    // Initialize the weights
    initialize_weights();
}

void MLP::initialize_weights()
{
    std::random_device rd;
    rng = std::mt19937(rd());

    // He initialization for each layer
    auto init_weights = [this](std::vector<std::vector<float>> &weights, int fan_in)
    {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
        for (auto &row : weights)
        {
            for (float &weight : row)
            {
                weight = dist(rng);
            }
        }
    };

    auto init_bias = [](std::vector<float> &bias)
    {
        std::fill(bias.begin(), bias.end(), 0.0f);
    };

    // Layer 1: 784 -> 400
    init_weights(fc1_weights, input_size);
    init_bias(fc1_bias);

    // Layer 2: 400 -> 120
    init_weights(fc2_weights, hidden_sizes[0]);
    init_bias(fc2_bias);

    // Layer 3: 120 -> 84
    init_weights(fc3_weights, hidden_sizes[1]);
    init_bias(fc3_bias);

    // Output layer: 84 -> 10
    init_weights(fc4_weights, hidden_sizes[2]);
    init_bias(fc4_bias);
}

float MLP::relu(float x)
{
    return std::max(0.0f, x);
}

float MLP::relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

std::vector<float> MLP::softmax(const std::vector<float> &x)
{
    std::vector<float> output(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); ++i)
    {
        output[i] = std::exp(x[i] - max_val);
        sum += output[i];
    }

    for (float &val : output)
    {
        val /= sum;
    }

    return output;
}

float MLP::cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target)
{
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        loss -= target[i] * std::log(std::max(predicted[i], 1e-7f));
    }
    return loss;
}

std::vector<float> MLP::forward(const std::vector<float> &input)
{
    // Layer 1: 784 -> 400
    fc1_output.resize(400);
#pragma omp parallel for
    for (int i = 0; i < 400; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 784; ++j)
        {
            sum += input[j] * fc1_weights[j][i];
        }
        fc1_output[i] = relu(sum + fc1_bias[i]);
    }

    // Layer 2: 400 -> 120
    fc2_output.resize(120);
#pragma omp parallel for
    for (int i = 0; i < 120; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 400; ++j)
        {
            sum += fc1_output[j] * fc2_weights[j][i];
        }
        fc2_output[i] = relu(sum + fc2_bias[i]);
    }

    // Layer 3: 120 -> 84
    fc3_output.resize(84);
#pragma omp parallel for
    for (int i = 0; i < 84; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 120; ++j)
        {
            sum += fc2_output[j] * fc3_weights[j][i];
        }
        fc3_output[i] = relu(sum + fc3_bias[i]);
    }

    // Output layer: 84 -> 10
    std::vector<float> output(10);
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 84; ++j)
        {
            sum += fc3_output[j] * fc4_weights[j][i];
        }
        output[i] = sum + fc4_bias[i];
    }

    final_output = softmax(output);
    return final_output;
}

void MLP::backward(const std::vector<float> &input, const std::vector<float> &target)
{
    // Output layer gradients
    std::vector<float> output_delta(10);
    for (size_t i = 0; i < 10; ++i)
    {
        output_delta[i] = final_output[i] - target[i];
    }

    // Layer 3 gradients (84)
    std::vector<float> fc3_delta(84);
#pragma omp parallel for
    for (int i = 0; i < 84; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 10; ++j)
        {
            sum += output_delta[j] * fc4_weights[i][j];
        }
        fc3_delta[i] = sum * relu_derivative(fc3_output[i]);
    }

    // Layer 2 gradients (120)
    std::vector<float> fc2_delta(120);
#pragma omp parallel for
    for (int i = 0; i < 120; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 84; ++j)
        {
            sum += fc3_delta[j] * fc3_weights[i][j];
        }
        fc2_delta[i] = sum * relu_derivative(fc2_output[i]);
    }

    // Layer 1 gradients (400)
    std::vector<float> fc1_delta(400);
#pragma omp parallel for
    for (int i = 0; i < 400; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 120; ++j)
        {
            sum += fc2_delta[j] * fc2_weights[i][j];
        }
        fc1_delta[i] = sum * relu_derivative(fc1_output[i]);
    }

// Update weights and biases
// Output layer
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 84; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            fc4_weights[i][j] -= learning_rate * fc3_output[i] * output_delta[j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
    {
        fc4_bias[i] -= learning_rate * output_delta[i];
    }

// Layer 3
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 120; ++i)
    {
        for (int j = 0; j < 84; ++j)
        {
            fc3_weights[i][j] -= learning_rate * fc2_output[i] * fc3_delta[j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < 84; ++i)
    {
        fc3_bias[i] -= learning_rate * fc3_delta[i];
    }

// Layer 2
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 400; ++i)
    {
        for (int j = 0; j < 120; ++j)
        {
            fc2_weights[i][j] -= learning_rate * fc1_output[i] * fc2_delta[j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < 120; ++i)
    {
        fc2_bias[i] -= learning_rate * fc2_delta[i];
    }

// Layer 1
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 784; ++i)
    {
        for (int j = 0; j < 400; ++j)
        {
            fc1_weights[i][j] -= learning_rate * input[i] * fc1_delta[j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < 400; ++i)
    {
        fc1_bias[i] -= learning_rate * fc1_delta[i];
    }
}

void MLP::train(const std::vector<std::vector<float>> &training_data,
                const std::vector<std::vector<float>> &training_labels,
                const std::vector<std::vector<float>> &test_data,
                const std::vector<std::vector<float>> &test_labels,
                int epochs,
                int batch_size)
{

    size_t num_samples = training_data.size();
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    float best_accuracy = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float total_loss = 0.0f;
        std::cout << "\nEpoch " << std::setw(4) << epoch + 1 << "/" << std::setw(4) << epochs << ":" << std::endl;

        // Shuffle indices for each epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        // Process mini-batches
        for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size)
        {
            size_t current_batch_size = std::min(batch_size, static_cast<int>(num_samples - batch_start));
            float batch_loss = 0.0f;

            // Process each sample in the batch
            for (size_t i = batch_start; i < batch_start + current_batch_size; ++i)
            {
                size_t idx = indices[i];
                std::vector<float> predicted = forward(training_data[idx]);
                batch_loss += cross_entropy_loss(predicted, training_labels[idx]);
                backward(training_data[idx], training_labels[idx]);

                // Print progress
                if ((i + 1) % 100 == 0)
                {
                    std::cout << "\rProcessed " << (i + 1) << "/" << num_samples
                              << " samples (Batch " << (batch_start / batch_size + 1)
                              << "/" << (num_samples + batch_size - 1) / batch_size << ")"
                              << std::flush;
                }
            }

            total_loss += batch_loss;
        }

        // Calculate metrics
        float avg_loss = total_loss / num_samples;
        float train_accuracy = evaluate(training_data, training_labels, false);
        float test_accuracy = evaluate(test_data, test_labels, false);

        // Save best model
        if (test_accuracy > best_accuracy)
        {
            best_accuracy = test_accuracy;
            save_best_model();
        }

        std::cout << "\nEpoch " << epoch + 1 << " - "
                  << "Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Train Acc: " << std::setprecision(2) << (train_accuracy * 100) << "%"
                  << ", Test Acc: " << std::setprecision(2) << (test_accuracy * 100) << "%"
                  << " (Best: " << std::setprecision(2) << (best_accuracy * 100) << "%)"
                  << std::endl;
    }

    // Restore best model
    restore_best_model();
}

float MLP::evaluate(const std::vector<std::vector<float>> &test_data,
                    const std::vector<std::vector<float>> &test_labels,
                    bool print_predictions)
{
    int correct = 0;
    int total = test_data.size();

    for (size_t i = 0; i < test_data.size(); ++i)
    {
        std::vector<float> predicted = forward(test_data[i]);
        int predicted_label = std::max_element(predicted.begin(), predicted.end()) - predicted.begin();
        int true_label = std::max_element(test_labels[i].begin(), test_labels[i].end()) - test_labels[i].begin();

        if (predicted_label == true_label)
        {
            correct++;
        }

        if (print_predictions && (i + 1) % 100 == 0)
        {
            std::cout << "Sample " << (i + 1) << "/" << total
                      << " - Predicted: " << predicted_label
                      << ", True: " << true_label << std::endl;
        }
    }

    return static_cast<float>(correct) / total;
}

void MLP::save_best_model()
{
    best_fc1_weights = fc1_weights;
    best_fc1_bias = fc1_bias;
    best_fc2_weights = fc2_weights;
    best_fc2_bias = fc2_bias;
    best_fc3_weights = fc3_weights;
    best_fc3_bias = fc3_bias;
    best_fc4_weights = fc4_weights;
    best_fc4_bias = fc4_bias;
}

void MLP::restore_best_model()
{
    fc1_weights = best_fc1_weights;
    fc1_bias = best_fc1_bias;
    fc2_weights = best_fc2_weights;
    fc2_bias = best_fc2_bias;
    fc3_weights = best_fc3_weights;
    fc3_bias = best_fc3_bias;
    fc4_weights = best_fc4_weights;
    fc4_bias = best_fc4_bias;
}

void MLP::save_weights(const std::string &filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not create weights file: " + filename);
    }

    // Lambda to write vector to file
    auto write_vector = [&file](const std::vector<float> &vec)
    {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(float));
    };

    // Lambda to write 2D vector to file
    auto write_2d_vector = [&file, &write_vector](const std::vector<std::vector<float>> &vec)
    {
        size_t rows = vec.size();
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        for (const auto &row : vec)
        {
            write_vector(row);
        }
    };

    try
    {
        write_2d_vector(fc1_weights);
        write_vector(fc1_bias);
        write_2d_vector(fc2_weights);
        write_vector(fc2_bias);
        write_2d_vector(fc3_weights);
        write_vector(fc3_bias);
        write_2d_vector(fc4_weights);
        write_vector(fc4_bias);

        if (!file)
        {
            throw std::runtime_error("Error writing weights file");
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("Failed to save weights: ") + e.what());
    }
}

void MLP::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open weights file: " + filename);
    }

    // Lambda to read vector from file
    auto read_vector = [&file](std::vector<float> &vec)
    {
        size_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));

        if (size > 1000000) // Sanity check for potential endianness issues
        {
            size = ((size & 0xFF000000) >> 24) |
                   ((size & 0x00FF0000) >> 8) |
                   ((size & 0x0000FF00) << 8) |
                   ((size & 0x000000FF) << 24);

            if (size > 1000000)
            {
                throw std::runtime_error("Invalid vector size: " + std::to_string(size));
            }
        }

        vec.resize(size);
        file.read(reinterpret_cast<char *>(vec.data()), size * sizeof(float));

        if (!file)
        {
            throw std::runtime_error("Error reading vector data");
        }
    };

    // Lambda to read 2D vector from file
    auto read_2d_vector = [&file, &read_vector](std::vector<std::vector<float>> &vec)
    {
        size_t rows;
        file.read(reinterpret_cast<char *>(&rows), sizeof(rows));

        if (rows > 1000000) // Sanity check for potential endianness issues
        {
            rows = ((rows & 0xFF000000) >> 24) |
                   ((rows & 0x00FF0000) >> 8) |
                   ((rows & 0x0000FF00) << 8) |
                   ((rows & 0x000000FF) << 24);

            if (rows > 1000000)
            {
                throw std::runtime_error("Invalid number of rows: " + std::to_string(rows));
            }
        }

        vec.resize(rows);
        for (auto &row : vec)
        {
            read_vector(row);
        }
    };

    try
    {
        read_2d_vector(fc1_weights);
        read_vector(fc1_bias);
        read_2d_vector(fc2_weights);
        read_vector(fc2_bias);
        read_2d_vector(fc3_weights);
        read_vector(fc3_bias);
        read_2d_vector(fc4_weights);
        read_vector(fc4_bias);

        if (!file)
        {
            throw std::runtime_error("Error reading weights file");
        }

        // Validate dimensions
        if (fc1_weights.size() != 784 || (fc1_weights.size() > 0 && fc1_weights[0].size() != 400) ||
            fc2_weights.size() != 400 || (fc2_weights.size() > 0 && fc2_weights[0].size() != 120) ||
            fc3_weights.size() != 120 || (fc3_weights.size() > 0 && fc3_weights[0].size() != 84) ||
            fc4_weights.size() != 84 || (fc4_weights.size() > 0 && fc4_weights[0].size() != 10))
        {
            throw std::runtime_error("Invalid weight dimensions in file");
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("Failed to load weights: ") + e.what());
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
MLP::load_mnist(const std::string &images_file, const std::string &labels_file, int num_images)
{
    std::cout << "Loading MNIST data from:\n"
              << "Images: " << images_file << "\n"
              << "Labels: " << labels_file << std::endl;

    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images || !labels)
    {
        throw std::runtime_error("Failed to open MNIST files");
    }

    // Read headers
    uint32_t magic_number, num_items, num_rows, num_cols;
    images.read(reinterpret_cast<char *>(&magic_number), 4);
    images.read(reinterpret_cast<char *>(&num_items), 4);
    images.read(reinterpret_cast<char *>(&num_rows), 4);
    images.read(reinterpret_cast<char *>(&num_cols), 4);

    // Convert from big-endian
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    // Read label header
    uint32_t label_magic, num_labels;
    labels.read(reinterpret_cast<char *>(&label_magic), 4);
    labels.read(reinterpret_cast<char *>(&num_labels), 4);
    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);

    // Verify format
    if (magic_number != 0x803 || label_magic != 0x801)
    {
        throw std::runtime_error("Invalid MNIST format");
    }

    // Adjust num_images if needed
    if (num_images > static_cast<int>(num_items))
    {
        num_images = num_items;
    }

    // Read data
    std::vector<std::vector<float>> image_data(num_images, std::vector<float>(784));
    std::vector<std::vector<float>> label_data(num_images, std::vector<float>(10, 0.0f));

    for (int i = 0; i < num_images; ++i)
    {
        // Read image
        for (size_t j = 0; j < 784; ++j)
        {
            unsigned char pixel;
            images.read(reinterpret_cast<char *>(&pixel), 1);
            image_data[i][j] = static_cast<float>(pixel) / 255.0f;
        }

        // Read label
        unsigned char label;
        labels.read(reinterpret_cast<char *>(&label), 1);
        label_data[i][label] = 1.0f;

        // Print progress
        if ((i + 1) % 1000 == 0)
        {
            std::cout << "Loaded " << (i + 1) << "/" << num_images << " images" << std::endl;
        }
    }

    return {image_data, label_data};
}