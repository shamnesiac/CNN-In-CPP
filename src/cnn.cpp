#include "../include/cnn.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <omp.h>

CNN::CNN(float lr) : learning_rate(lr)
{
    std::random_device rd;
    rng = std::mt19937(rd());
    initialize_weights();
}

void CNN::initialize_weights()
{
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Initialize conv1 weights (6 5x5x1 filters)
    conv1_weights.resize(6, std::vector<std::vector<std::vector<float>>>(
                                1, std::vector<std::vector<float>>(
                                       5, std::vector<float>(5))));

    for (auto &filter : conv1_weights)
    {
        for (auto &channel : filter)
        {
            for (auto &row : channel)
            {
                for (float &weight : row)
                {
                    weight = dist(rng);
                }
            }
        }
    }
    conv1_biases = std::vector<float>(6, 0.0f);

    // Initialize conv2 weights (16 5x5x6 filters)
    conv2_weights.resize(16, std::vector<std::vector<std::vector<float>>>(
                                 6, std::vector<std::vector<float>>(
                                        5, std::vector<float>(5))));

    for (auto &filter : conv2_weights)
    {
        for (auto &channel : filter)
        {
            for (auto &row : channel)
            {
                for (float &weight : row)
                {
                    weight = dist(rng);
                }
            }
        }
    }
    conv2_biases = std::vector<float>(16, 0.0f);

    // Initialize fully connected layers
    fc1_weights.resize(400, std::vector<float>(120));
    for (auto &row : fc1_weights)
    {
        for (float &weight : row)
        {
            weight = dist(rng);
        }
    }
    fc1_biases = std::vector<float>(120, 0.0f);

    fc2_weights.resize(120, std::vector<float>(84));
    for (auto &row : fc2_weights)
    {
        for (float &weight : row)
        {
            weight = dist(rng);
        }
    }
    fc2_biases = std::vector<float>(84, 0.0f);

    output_weights.resize(84, std::vector<float>(10));
    for (auto &row : output_weights)
    {
        for (float &weight : row)
        {
            weight = dist(rng);
        }
    }
    output_biases = std::vector<float>(10, 0.0f);
}

float CNN::relu(float x)
{
    return std::max(0.0f, x);
}

float CNN::relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

std::vector<float> CNN::softmax(const std::vector<float> &x)
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

float CNN::cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target)
{
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        loss -= target[i] * std::log(std::max(predicted[i], 1e-7f));
    }
    return loss;
}

std::vector<std::vector<std::vector<float>>> CNN::convolution(
    const std::vector<std::vector<std::vector<float>>> &input,
    const std::vector<std::vector<std::vector<std::vector<float>>>> &filters,
    const std::vector<float> &biases)
{
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int filter_size = filters[0][0].size();
    int num_filters = filters.size();
    int output_height = input_height - filter_size + 1;
    int output_width = input_width - filter_size + 1;

    std::vector<std::vector<std::vector<float>>> output(
        num_filters,
        std::vector<std::vector<float>>(
            output_height,
            std::vector<float>(output_width, 0.0f)));

#pragma omp parallel for collapse(2)
    for (int f = 0; f < num_filters; ++f)
    {
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                float sum = 0.0f;
                for (size_t c = 0; c < input.size(); ++c)
                {
                    for (int m = 0; m < filter_size; ++m)
                    {
                        for (int n = 0; n < filter_size; ++n)
                        {
                            sum += input[c][i + m][j + n] * filters[f][c][m][n];
                        }
                    }
                }
                output[f][i][j] = relu(sum + biases[f]);
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<float>>> CNN::max_pooling(
    const std::vector<std::vector<std::vector<float>>> &input)
{
    int channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = input_height / 2;
    int output_width = input_width / 2;

    std::vector<std::vector<std::vector<float>>> output(
        channels,
        std::vector<std::vector<float>>(
            output_height,
            std::vector<float>(output_width)));

#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                float max_val = input[c][2 * i][2 * j];
                max_val = std::max(max_val, input[c][2 * i][2 * j + 1]);
                max_val = std::max(max_val, input[c][2 * i + 1][2 * j]);
                max_val = std::max(max_val, input[c][2 * i + 1][2 * j + 1]);
                output[c][i][j] = max_val;
            }
        }
    }

    return output;
}

std::vector<float> CNN::forward(const std::vector<std::vector<float>> &input)
{
    // Convert 2D input to 3D (add channel dimension)
    std::vector<std::vector<std::vector<float>>> input_3d(1, input);

    // Convolution layer 1
    conv1_output = convolution(input_3d, conv1_weights, conv1_biases);

    // Max pooling layer 1
    pool1_output = max_pooling(conv1_output);

    // Convolution layer 2
    conv2_output = convolution(pool1_output, conv2_weights, conv2_biases);

    // Max pooling layer 2
    pool2_output = max_pooling(conv2_output);

    // Flatten the output
    std::vector<float> flattened;
    flattened.reserve(400); // Pre-allocate memory
    for (const auto &feature_map : pool2_output)
    {
        for (const auto &row : feature_map)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    // Fully connected layer 1
    fc1_output.resize(120);
#pragma omp parallel for
    for (int i = 0; i < 120; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 400 && j < flattened.size(); ++j)
        {
            sum += flattened[j] * fc1_weights[j][i];
        }
        fc1_output[i] = relu(sum + fc1_biases[i]);
    }

    // Fully connected layer 2
    fc2_output.resize(84);
#pragma omp parallel for
    for (int i = 0; i < 84; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 120; ++j)
        {
            sum += fc1_output[j] * fc2_weights[j][i];
        }
        fc2_output[i] = relu(sum + fc2_biases[i]);
    }

    // Output layer
    std::vector<float> output(10);
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 84; ++j)
        {
            sum += fc2_output[j] * output_weights[j][i];
        }
        output[i] = sum + output_biases[i];
    }

    final_output = softmax(output);
    return final_output;
}

void CNN::backward(const std::vector<std::vector<float>> &input, const std::vector<float> &target)
{
    // Output layer gradients
    std::vector<float> output_delta(10);
    for (size_t i = 0; i < 10; ++i)
    {
        output_delta[i] = final_output[i] - target[i];
    }

    // FC2 layer gradients
    std::vector<float> fc2_delta(84);
    for (size_t i = 0; i < 84; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 10; ++j)
        {
            sum += output_delta[j] * output_weights[i][j];
        }
        fc2_delta[i] = sum * relu_derivative(fc2_output[i]);
    }

    // FC1 layer gradients
    std::vector<float> fc1_delta(120);
    for (size_t i = 0; i < 120; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < 84; ++j)
        {
            sum += fc2_delta[j] * fc2_weights[i][j];
        }
        fc1_delta[i] = sum * relu_derivative(fc1_output[i]);
    }

    // Flatten pool2_output
    std::vector<float> flattened_pool2;
    for (const auto &feature_map : pool2_output)
    {
        for (const auto &row : feature_map)
        {
            flattened_pool2.insert(flattened_pool2.end(), row.begin(), row.end());
        }
    }

    // Conv2 layer gradients
    std::vector<std::vector<std::vector<float>>> conv2_delta(16,
                                                             std::vector<std::vector<float>>(5,
                                                                                             std::vector<float>(5, 0.0f)));

    // Update Conv2 weights
    for (size_t f = 0; f < 16 && f < conv2_weights.size(); ++f)
    {
        for (size_t c = 0; c < 6 && c < conv2_weights[f].size(); ++c)
        {
            for (size_t i = 0; i < 5 && i < conv2_weights[f][c].size(); ++i)
            {
                for (size_t j = 0; j < 5 && j < conv2_weights[f][c][i].size(); ++j)
                {
                    float grad = 0.0f;
                    for (size_t m = 0; m + i < pool1_output[c].size(); ++m)
                    {
                        for (size_t n = 0; n + j < pool1_output[c][0].size(); ++n)
                        {
                            grad += pool1_output[c][m][n] * conv2_delta[f][i][j];
                        }
                    }
                    conv2_weights[f][c][i][j] -= learning_rate * grad;
                }
            }
        }

        // Update biases
        float bias_grad = 0.0f;
        for (size_t i = 0; i < conv2_delta[f].size(); ++i)
        {
            for (size_t j = 0; j < conv2_delta[f][0].size(); ++j)
            {
                bias_grad += conv2_delta[f][i][j];
            }
        }
        if (f < conv2_biases.size())
        {
            conv2_biases[f] -= learning_rate * bias_grad;
        }
    }

    // Conv1 layer gradients
    std::vector<std::vector<std::vector<float>>> conv1_delta(6,
                                                             std::vector<std::vector<float>>(24,
                                                                                             std::vector<float>(24, 0.0f)));

    for (size_t f = 0; f < 6 && f < conv1_weights.size(); ++f)
    {
        for (size_t i = 0; i < 24 && i < conv1_output[f].size(); ++i)
        {
            for (size_t j = 0; j < 24 && j < conv1_output[f][0].size(); ++j)
            {
                float error = 0.0f;
                for (size_t k = 0; k < 16 && k < conv2_weights.size(); ++k)
                {
                    for (size_t m = 0; m < 5 && m < conv2_delta[k].size(); ++m)
                    {
                        for (size_t n = 0; n < 5 && n < conv2_delta[k][0].size(); ++n)
                        {
                            if (i / 2 + m < conv2_weights[k][f].size() && j / 2 + n < conv2_weights[k][f][0].size())
                            {
                                error += conv2_delta[k][m][n] * conv2_weights[k][f][i / 2 + m][j / 2 + n];
                            }
                        }
                    }
                }
                conv1_delta[f][i][j] = error * relu_derivative(conv1_output[f][i][j]);
            }
        }
    }

    // Update Conv1 weights
    for (size_t f = 0; f < 6 && f < conv1_weights.size(); ++f)
    {
        for (size_t i = 0; i < 5 && i < conv1_weights[f][0].size(); ++i)
        {
            for (size_t j = 0; j < 5 && j < conv1_weights[f][0][i].size(); ++j)
            {
                float grad = 0.0f;
                for (size_t m = 0; m < 24 && m < conv1_delta[f].size(); ++m)
                {
                    for (size_t n = 0; n < 24 && n < conv1_delta[f][0].size(); ++n)
                    {
                        if (m + i < input.size() && n + j < input[0].size())
                        {
                            grad += conv1_delta[f][m][n] * input[m + i][n + j];
                        }
                    }
                }
                conv1_weights[f][0][i][j] -= learning_rate * grad;
            }
        }

        // Update biases
        float bias_grad = 0.0f;
        for (size_t i = 0; i < conv1_delta[f].size(); ++i)
        {
            for (size_t j = 0; j < conv1_delta[f][0].size(); ++j)
            {
                bias_grad += conv1_delta[f][i][j];
            }
        }
        if (f < conv1_biases.size())
        {
            conv1_biases[f] -= learning_rate * bias_grad;
        }
    }

    // Update output layer weights
    for (size_t i = 0; i < 84; ++i)
    {
        for (size_t j = 0; j < 10; ++j)
        {
            output_weights[i][j] -= learning_rate * fc2_output[i] * output_delta[j];
        }
    }
    for (size_t i = 0; i < 10; ++i)
    {
        output_biases[i] -= learning_rate * output_delta[i];
    }

    // Update FC2 layer weights
    for (size_t i = 0; i < 120; ++i)
    {
        for (size_t j = 0; j < 84; ++j)
        {
            fc2_weights[i][j] -= learning_rate * fc1_output[i] * fc2_delta[j];
        }
    }
    for (size_t i = 0; i < 84; ++i)
    {
        fc2_biases[i] -= learning_rate * fc2_delta[i];
    }

    // Update FC1 layer weights
    for (size_t i = 0; i < flattened_pool2.size() && i < 400; ++i)
    {
        for (size_t j = 0; j < 120; ++j)
        {
            fc1_weights[i][j] -= learning_rate * flattened_pool2[i] * fc1_delta[j];
        }
    }
    for (size_t i = 0; i < 120; ++i)
    {
        fc1_biases[i] -= learning_rate * fc1_delta[i];
    }
}

void CNN::save_best_model()
{
    best_conv1_weights = conv1_weights;
    best_conv1_biases = conv1_biases;
    best_conv2_weights = conv2_weights;
    best_conv2_biases = conv2_biases;
    best_fc1_weights = fc1_weights;
    best_fc1_biases = fc1_biases;
    best_fc2_weights = fc2_weights;
    best_fc2_biases = fc2_biases;
    best_output_weights = output_weights;
    best_output_biases = output_biases;
}

void CNN::restore_best_model()
{
    conv1_weights = best_conv1_weights;
    conv1_biases = best_conv1_biases;
    conv2_weights = best_conv2_weights;
    conv2_biases = best_conv2_biases;
    fc1_weights = best_fc1_weights;
    fc1_biases = best_fc1_biases;
    fc2_weights = best_fc2_weights;
    fc2_biases = best_fc2_biases;
    output_weights = best_output_weights;
    output_biases = best_output_biases;
}

void CNN::train(const std::vector<std::vector<std::vector<float>>> &training_data,
                const std::vector<std::vector<float>> &training_labels,
                const std::vector<std::vector<std::vector<float>>> &test_data,
                const std::vector<std::vector<float>> &test_labels,
                int epochs,
                int batch_size)
{
    const size_t num_samples = training_data.size();
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Set number of threads for OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads for parallel processing" << std::endl;

    // Initialize best accuracy and save initial model
    best_accuracy = evaluate(test_data, test_labels, false);
    save_best_model();

    std::cout << "\nInitial Test Accuracy: " << std::fixed << std::setprecision(2)
              << (best_accuracy * 100) << "%" << std::endl;

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

                // Print progress for each image
                std::cout << "\rEpoch " << std::setw(4) << epoch + 1
                          << " - Processing image " << std::setw(5) << (i - batch_start + 1) << "/" << std::setw(5) << current_batch_size
                          << " (Batch " << (batch_start / batch_size + 1) << "/" << (num_samples + batch_size - 1) / batch_size << ")"
                          << std::flush;
            }

            total_loss += batch_loss;
        }

        // Calculate and print metrics for the epoch
        float avg_loss = total_loss / num_samples;
        float train_accuracy = evaluate(training_data, training_labels, false);
        float test_accuracy = evaluate(test_data, test_labels, false);

        std::cout << "\nEpoch " << std::setw(4) << epoch + 1 << "/" << std::setw(4) << epochs
                  << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " - Train Acc: " << std::fixed << std::setprecision(2) << (train_accuracy * 100) << "%"
                  << " - Test Acc: " << std::fixed << std::setprecision(2) << (test_accuracy * 100) << "%";

        // Update best model if we got better test accuracy
        if (test_accuracy > best_accuracy)
        {
            best_accuracy = test_accuracy;
            save_best_model();
            std::cout << " (Best)" << std::endl;
        }
        else
        {
            std::cout << std::endl;
        }
    }

    // Restore the best model
    restore_best_model();
    std::cout << "\nRestored best model with test accuracy: " << std::fixed << std::setprecision(2)
              << (best_accuracy * 100) << "%" << std::endl;
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>>
CNN::load_mnist(const std::string &images_file, const std::string &labels_file, int num_images)
{
    std::cout << "Attempting to open files:\n"
              << "Images: " << images_file << "\n"
              << "Labels: " << labels_file << std::endl;

    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images)
    {
        throw std::runtime_error("Failed to open MNIST images file: " + images_file);
    }
    if (!labels)
    {
        throw std::runtime_error("Failed to open MNIST labels file: " + labels_file);
    }

    // Read MNIST file headers
    uint32_t magic_number = 0;
    uint32_t num_items = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    // Read image file header
    images.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    images.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));
    images.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    images.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    // Convert from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    std::cout << "Image file header:\n"
              << "Magic number: " << magic_number << "\n"
              << "Number of items: " << num_items << "\n"
              << "Number of rows: " << num_rows << "\n"
              << "Number of columns: " << num_cols << std::endl;

    // Read label file header
    uint32_t label_magic = 0;
    uint32_t num_labels = 0;
    labels.read(reinterpret_cast<char *>(&label_magic), sizeof(label_magic));
    labels.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);

    std::cout << "Label file header:\n"
              << "Magic number: " << label_magic << "\n"
              << "Number of items: " << num_labels << std::endl;

    // Verify file format
    if (magic_number != 0x803 || label_magic != 0x801)
    {
        throw std::runtime_error("Invalid MNIST file format");
    }

    // Adjust num_images if it's larger than available data
    if (num_images > static_cast<int>(num_items))
    {
        std::cout << "Warning: Requested " << num_images << " images but only "
                  << num_items << " are available. Adjusting..." << std::endl;
        num_images = num_items;
    }

    std::vector<std::vector<std::vector<float>>> image_data(num_images,
                                                            std::vector<std::vector<float>>(num_rows, std::vector<float>(num_cols)));
    std::vector<std::vector<float>> label_data(num_images, std::vector<float>(10, 0.0f));

    std::cout << "Reading " << num_images << " images and labels..." << std::endl;

    for (int i = 0; i < num_images; ++i)
    {
        // Read image
        for (uint32_t r = 0; r < num_rows; ++r)
        {
            for (uint32_t c = 0; c < num_cols; ++c)
            {
                unsigned char pixel;
                images.read(reinterpret_cast<char *>(&pixel), 1);
                image_data[i][r][c] = static_cast<float>(pixel) / 255.0f;
            }
        }

        // Read label
        unsigned char label;
        labels.read(reinterpret_cast<char *>(&label), 1);
        if (label >= 10)
        {
            throw std::runtime_error("Invalid label value: " + std::to_string(label));
        }
        label_data[i][label] = 1.0f;

        // Print progress
        if ((i + 1) % 1000 == 0)
        {
            std::cout << "Processed " << (i + 1) << " images" << std::endl;
        }
    }

    std::cout << "Successfully loaded " << num_images << " images and labels" << std::endl;

    // Verify data
    std::cout << "Verifying data..." << std::endl;
    std::cout << "First image dimensions: " << image_data[0].size() << "x"
              << image_data[0][0].size() << std::endl;
    std::cout << "First label: ";
    for (int i = 0; i < 10; ++i)
    {
        if (label_data[0][i] == 1.0f)
        {
            std::cout << i;
            break;
        }
    }
    std::cout << std::endl;

    return {image_data, label_data};
}

int CNN::predict(const std::vector<std::vector<float>> &input)
{
    std::vector<float> output = forward(input);
    return std::max_element(output.begin(), output.end()) - output.begin();
}

float CNN::evaluate(const std::vector<std::vector<std::vector<float>>> &test_data,
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
            correct++;

        if (print_predictions && (i + 1) % 100 == 0)
        {
            std::cout << "Sample " << (i + 1) << "/" << total
                      << " - Predicted: " << predicted_label
                      << ", True: " << true_label << std::endl;
        }
    }

    return static_cast<float>(correct) / total;
}

void CNN::save_weights_to_file(const std::string &filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Helper lambda to write vector to file
    auto write_vector = [&file](const auto &vec)
    {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(vec[0]));
    };

    // Write conv1 weights and biases
    size_t conv1_size = conv1_weights.size();
    file.write(reinterpret_cast<const char *>(&conv1_size), sizeof(conv1_size));
    for (const auto &filter : conv1_weights)
    {
        for (const auto &channel : filter)
        {
            for (const auto &row : channel)
            {
                write_vector(row);
            }
        }
    }
    write_vector(conv1_biases);

    // Write conv2 weights and biases
    size_t conv2_size = conv2_weights.size();
    file.write(reinterpret_cast<const char *>(&conv2_size), sizeof(conv2_size));
    for (const auto &filter : conv2_weights)
    {
        for (const auto &channel : filter)
        {
            for (const auto &row : channel)
            {
                write_vector(row);
            }
        }
    }
    write_vector(conv2_biases);

    // Write FC1 weights and biases
    for (const auto &row : fc1_weights)
    {
        write_vector(row);
    }
    write_vector(fc1_biases);

    // Write FC2 weights and biases
    for (const auto &row : fc2_weights)
    {
        write_vector(row);
    }
    write_vector(fc2_biases);

    // Write output weights and biases
    for (const auto &row : output_weights)
    {
        write_vector(row);
    }
    write_vector(output_biases);
}

void CNN::load_weights_from_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // Helper lambda to read vector from file
    auto read_vector = [&file](auto &vec)
    {
        size_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char *>(vec.data()), size * sizeof(vec[0]));
    };

    // Read conv1 weights and biases
    size_t conv1_size;
    file.read(reinterpret_cast<char *>(&conv1_size), sizeof(conv1_size));
    conv1_weights.resize(conv1_size);
    for (auto &filter : conv1_weights)
    {
        filter.resize(1); // Input channel is 1
        for (auto &channel : filter)
        {
            channel.resize(5); // 5x5 filter
            for (auto &row : channel)
            {
                read_vector(row);
            }
        }
    }
    read_vector(conv1_biases);

    // Read conv2 weights and biases
    size_t conv2_size;
    file.read(reinterpret_cast<char *>(&conv2_size), sizeof(conv2_size));
    conv2_weights.resize(conv2_size);
    for (auto &filter : conv2_weights)
    {
        filter.resize(6); // 6 input channels
        for (auto &channel : filter)
        {
            channel.resize(5); // 5x5 filter
            for (auto &row : channel)
            {
                read_vector(row);
            }
        }
    }
    read_vector(conv2_biases);

    // Read FC1 weights and biases
    fc1_weights.resize(400); // Flattened input size
    for (auto &row : fc1_weights)
    {
        read_vector(row);
    }
    read_vector(fc1_biases);

    // Read FC2 weights and biases
    fc2_weights.resize(120); // FC1 output size
    for (auto &row : fc2_weights)
    {
        read_vector(row);
    }
    read_vector(fc2_biases);

    // Read output weights and biases
    output_weights.resize(84); // FC2 output size
    for (auto &row : output_weights)
    {
        read_vector(row);
    }
    read_vector(output_biases);
}