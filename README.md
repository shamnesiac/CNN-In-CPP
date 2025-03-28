# CNN Implementation in C++

This project implements a Convolutional Neural Network (CNN) from scratch in C++, based on the LeNet-5 architecture. The network is designed to recognize handwritten digits from the MNIST dataset.

## Architecture

The CNN consists of the following layers:
1. Input Layer (28x28 grayscale images)
2. Convolutional Layer 1 (6 filters of 5x5, stride 1)
3. Max Pooling Layer 1 (2x2 filter, stride 2)
4. Convolutional Layer 2 (16 filters of 5x5, stride 1)
5. Max Pooling Layer 2 (2x2 filter, stride 2)
6. Fully Connected Layer 1 (400 -> 120 neurons)
7. Fully Connected Layer 2 (120 -> 84 neurons)
8. Output Layer (84 -> 10 neurons)

## Prerequisites

- C++17 compatible compiler
- CMake (version 3.10 or higher)
- MNIST dataset files (not included)

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure and build with CMake:
```bash
cmake ..
cmake --build .
```

## Preparing the Dataset

1. Download the MNIST dataset files:
   - train-images.idx3-ubyte
   - train-labels.idx1-ubyte
   - t10k-images.idx3-ubyte
   - t10k-labels.idx1-ubyte

2. Place these files in the `data` directory at the project root.

## Running the Program

After building, run the executable from the build directory:
```bash
./CNNCPP
```

The program will:
1. Load the MNIST training and test data
2. Train the network for 10 epochs
3. Display the training loss after each epoch
4. Evaluate the model on the test set and display the accuracy

## Implementation Details

- The network uses ReLU activation functions for all layers except the output layer
- The output layer uses softmax activation
- Cross-entropy loss is used as the loss function
- The network is trained using stochastic gradient descent
- The implementation includes data normalization (pixel values scaled to 0-1)

## Performance Optimization

- OpenMP is used for parallel processing when available
- Release build includes compiler optimizations (-O3 for GCC/Clang, /O2 for MSVC)

## License

This project is open source and available under the MIT License. 