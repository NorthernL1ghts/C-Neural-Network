# XOR Neural Network in C

A simple multi-layer perceptron neural network written in C++ to solve the classic XOR problem using backpropagation. The network has two hidden layers and trains using a mean squared error loss function.

## Features

- **Fully connected neural network** with:
  - 2 input neurons
  - 2 hidden layers (each with 4 neurons)
  - 1 output neuron
- **Sigmoid activation function**
- **Backpropagation algorithm** for training
- Supports **batch gradient descent**
- Adjustable **learning rate** with decay
- Network training over multiple epochs with optional **data shuffling** for improved convergence

## Tech Stack

- **Language:** C++
- **Standard:** C++20
- **Libraries:** `cmath`, `cstdlib`, `ctime`
- **Operating System:** Tested on Linux and Windows

## Project Structure

```
.
├── bin/
│   └── xor_nn           # Compiled executable
├── bin-int/
│   └── build/           # Intermediate build files
└── src/
    └── main.cpp         # Neural network source code
```

## Build and Run Instructions

### Requirements

- A C++ compiler supporting C++20 (e.g., `g++` on Linux, `MinGW` on Windows).

### Build Instructions

Compile the program using the following command:

```bash
g++ -std=c++20 -o bin/xor_nn src/main.cpp -lm
```

- `-std=c++20`: Use the C++20 standard.
- `-o bin/xor_nn`: Output the executable to the `bin` directory.
- `-lm`: Link the math library for functions like `exp()`.

### Run the Program

Execute the compiled program:

```bash
./bin/xor_nn
```

### Sample Output

```
Epoch 0
Input: 0.00 0.00, Output: 0.01, Expected: 0.00
Input: 0.00 1.00, Output: 0.99, Expected: 1.00
Input: 1.00 0.00, Output: 0.98, Expected: 1.00
Input: 1.00 1.00, Output: 0.03, Expected: 0.00

Epoch 10000
Input: 0.00 0.00, Output: 0.00, Expected: 0.00
Input: 0.00 1.00, Output: 1.00, Expected: 1.00
Input: 1.00 0.00, Output: 1.00, Expected: 1.00
Input: 1.00 1.00, Output: 0.00, Expected: 0.00
```

## Customization

### Parameters

You can modify the following parameters in `main.cpp`:

- **Learning Rate**: Adjust the `learningRate` variable.
- **Epochs**: Set the number of training epochs with the `epochs` variable.

## License

This project is licensed under the MIT License.
#   C - N e u r a l - N e t w o r k  
 