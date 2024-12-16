# XOR Neural Network in C

A simple multi-layer perceptron neural network written in C to solve the classic XOR problem using backpropagation. The network has two hidden layers and trains using a mean squared error loss function.

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

- **Language:** C
- **Standard:** C20
- **Libraries:** `math.h`, `stdlib.h`, `time.h`
- **Operating System:** Tested on Linux and Windows

## Project Structure

```
.
├── bin/
│   └── Debug/
│       └── x64/
│           └── CNeuralNetwork/
│               └── CNeuralNetwork.exe
├── bin-int/
│   └── Debug/
│       └── x64/
│           └── CNeuralNetwork/
│               └── main.obj
└── CNeuralNetwork/
    ├── CNeuralNetwork.vcxproj
    ├── CNeuralNetwork.vcxproj.filters
    └── src/
        └── main.c
```

## Build and Run Instructions

### Requirements

- A C compiler supporting C17 (e.g., `gcc` on Linux, Visual Studio on Windows).

### Build Instructions

#### Using GCC

Compile the program using the following command:

```bash
gcc -std=c17 -o bin/Debug/x64/CNeuralNetwork/CNeuralNetwork src/main.c -lm
```

- `-std=c17`: Use the C17 standard.
- `-o bin/Debug/x64/CNeuralNetwork/CNeuralNetwork`: Output the executable to the `bin` directory.
- `-lm`: Link the math library for functions like `exp()`.

#### Using Visual Studio

1. Open the `CNeuralNetwork.sln` file in Visual Studio.
2. Select the appropriate build configuration (e.g., Debug x64).
3. Build the solution (`Ctrl+Shift+B`).

### Run the Program

Execute the compiled program:

```bash
./bin/Debug/x64/CNeuralNetwork/CNeuralNetwork.exe
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

You can modify the following parameters in `main.c`:

- **Learning Rate**: Adjust the `learningRate` variable.
- **Epochs**: Set the number of training epochs with the `epochs` variable.

## License

This project is licensed under the MIT License.
