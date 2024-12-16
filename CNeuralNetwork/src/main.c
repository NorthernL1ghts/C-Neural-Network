#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN_NEURONS 4
#define NUM_HIDDEN_NEURONS_2 4
#define NUM_OUTPUTS 1
#define MAX_EPOCHS 20000
#define LEARNING_RATE 0.01
#define BATCH_SIZE 4

// Activation function: Sigmoid
double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid function
double SigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Random number generator
double Random() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// Normalize the data
double Normalize(double value, double mean, double stddev) {
    return (value - mean) / stddev;
}

// Initialize weights with small random values
void Initialize(double weights[][NUM_INPUTS], int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            weights[i][j] = Random();
}

// Calculate the dot product of inputs and weights
double Activate(double* weights, double* inputs, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++)
        sum += weights[i] * inputs[i];
    return sum;
}

// Calculate Mean Squared Error (MSE) for regression
double CalculateMSE(double* outputs, double* expected) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OUTPUTS; i++)
        sum += pow(outputs[i] - expected[i], 2);
    return sum / NUM_OUTPUTS;
}

// Shuffle the data (optional for better convergence)
void Shuffle(double* data, int dataSize) {
    for (int i = 0; i < dataSize; i++) {
        int j = rand() % dataSize;
        double temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

// Forward propagation (with 2 hidden layers)
void ForwardPropagation(double inputs[], double hiddenLayer[], double hiddenLayer2[], double outputs[], double weightsIH[][NUM_HIDDEN_NEURONS], double weightsIH2[][NUM_HIDDEN_NEURONS_2], double weightsHO2[][NUM_OUTPUTS]) {
    // First hidden layer
    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++)
        hiddenLayer[i] = Sigmoid(Activate(weightsIH[i], inputs, NUM_INPUTS));

    // Second hidden layer
    for (int i = 0; i < NUM_HIDDEN_NEURONS_2; i++)
        hiddenLayer2[i] = Sigmoid(Activate(weightsIH2[i], hiddenLayer, NUM_HIDDEN_NEURONS));

    // Output layer
    for (int i = 0; i < NUM_OUTPUTS; i++)
        outputs[i] = Sigmoid(Activate(weightsHO2[i], hiddenLayer2, NUM_HIDDEN_NEURONS_2));
}

// Backpropagation for weight adjustments
void Backpropagation(double inputs[], double hiddenLayer[], double hiddenLayer2[], double outputs[], double expected[], double weightsIH[][NUM_HIDDEN_NEURONS], double weightsIH2[][NUM_HIDDEN_NEURONS_2], double weightsHO2[][NUM_OUTPUTS], double learningRate) {
    // Calculate output layer error and delta
    double outputError[NUM_OUTPUTS];
    double outputDelta[NUM_OUTPUTS];

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        outputError[i] = expected[i] - outputs[i];
        outputDelta[i] = outputError[i] * SigmoidDerivative(outputs[i]);
    }

    // Calculate second hidden layer error and delta
    double hiddenLayer2Error[NUM_HIDDEN_NEURONS_2];
    double hiddenLayer2Delta[NUM_HIDDEN_NEURONS_2];

    for (int i = 0; i < NUM_HIDDEN_NEURONS_2; i++) {
        hiddenLayer2Error[i] = 0.0;
        for (int j = 0; j < NUM_OUTPUTS; j++)
            hiddenLayer2Error[i] += outputDelta[j] * weightsHO2[j][i];

        hiddenLayer2Delta[i] = hiddenLayer2Error[i] * SigmoidDerivative(hiddenLayer2[i]);
    }

    // Calculate first hidden layer error and delta
    double hiddenLayerError[NUM_HIDDEN_NEURONS];
    double hiddenLayerDelta[NUM_HIDDEN_NEURONS];

    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++) {
        hiddenLayerError[i] = 0.0;
        for (int j = 0; j < NUM_HIDDEN_NEURONS_2; j++)
            hiddenLayerError[i] += hiddenLayer2Delta[j] * weightsIH2[j][i];

        hiddenLayerDelta[i] = hiddenLayerError[i] * SigmoidDerivative(hiddenLayer[i]);
    }

    // Update weights for output layer
    for (int i = 0; i < NUM_OUTPUTS; i++)
        for (int j = 0; j < NUM_HIDDEN_NEURONS_2; j++)
            weightsHO2[i][j] += learningRate * outputDelta[i] * hiddenLayer2[j];

    // Update weights for second hidden layer
    for (int i = 0; i < NUM_HIDDEN_NEURONS_2; i++)
        for (int j = 0; j < NUM_HIDDEN_NEURONS; j++)
            weightsIH2[i][j] += learningRate * hiddenLayer2Delta[i] * hiddenLayer[j];

    // Update weights for first hidden layer
    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++)
        for (int j = 0; j < NUM_INPUTS; j++)
            weightsIH[i][j] += learningRate * hiddenLayerDelta[i] * inputs[j];
}

// Training the network
void Train(double trainingData[][NUM_INPUTS], double expectedData[][NUM_OUTPUTS], int dataSize) {
    double weightsIH[NUM_HIDDEN_NEURONS][NUM_INPUTS];
    double weightsIH2[NUM_HIDDEN_NEURONS_2][NUM_HIDDEN_NEURONS];
    double weightsHO2[NUM_OUTPUTS][NUM_HIDDEN_NEURONS_2];

    Initialize(weightsIH, NUM_HIDDEN_NEURONS, NUM_INPUTS);
    Initialize(weightsIH2, NUM_HIDDEN_NEURONS_2, NUM_HIDDEN_NEURONS);
    Initialize(weightsHO2, NUM_OUTPUTS, NUM_HIDDEN_NEURONS_2);

    double inputs[NUM_INPUTS];
    double hiddenLayer[NUM_HIDDEN_NEURONS];
    double hiddenLayer2[NUM_HIDDEN_NEURONS_2];
    double outputs[NUM_OUTPUTS];

    // Train for multiple epochs
    double learningRate = LEARNING_RATE;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        Shuffle(trainingData, dataSize);

        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < NUM_INPUTS; j++)
                inputs[j] = trainingData[i][j];

            ForwardPropagation(inputs, hiddenLayer, hiddenLayer2, outputs, weightsIH, weightsIH2, weightsHO2);
            Backpropagation(inputs, hiddenLayer, hiddenLayer2, outputs, expectedData[i], weightsIH, weightsIH2, weightsHO2, learningRate);
        }

        // Reduce learning rate every 1000 epochs
        if (epoch % 1000 == 0)
            learningRate *= 0.99;

        // Print results every 1000 epochs
        if (epoch % 1000 == 0) {
            printf("Epoch %d\n", epoch);
            for (int i = 0; i < dataSize; i++) {
                ForwardPropagation(trainingData[i], hiddenLayer, hiddenLayer2, outputs, weightsIH, weightsIH2, weightsHO2);
                printf("Input: %.2f %.2f, Expected: %.2f, Prediction: %.2f\n", trainingData[i][0], trainingData[i][1], expectedData[i][0], outputs[0]);
            }
        }
    }
}

int main() {
    srand(time(NULL));

    // Training data for XOR problem
    double trainingData[4][NUM_INPUTS] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    double expectedData[4][NUM_OUTPUTS] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    Train(trainingData, expectedData, 4);
    return 0;
}