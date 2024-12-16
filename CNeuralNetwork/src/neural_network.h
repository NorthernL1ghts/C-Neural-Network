#pragma once

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Constants
#define NUM_INPUTS 2
#define NUM_HIDDEN_NEURONS 4
#define NUM_HIDDEN_NEURONS_2 4
#define NUM_OUTPUTS 1
#define MAX_EPOCHS 20000
#define LEARNING_RATE 0.01
#define BATCH_SIZE 4

// Function prototypes
double Sigmoid(double x);
double SigmoidDerivative(double x);
double Random();
double Normalize(double value, double mean, double stddev);
void Initialize(double weights[][NUM_INPUTS], int rows, int cols);
double Activate(double* weights, double* inputs, int len);
double CalculateMSE(double* outputs, double* expected);
void Shuffle(double* data, int dataSize);
void ForwardPropagation(double inputs[], double hiddenLayer[], double hiddenLayer2[], double outputs[], double weightsIH[][NUM_HIDDEN_NEURONS], double weightsIH2[][NUM_HIDDEN_NEURONS_2], double weightsHO2[][NUM_OUTPUTS]);
void Backpropagation(double inputs[], double hiddenLayer[], double hiddenLayer2[], double outputs[], double expected[], double weightsIH[][NUM_HIDDEN_NEURONS], double weightsIH2[][NUM_HIDDEN_NEURONS_2], double weightsHO2[][NUM_OUTPUTS], double learningRate);
void Train(double trainingData[][NUM_INPUTS], double expectedData[][NUM_OUTPUTS], int dataSize);

#endif // NEURAL_NETWORK_H
