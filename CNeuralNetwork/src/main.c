#include "neural_network.h"

int main() 
{
    srand(time(NULL));

    double trainingData[4][NUM_INPUTS] = 
    {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    double expectedData[4][NUM_OUTPUTS] = 
    {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    Train(trainingData, expectedData, 4);
    return 0;
}
