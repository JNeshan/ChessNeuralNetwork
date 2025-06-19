#ifndef NEURAL_H
#define NEURAL_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);
    std::vector<double> predict(const std::vector<double>& input) const;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    void initialize();
    double activationFunction(double x) const;
};

#endif // NEURAL_H