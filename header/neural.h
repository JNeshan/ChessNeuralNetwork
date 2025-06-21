#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include "layer.h"


class NeuralNetwork {
public:
    NeuralNetwork(std::string f);
    ~NeuralNetwork();
    void RunNetwork(Tensor input); //to start a run of the network,temporary placeholder
    void SaveNetwork();
    void SetLayers(std::vector<std::unique_ptr<Layer>> layers);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::string file; //file name used for storing tensors
    void initialize();
    void saveTensors();
    void loadTensors();
};

#endif // NEURAL_H