#include "Image.h"
#include <math.h>
#include <cstdlib>
#include <ctime>

#pragma once

class NeuralNetwork {

    int num_hidden_layers;
    int num_hidden_layer_nodes;

    vector<double> input_layer;
    vector<vector<double>> hidden_layers;
    vector<double> output_layer;
    vector<double> bias_nodes;

    //weight_layers[layer][layer1node][layer2node] where this value represents
    //the weighted connection between layer1node and layer2node
    vector<vector<vector<double>>> weight_layers;

    void forwardpropogate();
    void backpropogate();
    //advantageous as derivative is ( S(x)*(1-S(x)) )
    double sigmoid(double x);
    //randomly initialize weights and init layers to predetermined size
    void randomInit();

public:
    NeuralNetwork(int hl, int hn);

    void train(vector<Image> images);
    void think(vector<Image> images);

};