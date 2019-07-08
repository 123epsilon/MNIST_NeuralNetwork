#include "Image.h"
#include <math.h>
#include <cstdlib>
#include <random>

#pragma once

class NeuralNetwork {

    int num_hidden_layers;
    int num_hidden_layer_nodes;
    double learning_rate;

    vector<double> input_layer;
    vector<vector<double>> hidden_layers;
    vector<double> output_layer;
    //one bias node (+!) for each hidden layer and output layer - connects via weights
    vector<vector<double>> bias_nodes;

    //weight_layers[layer][layer1node][layer2node] where this value represents
    //the weighted connection between layer1node and layer2node
    vector<vector<vector<double>>> weight_layers;

    //take nodes from input layer and propogate their value forward by normalizing and then doing a dot product in order to
    //fill the next layer
    void forwardpropogate();
    void backpropogate(int label);
    //advantageous as derivative is ( S(x)*(1-S(x)) )
    double sigmoid(double x);
    double derivative(double x);
    //randomly initialize weights and init layers to predetermined size
    void init();

public:
    NeuralNetwork(int hl, int hn, double lr);

    void train(vector<Image> images, int iterations);
    void think(vector<Image> images);
    void printWeights();

};