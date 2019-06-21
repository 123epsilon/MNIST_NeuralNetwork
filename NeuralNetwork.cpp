#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(int hl, int hn) {
    num_hidden_layers = hl;
    num_hidden_layer_nodes = hn;

    randomInit();

}

void NeuralNetwork::randomInit() {

    //init input, output, and bias to set size
    for(int i = 0; i < 784; i++){
        input_layer.push_back(0);
        if(i < 10) output_layer.push_back(0);
    }

    for(int i = 0; i < num_hidden_layers; i++){
        hidden_layers.push_back(vector<double>());
        for(int j = 0; j < num_hidden_layer_nodes; j++){
            hidden_layers[i].push_back(0);
        }
    }

    for(int i = 0; i < 2+num_hidden_layers; i++) bias_nodes.push_back(1.0);

    //randomly initialize weights
    int num_nodes;
    int second_num_nodes;

    srand(0);//srand(time(NULL));

    for(int current_layer = 0; current_layer < 1+num_hidden_layers; current_layer++){
        vector<vector<double>> temp;
        num_nodes = num_hidden_layer_nodes;
        second_num_nodes = num_hidden_layer_nodes;
        if(current_layer == 0) num_nodes = 784;
        if(current_layer == num_hidden_layers) second_num_nodes = 10;

        for(int layer1node = 0; layer1node < num_nodes; layer1node++){
            vector<double> temp2;
            for(int layer2node = 0; layer2node < second_num_nodes; layer2node++){
                temp2.push_back(((double) rand() / (RAND_MAX)));
            }
            temp.push_back(temp2);
        }
        weight_layers.push_back(temp);
    }
}

double NeuralNetwork::sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

void NeuralNetwork::think(vector<Image> images) {}

void NeuralNetwork::train(vector<Image> images) {

    for(int current_img = 0; current_img < images.size(); current_img++){
        //load input layer with normalized pixel values
        for(int i = 0; i < 784; i++){
            input_layer[i] = images[current_img].get(i) / 255.0;
        }

        forwardpropogate();
        backpropogate();
    }

}

void NeuralNetwork::forwardpropogate() {
    //propogate fwd from input to first hidden layer
    for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
       //reset this node's value
        hidden_layers[0][hidden_node] = 0;
        for(int input_node = 0; input_node < input_layer.size(); input_node++){
            //fill with sum of weights * inputs, normalize w/ sigmoid function
            hidden_layers[0][hidden_node] += input_layer[input_node] * weight_layers[0][input_node][hidden_node];
        }
        //normalize value
        hidden_layers[0][hidden_node] = sigmoid(hidden_layers[0][hidden_node]);
    }

    //propogate between all hidden layers, from first hidden layer to last
    for(int second_layer = 1; second_layer < num_hidden_layers; second_layer++){
        for(int second_layer_node = 0; second_layer_node < num_hidden_layer_nodes; second_layer_node++){
            //reset this node's value
            hidden_layers[second_layer][second_layer_node] = 0;
            for(int first_layer_node = 0; first_layer_node < num_hidden_layer_nodes; first_layer_node++){
                hidden_layers[second_layer][second_layer_node] += hidden_layers[second_layer-1][first_layer_node] * weight_layers[second_layer][first_layer_node][second_layer_node];
            }
            //normalize value
            hidden_layers[second_layer][second_layer_node] = sigmoid(hidden_layers[second_layer][second_layer_node]);
        }
    }

    //propogate from last hidden layer to the output layer
    for(int output_node = 0; output_node < output_layer.size(); output_node++){
        //reset this nodes value
        output_layer[output_node] = 0;
        for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
            output_layer[output_node] += hidden_layers[num_hidden_layers-1][hidden_node] * weight_layers[num_hidden_layers+1][hidden_node][output_node];
        }
        //normalize value
        output_layer[output_node] = sigmoid(output_layer[output_node]);
    }

}

void NeuralNetwork::backpropogate() {}