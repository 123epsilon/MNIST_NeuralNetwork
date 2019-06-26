#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(int hl, int hn, double lr) {
    num_hidden_layers = hl;
    num_hidden_layer_nodes = hn;
    learning_rate = lr;

    randomInit();

}

void NeuralNetwork::randomInit() {
    //init hidden layers
    for(int i = 0; i < num_hidden_layers; i++){
        hidden_layers.push_back(vector<double>());
        bias_nodes.push_back(vector<double>());
        for(int j = 0; j < num_hidden_layer_nodes; j++){
            hidden_layers[i].push_back(0);
            bias_nodes[i].push_back(((double) rand() / (RAND_MAX)));
        }
    }

    bias_nodes.push_back(vector<double>());

    //init input, output
    for(int i = 0; i < 784; i++){
        input_layer.push_back(0);
        if(i < 10) {
            output_layer.push_back(0);
            bias_nodes[bias_nodes.size()-1].push_back(((double) rand() / (RAND_MAX)));
        }
    }

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

void NeuralNetwork::think(vector<Image> images) {
    for(int current_img = 0; current_img < images.size(); current_img++) {
        //load input layer with pixel values
        for (int i = 0; i < 784; i++) {
            input_layer[i] = images[current_img].get(i);
        }

        forwardpropogate();
        int max = 0;
        double error = 0;
        double expected;
        for(int i = 0; i < output_layer.size(); i++){
            if(output_layer[i] > output_layer[max]) max = i;
            i == images[current_img].getLabel() ? expected = 1.0 : expected = 0.0;
            error += 0.5 * pow(expected - sigmoid(output_layer[i]),2);
            cout << sigmoid(output_layer[i]) << " ; ";
        }
        cout << "Label: " << images[current_img].getLabel() << " | ";
        cout << "\nPrediction: " << max << " | Error: " << error << endl;
    }
}

void NeuralNetwork::train(vector<Image> images, int iterations) {
    for(int x = 0; x < iterations; x++){
        for(int current_img = 0; current_img < images.size(); current_img++) {
            //load input layer with pixel values
            for (int i = 0; i < 784; i++) {
                input_layer[i] = images[current_img].get(i);
            }

            forwardpropogate();
            backpropogate(images[current_img].getLabel());
        }
    }
}

void NeuralNetwork::forwardpropogate() {
    //propogate fwd from input to first hidden layer
    for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
       //reset this node's value
        hidden_layers[0][hidden_node] = 0;
        for(int input_node = 0; input_node < input_layer.size(); input_node++){
            //fill with sum of weights * inputs + bias
            hidden_layers[0][hidden_node] += (sigmoid(input_layer[input_node]) * weight_layers[0][input_node][hidden_node])
                    + bias_nodes[0][hidden_node];
        }
    }

    //propogate between all hidden layers, from first hidden layer to last
    for(int second_layer = 1; second_layer < num_hidden_layers; second_layer++){
        for(int second_layer_node = 0; second_layer_node < num_hidden_layer_nodes; second_layer_node++){
            //reset this node's value
            hidden_layers[second_layer][second_layer_node] = 0;
            for(int first_layer_node = 0; first_layer_node < num_hidden_layer_nodes; first_layer_node++){
                hidden_layers[second_layer][second_layer_node] += (sigmoid(hidden_layers[second_layer-1][first_layer_node]) * weight_layers[second_layer][first_layer_node][second_layer_node])
                        + bias_nodes[second_layer][second_layer_node];
            }
        }
    }

    //propogate from last hidden layer to the output layer
    for(int output_node = 0; output_node < output_layer.size(); output_node++){
        //reset this nodes value
        output_layer[output_node] = 0;
        for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
            output_layer[output_node] += (sigmoid(hidden_layers[hidden_layers.size()-1][hidden_node]) * weight_layers[num_hidden_layers][hidden_node][output_node])
                    + bias_nodes[bias_nodes.size()-1][output_node];
        }
    }
}

void NeuralNetwork::backpropogate(int label) {
    //calculate cost using mean squared for output layer (BP1)
    vector<double> output_error;
    for(int i = 0; i < 10; i++){
        double activation = sigmoid(output_layer[i]);
        if(i == label) {
            output_error.push_back( activation - 1.0 );
        } else {
            output_error.push_back( activation );
        }
        //multiply by derivative of sigmoid at this point
        output_error[i] *= ( activation * ( 1.0 - activation ) );
    }

    vector<vector<double>> hidden_error(num_hidden_layers);
    //propogate error backwards from output to last hidden layer (BP2)
    for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
        hidden_error[hidden_error.size()-1].push_back(0);
        double activation = sigmoid(hidden_layers[hidden_layers.size()-1][hidden_node]);
        for(int output_error_node = 0; output_error_node < output_error.size(); output_error_node++){
            hidden_error[hidden_error.size()-1][hidden_node] +=
                    ( ( weight_layers[weight_layers.size()-1][hidden_node][output_error_node] * output_error[output_error_node] )
                    * ( activation * ( 1.0 - activation ) ) );
        }
    }

    //propogate error backwards from last hidden layer to first hidden layer
    for(int layer = num_hidden_layers-2; layer >= 0; layer--){
        for(int layer1node = 0; layer1node < num_hidden_layer_nodes; layer1node++){
            hidden_error[layer].push_back(0);
            double activation = sigmoid(hidden_layers[layer][layer1node]);
            for(int layer2node = 0; layer2node < num_hidden_layer_nodes; layer2node++){
                hidden_error[layer][layer1node] +=
                        ( ( weight_layers[layer+1][layer1node][layer2node] * hidden_error[layer+1][layer2node] )
                        * ( activation * ( 1.0 - activation ) ) );
            }
        }
    }

    //(BP3,BP4)
    //adjust weights and biases from input to first hidden layer
    for(int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++){
        //adjust weights
        for(int input_node = 0; input_node < input_layer.size(); input_node++){
            weight_layers[0][input_node][hidden_node] -=
                    learning_rate * sigmoid(input_layer[input_node]) * hidden_error[0][hidden_node];
        }
        //adjust biases
        bias_nodes[0][hidden_node] -= learning_rate * hidden_error[0][hidden_node];
    }

    //adjust weights and biases from 1st hidden layer to last hidden layer
    for(int weight_layer = 1; weight_layer < weight_layers.size()-1; weight_layer++){
        for(int layer2node = 0; layer2node < num_hidden_layer_nodes; layer2node++){
            //adjust weights
            for(int layer1node = 0; layer1node < num_hidden_layer_nodes; layer1node++){
                weight_layers[weight_layer][layer1node][layer2node] -=
                        learning_rate * sigmoid(hidden_layers[weight_layer-1][layer1node]) * hidden_error[weight_layer][layer2node];
            }
            //adjust biases
            bias_nodes[weight_layer][layer2node] -= learning_rate * hidden_error[weight_layer][layer2node];
        }
    }

    //adjust weights and biases from last hidden layer to output layer
    for(int output_node = 0; output_node < output_layer.size(); output_node++) {
        //adjust weights
        for (int hidden_node = 0; hidden_node < num_hidden_layer_nodes; hidden_node++) {
            weight_layers[weight_layers.size() - 1][hidden_node][output_node] -=
                    learning_rate * sigmoid(hidden_layers[hidden_layers.size() - 1][hidden_node]) *
                    output_error[output_node];
        }
        //adjust biases
        bias_nodes[bias_nodes.size() - 1][output_node] -= learning_rate * output_error[output_node];
    }
}