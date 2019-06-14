#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "Image.h"

using namespace std;

void readMnistCSV(vector<Image> &images, string filename, int size){

    ifstream file;
    file.open(filename, ios::in);

    if(file.is_open()){

        vector<int> temp;
        for(int current_img = 0; current_img < size; current_img++){

            string line;
            string item;

            getline(file,line);
            stringstream stream(line);

            getline(stream,item,',');
            images.push_back(Image(stoi(item)));
            temp.clear();
            while(getline(stream,item,',')){

                temp.push_back(stoi(item));

            }

            images[current_img].read(temp);

        }

    }
    else
        cout << "Error opening " + filename << endl;

}

int main() {

    const int MNIST_TRAIN_SIZE = 60000;
    const int MNIST_TEST_SIZE = 10000;

    vector<Image> training_set;
    vector<Image> testing_set;
    readMnistCSV(training_set,"mnist_train.csv", MNIST_TRAIN_SIZE);
    readMnistCSV(testing_set,"mnist_test.csv", MNIST_TEST_SIZE);




    return 0;
}