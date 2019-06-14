#include <vector>
#include <string>
#pragma once

using namespace std;

class Image {

    int label;
    vector<int> pixels;

public:

    Image(int l);
    int getLabel();
    void read(vector<int> input);
    int get(int index);

};
