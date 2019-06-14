#include "Image.h"

Image::Image(int l) { label = l; }

int Image::getLabel() { return label; }

int Image::get(int index) { return pixels[index]; }

void Image::read(vector<int> input) {
    for(int i = 0; i < input.size(); i++){
        pixels.push_back(input[i]);
    }
}