#include <iostream>
#include <vector>
#include <memory>
#include "../include/mnist_reader.hpp"
#include "../include/mnist_utils.hpp"
#include "../src/neuralnetwork.cpp"  // Directly include the implementation file (not ideal but functional).

using namespace std;


int main() {

    // auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/home/digit_recognition/data");
    
    // // Normalize all the images in the data set to a zero mean and unit variance.
    // mnist::normalize_dataset(dataset);

    // std::cout << "Number of training images: " << dataset.training_images.size() << '\n';
    // std::cout << "Number of test images: " << dataset.test_images.size() << '\n';
    // std::cout << "sadasdasd";

    
    Network a({14, 5, 2});
    a.forward();



    // a.forward();
    return 0;
}

