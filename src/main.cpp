#include <iostream>
#include <vector>
#include <memory>
#include "../include/mnist_reader.hpp"
#include "../include/mnist_utils.hpp"
#include "../src/neuralnetwork.cpp"  // Directly include the implementation file (not ideal but functional).
#include <typeinfo>
using namespace std;


int main() {

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/home/nbl/Digit_Recognition/data");
    
    // // // Normalize all the images in the data set to a zero mean and unit variance.
    mnist::normalize_dataset(dataset);

    // std::cout << "Number of training images: " << dataset.training_images.size() << '\n';
    // std::cout << static_cast<int>(dataset.training_labels[0]) << '\n';



    
    
    Network a({10, 5, 2});
    // a.forward(dataset.training_images[0]);
    a.backpropagation(static_cast<int>(dataset.training_labels[10]), a.get_layers()[a.get_layers().size() - 1]);



    // loss = a.cross_entropy(static_cast<int>(dataset.training_labels[10]), a.get_layers()[a.get_layers().size() - 1]); // input: label, last layer



    // a.forward();
    return 0;
}

