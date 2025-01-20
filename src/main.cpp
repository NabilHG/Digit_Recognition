#include <iostream>
#include <vector>
#include <memory>
#include "../include/mnist_reader.hpp"
#include "../include/mnist_utils.hpp"
#include "../src/neuralnetwork.cpp"  // Directly include the implementation file (not ideal but functional).
#include <typeinfo>
#include <algorithm>
#include <random>



using namespace std;

void validate(Network& network, const std::vector<std::vector<uint8_t>>& test_images, const std::vector<uint8_t>& test_labels) {
    double total_loss = 0.0;
    int correct_predictions = 0;

    for (size_t i = 0; i < test_images.size(); ++i) {
        network.forward(test_images[i]);

        total_loss += network.cross_entropy(static_cast<int>(test_labels[i]), network.get_layers()[network.get_layers().size() - 1]);

        // get predicted label (index of max output)
        auto output_neurons = network.get_layers()[network.get_layers().size() - 1]->get_neurons();
        int predicted_label = std::distance(output_neurons.begin(), std::max_element(output_neurons.begin(), output_neurons.end(),
                                            [](const auto& a, const auto& b) {
                                                return a->get_value() < b->get_value();
                                            }));

        // Check if the prediction matches the true label
        if (predicted_label == static_cast<int>(test_labels[i])) {
            ++correct_predictions;
        }
    }

    // compute metrics
    double average_loss = total_loss / test_images.size();
    double accuracy = static_cast<double>(correct_predictions) / test_images.size();
    
    std::cout << "Validation Loss: " << average_loss << "\n";
    std::cout << "Validation Accuracy: " << accuracy * 100 << "%\n";
}

void shuffle_dataset(std::vector<std::vector<uint8_t>>& images, std::vector<uint8_t>& labels) {
    auto rng = std::default_random_engine{};
    std::shuffle(images.begin(), images.end(), rng);
    std::shuffle(labels.begin(), labels.end(), rng);
}


int main() {
    
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data");
    
    // normalize all the images in the data set to a zero mean and unit variance.
    // mnist::normalize_dataset(dataset);

    Network network({784, 128, 10});

    int epochs = 100;
    int batch_size = 469; // 60000 / 128
    int index = 0;
    double loss = 0;
    for(int i = 1; i <= epochs; i++){
        shuffle_dataset(dataset.training_images, dataset.training_labels);

        for(int j = 0; j < batch_size && index < dataset.training_images.size(); ++j){
            //iterate the 60000 imagen in groups of 128
            for(int j = 0; j < batch_size && index < dataset.training_images.size(); ++j){
                network.forward(dataset.training_images[index]);
                loss += network.cross_entropy(static_cast<int>(dataset.training_labels[index]), network.get_layers()[network.get_layers().size() - 1]); // input: label, last layer
                network.backpropagation(static_cast<int>(dataset.training_labels[index]), network.get_layers()[network.get_layers().size() - 1]);
                index++;
            }
        }
        index = 0;
        std::cout << "Epoch " << i << "/" << epochs << '\n';
        validate(network, dataset.test_images, dataset.test_labels);

        if (i == epochs) { 
            network.save_model("trained_model.txt");
            std::cout << "Model saved to trained_model.txt\n";
        }

    }

    // reload the model for further evaluation
    // network.load_model("trained_model.txt");
    // validate(network, dataset.test_images, dataset.test_labels);
    
    return 0;
}

