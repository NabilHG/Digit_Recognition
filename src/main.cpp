#include <iostream>
#include <vector>
#include "../include/mnist_reader.hpp"
#include "../include/mnist_utils.hpp"
#include "../src/neuralnetwork.cpp"  // Directly include the implementation file (not ideal but functional).
#include <algorithm>
#include <random>

void shuffle_dataset(std::vector<std::vector<float>>& images, std::vector<uint8_t>& labels) {
    std::vector<std::pair<std::vector<float>, uint8_t>> combined_dataset;
    for (size_t i = 0; i < images.size(); ++i) {
        combined_dataset.emplace_back(images[i], labels[i]);
    }

    auto rng = std::default_random_engine{};
    std::shuffle(combined_dataset.begin(), combined_dataset.end(), rng);

    for (size_t i = 0; i < combined_dataset.size(); ++i) {
        images[i] = combined_dataset[i].first;
        labels[i] = combined_dataset[i].second;
    }
}

void validate(Network& network, const std::vector<std::vector<float>>& test_images, const std::vector<uint8_t>& test_labels) {
    double total_loss = 0.0;
    int correct_predictions = 0;

    for (size_t i = 0; i < test_images.size(); ++i) {
        network.forward(test_images[i]);

        total_loss += network.cross_entropy(static_cast<int>(test_labels[i]), network.get_layers().back());

        // obatining the index of the prediction
        auto output_neurons = network.get_layers().back()->get_neurons();
        int predicted_label = std::distance(output_neurons.begin(), std::max_element(output_neurons.begin(), output_neurons.end(),
                                              [](const auto& a, const auto& b) {
                                                  return a->get_value() < b->get_value();
                                              }));

        if (predicted_label == static_cast<int>(test_labels[i])) {
            ++correct_predictions;
        }
    }

    double average_loss = total_loss / test_images.size();
    double accuracy = static_cast<double>(correct_predictions) / test_images.size();

    std::cout << "  Loss: " << average_loss << "\n";
    std::cout << "  Accuracy: " << accuracy * 100 << "%\n";
}

int main() {

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data");

    // normalizing images [0, 1]
    std::vector<std::vector<float>> normalized_images;
    for (const auto& image : dataset.training_images) {
        std::vector<float> normalized_image;
        for (uint8_t pixel : image) {  
            normalized_image.push_back(static_cast<float>(pixel) / 255.0f);
        }
        normalized_images.push_back(std::move(normalized_image));
    }



    std::vector<std::vector<float>> normalized_images_test;
    for (const auto& image : dataset.test_images) {
        std::vector<float> normalized_image_test;
        for (uint8_t pixel : image) { 
            normalized_image_test.push_back(static_cast<float>(pixel) / 255.0f);
        }
        normalized_images_test.push_back(std::move(normalized_image_test));
    }

    // init network
    Network network({784, 30, 30, 10});

    int epochs = 40;
    int batch_size = 250;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        shuffle_dataset(normalized_images, dataset.training_labels);
        std::cout << "Epoch " << epoch << "/" << epochs << '\n';

        for (size_t batch_start = 0; batch_start < normalized_images.size(); batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, normalized_images.size());

            for (size_t i = batch_start; i < batch_end; ++i) {
                // forward pass
                network.forward(normalized_images[i]);

                // backpropagation
                network.backpropagation(static_cast<int>(dataset.training_labels[i]), network.get_layers().back());
            }
        }

        // validation after each epoch
        validate(network, normalized_images_test, dataset.test_labels);

        // saving model
        if (epoch == epochs) {
            network.save_model("trained_model.txt");
            std::cout << "Model saved to trained_model.txt\n";
        }
    }

    return 0;
}