#define _USE_MATH_DEFINES 
#include <iostream>
#include <vector>
#include <memory>
#include <ctime>   
#include <cmath>
#include <fstream>

class Neuron{
    private:
        double value;
        double gradient;
        std::vector<std::shared_ptr<Neuron>> parents; // vector of pointers of parents neurons
    
    public:
        Neuron(double val = 0.0){
            this->value = val;
            this->gradient = 0;
        }   
    
        // ReLu (Rectified Linear Unit) for forward pass
        void relu(double &val) {
            this->value = std::max(0.1, val);
        }

        // derivate of Relu for backprop
        double relu_derivative(double val){
            return (val > 0.0) ? 1.0 : 0.1;
        }
        

        double get_value(){
            return this->value;
        }

        void set_value(double val){
            this->value = val;
        }

        double get_gradient(){
            return this->gradient;
        }

        void set_gradient(double grad){
            this->gradient = grad;
        }
};

class Layer{
    private:
        std::vector<std::shared_ptr<Neuron>> neurons;
    
    public:
        Layer(int size){
            for (int i = 0; i < size; i++){
                this->neurons.push_back(std::make_shared<Neuron>());
            }    
        }

        std::vector<std::shared_ptr<Neuron>> get_neurons(){
            return this->neurons;
        }

};

class Network{
    private:
        std::vector<std::shared_ptr<Layer>> layers;
        std::vector<std::vector<std::vector<double>>> weights; // matrix form
        std::vector<std::vector<double>> biases; // matrix form

    public:
        Network(const std::vector<int>& layer_sizes) { // Rename the parameter to `layer_sizes`
            // Create layers
            for (int size : layer_sizes) {
                // std::cout << size;
                this->layers.push_back(std::make_shared<Layer>(size));
            }

            // Seed the random number generator for reproducibility
            srand(static_cast<unsigned>(time(0)));

            // Initialize weights and biases for each layer connection
            for (size_t i = 0; i < layer_sizes.size()-1; ++i) {
                int current_layer_size = layer_sizes[i];     // Number of neurons in the current layer
                int next_layer_size = layer_sizes[i + 1];    // Number of neurons in the next layer
                // Initialize weight matrix for the current -> next layer
                // std::cout << "current_layer_size: " << current_layer_size << ", next_layer_size: " << next_layer_size << '\n';
                std::vector<std::vector<double>> weight_matrix(current_layer_size, std::vector<double>(next_layer_size));
                for (auto& row : weight_matrix) {
                    for (auto& weight : row) {
                        weight = static_cast<double>(rand()) / RAND_MAX; // Random value [0, 1) 
                    }
                }
                this->weights.push_back(weight_matrix); // Add the weight matrix for this layer connection
                // Initialize bias vector for the next layer
                std::vector<double> bias_vector(next_layer_size, 0.0); // A single vector of biases initialized to 0.0
                this->biases.push_back(bias_vector); // Add biases for this layer
            }
            // this->visualize_weights();
        }

        std::vector<std::vector<std::vector<double>>> get_weights(){
            return this->weights;
        }

        void visualize_weights(){
            for(auto matrix : this->weights){
                for(auto row : matrix){
                    for(auto weight : row){
                        std::cout << weight << " ";
                    }
                    std::cout << '\n';
                }
                std::cout << '\n';
            }
        }
        
        std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
            int rows = matrix.size();
            int cols = matrix[0].size();
            // std::cout << "n rows: " << rows << ", n cols: " << cols << '\n';
            // create the transposed matrix with dimensions cols x rows
            std::vector<std::vector<double>> t_matrix(cols, std::vector<double>(rows));

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    t_matrix[j][i] = matrix[i][j];
                }
            }
            // std::cout << "t_matrix, rows: " << t_matrix.size() << ", cols: " << t_matrix[0].size() << '\n'; 
            return t_matrix;
        }

       void forward(const std::vector<uint8_t>& input) {
            //init first layer values
            // std::cout << "SIZE: " << input.size() << '\n';
            // for each layer, compute the weighted sum ((value * weight) + bias), then apply ReLU
            const auto& first_layer_neurons = this->layers[0]->get_neurons();
            for (size_t i = 0; i < input.size(); ++i) {
                first_layer_neurons[i]->set_value(static_cast<double>(input[i]));  // Set input value to the first layer's neurons
                // std::cout << "Neuron " << i << " value: " << first_layer_neurons[i]->get_value() << '\n';

            }
            
            for (int i = 1; i < this->weights.size() + 1; i++) { // start from i=1 to avoid out-of-bounds access
                std::vector<std::vector<double>> t_matrix = transpose(this->weights[i - 1]);
                
                // Access the previous and next layer's neurons for efficiency
                const auto& prev_layer_neurons = this->layers[i - 1]->get_neurons();
                const auto& next_layer_neurons = this->layers[i]->get_neurons();
                // std::cout << "Weight matrix size: " << t_matrix.size() << " x " << t_matrix[0].size() << '\n';
            
                // Ensure bounds check when accessing prev_layer_neurons and weights
                for (int j = 0; j < t_matrix.size(); j++) {
                    double total {0};
                    for (int k = 0; k < t_matrix[j].size(); k++) {
                        if (k < prev_layer_neurons.size()) {
                            // std::cout << "valor anterior neurona " << prev_layer_neurons[k]->get_value() << '\n';
                            // std::cout << "weight " << t_matrix[j][k] << '\n';
                            total += prev_layer_neurons[k]->get_value() * t_matrix[j][k]; // value * weight
                            // std::cout << "total " << total << '\n';
                        }
                    }
                    total += this->biases[i - 1][j];
                    // std::cout << "bias " << this->biases[i - 1][j] << '\n';
                    // std::cout << "antes de aplicar relu " << total << '\n';
                    next_layer_neurons[j]->relu(total);
                    // std::cout << "despues de aplicar relu " << next_layer_neurons[j]->get_value() << '\n';
                }

                // std::cout << '\n';
            }

            //apply softmax
            this->softmax(this->layers[this->layers.size()-1]); 

        }


        void backpropagation(int input, std::shared_ptr<Layer>& layer){
            double learning_rate = 0.001;

            //computing gradients in the output layer
            int index = 0;
            for (auto& neuron : layer->get_neurons()) {  
                if (index == input) {
                    // Calculate and set the gradient for the correct class neuron
                    double gradient = neuron->get_value() - 1;
                    neuron->set_gradient(gradient);
                }
                ++index;
            }

            // backpropagate, compute gradients in previous layer (except input layer)  
            // for each neuron, sum of all weigth * gradient * derivate ReLu(pre-activation value)
            // also update weights and biases
            for (int i = this->layers.size() - 2; i >= 0; --i) {  // Start from the second-to-last layer
                const auto& current_layer_neurons = this->layers[i + 1]->get_neurons();
                const auto& prev_layer_neurons = this->layers[i]->get_neurons();
                auto& weights_matrix = this->weights[i];
                auto& biases = this->biases[i];

                for (int j = 0; j < prev_layer_neurons.size(); ++j) {
                    double gradient_sum = 0.0;

                    // sum gradients from the next layer
                    for (int k = 0; k < current_layer_neurons.size(); ++k) {
                        gradient_sum += current_layer_neurons[k]->get_gradient() * weights_matrix[k][j];
                        //updating weight and biases
                        weights_matrix[j][k] = weights_matrix[j][k] - learning_rate * current_layer_neurons[k]->get_gradient() * prev_layer_neurons[j]->get_value();
                        biases[k] = biases[k] - learning_rate * current_layer_neurons[k]->get_gradient();
                    }

                    // multiply by ReLU derivative of the activation
                    // std::cout << "gradient before relu derivative" << gradient_sum << '\n';
                    gradient_sum *= prev_layer_neurons[j]->relu_derivative(prev_layer_neurons[j]->get_value());
                    // std::cout << "gradient after relu derivative" << gradient_sum << '\n';
                    // std::cout << "gradient before setting new one" << prev_layer_neurons[j]->get_gradient() << '\n';
                    prev_layer_neurons[j]->set_gradient(gradient_sum);
                }
            }

            // std::cout << "no problems" << '\n';

        }

        double cross_entropy(int input, std::shared_ptr<Layer>& layer){
            double epsilon = 1e-15; // adding a very small number just for when value 0 not having problems
            return log(layer->get_neurons()[input]->get_value() + epsilon) * (-1);
        }

        // apply softmax to last layer to convert logits into probability. Cross-Entropy require it
        void softmax(std::shared_ptr<Layer>& layer){
            // std::cout << layer->get_neurons()[0]->get_value() << '\n';

            // lambda function to compute the sum of all exponentiated values on this layer (e^value)
            auto total_e_values = [](std::shared_ptr<Layer>& layer) -> double {
                double total {0};
                for (const auto& neuron : layer->get_neurons()) {
                    total += exp(neuron->get_value());  
                }
                return total;
            };

            double sum_e_values = total_e_values(layer);

            // apply softmax to each neuron
            for (auto& neuron : layer->get_neurons()) {
                double exponentiated_value = exp(neuron->get_value());  
                neuron->set_value(exponentiated_value / sum_e_values);  // normalize by sum of e^values
            }
        }

        void save_model(const std::string& filename) {
            std::ofstream file(filename, std::ios::out);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for saving the model.");
            }

            // save architecture
            file << layers.size() << '\n'; 

            for (const auto& layer : layers) {
                file << layer->get_neurons().size() << ' '; 
            }

            file << '\n';

            // save weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                for (const auto& row : weights[i]) {
                    for (double weight : row) {
                        file << weight << ' ';
                    }
                    file << '\n';
                }
                for (double bias : biases[i]) {
                    file << bias << ' ';
                }
                file << '\n';
            }

            file.close();
        }
        
        void load_model(const std::string& filename) {
            std::ifstream file(filename, std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for loading the model.");
            }

            // load architecture
            size_t num_layers;
            file >> num_layers;
            std::vector<int> architecture(num_layers);
            for (size_t i = 0; i < num_layers; ++i) {
                file >> architecture[int(i)];
            }

            // Reinitialize the network with the loaded architecture
            *this = Network(architecture);

            // load weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                for (auto& row : weights[i]) {
                    for (double& weight : row) {
                        file >> weight;
                    }
                }
                for (double& bias : biases[i]) {
                    file >> bias;
                }
            }

            file.close();
        }

        std::vector<std::shared_ptr<Layer>> get_layers(){
            return this->layers;
        }
};