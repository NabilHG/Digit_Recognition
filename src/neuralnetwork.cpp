#define _USE_MATH_DEFINES 
#include <iostream>
#include <vector>
#include <memory>
#include <ctime>   
#include <cmath>
#include <fstream>
#include <random>

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

        void relu(double &val) {
            this->value = std::max(0.0, val);
        }

        // derivate of Relu for backprop
        double relu_derivative(double val){
            return (val >= 0.0) ? 1.0 : 0.0;
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

        std::vector<std::shared_ptr<Neuron>>& get_neurons(){
            return this->neurons;
        }

};

class Network {
    private:
        std::vector<std::shared_ptr<Layer>> layers;
        std::vector<std::vector<std::vector<double>>> weights;  
        std::vector<std::vector<double>> biases;  

        std::vector<std::vector<std::vector<double>>> velocity_weights;
        std::vector<std::vector<double>> velocity_biases;  

        double momentum = 0.4; 
        double learning_rate = 0.0001; 

    public:
        Network(const std::vector<int>& layer_sizes) {
            for (int size : layer_sizes) {
                layers.push_back(std::make_shared<Layer>(size));
            }

            // init weights, biases and momentum
            srand(static_cast<unsigned>(time(0)));
            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                int current_layer_size = layer_sizes[i];
                int next_layer_size = layer_sizes[i + 1];

                // He Initialization
                double stddev = std::sqrt(2.0 / current_layer_size);
                std::vector<std::vector<double>> weight_matrix(current_layer_size, std::vector<double>(next_layer_size));
                std::vector<std::vector<double>> velocity_w(current_layer_size, std::vector<double>(next_layer_size, 0.0));

                for (auto& row : weight_matrix) {
                    for (auto& weight : row) {
                        weight = stddev * (static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0);
                    }
                }
                weights.push_back(weight_matrix);
                velocity_weights.push_back(velocity_w);

                
                biases.push_back(std::vector<double>(next_layer_size, 0.0));
                velocity_biases.push_back(std::vector<double>(next_layer_size, 0.0));
            }
        }


        std::vector<std::vector<std::vector<double>>> get_weights(){
            return this->weights;
        }

        std::vector<std::vector<double>> get_biases(){
            return this->biases;
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
            // create the transposed matrix with dimensions cols x rows
            std::vector<std::vector<double>> t_matrix(cols, std::vector<double>(rows));

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    t_matrix[j][i] = matrix[i][j];
                }
            }

            return t_matrix;
        }

        void forward(const std::vector<float>& input) {
            // feed values into first layer
            const auto& first_layer_neurons = this->layers[0]->get_neurons();
            for (size_t i = 0; i < input.size(); ++i) {
                first_layer_neurons[i]->set_value(static_cast<double>(input[i]));
            }

            // forward pass
            for (int i = 1; i < this->weights.size() + 1; i++) { 
                std::vector<std::vector<double>> t_matrix = transpose(this->weights[i - 1]);

                const auto& prev_layer_neurons = this->layers[i - 1]->get_neurons();
                const auto& next_layer_neurons = this->layers[i]->get_neurons();

                for (int j = 0; j < t_matrix.size(); j++) {
                    double total = 0;
                    for (int k = 0; k < t_matrix[j].size(); k++) {
                        if (k < prev_layer_neurons.size()) {
                            total += prev_layer_neurons[k]->get_value() * t_matrix[j][k];
                        }
                    }
                    total += this->biases[i - 1][j];

                    if (i < this->weights.size()) {
                        next_layer_neurons[j]->relu(total);
                    } else {
                        next_layer_neurons[j]->set_value(total); 
                    }
                }
            }

            // softmax last layer
            this->softmax(this->layers[this->layers.size() - 1]);
        }
        
        // apply softmax to last layer to convert logits into probability. Cross-Entropy require it
        void softmax(std::shared_ptr<Layer>& layer) {
            double max_val = -std::numeric_limits<double>::infinity();
            for (const auto& neuron : layer->get_neurons()) {
                if (neuron->get_value() > max_val) {
                    max_val = neuron->get_value();
                }
            }

            double sum_e_values = 0.0;
            for (auto& neuron : layer->get_neurons()) {
                neuron->set_value(exp(neuron->get_value() - max_val));
                sum_e_values += neuron->get_value();
            }

            for (auto& neuron : layer->get_neurons()) {
                neuron->set_value(neuron->get_value() / sum_e_values);
            }
        }

        void backpropagation(int input, std::shared_ptr<Layer>& output_layer) {
            // gradients last layer
            int index = 0;
            for (auto& neuron : output_layer->get_neurons()) {
                neuron->set_gradient((index == input) ? (neuron->get_value() - 1) : neuron->get_value());
                ++index;
            }

            // backpropagation
            for (int i = layers.size() - 2; i >= 0; --i) {
                auto& current_layer = layers[i + 1]->get_neurons();
                auto& prev_layer = layers[i]->get_neurons();
                auto& weights_matrix = weights[i];
                auto& biases_vector = biases[i];
                auto& velocity_w = velocity_weights[i];
                auto& velocity_b = velocity_biases[i];

                for (size_t j = 0; j < prev_layer.size(); ++j) {
                    double gradient_sum = 0.0;
                    //computing gradient for neuron j
                    for (size_t k = 0; k < current_layer.size(); ++k) {
                        double gradient = current_layer[k]->get_gradient() * prev_layer[j]->get_value();
                        
                        velocity_w[j][k] = momentum * velocity_w[j][k] - learning_rate * gradient;
                        gradient_sum += current_layer[k]->get_gradient() * weights_matrix[j][k];
                    }

                    for (size_t k = 0; k < current_layer.size(); ++k) {
                        weights_matrix[j][k] += velocity_w[j][k];
                    }
                    //setting gradient for neuron j
                    prev_layer[j]->set_gradient(gradient_sum * prev_layer[j]->relu_derivative(prev_layer[j]->get_value()));
                }


                for (size_t k = 0; k < current_layer.size(); ++k) {
                    double gradient = current_layer[k]->get_gradient();

                    // momentum into biases
                    velocity_b[k] = momentum * velocity_b[k] - learning_rate * gradient;
                    biases_vector[k] += velocity_b[k];
                }
            }
        }

        double cross_entropy(int target, std::shared_ptr<Layer>& layer) {
            double epsilon = 1e-15; 
            double predicted = layer->get_neurons()[target]->get_value();
            return -log(std::max(predicted, epsilon));
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