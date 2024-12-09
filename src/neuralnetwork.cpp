#include <iostream>
#include <vector>
#include <memory>

class Neuron{
    private:
        double value;
        double gradient;
        std::vector<std::shared_ptr<Neuron>> parents; // vector of pointers of parents neurons
    
    public:
        Neuron(double val = 0.0){
            this->value;
            this->gradient = 0;
        }   
    
        // ReLu (Rectified Linear Unit) for forward pass
        void relu() {
            value = std::max(0.0, value);
        }

        // derivate of Relu for backprop
        double relu_derivative(){
            return (value > 0.0) ? 1.0 : 0.0;
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
};

class Network{
    private:
        std::vector<std::shared_ptr<Layer>> layers;
        std::vector<std::vector<double>> weights; // matrix form
        std::vector<std::vector<double>> biases; // matrix form

    public:
        Network(const std::vector<int>& layer_sizes) { // Rename the parameter to `layer_sizes`
            // Create layers
            for (int size : layer_sizes) {
                this->layers.push_back(std::make_shared<Layer>(size));
            }

            // Initialize weights and biases
            // push a 2D vector into weights(1D)
        }

        void forward(){

        }

        void backpropagation(){

        }

        void update(){

        }
        
};