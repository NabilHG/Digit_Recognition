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
