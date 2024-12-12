#include <iostream>
#include <vector>
#include <memory>
#include <ctime>   

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
        void relu() {
            value = std::max(0.0, value);
        }

        // derivate of Relu for backprop
        double relu_derivative(){
            return (value > 0.0) ? 1.0 : 0.0;
        }

        double get_value(){
            return this->value;
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
            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                int current_layer_size = layer_sizes[i];     // Number of neurons in the current layer
                int next_layer_size = layer_sizes[i + 1];    // Number of neurons in the next layer
                // Initialize weight matrix for the current -> next layer
                std::vector<std::vector<double>> weight_matrix(current_layer_size, std::vector<double>(next_layer_size));
                for (auto& row : weight_matrix) {
                    for (auto& weight : row) {
                        weight = static_cast<double>(rand()) / RAND_MAX; // Random value [0, 1) 
                    }
                }
                this->weights.push_back(weight_matrix); // Add the weight matrix for this layer connection
                // Initialize bias vector for the next layer
                std::vector<double> bias_vector(next_layer_size, 0.0); // Biases initialized to 0.0 for simplicity
                this->biases.push_back(bias_vector); // Add biases for this layer
            }
            // std::cout << this->get_weights()[0][0][0] << '\n';
            // std::cout << '\n';

            this->visualize_weights();
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
            // create the transposed matrix with dimensions cols x rows
            std::vector<std::vector<double>> t_matrix(cols, std::vector<double>(rows));

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    t_matrix[j][i] = matrix[i][j];
                }
            }
            return t_matrix;
        }

        // in this function we are going to propagate the values from the input layer to the output layer
        //multiplying each value for the correspongind weight and the adding the bias 
        //to finally call the activation function (relu) and repeat for the next layer
        void forward(){
            //possible approach?
            //for each column of the weights matrix compute value*weight + bias, N1*w0,1+N2*w0,1+...+Nnw0,n + B (bias)
            //then call RelU(val) on the corresponding neuron of the column n in the weight matrix (n=0, first neuron of current_layer +1)     


            for(int i = 0; i < this->weights.size(); i++){
                // transpose to iterate directly into weights of neuron of the next layer
                std::vector<std::vector<double>> t_matrix = transpose(this->weights[i]); 
                for(int j = 0; j < t_matrix.size(); j++){
                    for(int z = 0; z < t_matrix[j].size(); z++){
                        std::cout << t_matrix[j][z] << " ";
                    }
                    std::cout << '\n';
                }
                std::cout << '\n';
            }

        }

        void backpropagation(){

        }

        void update(){

        }
        
};