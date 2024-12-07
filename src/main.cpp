#include <iostream>
#include <vector>
#include <memory>
#include "../include/mnist_reader.hpp"
#include "../include/mnist_utils.hpp"

using namespace std;


int main() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/home/digit_recognition/data");

    std::cout << "Number of training images: " << dataset.training_images.size() << '\n';
    std::cout << "Number of test images: " << dataset.test_images.size() << '\n';

    return 0;
}

// class Value : public enable_shared_from_this<Value>{
// 	private:
// 		double value;
// 		vector<shared_ptr<Value>> parents; // Vector of smart pointers to Value objects, to avoid recursion problems

// 	public:
// 		Value(double val = 0){
// 			this->value = val;
// 		}

// 		double getValue() const{
// 			return value;
// 		}

// 		void setValue(double val) {			
// 			value = val;
// 		}

// 		void addParent(shared_ptr<Value> obj){
// 			parents.push_back(obj);
// 		}

// 		void getParents() {
// 			if (!this->parents.empty()) {
// 				for (const auto& parent : this->parents) {
// 					cout << "Value object: " << this->value << ", has parent Value object: " << parent->value << '\n';
// 					// Recursively call getParents on the parent
// 					parent->getParents();
// 				}
// 			} else {
// 				// Base case: No more parents
// 				cout << "Value object: " << this->value << " has no more parents." << endl;
// 			}
// 		}

// 		// overwriting operators
// 		shared_ptr<Value> operator+(Value& other) { // const for not modifying neither the other object nor this
// 			auto res = make_shared<Value>(this->value + other.value);
// 			// tracking this and other as parents
// 			res->addParent(shared_from_this());
// 			res->addParent(other.shared_from_this());
// 			return res;
// 		}

// 		shared_ptr<Value> operator-( Value& other) {
// 			auto res = make_shared<Value>(this->value - other.value); 
// 			// tracking this and other as parents
// 			res->addParent(shared_from_this());
// 			res->addParent(other.shared_from_this());
// 			return res;
// 		}

// 		shared_ptr<Value> operator*(Value& other) {
// 			auto res  = make_shared<Value>(this->value * other.value);
// 			// tracking this and other as parents
// 			res->addParent(shared_from_this());
// 			res->addParent(other.shared_from_this());
// 			return make_shared<Value>(this->value * other.value);
// 		}

// 		shared_ptr<Value> operator/(Value& other) {
// 			auto res = make_shared<Value>(0.0);
// 			if (other.value == 0){
// 				res->value = -1.0;
// 			} else {
// 				res->value = this->value / other.value;
// 			}
// 			// tracking this and other as parents
// 			res->addParent(shared_from_this());
// 			res->addParent(other.shared_from_this());
			
// 			return res;
// 		}

// 		shared_ptr<Value> pow(double base, double exp){
// 			double res {}, aux {base};

// 			for(int i {1}; i < exp; i++){
// 				aux = aux * base;
// 				res = aux;
// 				aux = res;	
// 			}
			
// 			return make_shared<Value>(res);
// 		}
// };




// int main(){
	
// 	int opt {};
// 	bool menu {true};

// 	 cout << "Hello user this is a calculator in c++" << '\n';

// 	while (menu) {
// 		cout << "Enter a desire option" << '\n';
// 		cout << "[1] Addition \n[2] Substraction \n[3] Multiplication \n[4] Division \n[5] Power \n[6] Exit" << '\n';
// 		cin >> opt;
// 		auto num1 = make_shared<Value>(); // auto to avoid writeing "shared_ptr<Value>""
// 		auto num2 = make_shared<Value>();
// 		auto res = make_shared<Value>();
// 		double input {};
// 		switch (opt){
// 			case 1:
// 				cout << "Enter the two numbers to be added:";
// 				cin >> input;
// 				num1->setValue(input);
// 				cin >> input; 	
// 				num2->setValue(input);
// 				res = *num1 + *num2; // dereferencing *var, to acces the variable not the pointer 
// 				cout << "Solution: " << res->getValue() << '\n';
// 				res->getParents();
// 				break;
// 			case 2:
// 				cout << "Enter the two numbers to be substracted:";
// 				cin >> input;
// 				num1->setValue(input);
// 				cin >> input; 	
// 				num2->setValue(input);
// 				res = *num1 - *num2; // dereferencing *var, to acces the variable not the pointer 
// 				cout << "Solution: " << res->getValue() << '\n';
// 				res->getParents();
// 				break;
// 			case 3:
// 				cout << "Enter the two numbers to be multiplied:";
// 				cin >> input;
// 				num1->setValue(input);
// 				cin >> input; 	
// 				num2->setValue(input);
// 				res = *num1 * *num2; // dereferencing *var, to acces the variable not the pointer 
// 				cout << "Solution: " << res->getValue() << '\n';
// 				res->getParents();
// 				break;
// 			case 4:
// 				cout << "Enter the two numbers to be divided:";
// 				cin >> input;
// 				num1->setValue(input);
// 				cin >> input; 	
// 				num2->setValue(input);
// 				res = *num1 / *num2; // dereferencing *var, to acces the variable not the pointer 
// 				cout << "Solution: " << res->getValue() << '\n';
// 				if (res->getValue() == -1) 	
// 					cout << "Can't divided zero" << '\n';
// 				else
// 					cout << "Solution: " << res->getValue() << '\n';
// 					res->getParents();
// 				break;
// 			case 5:
// 				cout << "Enter the two numbers to be exponentiated:";
// 				cin >> input;
// 				num1->setValue(input);
// 				cin >> input; 	
// 				num2->setValue(input);
// 				res = res->pow(num1->getValue(), num2->getValue()); 
// 				cout << "Solution: " << res->getValue() << '\n';
// 				res->getParents();
// 				break;
// 			case 6:
// 				menu = false;
// 				break;
// 			default:
// 				cout << "Wrong option try again!" << '\n';		
// 				break;
// 		}
// 	}

// 	return 0;
// }
