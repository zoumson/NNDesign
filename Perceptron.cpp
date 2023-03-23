#include "Perceptron.h"


namespace za
{
	namespace nn
	{
		Perceptron::Perceptron(size_t _inputs_size, double _bias)
		{
			__bias = _bias;
			__inputs_size = _inputs_size;
			//add 1 for bias
			__inputs_bias_size = __inputs_size + 1;			
			__inputs.resize(__inputs_size);
			__inputs_bias.resize(__inputs_bias_size);
			__weights.resize(__inputs_size);
			__weights_bias.resize(__inputs_bias_size);
			//randomize initial weigths for training 
			std::generate(__weights_bias.begin(), __weights_bias.end(), com::frand);
			__weights = { __weights_bias.begin(), __weights_bias.end() - 1 };
		};
		double Perceptron::run(const std::vector<double>& _inputs)
		{
			__inputs = _inputs;
			__inputs_bias = _inputs;
			__inputs_bias.push_back(__bias);
			__z = std::inner_product(__inputs_bias.begin(), __inputs_bias.end(), __weights_bias.begin(), (double)0.0);
			__output = com::sigmoid(__z);
			return __output;
		}
		void Perceptron::set_weights(const std::vector<double>& _weights)
		{
			__weights = _weights;
			__weights_bias = _weights;
			__weights_bias.emplace_back(com::frand());
		}				
		void Perceptron::set_weights_bias(const std::vector<double>& _weights_bias)
		{
			__weights_bias = _weights_bias;
			__weights = { __weights_bias.begin(), __weights_bias.end() - 1 };
		}
		void Perceptron::get_weights_bias(std::vector<double>& _weights_bias)
		{
			_weights_bias = __weights_bias;
		}
	}
}
