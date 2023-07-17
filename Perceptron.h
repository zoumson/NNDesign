#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include "NNUtilities.h"

namespace za
{
	namespace nn
	{
		class Perceptron
		{
		private:

			std::vector<double> __weights;
			std::vector<double> __weights_bias;
			std::vector<double> __inputs;
			std::vector<double> __inputs_bias;
			double __output;
			size_t __inputs_size;
			size_t __inputs_bias_size;
			double __z;
			double __bias;

		public:

			Perceptron(size_t _inputs_size, double _bias = 1.0);

			//sum of weighted input to the perceptron fed to the sigmoid function as 
			//the activation function 
			//nucleus performing calculation to yield an output 
			double run(const std::vector<double>& _inputs);

			//set initial weight linking input data to current perceptron 
			//dendrite to nucleus 
			void set_weights(const std::vector<double>& _weights);

			void set_weights_bias(const std::vector<double>& _weights_bias);

			//setting the input data of the current percetron 
			void set_inputs(const std::vector<double>& _inputs);

			//by default the bias is 1, a default input that helps to  adjust the 
			//decision boundary when the later is a line 
			//otherwise the line just pass throught the origins 
			void set_bias(const double& _bias_ini);

			//the input size/dendrtite/ramification of the current neurons 
			void set_inputs_size(const size_t& _input_size);

			//getting the weight entering current perceptron 
			void get_weights(std::vector<double>& _weights);

			//get current perceptron input 
			void get_inputs(std::vector<double>& _inputs);
			void get_inputs_bias(std::vector<double>& _inputs_bias);
			void get_weights_bias(std::vector<double>& _weights_bias);
			//bias always 1, but weight is not 
			void get_bias(double& _bias);

			//output obtained for applying the activation sigmoid function on the sum 
			//of the weighted sum of incoming input 
			void get_z(double& _z);

			//get output of the neuron 
			void get_output(double& _output);

			//get input data
			void get_inputs_size(size_t& _inputs_size);
			void get_inputs_bias_size(size_t& _inputs_bias_size);





		};
	}
}