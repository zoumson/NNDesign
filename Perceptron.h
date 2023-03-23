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

			double run(const std::vector<double>& _inputs);

			void set_weights(const std::vector<double>& _weights);
			void set_weights_bias(const std::vector<double>& _weights_bias);
			void set_inputs(const std::vector<double>& _inputs);
			void set_bias(const double& _bias_ini);
			void set_inputs_size(const size_t& _input_size);


			void get_weights(std::vector<double>& _weights);
			void get_inputs(std::vector<double>& _inputs);
			void get_inputs_bias(std::vector<double>& _inputs_bias);
			void get_weights_bias(std::vector<double>& _weights_bias);
			void get_bias(double& _bias);
			void get_z(double& _z);
			void get_output(double& _output);
			void get_inputs_size(size_t& _inputs_size);
			void get_inputs_bias_size(size_t& _inputs_bias_size);





		};
	}
}