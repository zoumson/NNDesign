#pragma once
#include "Perceptron.h"

namespace za
{
	namespace nn
	{

		class MultiLayerPerceptron
		{
		private:
			std::vector<size_t> __layers_nodes_num;
			size_t __layers_num;
			double __bias;
			double __learn_rate;
			double __mse;
			std::vector<std::vector<Perceptron>> __network;
			std::vector<std::vector<double>> __output_node_vals;
			std::vector<std::vector<double>> __d;
			std::vector < std::vector<std::vector<double>>> __weights_bias;

		public:

			MultiLayerPerceptron(std::vector<size_t> _layers_nodes_num, double _bias = 1.0, double _learn_rate = 0.5);
			std::vector<double> run(std::vector<double>& _inputs);
			std::vector<double> run(std::vector<double>&& _inputs);
			double train(std::vector < std::vector<double>>& _inputs, std::vector < std::vector<double>>& _labels);
			double train(std::vector < std::vector<double>>&& _inputs, std::vector < std::vector<double>>&& _labels);
			std::vector<double> train_epoch(std::vector < std::vector<double>>& _inputs, std::vector < std::vector<double>>& _labels, size_t& _epoch);
			std::vector<double> train_epoch(std::vector < std::vector<double>>&& _inputs, std::vector < std::vector<double>>&& _labels, size_t&& _epoch);
			double back_pro(std::vector<double> _inputs, std::vector<double> _labels);
			void set_weights(const std::vector<std::vector<double>>& _weights);
			void set_weights_bias(const std::vector < std::vector<std::vector<double>>>& _weights_bias);
			void set_inputs(const std::vector<double>& _inputs);
			void set_bias(const double& _bias_ini);
			void set_inputs_size(const size_t& _input_size);


			void get_weights(std::vector < std::vector<double>>& _weights);
			void get_inputs(std::vector<double>& _inputs);
			void get_inputs_bias(std::vector<double>& _inputs_bias);
			void get_weights_bias(std::vector<std::vector<double>> & _weights_bias);
			void get_bias(double& _bias);
			void get_output(std::vector<double>& _output);


			void print_weights();
			void print_weights_bias();



		};
	}
}