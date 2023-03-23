#include "MultiLayerPerceptron.h"


namespace za
{
	namespace nn
	{
		//always remove the default values from implementation
		MultiLayerPerceptron::MultiLayerPerceptron(std::vector<size_t> _layers_nodes_num, double _bias, double _learn_rate)
		{
			__layers_nodes_num = _layers_nodes_num;
			__bias = _bias;
			__learn_rate = _learn_rate;
			__layers_num = _layers_nodes_num.size();
			size_t layers_nodes_num_curr = 0;
			for (size_t layer_num_i = 0; layer_num_i < __layers_num; layer_num_i++)
			{
				layers_nodes_num_curr = __layers_nodes_num[layer_num_i];
				__output_node_vals.push_back(std::vector<double>(layers_nodes_num_curr, 0.0));
				__d.push_back(std::vector<double>(layers_nodes_num_curr, 0.0));
				__network.push_back(std::vector<Perceptron>());
				//input layer has no nodes
				if (layer_num_i > 0)
				{

					for (size_t layer_num_i_node_j = 0; layer_num_i_node_j < layers_nodes_num_curr; layer_num_i_node_j++)
					{
						__network[layer_num_i].push_back(Perceptron(__layers_nodes_num[layer_num_i - 1], __bias));
					}
				}
			}
		}
		std::vector<double> MultiLayerPerceptron::run(std::vector<double>& _inputs)
		{
			__output_node_vals[0] = _inputs;
			for (size_t i = 1; i < __layers_num; i++)
			{
				for (size_t j = 0; j < __layers_nodes_num[i]; j++)
				{

					__output_node_vals[i][j] = __network[i][j].run(__output_node_vals[i - 1]);
				}
			}
			return __output_node_vals.back();
		};
		void MultiLayerPerceptron::set_weights_bias(const std::vector < std::vector<std::vector<double>>>& _weights_bias)
		{
			__weights_bias = _weights_bias;
			size_t layers_num_loc = __weights_bias.size();
			size_t layer_nodes_num_loc = 0;
			for (size_t layers_num_loc_i = 0; layers_num_loc_i < layers_num_loc; layers_num_loc_i++)
			{
				layer_nodes_num_loc = __weights_bias[layers_num_loc_i].size();
				for (size_t layer_nodes_num_loc_j = 0; layer_nodes_num_loc_j < layer_nodes_num_loc; layer_nodes_num_loc_j++)
				{
					//First layer is input layer, no weight
					__network[layers_num_loc_i + 1][layer_nodes_num_loc_j].set_weights_bias(__weights_bias[layers_num_loc_i][layer_nodes_num_loc_j]);
				}
			}
		}
		void MultiLayerPerceptron::print_weights_bias()
		{
			std::cout << std::endl;
			std::vector<double> weights_bias_node;
			for (size_t i = 1; i < __network.size(); i++)
			{
				for (size_t j = 0; j < __layers_nodes_num[i]; j++)
				{
					std::cout << "Layer " << i + 1 << " Neuron " << j << ": ";
					__network[i][j].get_weights_bias(weights_bias_node);
					for (auto& it : weights_bias_node)
					{
						std::cout << it << "  ";
					}
					std::cout << std::endl;
				}
			}

		}
		double MultiLayerPerceptron::back_pro(std::vector<double> _inputs, std::vector<double> _labels)
		{
			//Backpropagagtion
			//Step 1: Feed a sample to the network
			std::vector<double> outputs = run(_inputs);

			//Step 2: Calculate the MSE
			std::vector<double> err;
			double MSE = 0.0;

			for (size_t i = 0; i < _labels.size(); i++)
			{
				err.push_back(_labels[i] - outputs[i]);
				MSE += err[i] * err[i];
			}

			//Divide by the number of neuron on output layer
			MSE /= __layers_nodes_num.back();

			//Step 3: Calculate the output error terms
			for (size_t i = 0; i < outputs.size(); i++)
			{
				__d.back()[i] = outputs[i] * (1 - outputs[i]) * (err[i]);
			}

			//Step 4: Calculate the error term of each unit on each layer
			// //backward
			// input layer and output layer excluded
			std::vector<double> weights_bias_node;
			for (size_t i = __layers_num - 2; i > 0; i--)
			{
				//hidden layers 
				for (size_t j = 0; j < __layers_nodes_num[i]; j++)
				{
					// A node at a time
					// Next layer weights linked to current node
					// Next layer error node linked to current 
					// Sum
					double fwd_err = 0;
					for (size_t k = 0; k < __layers_nodes_num[i + 1]; k++)
					{
						__network[i + 1][k].get_weights_bias(weights_bias_node);
						fwd_err += weights_bias_node[j] * __d[i + 1][k];
					}

					__d[i][j] = __output_node_vals[i][j] * (1 - __output_node_vals[i][j]) * fwd_err;
				}
			}

			// Step 5 & 6: Calculate the deltas and update the weights

			for (size_t i = 1; i < __layers_num; i++)
			{
				for (size_t j = 0; j < __layers_nodes_num[i]; j++)
				{
					//+1 for bias
					std::vector<double> weights_bias_node_curr;
					std::vector<double> weights_bias_node_next;

					__network[i][j].get_weights_bias(weights_bias_node_curr);
					for (size_t k = 0; k < __layers_nodes_num[i - 1] + 1; k++)
					{

						double delt;


						if (k == __layers_nodes_num[i - 1])
						{
							delt = __learn_rate * __d[i][j] * __bias;
						}
						else
						{
							delt = __learn_rate * __d[i][j] * __output_node_vals[i - 1][k];
						}

						weights_bias_node_next.push_back(weights_bias_node_curr[k] + delt);
					}
					__network[i][j].set_weights_bias(weights_bias_node_next);
				}
			}
			return MSE;
		}
		std::vector<double> MultiLayerPerceptron::run(std::vector<double>&& _inputs)
		{
			__output_node_vals[0] = _inputs;
			for (size_t i = 1; i < __layers_num; i++)
			{
				for (size_t j = 0; j < __layers_nodes_num[i]; j++)
				{

					__output_node_vals[i][j] = __network[i][j].run(__output_node_vals[i - 1]);
				}
			}
			return __output_node_vals.back();
		};
		double MultiLayerPerceptron::train(std::vector < std::vector<double>>& _inputs, std::vector < std::vector<double>>& _labels)
		{
			double mse_curr = 0;

			for (size_t i = 0; i < _inputs.size(); i++)
			{
				auto x = _inputs[i];
				auto y = _labels[i];
				mse_curr += back_pro(x, y);
			}
			mse_curr /= _inputs.size();
			return mse_curr;
		};		
		double MultiLayerPerceptron::train(std::vector < std::vector<double>>&& _inputs, std::vector < std::vector<double>>&& _labels)
		{
			double mse_curr = 0;

			for (size_t i = 0; i < _inputs.size(); i++)
			{
				auto x = _inputs[i];
				auto y = _labels[i];
				mse_curr += back_pro(x, y);
			}
			mse_curr /= _inputs.size();
			return mse_curr;
		};		
		std::vector<double> MultiLayerPerceptron::train_epoch(std::vector < std::vector<double>>& _inputs, std::vector < std::vector<double>>& _labels, size_t& _epoch)
		{
			double mse_curr = 0;
			std::vector<double> mse_all;

			for (size_t i = 0; i < _epoch; i++)
			{
				mse_curr = train(_inputs, _labels);
				mse_all.push_back(mse_curr);
			}

			return mse_all;
		};
		std::vector<double> MultiLayerPerceptron::train_epoch(std::vector < std::vector<double>>&& _inputs, std::vector < std::vector<double>>&& _labels, size_t&& _epoch)
		{
			double mse_curr = 0;
			std::vector<double> mse_all;

			for (size_t i = 0; i < _epoch; i++)
			{
				mse_curr = train(_inputs, _labels);
				mse_all.push_back(mse_curr);
			}

			return mse_all;
		};

	}
}