#include "NNModels.h"

namespace za
{
	namespace nn
	{
		namespace mm
		{
			void ANDGate()
			{
				std::cout << "\n\n -----------Hardcoded AND Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::Perceptron* p = new za::nn::Perceptron(2);
				p->set_weights_bias({ 10, 10, -15 });
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, p->run({ 0, 0 }) });
				com::print_row_table<double>({ 0, 1, p->run({ 0, 1 }) });
				com::print_row_table<double>({ 1, 0, p->run({ 1, 0 }) });
				com::print_row_table<double>({ 1, 1, p->run({ 1, 1 }) });


				std::cout << "\n\n -----------Trained AND Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({ 2, 1 });

				std::vector<std::vector<double>> x = {
					{ 0, 0 },
					{ 0, 1 },
					{ 1, 0 },
					{ 1, 1 }};

				std::vector<std::vector<double>> y = {
					{ 0 },
					{ 0 },
					{ 0 },
					{ 1 }};

				size_t ep = 1000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}
				mlp->print_weights_bias();
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0] });
				com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0] });
				com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0] });
				com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0] });


			}
			void ORGate()
			{
				std::cout << "\n\n -----------Hardcoded OR Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::Perceptron* p = new za::nn::Perceptron(2);
				p->set_weights_bias({ 15, 15, -10 });
				com::print_row_table<std::string>({"Input 1", "Input 2", "Output"});
				com::print_row_table<double>({0, 0, p->run({ 0, 0 }) });
				com::print_row_table<double>({0, 1, p->run({ 0, 1 }) });
				com::print_row_table<double>({1, 0, p->run({ 1, 0 }) });
				com::print_row_table<double>({1, 1, p->run({ 1, 1 }) });


				std::cout << "\n\n -----------Trained OR Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({ 2, 1 });

				std::vector<std::vector<double>> x = {
					{ 0, 0 },
					{ 0, 1 },
					{ 1, 0 },
					{ 1, 1 } };

				std::vector<std::vector<double>> y = {
					{  0 },
					{  1 },
					{  1 },
					{  1 } };

				size_t ep = 1000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}

				mlp->print_weights_bias();
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0] });
				com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0] });
				com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0] });
				com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0] });

			}			
			void NANDGate()
			{
				std::cout << "\n\n -----------Hardcoded NAND Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::Perceptron* p = new za::nn::Perceptron(2);
				p->set_weights_bias({ -10, -10, 15 });
				com::print_row_table<std::string>({"Input 1", "Input 2", "Output"});
				com::print_row_table<double>({0, 0, p->run({ 0, 0 }) });
				com::print_row_table<double>({0, 1, p->run({ 0, 1 }) });
				com::print_row_table<double>({1, 0, p->run({ 1, 0 }) });
				com::print_row_table<double>({1, 1, p->run({ 1, 1 }) });



				std::cout << "\n\n -----------Trained NAND Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({ 2, 1 });


				std::vector<std::vector<double>> x = {
					{ 0, 0 },
					{ 0, 1 },
					{ 1, 0 },
					{ 1, 1 } };

				std::vector<std::vector<double>> y = {
					{  1 },
					{  0 },
					{  0 },
					{  0 } };

				size_t ep = 1000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}

				mlp->print_weights_bias();
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0] });
				com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0] });
				com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0] });
				com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0] });
			}
			void XORGate()
			{
				std::cout << "\n\n -----------Hardcoded XOR Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({2, 2, 1});

				/*
					First layer, layer 0,  is input layer, 2 inputs, but no weights
					Second layer, layer 1,  are NAND + OR gates, 2 neuron, 2x2 weights
					Third layer, layer 2, is AND gate, 1 neuron, 1x2 weigths 
				*/
				mlp->set_weights_bias({ {{ -10, -10, 15}, { 15, 15, -10}},{{10, 10, -15}} });
				std::cout << "Weights" << std::endl;
				mlp->print_weights_bias();
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0]});
				com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0]});
				com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0]});
				com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0]});

				std::cout << "\n\n -----------Trained XOR Logic Gate-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlpt = new za::nn::MultiLayerPerceptron({ 2, 2, 1 });

				std::vector<std::vector<double>> x = {
					{ 0, 0 },
					{ 0, 1 },
					{ 1, 0 },
					{ 1, 1 } };

				std::vector<std::vector<double>> y = {
					{  0 },
					{  1 },
					{  1 },
					{  0 } };

				size_t ep = 3000;
				auto mse_train = mlpt->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}

				mlpt->print_weights_bias();
				com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				com::print_row_table<double>({ 0, 0, mlpt->run({ 0, 0 })[0] });
				com::print_row_table<double>({ 0, 1, mlpt->run({ 0, 1 })[0] });
				com::print_row_table<double>({ 1, 0, mlpt->run({ 1, 0 })[0] });
				com::print_row_table<double>({ 1, 1, mlpt->run({ 1, 1 })[0] });

			}
			void SDR7To10()
			{
				std::cout << "\n\n -----------SDR 7 to 10-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 10});

				std::vector<std::vector<double>> x = {
														{ 1,1,1,1,1,1,0 },
														{ 0,1,1,0,0,0,0 },
														{ 1,1,0,1,1,0,1 },
														{ 1,1,1,1,0,0,1 },
														{ 0,1,1,0,0,1,1 },
														{ 1,0,1,1,0,1,1 },
														{ 1,0,1,1,1,1,1 },
														{ 1,1,1,0,0,0,0 },
														{ 1,1,1,1,1,1,1 },
														{ 1,1,1,1,0,1,1 }};

				std::vector<std::vector<double>> y = {
														{ 1,0,0,0,0,0,0,0,0,0 }, //0 pattern
														{ 0,1,0,0,0,0,0,0,0,0 }, //1 pattern
														{ 0,0,1,0,0,0,0,0,0,0 }, //2 pattern
														{ 0,0,0,1,0,0,0,0,0,0 }, //3 pattern
														{ 0,0,0,0,1,0,0,0,0,0 }, //4 pattern
														{ 0,0,0,0,0,1,0,0,0,0 }, //5 pattern
														{ 0,0,0,0,0,0,1,0,0,0 }, //6 pattern
														{ 0,0,0,0,0,0,0,1,0,0 }, //7 pattern
														{ 0,0,0,0,0,0,0,0,1,0 }, //8 pattern
														{ 0,0,0,0,0,0,0,0,0,1 }  //9 pattern
				};
				size_t ep = 3000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}

				
				mlp->print_weights_bias();

				com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				auto outputs1 = mlp->run(inputs1);
				auto label1 = std::distance(outputs1.begin(), std::max_element(outputs1.begin(), outputs1.end()));
				auto display1 = inputs1;
				display1.push_back(label1);
				com::print_row_table<double>(display1);

			}			
			void SDR7To7()
			{
				std::cout << "\n\n -----------SDR 7 to 7-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 7});

				std::vector<std::vector<double>> x = {
														{ 1,1,1,1,1,1,0 },
														{ 0,1,1,0,0,0,0 },
														{ 1,1,0,1,1,0,1 },
														{ 1,1,1,1,0,0,1 },
														{ 0,1,1,0,0,1,1 },
														{ 1,0,1,1,0,1,1 },
														{ 1,0,1,1,1,1,1 },
														{ 1,1,1,0,0,0,0 },
														{ 1,1,1,1,1,1,1 },
														{ 1,1,1,1,0,1,1 } };

				std::vector<std::vector<double>> y = {
														{ 1,1,1,1,1,1,0 }, //0 pattern
														{ 0,1,1,0,0,0,0 }, //1 pattern
														{ 1,1,0,1,1,0,1 }, //2 pattern
														{ 1,1,1,1,0,0,1 }, //3 pattern
														{ 0,1,1,0,0,1,1 }, //4 pattern
														{ 1,0,1,1,0,1,1 }, //5 pattern
														{ 1,0,1,1,1,1,1 }, //6 pattern
														{ 1,1,1,0,0,0,0 }, //7 pattern
														{ 1,1,1,1,1,1,1 }, //8 pattern
														{ 1,1,1,1,0,1,1 }  //9 pattern
				};
				size_t ep = 3000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}

				mlp->print_weights_bias();

				com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				auto outputs1 = mlp->run(inputs1);

				com::print_row_table<double>(inputs1);
				com::print_row_table<double>(outputs1);

			}
			void SDR7To1()
			{
				std::cout << "\n\n -----------SDR 7 to 1-------------\n\n";
				//initialize random number generator
				srand(time(nullptr));
				//ignore return value
				static_cast<void>(rand());
				za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 1});
				std::vector<std::vector<double>> x = {
															{ 1,1,1,1,1,1,0 },
															{ 0,1,1,0,0,0,0 },
															{ 1,1,0,1,1,0,1 },
															{ 1,1,1,1,0,0,1 },
															{ 0,1,1,0,0,1,1 },
															{ 1,0,1,1,0,1,1 },
															{ 1,0,1,1,1,1,1 },
															{ 1,1,1,0,0,0,0 },
															{ 1,1,1,1,1,1,1 },
															{ 1,1,1,1,0,1,1 } };

				std::vector<std::vector<double>> y = {
														{ 0.05 }, //0 pattern
														{ 0.15 }, //1 pattern
														{ 0.25 }, //2 pattern
														{ 0.35 }, //3 pattern
														{ 0.45 }, //4 pattern
														{ 0.55 }, //5 pattern
														{ 0.65 }, //6 pattern
														{ 0.75 }, //7 pattern
														{ 0.85 }, //8 pattern
														{ 0.95 }  //9 pattern
				};
				size_t ep = 3000;
				auto mse_train = mlp->train_epoch(x, y, ep);
				for (size_t i = 0; i < ep; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mse_train[i] << std::endl;
					}
				}
				mlp->print_weights_bias();

				com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				auto outputs1 = mlp->run(inputs1);
				auto label1 = std::trunc(10*outputs1[0]);
				auto display1 = inputs1;
				display1.push_back(label1);
				com::print_row_table<double>(display1);
			}			
		}
	}
}