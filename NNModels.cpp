#include "NNModels.h"

namespace za
{
	namespace nn
	{
		namespace mm
		{
			using namespace nn;
			using namespace com;
			void LogicGateSetUp(std::vector<data_t> dataInput, std::vector<data_t> dataLabel, data_t Weigthts)
			{
#pragma region data 

				size_t DATA_SIZE = 2;
				std::vector<double>data1 = { 0, 0 };
				double label1 = 0;
				std::vector<double>data2 = { 0, 1 };
				double label2 = 0;
				std::vector<double>data3 = { 1, 0 };
				double label3 = 0;
				std::vector<double>data4 = { 1, 1 };
				double label4 = 1;
				std::vector<std::vector<double>> data = { data1, data2, data3, data4 };
				std::vector<double> label = { label1, label2, label3, label4 };
				size_t DATA_SET_SIZE = data.size();
				//always initialize 
				std::vector<double> output(DATA_SET_SIZE, 0);

#pragma endregion data 

#pragma region weight 

				//// wo, w1, w2 for bias
				//double inputWeight1 = 10;
				//double inputWeight2 = 10;
				//double biasWeight = -15;
				std::vector<double> neuronWeights = Weigthts;

#pragma endregion weight

#pragma region neuron 

				std::unique_ptr<Perceptron> neuron(new Perceptron(DATA_SIZE));
				neuron->set_weights_bias(neuronWeights);

#pragma endregion neuron 

#pragma region display 

				// Data Output(1) Lablel(1)
				size_t ROW_LENGTH = DATA_SIZE + 1 + 1;
				std::vector<double> row(ROW_LENGTH, 0);
				int rowCol = 0;
				std::vector<std::string> rowHeader = { "Input 1", "Input 2", "Output", "Expected" };
				print_row_table<std::string>(rowHeader);

#pragma endregion display

#pragma region prediction

				for (int i = 0; i < DATA_SET_SIZE; i++)
				{
					//estimate current input data label  
					output[i] = neuron->run(data[i]);

					rowCol = 0;
					//data for display 
					for (int jData = 0; jData < data[i].size(); jData++)
					{
						row[rowCol++] = data[i][jData];
					}
					//output 
					row[rowCol++] = output[i];
					//label 
					row[rowCol] = label[i];

					//print the row input, output, label 
					print_row_table<double>(row);
				}

#pragma endregion prediction
			}
			void ANDGate()
			{
				std::cout << "Start AND GATE" << std::endl;


				auto hardcodeWeight = [&]()
				{
					std::cout << std::endl << std::endl << "-----------Hardcoded AND Logic Gate-------------" << std::endl << std::endl;
					//LogicGateSetUp(std::vector<data_t> dataInput, std::vector<data_t> dataLabel, data_t Weigthts);

#pragma region data 

					size_t DATA_SIZE = 2;
					std::vector<double>data1 = { 0, 0 };
					double label1 = 0;
					std::vector<double>data2 = { 0, 1 };
					double label2 = 0;
					std::vector<double>data3 = { 1, 0 };
					double label3 = 0;
					std::vector<double>data4 = { 1, 1 };
					double label4 = 1;
					std::vector<std::vector<double>> data = { data1, data2, data3, data4 };
					std::vector<double> label = { label1, label2, label3, label4 };
					size_t DATA_SET_SIZE = data.size();
					//always initialize 
					std::vector<double> output(DATA_SET_SIZE, 0);

#pragma endregion data 

#pragma region weight 

					// wo, w1, w2 for bias
					double inputWeight1 = 10;
					double inputWeight2 = 10;
					double biasWeight = -15;
					std::vector<double> neuronWeights = { inputWeight1 , inputWeight2, biasWeight };

#pragma endregion weight
//
//#pragma region gate 
//					LogicGateSetUp();
//#pragma endregion gate 
#pragma region neuron 

					std::unique_ptr<Perceptron> neuron(new Perceptron(DATA_SIZE));
					neuron->set_weights_bias(neuronWeights);

#pragma endregion neuron 

#pragma region display 

					// Data Output(1) Lablel(1)
					size_t ROW_LENGTH = DATA_SIZE + 1 + 1;
					std::vector<double> row(ROW_LENGTH, 0);
					int rowCol = 0;
					std::vector<std::string> rowHeader = { "Input 1", "Input 2", "Output", "Expected" };
					print_row_table<std::string>(rowHeader);

#pragma endregion display

#pragma region prediction

					for (int i = 0; i < DATA_SET_SIZE; i++)
					{
						//estimate current input data label  
						output[i] = neuron->run(data[i]);
	
						rowCol = 0;
						//data for display 
						for (int jData = 0; jData < data[i].size(); jData++)
						{
							row[rowCol++] = data[i][jData];
						}
						//output 
						row[rowCol++] = output[i];
						//label 
						row[rowCol] = label[i];

						//print the row input, output, label 
						print_row_table<double>(row);
					}

#pragma endregion prediction
												
				};

				auto trainedWeight = [&]()
				{
					std::cout << "\n\n -----------Trained AND Logic Gate-------------\n\n";

#pragma region data 
					size_t INPUT_DATA_SIZE = 2;
					size_t OUTPUT_DATA_SIZE = 1;
					std::vector<double>feature1 = { 0, 0 };
					std::vector<double> label1 = { 0 };
					std::vector<double>feature2 = { 0, 1 };
					std::vector<double> label2 = { 0 };
					std::vector<double>feature3 = { 1, 0 };
					std::vector<double> label3 = { 0 };
					std::vector<double>feature4 = { 1, 1 };
					std::vector<double> label4 = { 1 };
					std::vector<std::vector<double>> features = { feature1 , feature2, feature3, feature4};
					std::vector<std::vector<double>> labels = { label1 , label2, label3, label4 };
					size_t DATA_SET_SIZE = features.size();
					std::vector<double> output(DATA_SET_SIZE, 0);

#pragma endregion data 

#pragma region neuron 

					//layer 0 ==> input layer 
					//last layer ==> output layer 
					//1 neuron ==> 2 layers ==> 1 input and 1 output 
					//2 input data to predict one result 
					size_t LAYER_SIZE = 2;
					std::vector<size_t> neuronPerLayer(LAYER_SIZE, 0);
					neuronPerLayer[0] = INPUT_DATA_SIZE;
					neuronPerLayer[1] = OUTPUT_DATA_SIZE;
					//MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({ 2, 1 });
					std::unique_ptr<MultiLayerPerceptron> neuron(new MultiLayerPerceptron(neuronPerLayer));


#pragma endregion neuron 

#pragma region training  

					//Initialize random number generator
					//Timer used as seed to get different initial weight values when run 
					srand(time(nullptr));
					//gnore return value
					static_cast<void>(rand());
					size_t EPOCH_SIZE = 1000;
					auto mse_train = neuron->train_epoch(features, labels, EPOCH_SIZE);
					for (size_t iEpoch = 0; iEpoch < EPOCH_SIZE; iEpoch++)
					{

						if (iEpoch % 100 == 0)
						{
							std::cout << "MSE = " << mse_train[iEpoch] << std::endl;
						}
					}

					neuron->print_weights_bias();

#pragma endregion training 

#pragma region display 

					// Data Output(1) Lablel(1)
					size_t ROW_LENGTH = INPUT_DATA_SIZE + 1 + 1;
					std::vector<double> row(ROW_LENGTH, 0);
					int rowCol = 0;
					std::vector<std::string> rowHeader = { "Input 1", "Input 2", "Output", "Expected" };
					print_row_table<std::string>(rowHeader);

#pragma endregion display

#pragma region prediction 

					for (int i = 0; i < DATA_SET_SIZE; i++)
					{
						//estimate current input data label  
						output[i] = neuron->run(features[i])[0];

						rowCol = 0;
						//data for display 
						for (int jData = 0; jData < features[i].size(); jData++)
						{
							row[rowCol++] = features[i][jData];
						}
						//output 
						row[rowCol++] = output[i];
						//label 
						row[rowCol] = labels[i][0];

						//print the row input, output, label 
						print_row_table<double>(row);
					}

#pragma endregion prediction 


				};

				//hardcodeWeight();
				trainedWeight();

				std::cout << std::endl << std::endl << "End AND GATE" << std::endl;
			}


			void ORGate()
			{
				std::cout << "Start OR GATE" << std::endl;
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
				std::cout << "End OR GATE" << std::endl;
			}			
			
			void NANDGate()
			{
				//std::cout << "Start NAND GATE" << std::endl;
				//std::cout << "\n\n -----------Hardcoded NAND Logic Gate-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::Perceptron* p = new za::nn::Perceptron(2);
				//p->set_weights_bias({ -10, -10, 15 });
				//com::print_row_table<std::string>({"Input 1", "Input 2", "Output"});
				//com::print_row_table<double>({0, 0, p->run({ 0, 0 }) });
				//com::print_row_table<double>({0, 1, p->run({ 0, 1 }) });
				//com::print_row_table<double>({1, 0, p->run({ 1, 0 }) });
				//com::print_row_table<double>({1, 1, p->run({ 1, 1 }) });



				//std::cout << "\n\n -----------Trained NAND Logic Gate-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({ 2, 1 });


				//std::vector<std::vector<double>> x = {
				//	{ 0, 0 },
				//	{ 0, 1 },
				//	{ 1, 0 },
				//	{ 1, 1 } };

				//std::vector<std::vector<double>> y = {
				//	{  1 },
				//	{  0 },
				//	{  0 },
				//	{  0 } };

				//size_t ep = 1000;
				//auto mse_train = mlp->train_epoch(x, y, ep);
				//for (size_t i = 0; i < ep; i++)
				//{

				//	if (i % 100 == 0)
				//	{
				//		std::cout << "MSE = " << mse_train[i] << std::endl;
				//	}
				//}

				//mlp->print_weights_bias();
				//com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				//com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0] });
				//com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0] });
				//com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0] });
				//com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0] });

				//std::cout << "End NAND GATE" << std::endl;
			}
			void XORGate()
			{
				//std::cout << "Start XOR GATE" << std::endl;
				//std::cout << "\n\n -----------Hardcoded XOR Logic Gate-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({2, 2, 1});

				///*
				//	First layer, layer 0,  is input layer, 2 inputs, but no weights
				//	Second layer, layer 1,  are NAND + OR gates, 2 neuron, 2x2 weights
				//	Third layer, layer 2, is AND gate, 1 neuron, 1x2 weigths 
				//*/
				//mlp->set_weights_bias({ {{ -10, -10, 15}, { 15, 15, -10}},{{10, 10, -15}} });
				//std::cout << "Weights" << std::endl;
				//mlp->print_weights_bias();
				//com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				//com::print_row_table<double>({ 0, 0, mlp->run({ 0, 0 })[0]});
				//com::print_row_table<double>({ 0, 1, mlp->run({ 0, 1 })[0]});
				//com::print_row_table<double>({ 1, 0, mlp->run({ 1, 0 })[0]});
				//com::print_row_table<double>({ 1, 1, mlp->run({ 1, 1 })[0]});

				//std::cout << "\n\n -----------Trained XOR Logic Gate-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlpt = new za::nn::MultiLayerPerceptron({ 2, 2, 1 });

				//std::vector<std::vector<double>> x = {
				//	{ 0, 0 },
				//	{ 0, 1 },
				//	{ 1, 0 },
				//	{ 1, 1 } };

				//std::vector<std::vector<double>> y = {
				//	{  0 },
				//	{  1 },
				//	{  1 },
				//	{  0 } };

				//size_t ep = 3000;
				//auto mse_train = mlpt->train_epoch(x, y, ep);
				//for (size_t i = 0; i < ep; i++)
				//{

				//	if (i % 100 == 0)
				//	{
				//		std::cout << "MSE = " << mse_train[i] << std::endl;
				//	}
				//}

				//mlpt->print_weights_bias();
				//com::print_row_table<std::string>({ "Input 1", "Input 2", "Output" });
				//com::print_row_table<double>({ 0, 0, mlpt->run({ 0, 0 })[0] });
				//com::print_row_table<double>({ 0, 1, mlpt->run({ 0, 1 })[0] });
				//com::print_row_table<double>({ 1, 0, mlpt->run({ 1, 0 })[0] });
				//com::print_row_table<double>({ 1, 1, mlpt->run({ 1, 1 })[0] });
				//std::cout << "Start XOR GATE" << std::endl;
			}
			void SDR7To10()
			{
				//std::cout << "Start SDR 7 inputs To 10 outputs" << std::endl;
				//std::cout << "\n\n -----------SDR 7 to 10-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 10});

				//std::vector<std::vector<double>> x = {
				//										{ 1,1,1,1,1,1,0 },
				//										{ 0,1,1,0,0,0,0 },
				//										{ 1,1,0,1,1,0,1 },
				//										{ 1,1,1,1,0,0,1 },
				//										{ 0,1,1,0,0,1,1 },
				//										{ 1,0,1,1,0,1,1 },
				//										{ 1,0,1,1,1,1,1 },
				//										{ 1,1,1,0,0,0,0 },
				//										{ 1,1,1,1,1,1,1 },
				//										{ 1,1,1,1,0,1,1 }};

				//std::vector<std::vector<double>> y = {
				//										{ 1,0,0,0,0,0,0,0,0,0 }, //0 pattern
				//										{ 0,1,0,0,0,0,0,0,0,0 }, //1 pattern
				//										{ 0,0,1,0,0,0,0,0,0,0 }, //2 pattern
				//										{ 0,0,0,1,0,0,0,0,0,0 }, //3 pattern
				//										{ 0,0,0,0,1,0,0,0,0,0 }, //4 pattern
				//										{ 0,0,0,0,0,1,0,0,0,0 }, //5 pattern
				//										{ 0,0,0,0,0,0,1,0,0,0 }, //6 pattern
				//										{ 0,0,0,0,0,0,0,1,0,0 }, //7 pattern
				//										{ 0,0,0,0,0,0,0,0,1,0 }, //8 pattern
				//										{ 0,0,0,0,0,0,0,0,0,1 }  //9 pattern
				//};
				//size_t ep = 3000;
				//auto mse_train = mlp->train_epoch(x, y, ep);
				//for (size_t i = 0; i < ep; i++)
				//{

				//	if (i % 100 == 0)
				//	{
				//		std::cout << "MSE = " << mse_train[i] << std::endl;
				//	}
				//}

				//
				//mlp->print_weights_bias();

				//com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				//std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				//auto outputs1 = mlp->run(inputs1);
				//auto label1 = std::distance(outputs1.begin(), std::max_element(outputs1.begin(), outputs1.end()));
				//auto display1 = inputs1;
				//display1.push_back(label1);
				//com::print_row_table<double>(display1);
				//std::cout << "End SDR 7 inputs To 10 outputs" << std::endl;
			}			
			void SDR7To7()
			{
				//std::cout << "Start SDR 7 inputs To 7 outputs" << std::endl;
				//std::cout << "\n\n -----------SDR 7 to 7-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 7});

				//std::vector<std::vector<double>> x = {
				//										{ 1,1,1,1,1,1,0 },
				//										{ 0,1,1,0,0,0,0 },
				//										{ 1,1,0,1,1,0,1 },
				//										{ 1,1,1,1,0,0,1 },
				//										{ 0,1,1,0,0,1,1 },
				//										{ 1,0,1,1,0,1,1 },
				//										{ 1,0,1,1,1,1,1 },
				//										{ 1,1,1,0,0,0,0 },
				//										{ 1,1,1,1,1,1,1 },
				//										{ 1,1,1,1,0,1,1 } };

				//std::vector<std::vector<double>> y = {
				//										{ 1,1,1,1,1,1,0 }, //0 pattern
				//										{ 0,1,1,0,0,0,0 }, //1 pattern
				//										{ 1,1,0,1,1,0,1 }, //2 pattern
				//										{ 1,1,1,1,0,0,1 }, //3 pattern
				//										{ 0,1,1,0,0,1,1 }, //4 pattern
				//										{ 1,0,1,1,0,1,1 }, //5 pattern
				//										{ 1,0,1,1,1,1,1 }, //6 pattern
				//										{ 1,1,1,0,0,0,0 }, //7 pattern
				//										{ 1,1,1,1,1,1,1 }, //8 pattern
				//										{ 1,1,1,1,0,1,1 }  //9 pattern
				//};
				//size_t ep = 3000;
				//auto mse_train = mlp->train_epoch(x, y, ep);
				//for (size_t i = 0; i < ep; i++)
				//{

				//	if (i % 100 == 0)
				//	{
				//		std::cout << "MSE = " << mse_train[i] << std::endl;
				//	}
				//}

				//mlp->print_weights_bias();

				//com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				//std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				//auto outputs1 = mlp->run(inputs1);

				//com::print_row_table<double>(inputs1);
				//com::print_row_table<double>(outputs1);
				//std::cout << "End SDR 7 inputs To 7 outputs" << std::endl;
			}
			void SDR7To1()
			{
				//std::cout << "Start SDR 7 inputs To 1 output" << std::endl;
				//std::cout << "\n\n -----------SDR 7 to 1-------------\n\n";
				////initialize random number generator
				//srand(time(nullptr));
				////ignore return value
				//static_cast<void>(rand());
				//za::nn::MultiLayerPerceptron* mlp = new za::nn::MultiLayerPerceptron({7, 7, 1});
				//std::vector<std::vector<double>> x = {
				//											{ 1,1,1,1,1,1,0 },
				//											{ 0,1,1,0,0,0,0 },
				//											{ 1,1,0,1,1,0,1 },
				//											{ 1,1,1,1,0,0,1 },
				//											{ 0,1,1,0,0,1,1 },
				//											{ 1,0,1,1,0,1,1 },
				//											{ 1,0,1,1,1,1,1 },
				//											{ 1,1,1,0,0,0,0 },
				//											{ 1,1,1,1,1,1,1 },
				//											{ 1,1,1,1,0,1,1 } };

				//std::vector<std::vector<double>> y = {
				//										{ 0.05 }, //0 pattern
				//										{ 0.15 }, //1 pattern
				//										{ 0.25 }, //2 pattern
				//										{ 0.35 }, //3 pattern
				//										{ 0.45 }, //4 pattern
				//										{ 0.55 }, //5 pattern
				//										{ 0.65 }, //6 pattern
				//										{ 0.75 }, //7 pattern
				//										{ 0.85 }, //8 pattern
				//										{ 0.95 }  //9 pattern
				//};
				//size_t ep = 3000;
				//auto mse_train = mlp->train_epoch(x, y, ep);
				//for (size_t i = 0; i < ep; i++)
				//{

				//	if (i % 100 == 0)
				//	{
				//		std::cout << "MSE = " << mse_train[i] << std::endl;
				//	}
				//}
				//mlp->print_weights_bias();

				//com::print_row_table<std::string>({ "Input 1", "Output" }, 15);
				//std::vector<double> inputs1 = { 1,1,1,1,1,1,0.4 };
				//auto outputs1 = mlp->run(inputs1);
				//auto label1 = std::trunc(10*outputs1[0]);
				//auto display1 = inputs1;
				//display1.push_back(label1);
				//com::print_row_table<double>(display1);
				//std::cout << "Start SDR 7 inputs To 1 output" << std::endl;
			}			
		}
	}
}