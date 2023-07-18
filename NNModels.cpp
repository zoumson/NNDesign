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
				std::cout << std::endl << std::endl << "Start SDR 7 inputs To 10 outputs" << std::endl << std::endl;
				std::cout << std::endl << std::endl << "----------- SDR 7 to 10------------- " << std::endl << std::endl;


#pragma region data 
				//Input data has 7 entries 
				//Data a, b, c, d, e, f, g
				/*
				*		a
				*	f		b
				*		g
				*	e		c
				*		d
				*/

				//Output data has 10 entries 
				//Output 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
				//Each entrie belong to a class, 
				//The class wit the highest value is assigned as label of the data
				
				
											//a, b, c, d, e, f, g
				std::vector<double> data0 = { 1, 1, 1, 1, 1, 1, 0 };
											 //0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
				std::vector<double> label0 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				std::vector<double> data1 = { 0, 1, 1, 0, 0, 0, 0 };
				std::vector<double> label1 = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

				std::vector<double> data2 = { 1, 1, 0, 1, 1, 0, 1 };
				std::vector<double> label2 = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };

				std::vector<double> data3 = { 1, 1, 1, 1, 0, 0, 1 };
				std::vector<double> label3 = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };

				std::vector<double> data4 = { 0, 1, 1, 0, 0, 1, 1 };
				std::vector<double> label4 = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };

				std::vector<double> data5 = { 1, 0, 1, 1, 0, 1, 1 };
				std::vector<double> label5 = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };

				std::vector<double> data6 = { 1, 0, 1, 1, 1, 1, 1 };
				std::vector<double> label6 = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };

				std::vector<double> data7 = { 1, 1, 1, 0, 0, 0, 0 };
				std::vector<double> label7 = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };

				std::vector<double> data8 = { 1, 1, 1, 1, 1, 1, 1 };
				std::vector<double> label8 = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };

				std::vector<double> data9 = { 1, 1, 1, 1, 0, 1, 1 };
				std::vector<double> label9 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

				std::vector<std::vector<double>> data = {
															data0,
															data1,
															data2,
															data3,
															data4,
															data5,
															data6,
															data7,
															data8,
															data9,
														};
				
				std::vector<std::vector<double>> labels = {
															label0,
															label1,
															label2,
															label3,
															label4,
															label5,
															label6,
															label7,
															label8,
															label9,
														};

#pragma endregion data 

#pragma region neurons 
				size_t INPUT_DATA_ENTRIES_SIZE = 7;
				size_t OUTPUT_DATA_CLASSES_SIZE = 10;
				size_t HIDDEN_LAYER_NEURONS_SIZE = 7;
				//Number of Layers: 3
				// Layer 0 ==> Input Layer ==> Data Entries == > 7
				// Layer 1 ==> Hidden Layer ==> 7
				// Layer 2 ==> Output Layer ==> Output Classes ==> 10
				//MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ 7, 7, 10 });

				size_t LAYER_SIZE = 3;
				std::vector<size_t> neuronPerLayer(LAYER_SIZE, 0);
				neuronPerLayer[0] = INPUT_DATA_ENTRIES_SIZE;
				neuronPerLayer[1] = HIDDEN_LAYER_NEURONS_SIZE;
				neuronPerLayer[2] = OUTPUT_DATA_CLASSES_SIZE;
				//Initialize random number generator for weights initialization 
				srand(time(nullptr));
				//Ignore return value
				static_cast<void>(rand());
				std::unique_ptr<MultiLayerPerceptron> mlp (new MultiLayerPerceptron(neuronPerLayer));
#pragma endregion neurons 

#pragma region training 

	
				size_t EPOCH_SIZE = 3000;
				auto mseTrain = mlp->train_epoch(data, labels, EPOCH_SIZE);
				for (size_t i = 0; i < EPOCH_SIZE; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mseTrain[i] << std::endl;
					}
				}

				
				mlp->print_weights_bias();
#pragma endregion training 

#pragma region display 

				std::vector<std::string> rowHeader = { "Input 1", "Output" };
				std::vector<double> row;
				print_row_table<std::string>(rowHeader, 15);

#pragma endregion display 

#pragma region prediction 
				std::vector<double> testData1 = { 1,1,1,1,1,1,0.4 };
				auto outputClasses1 = mlp->run(testData1);

				//The class with the highest value raise its hand
				auto outputClasse1 = std::distance(outputClasses1.begin(), std::max_element(outputClasses1.begin(), outputClasses1.end()));
				row = testData1;
				row.push_back(outputClasse1);
				print_row_table<double>(row);
				//std::vector<double> testData1 = { 1,1,1,1,1,1,0.4 };
#pragma endregion prediction 

				std::cout << std::endl << std::endl << "End SDR 7 inputs To 10 outputs" << std::endl << std::endl;
			}			
			void SDR7To7()
			{
				std::cout << std::endl << std::endl << "Start SDR 7 inputs To 7 outputs" << std::endl << std::endl;
				std::cout << std::endl << std::endl << "----------- SDR 7 to 7------------- " << std::endl << std::endl;


#pragma region data 
				//Input data has 7 entries 
				//Data a, b, c, d, e, f, g
				/*
				*		a
				*	f		b
				*		g
				*	e		c
				*		d
				*/

				//Output data has 7 entries 
				//Output = Input 
	


											//a, b, c, d, e, f, g
				std::vector<double> data0 = { 1, 1, 1, 1, 1, 1, 0 };
				std::vector<double> label0 = { 1, 1, 1, 1, 1, 1, 0 };

				std::vector<double> data1 = { 0, 1, 1, 0, 0, 0, 0 };
				std::vector<double> label1 = { 0, 1, 1, 0, 0, 0, 0 };

				std::vector<double> data2 = { 1, 1, 0, 1, 1, 0, 1 };
				std::vector<double> label2 = { 1, 1, 0, 1, 1, 0, 1 };

				std::vector<double> data3 = { 1, 1, 1, 1, 0, 0, 1 };
				std::vector<double> label3 = { 1, 1, 1, 1, 0, 0, 1 };

				std::vector<double> data4 = { 0, 1, 1, 0, 0, 1, 1 };
				std::vector<double> label4 = { 0, 1, 1, 0, 0, 1, 1 };

				std::vector<double> data5 = { 1, 0, 1, 1, 0, 1, 1 };
				std::vector<double> label5 = { 1, 0, 1, 1, 0, 1, 1 };

				std::vector<double> data6 = { 1, 0, 1, 1, 1, 1, 1 };
				std::vector<double> label6 = { 1, 0, 1, 1, 1, 1, 1 };

				std::vector<double> data7 = { 1, 1, 1, 0, 0, 0, 0 };
				std::vector<double> label7 = { 1, 1, 1, 0, 0, 0, 0 };

				std::vector<double> data8 = { 1, 1, 1, 1, 1, 1, 1 };
				std::vector<double> label8 = { 1, 1, 1, 1, 1, 1, 1 };

				std::vector<double> data9 = { 1, 1, 1, 1, 0, 1, 1 };
				std::vector<double> label9 = { 1, 1, 1, 1, 0, 1, 1 };

				std::vector<std::vector<double>> data = {
															data0,
															data1,
															data2,
															data3,
															data4,
															data5,
															data6,
															data7,
															data8,
															data9,
				};

				std::vector<std::vector<double>> labels = {
															label0,
															label1,
															label2,
															label3,
															label4,
															label5,
															label6,
															label7,
															label8,
															label9,
				};

#pragma endregion data 

#pragma region neurons 
				size_t INPUT_DATA_ENTRIES_SIZE = 7;
				size_t OUTPUT_DATA_CLASSES_SIZE = 7;
				size_t HIDDEN_LAYER_NEURONS_SIZE = 7;
				//Number of Layers: 3
				// Layer 0 ==> Input Layer ==> Data Entries == > 7
				// Layer 1 ==> Hidden Layer ==> 7
				// Layer 2 ==> Output Layer ==> Output Classes ==> 7

				size_t LAYER_SIZE = 3;
				std::vector<size_t> neuronPerLayer(LAYER_SIZE, 0);
				neuronPerLayer[0] = INPUT_DATA_ENTRIES_SIZE;
				neuronPerLayer[1] = HIDDEN_LAYER_NEURONS_SIZE;
				neuronPerLayer[2] = OUTPUT_DATA_CLASSES_SIZE;
				//Initialize random number generator for weights initialization 
				srand(time(nullptr));
				//Ignore return value
				static_cast<void>(rand());
				std::unique_ptr<MultiLayerPerceptron> mlp(new MultiLayerPerceptron(neuronPerLayer));
#pragma endregion neurons 

#pragma region training 


				size_t EPOCH_SIZE = 3000;
				auto mseTrain = mlp->train_epoch(data, labels, EPOCH_SIZE);
				for (size_t i = 0; i < EPOCH_SIZE; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mseTrain[i] << std::endl;
					}
				}


				mlp->print_weights_bias();
#pragma endregion training 

#pragma region display 

				std::vector<std::string> rowHeader = { "Input 1", "Output" };
				std::vector<double> row;
				print_row_table<std::string>(rowHeader, 15);

#pragma endregion display 

#pragma region prediction 
				std::vector<double> testData1 = { 1,1,1,1,1,1,0.4 };
				auto outputClasses1 = mlp->run(testData1);

				//The class with the highest value raise its hand
				auto outputClasse1 = std::distance(outputClasses1.begin(), std::max_element(outputClasses1.begin(), outputClasses1.end()));
				row = testData1;
				row.push_back(outputClasse1);
				print_row_table<double>(row);
				//std::vector<double> testData1 = { 1,1,1,1,1,1,0.4 };
#pragma endregion prediction 

				std::cout << std::endl << std::endl << "End SDR 7 inputs To 7 outputs" << std::endl << std::endl;

			}
			void SDR7To1()
			{
				std::cout << std::endl << std::endl << "Start SDR 7 inputs To 1 output" << std::endl << std::endl;
				std::cout << std::endl << std::endl << "----------- SDR 7 to 1------------- " << std::endl << std::endl;

#pragma region data 
				//Input data has 7 entries 
				//Data a, b, c, d, e, f, g
				/*
				*		a
				*	f		b
				*		g
				*	e		c
				*		d
				*/

				//Output data has 1 entry
				
				using data_t = std::vector<double>;
				using label_t = std::vector<double>;
				using dataSet_t = std::vector<data_t>;
				using labelSet_t = std::vector<label_t>;

				dataSet_t dataSet;
				labelSet_t labelSet;

				data_t thresholdLabel = { 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1};
				
				//Middle point of the boundary values 
				auto middleOfBoundaries = [&](const std::vector<double> boundaries) -> double
				{
					auto const count = static_cast<float>(boundaries.size());
					return (std::accumulate(boundaries.begin(), boundaries.end(), 0.0) / count);
				};
				
				auto encodeLabel = [&](const DIGIT_LABEL& dig) -> label_t
				{

					double encodeVal = 0;
					switch (dig)
					{
					case DIGIT_LABEL::DIGIT_ZERO:
						encodeVal = middleOfBoundaries({ thresholdLabel[1] , thresholdLabel[0] });
						break;
					case DIGIT_LABEL::DIGIT_ONE:
						encodeVal = middleOfBoundaries({ thresholdLabel[2] , thresholdLabel[1] });
						break;					
					case DIGIT_LABEL::DIGIT_TWO:
						encodeVal = middleOfBoundaries({ thresholdLabel[3] , thresholdLabel[2] });
						break;					
					case DIGIT_LABEL::DIGIT_THREE:
						encodeVal = middleOfBoundaries({ thresholdLabel[4] , thresholdLabel[3] });
						break;					
					case DIGIT_LABEL::DIGIT_FOUR:
						encodeVal = middleOfBoundaries({ thresholdLabel[5] , thresholdLabel[4] });
						break;					
					case DIGIT_LABEL::DIGIT_FIVE:
						encodeVal = middleOfBoundaries({ thresholdLabel[6] , thresholdLabel[5] });
						break;					
					case DIGIT_LABEL::DIGIT_SIX:
						encodeVal = middleOfBoundaries({ thresholdLabel[7] , thresholdLabel[6] });
						break;
					case DIGIT_LABEL::DIGIT_SEVEN:
						encodeVal = middleOfBoundaries({ thresholdLabel[8] , thresholdLabel[7] });
						break;
					case DIGIT_LABEL::DIGIT_EIGHT:
						encodeVal = middleOfBoundaries({ thresholdLabel[9] , thresholdLabel[8] });
						break;					
					case DIGIT_LABEL::DIGIT_NINE:
						encodeVal = middleOfBoundaries({ thresholdLabel[10] , thresholdLabel[9] });
						break;
					default:
						break;
					};
					return { encodeVal };
				};

				auto decodeLabel = [&](const data_t& lab) -> DIGIT_LABEL
				{

					//decodedValTmp = (unsigned int)std::trunc(10 * lab[0]);
					DIGIT_LABEL decodedVal;

					auto call0 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_ZERO;
					};
					
					auto call1 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_ONE;
					};
									
					auto call2 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_TWO;
					};					
									
					auto call3 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_THREE;
					};
									
					auto call4 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_FOUR;
					};					
									
					auto call5 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_FIVE;
					};
									
					auto call6 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_SIX;
					};
									
					auto call7 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_SEVEN;
					};					
									
					auto call8 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_EIGHT;
					};					
									
					auto call9 = [&]()
					{
						decodedVal = DIGIT_LABEL::DIGIT_NINE;
					};

					//To avoid if else on range based 
					std::map<double, std::function<void(void)>> mapLabValToLabel{
						
						{thresholdLabel[1], call0},
						{thresholdLabel[2], call1},
						{thresholdLabel[3], call2},
						{thresholdLabel[4], call3},
						{thresholdLabel[5], call4},
						{thresholdLabel[6], call5},
						{thresholdLabel[7], call6},
						{thresholdLabel[8], call7},
						{thresholdLabel[9], call8},
						{thresholdLabel[10], call9},
					};

					 mapLabValToLabel.lower_bound(lab[0])->second();	
			
					return decodedVal;
				};

				auto initializeDataAndLabel = [&](dataSet_t& _dataSet, labelSet_t& _labelSet)
				{

					//a, b, c, d, e, f, g
					data_t data0 = { 1, 1, 1, 1, 1, 1, 0 };
					label_t label0 = encodeLabel(DIGIT_LABEL::DIGIT_ZERO);

					data_t data1 = { 0, 1, 1, 0, 0, 0, 0 };
					label_t label1 = encodeLabel(DIGIT_LABEL::DIGIT_ONE);

					data_t data2 = { 1, 1, 0, 1, 1, 0, 1 };
					label_t label2 = encodeLabel(DIGIT_LABEL::DIGIT_TWO);

					data_t data3 = { 1, 1, 1, 1, 0, 0, 1 };
					label_t label3 = encodeLabel(DIGIT_LABEL::DIGIT_THREE);

					data_t data4 = { 0, 1, 1, 0, 0, 1, 1 };
					label_t label4 = encodeLabel(DIGIT_LABEL::DIGIT_FOUR);

					data_t data5 = { 1, 0, 1, 1, 0, 1, 1 };
					label_t label5 = encodeLabel(DIGIT_LABEL::DIGIT_FIVE);

					data_t data6 = { 1, 0, 1, 1, 1, 1, 1 };
					label_t label6 = encodeLabel(DIGIT_LABEL::DIGIT_SIX);

					data_t data7 = { 1, 1, 1, 0, 0, 0, 0 };
					label_t label7 = encodeLabel(DIGIT_LABEL::DIGIT_SEVEN);

					data_t data8 = { 1, 1, 1, 1, 1, 1, 1 };
					label_t label8 = encodeLabel(DIGIT_LABEL::DIGIT_EIGHT);

					data_t data9 = { 1, 1, 1, 1, 0, 1, 1 };
					label_t label9 = encodeLabel(DIGIT_LABEL::DIGIT_NINE);


					_dataSet = {
																data0,
																data1,
																data2,
																data3,
																data4,
																data5,
																data6,
																data7,
																data8,
																data9,
					};

					_labelSet = {
																label0,
																label1,
																label2,
																label3,
																label4,
																label5,
																label6,
																label7,
																label8,
																label9,
					};
				};
	
				initializeDataAndLabel(dataSet, labelSet);
#pragma endregion data 

#pragma region neurons 
				size_t INPUT_DATA_ENTRIES_SIZE = 7;
				size_t OUTPUT_DATA_CLASSES_SIZE = 1;
				size_t HIDDEN_LAYER_NEURONS_SIZE = 7;
				//Number of Layers: 3
				// Layer 0 ==> Input Layer ==> Data Entries == > 7
				// Layer 1 ==> Hidden Layer ==> 7
				// Layer 2 ==> Output Layer ==> Output Class ==> 1

				size_t LAYER_SIZE = 3;
				std::vector<size_t> neuronPerLayer(LAYER_SIZE, 0);
				neuronPerLayer[0] = INPUT_DATA_ENTRIES_SIZE;
				neuronPerLayer[1] = HIDDEN_LAYER_NEURONS_SIZE;
				neuronPerLayer[2] = OUTPUT_DATA_CLASSES_SIZE;
				//Initialize random number generator for weights initialization 
				srand(time(nullptr));
				//Ignore return value
				static_cast<void>(rand());
				std::unique_ptr<MultiLayerPerceptron> mlp(new MultiLayerPerceptron(neuronPerLayer));
#pragma endregion neurons 

#pragma region training 


				size_t EPOCH_SIZE = 3000;
				auto mseTrain = mlp->train_epoch(dataSet, labelSet, EPOCH_SIZE);
				for (size_t i = 0; i < EPOCH_SIZE; i++)
				{

					if (i % 100 == 0)
					{
						std::cout << "MSE = " << mseTrain[i] << std::endl;
					}
				}


				mlp->print_weights_bias();
#pragma endregion training 

#pragma region display 

				std::vector<std::string> rowHeader = { "Input 1", "Output" };
				std::vector<double> row;
				print_row_table<std::string>(rowHeader, 15);

#pragma endregion display 

#pragma region prediction 

				//std::vector<double> testData1 = { 1,1,1,1,1,1,0.4 };
				std::vector<double> testData1 = { 1, 1, 1, 1, 1, 1, 0.2 };
				auto outputClasses1 = mlp->run(testData1);

				//The class with the highest value raise its hand
				//auto outputClasse1 = std::trunc(10 * outputClasses1[0]);
				auto outputClasse1 = decodeLabel(outputClasses1);

				std::stringstream bufferStr;
				bufferStr << outputClasse1;
				std::string outputClasseCPPStr1 = bufferStr.str();
				const char* outputClasseCStr1 = outputClasseCPPStr1.c_str();

				////int outputDigit1 = atoi(outputClasseCStr1);
				int outputDigit1 = std::stoi(outputClasseCPPStr1);


				row = testData1;
				row.push_back(outputDigit1);
				print_row_table<double>(row);

#pragma endregion prediction 

				std::cout << std::endl << std::endl << "End SDR 7 inputs To 1 output" << std::endl << std::endl;

			}			
		}
	}
}