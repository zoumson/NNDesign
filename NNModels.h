#pragma once
#include "Perceptron.h"
#include "MultiLayerPerceptron.h"


namespace za
{
	namespace nn
	{
		namespace mm
		{

        enum class MODEL
        {
            SINGLE_PERCEPTRON_AND_GATE,
            SINGLE_PERCEPTRON_NAND_GATE,
            SINGLE_PERCEPTRON_OR_GATE,

            MULTIPLE_PERCEPTRON_XOR_GATE,
            MULTIPLE_PERCEPTRON_SDR_7_IN_1_OUT,
            MULTIPLE_PERCEPTRON_SDR_7_IN_7_OUT,
            MULTIPLE_PERCEPTRON_SDR_7_IN_10_OUT,
        };


			 //   case 1:
    //    za::nn::mm::ANDGate();
    //    break;    
    //case 2:
    //    za::nn::mm::ORGate();
    //    break;    
    //case 3:
    //    za::nn::mm::NANDGate();
    //    break;    
    //case 4:
    //    za::nn::mm::XORGate();
    //    break;    
    //case 5:
    //    za::nn::mm::SDR7To10();
    //    break;    
    //case 6:
    //    za::nn::mm::SDR7To7();
    //    break;    
    //case 7:
    //    za::nn::mm::SDR7To1();
    //    break;
    //default:
    //    za::nn::mm::ANDGate();
            using data_t = std::vector<double>;

			void LogicGateSetUp(std::vector<data_t> dataInput, std::vector<data_t> dataLabel, data_t Weigthts);
			void ANDGate();
			void ORGate();
			void NANDGate();
			void XORGate();
			void SDR7To10();
			void SDR7To7();
			void SDR7To1();
		}
	}
}