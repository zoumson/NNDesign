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
        enum class DIGIT_LABEL
        {
            DIGIT_ZERO,
            DIGIT_ONE,
            DIGIT_TWO,
            DIGIT_THREE,
            DIGIT_FOUR,
            DIGIT_FIVE,
            DIGIT_SIX,
            DIGIT_SEVEN,
            DIGIT_EIGHT,
            DIGIT_NINE,
        };

        static std::ostream& operator<<(std::ostream& os, const MODEL type)
        {
            std::string disp = "";
            switch (type)
            {

            case MODEL::SINGLE_PERCEPTRON_AND_GATE:
                disp =  "Single Perceptron AND Gate";
                break;
            case MODEL::SINGLE_PERCEPTRON_NAND_GATE:
                disp = "Single Perceptron NAND Gate";
                break;
            case MODEL::SINGLE_PERCEPTRON_OR_GATE:
                disp = "Single Perceptron OR Gate";
                break;
            case MODEL::MULTIPLE_PERCEPTRON_XOR_GATE:
                disp = "Multiple Perceptron XOR Gate";
                break;
            case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_1_OUT:
                disp = "Multiple Perceptron SDR Network 7 Input Entries and 1 Output Class";
                break;
            case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_7_OUT:
                disp = "Multiple Perceptron SDR Network 7 Input Entries and 7 Output Classes";
                break;
            case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_10_OUT:
                disp = "Multiple Perceptron SDR Network 7 Input Entries and 10 Output Classes";
                break;

                           
           default:
               
                break;


            }
            os << disp;
            return os;
        }
        static std::ostream& operator<<(std::ostream& os, const DIGIT_LABEL type)
        {
            std::string disp = "";
            switch (type)
            {

            case DIGIT_LABEL::DIGIT_ZERO:
                disp =  "0";
                break;
            case DIGIT_LABEL::DIGIT_ONE:
                disp =  "1";
                break;
            case DIGIT_LABEL::DIGIT_TWO:
                disp =  "2";
                break;
            case DIGIT_LABEL::DIGIT_THREE:
                disp =  "3";
                break;
            case DIGIT_LABEL::DIGIT_FOUR:
                disp =  "4";
                break;
            case DIGIT_LABEL::DIGIT_FIVE:
                disp =  "5";
                break;
            case DIGIT_LABEL::DIGIT_SIX:
                disp =  "6";
                break;
            case DIGIT_LABEL::DIGIT_SEVEN:
                disp =  "7";
                break;
            case DIGIT_LABEL::DIGIT_EIGHT:
                disp =  "8";
                break;            
            case DIGIT_LABEL::DIGIT_NINE:
                disp =  "9";
                break;
                            
           default:
               
                break;


            }
            os << disp;
            return os;
        }


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