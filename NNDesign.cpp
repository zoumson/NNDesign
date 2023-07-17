// NNDesign.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "NNModels.h"
#include <iostream>
using namespace za::nn::mm;
void runModel(const MODEL& model);


int main()
{
     
    MODEL model = MODEL::SINGLE_PERCEPTRON_AND_GATE;
    runModel(model);
    return 0;
}

void runModel(const MODEL& model)
{

    switch (model)
    {
#pragma region Single
    case MODEL::SINGLE_PERCEPTRON_AND_GATE:
        ANDGate();
        break;
    case MODEL::SINGLE_PERCEPTRON_NAND_GATE:
        NANDGate();   
        break;
    case MODEL::SINGLE_PERCEPTRON_OR_GATE:
        ORGate();
        break;    

#pragma endregion Single
#pragma region Multiple
    case MODEL::MULTIPLE_PERCEPTRON_XOR_GATE:
        XORGate();    
        break;
    case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_1_OUT:
        SDR7To1();
        break;    
    case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_7_OUT:
        SDR7To7();
        break;    
    case MODEL::MULTIPLE_PERCEPTRON_SDR_7_IN_10_OUT:
        SDR7To10();
        break; 
#pragma endregion Multiple
    default:

        break;
    }

}