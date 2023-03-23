// NNDesign.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "NNModels.h"
#include <iostream>

int main()
{
    size_t test_num = 1;
    switch (test_num)
    {
    case 1:
        za::nn::mm::ANDGate();
        break;    
    case 2:
        za::nn::mm::ORGate();
        break;    
    case 3:
        za::nn::mm::NANDGate();
        break;    
    case 4:
        za::nn::mm::XORGate();
        break;    
    case 5:
        za::nn::mm::SDR7To10();
        break;    
    case 6:
        za::nn::mm::SDR7To7();
        break;    
    case 7:
        za::nn::mm::SDR7To1();
        break;
    default:
        za::nn::mm::ANDGate();
        break;
    }

    return 0;
}

