#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include <string>
#include <iomanip>
#include <tuple>
#include <vector>




namespace za
{
	namespace nn
	{
		namespace com
		{
			double frand();
			double sigmoid(const double& _z);
			template<typename T> void print_cell_table(T t, int width = 8, char separator = ' ')
			{
				std::cout << std::left << std::setw(width) << std::setfill(separator) << t;
			};			
			template<typename T> void print_row_table(std::vector<T> t, int width = 8, char separator = ' ')
			{
				for (auto cell : t)
				{
					std::cout << std::left << std::setw(width) << std::setfill(separator) << cell;
				}
				std::cout << std::endl;
			};						
			template<typename T> void print_table(std::vector<std::vector<T>> t, int width = 8, char separator = ' ')
			{
				size_t row_size = t.size();
				size_t col_size = t[0].size();
				for (size_t row_i = 0; row_i < row_size; row_i++)
				{
					for (size_t col_i = 0; col_i < col_size; col_i++)
					{
						std::cout << std::left << std::setw(width) << std::setfill(separator) << t[row_i][col_size];
					}
					std::cout << std::endl;
				}
				
			};




		}
	}
}