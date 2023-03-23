#include "NNUtilities.h"

namespace za
{
	namespace nn
	{
		namespace com
		{
			double frand()
			{
				return (2.0 * (double)rand() / RAND_MAX) - 1;
			}
			double sigmoid(const double& _z)
			{
				double __output_tmp = 1.0 / (1.0 + exp(-_z));				
				return __output_tmp;
			}
		}
	}
}