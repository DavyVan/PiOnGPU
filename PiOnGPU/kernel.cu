#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernel.h"
#include "functions.h"

__global__ void PiOnGPU_Kernel(
	__int64 precision_init,
	__int64 precision_hex,
	__int64 dLeap,
	unsigned char *result_dec,
	char *result_hex)
{
	__int64 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	__int64 precision_current = precision_init + tid * dLeap;
	if (precision_current > precision_hex)
		return;
	double _16dPi = 0;

	_16dPi = 4 * _16dSj(precision_current, 1) - 2 * _16dSj(precision_current, 4) - _16dSj(precision_current, 5) - _16dSj(precision_current, 6);
	_16dPi = (_16dPi > 0) ? (_16dPi - (int) _16dPi) : (_16dPi - (int) _16dPi + 1);

	for (__int64 i = precision_current; i < precision_current + dLeap; i++)
	{
		_16dPi *= 16;
		result_dec[i] = (int) _16dPi;
		_16dPi -= result_dec[i];

		DecToHexModDiv(result_dec[i], result_hex[i]);
	}
}