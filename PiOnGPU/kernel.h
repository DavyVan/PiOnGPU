#include <cuda_runtime.h>

__global__ void PiOnGPU_Kernel(
	__int64 precision_init,
	__int64 precision_hex,
	__int64 dLeap,
	unsigned char *result_dec,
	char *result_hex);