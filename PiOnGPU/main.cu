/*
* Copyright for DeviceQuery 
*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* PiOnGPU computes Pi on GPU for 2,000,000 Hex
*/

#include<memory>
#include<iostream>
#include<cstdlib>
#include<cassert>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include"CUDA Helper\helper_cuda.h"
//#include "functions.h"
//#include "kernel.h"

#define ENABLE_CONSTANT_MEM
#define ENABLE_SHARED_MEM

#define dLEAP 2
#define PRECISION_INIT 2
#define PRECISION_HEX 20008	//of digits in hex
#define PRECISION_OUTPUT_START (PRECISION_HEX - 8)
#define PRECISION_OUTPUT_END (PRECISION_HEX)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 2	//1024 threads per block
#define THREADS_PER_BLOCK ((WARPS_PER_BLOCK) * (WARP_SIZE))
#define BLOCKS_NUM ((PRECISION_HEX/dLEAP/THREADS_PER_BLOCK) + 1)



void errorPrint(cudaError_t error_id)
{
	printf("ERROR: %d - %s\n", (int) error_id, cudaGetErrorString(error_id));
	//printf("Error occured, terminating...\n");
	return;
}


__device__ __int64 biexp(__int64 a, __int64 b, __int64 mod)
{
	__int64 ret = 1;
	while (b != 0)
	{
		if (b % 2)
			ret = (ret * a) % mod;
		a = (a * a) % mod;
		b /= 2;
	}
	return ret;
}

char hex_table1[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
#ifndef ENABLE_CONSTANT_MEM
char *hex_table = NULL;
#else ENABLE_CONSTANT_MEM
__constant__ char hex_table[16];
#endif

__host__ void hex_table_init()
{
#ifndef ENABLE_CONSTANT_MEM
	checkCudaErrors(cudaMalloc((void **) &hex_table, 16 * sizeof(char)));
	checkCudaErrors(cudaMemcpy(hex_table, hex_table1, 16 * sizeof(char), cudaMemcpyHostToDevice));
#else
	checkCudaErrors(cudaMemcpyToSymbol(hex_table, hex_table1, 16 * sizeof(char)));
#endif
}

__device__ void DecToHexModDiv(unsigned char dec, char &hex
#ifndef ENABLE_CONSTANT_MEM
	, char* hex_table
#endif
	)
{
	hex = hex_table[dec];
}

__device__ double F16dSj(__int64 d, int j)
{
	double sum = 0;
	for (__int64 k = 0; k <= d; k++)
	{
		sum += (double) biexp(16, d - k, 8 * k + j) / (8 * k + j);
	}

	return sum - (int) sum;
}

__global__ void PiOnGPU_Kernel(
	__int64 precision_init,
	__int64 precision_hex,
	__int64 dLeap,
	unsigned char *result_dec,
	char *result_hex
#ifndef ENABLE_CONSTANT_MEM
	,char *hex_table
#endif
	)
{
#ifdef ENABLE_SHARED_MEM
	__shared__ unsigned char shared_result_dec[THREADS_PER_BLOCK * dLEAP];
	__shared__ char shared_result_hex[THREADS_PER_BLOCK * dLEAP];
	for (int i = threadIdx.x; i < threadIdx.x + dLeap; i++)
	{
		shared_result_dec[i] = 0;
		shared_result_hex[i] = 0;
	}

	__syncthreads();
#endif
	__int64 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	__int64 precision_current = precision_init + tid * dLeap;
	if (precision_current >= precision_hex-1)
		return;
	double _16dPi = 0;

	_16dPi = 4 * F16dSj(precision_current, 1) - 2 * F16dSj(precision_current, 4) - F16dSj(precision_current, 5) - F16dSj(precision_current, 6);
	_16dPi = (_16dPi > 0) ? (_16dPi - (int) _16dPi) : (_16dPi - (int) _16dPi + 1);

#ifndef ENABLE_SHARED_MEM
	for (__int64 i = precision_current; i < precision_current + dLeap; i++)
	{
		_16dPi *= 16;
		result_dec[i] = (int) _16dPi;
		_16dPi -= result_dec[i];

		DecToHexModDiv(result_dec[i], result_hex[i]
#ifndef ENABLE_CONSTANT_MEM
			, hex_table
#endif
			);
	}
#else
	for (__int64 i = threadIdx.x * dLeap; i < threadIdx.x * dLeap + dLeap; i++)
	{
		_16dPi *= 16;
		shared_result_dec[i] = (int) _16dPi;
		_16dPi -= shared_result_dec[i];

		DecToHexModDiv(shared_result_dec[i], shared_result_hex[i]
#ifndef ENABLE_CONSTANT_MEM
			, hex_table
#endif
			);
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (__int64 i = precision_current; i < precision_current + THREADS_PER_BLOCK * dLeap; i++)
		{
			result_dec[i] = shared_result_dec[i - precision_current];
			result_hex[i] = shared_result_hex[i - precision_current];
		}
	}
#endif
}


int *pArgc = NULL;
char **pArgv = NULL;
unsigned char result_dec_cpu[PRECISION_HEX] = { 0 };
char result_hex_cpu[PRECISION_HEX + 1] = { 0 };
unsigned char *result_dec_gpu = NULL;
char *result_hex_gpu = NULL;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;

	printf("Starting...\n");

	/**********************************DeviceQuery*****************************************/
	printf("Device checking...\n");
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
	{
		errorPrint(error_id);
		printf("Device check failed, terminating...\n");
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0)
	{
		printf("ERROR: No CUDA device available\n");
		printf("Device check failed, terminating...\n");
		exit(EXIT_FAILURE);
	}
	else
		printf("\nDetected %d CUDA Capable device(s)\n\n", deviceCount);

	int deviceIterator, driverVersion, runtimeVersion;
	for (deviceIterator = 0; deviceIterator < deviceCount; deviceIterator++)
	{
		cudaSetDevice(deviceIterator);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceIterator);

		printf("Device %d: %s\n", deviceIterator, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", 
			driverVersion / 1000, 
			(driverVersion % 100) / 10, 
			runtimeVersion / 1000, 
			(runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
			(float) deviceProp.totalGlobalMem / 1048576.0f,
			(unsigned long long) deviceProp.totalGlobalMem);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}
#endif
		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", 
			(deviceProp.deviceOverlap ? "Yes" : "No"), 
			deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Warp Size:                                     %d\n", deviceProp.warpSize);
	}
	printf("Device check done...\n");

	/**********************************Pi Computing*****************************************/
	assert(PRECISION_OUTPUT_START < PRECISION_HEX && PRECISION_OUTPUT_END <= PRECISION_HEX);
	printf("Start to compute Pi...\n");
	cudaEvent_t start, stop;
	float elapseTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	hex_table_init();
	checkCudaErrors(cudaMalloc((void **) &result_dec_gpu, PRECISION_HEX * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void **) &result_hex_gpu, (PRECISION_HEX + 1) * sizeof(unsigned char)));

	checkCudaErrors(cudaEventRecord(start));
	PiOnGPU_Kernel<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(PRECISION_INIT, PRECISION_HEX, dLEAP, result_dec_gpu, result_hex_gpu
#ifndef ENABLE_CONSTANT_MEM
		, hex_table
#endif
		);
	checkCudaErrors(cudaEventRecord(stop));

	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapseTime, start, stop));

	//Copy back
	checkCudaErrors(cudaMemcpy(result_hex_cpu, result_hex_gpu, (PRECISION_HEX + 1) * sizeof(char), cudaMemcpyDeviceToHost));

	//Print
	for (__int64 i = PRECISION_OUTPUT_START; i < PRECISION_OUTPUT_END; i++)
		printf("%c", result_hex_cpu[i]);
	printf("\n");

	printf("Total Time: %f\n", elapseTime);

	cudaDeviceReset();
	//system("pause");
	exit(EXIT_SUCCESS);
}