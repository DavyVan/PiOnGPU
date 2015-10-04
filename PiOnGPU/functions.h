#include <cuda_runtime.h>

//Binary exponentiation, (a ^ b) % mod
__device__ __int64 biexp(__int64 a, __int64 b, __int64 mod);

//Decimal --> Hex in char, 1 decimal digit convert to 2 hexadecimal characters
__device__ void DecToHexSingle(unsigned char dec, char * hex);

//Decimal --> Hex in char, convert all digits in a array
__device__ void DecToHexArray(unsigned char dec[], char hex[], int count);

__device__ void DecToHexModDiv(unsigned char dec, char &hex);


__device__ double _16dSj(__int64 d, int j);