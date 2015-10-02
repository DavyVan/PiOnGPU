#include "functions.h"

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

char hex_table[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

__device__ void DecToHexSingle(unsigned char dec, char * hex)
{
	*hex = hex_table[dec >> 4];
	*(hex + 1) = hex_table[dec & 0x0F];
}

__device__ void DecToHexArray(unsigned char dec[], char hex[], int count)
{
	for (int i = 0; i < count; i++)
		DecToHexSingle(dec[i], &hex[2 * i]);
}