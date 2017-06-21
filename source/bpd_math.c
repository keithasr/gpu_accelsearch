#include "../headers/bpd_math.h"

/// Return x such that 2**x = n
int fast_log2(int n)
{
	int x = 0;

	while (n > 1)
	{
		n >>= 1;
		x++;
	}
	return x;
}

/// This Quickselect routine is based on the algorithm described in
/// "Numerical recipies in C", Second Edition,  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
/// Fast computation of the median of an array.
/// Note:  It messes up the order!
float fast_median(float arr[], int n)
{
	int low, high;
	int median;
	int middle, ll, hh;

	low = 0;
	high = n - 1;
	median = (low + high) / 2;
	for (;;)
	{
		if (high <= low) /* One element only */
		{
			return arr[median];
		}

		if (high == low + 1)
		{ /* Two elements only */
			if (arr[low] > arr[high])
				ELEM_SWAP(arr[low], arr[high]);
			return arr[median];
		}

		/* Find median of low, middle and high items; swap into position low */
		middle = (low + high) / 2;
		if (arr[middle] > arr[high])
			ELEM_SWAP(arr[middle], arr[high]);
		if (arr[low] > arr[high])
			ELEM_SWAP(arr[low], arr[high]);
		if (arr[middle] > arr[low])
			ELEM_SWAP(arr[middle], arr[low]);

		/* Swap low item (now in position middle) into position (low+1) */
		ELEM_SWAP(arr[middle], arr[low + 1]);

		/* Nibble from each end towards middle, swapping items when stuck */
		ll = low + 1;
		hh = high;
		for (;;)
		{
			do
			{
				ll++;
			} while (arr[low] > arr[ll]);
			do
			{
				hh--;
			} while (arr[hh] > arr[low]);

			if (hh < ll)
			{
				break;
			}

			ELEM_SWAP(arr[ll], arr[hh]);
		}

		/* Swap middle item (in position low) back into correct position */
		ELEM_SWAP(arr[low], arr[hh]);

		/* Re-set active partition */
		if (hh <= median)
		{
			low = ll;
		}

		if (hh >= median)
		{
			high = hh - 1;
		}
	}
}

/// Return inverse square root of x using Fast inverse square root Algorithm. See wikipedia
float inverse_square_root(float x)
{
	union
	{
		float f;
		int i;
	} tmp;
	tmp.f = x;
	tmp.i = 0x5f3759df - (tmp.i >> 1);
	float y = tmp.f;
	return y * (1.5f - 0.5f * x * y * y);
}

int toNearestInt(double x)
{
	return (x < 0) ? (int) (x - 0.5) : (int) (x + 0.5);
}

/// Return the first value of 2^n >= x
long long get2PowerNGreaterThan(long long x)
{
	long long i = 1;

	while (i < x)
	{
		i <<= 1;
	}
	return i;
}

float** create2DMatrix(unsigned rows, unsigned cols)
{
	// Allocate data on the heap.
	int **arr;
	arr = (int **) malloc(sizeof(int *) * rows);
	arr[0] = (int *) malloc(sizeof(int) * cols * rows);

	// Set pointers position.
	int i;
	for (i = 1; i < rows; i++)
	{
		arr[i] = arr[i - 1] + cols;

	}

	return arr;
}
