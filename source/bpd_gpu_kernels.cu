#include "../headers/bpd_gpu_kernels.cuh"

// Helper functions and utilities to work with CUDA

/*
/// Perform a Fourier Domain convolution of each segment with all
/// templates (complex multiply - scale), 2-dimensional f − f-dot plane produced.
__global__ void convolveSignal(int numzs, cufftComplex* data, cufftComplex *kernels, int kernelTemplateLength, cufftComplex* outputdata)
{
	//int currentnumzs = threadIdx.x + (gridDim.y * blockDim.x * blockIdx.y);
	int currentnumzs = threadIdx.x + (blockDim.x * blockIdx.y);
	int currentfft = blockIdx.x;
	int idx = blockIdx.x + (gridDim.x * currentnumzs);

	if ( currentnumzs < numzs)
	{
		float data_real = data[currentfft].x;
		float data_imag = data[currentfft].y;
		float kern_real = kernels[idx].x;
		float kern_imag = kernels[idx].y;

		outputdata[idx].x = data_real * kern_real + data_imag * kern_imag;
		outputdata[idx].y = data_imag * kern_real - data_real * kern_imag;

//		cufftComplex dataVal = data[currentfft];
//		cufftComplex kernVal = kernels[idx];
//		cufftComplex result;
//		result.x = dataVal.x * kernVal.x + dataVal.y * kernVal.y;
//		result.y = dataVal.y * kernVal.x - dataVal.x * kernVal.y;
//		outputdata[idx] = result;

	}
//	if (currentnumzs <5 & currentfft < 5)
//	{
//		printf("Convolved signal #%i %i [%i] X: %f Y: %f\n", currentnumzs, currentfft, idx, outputdata[idx].x, outputdata[idx].y);
//	}
}*/

__global__ void convolveSignal(int kernelTotalNumzs, int fftlen, cufftComplex* data, cufftComplex *kernels, int kernelTemplateLength, cufftComplex* outputdata)
{
	register int currentFFT = threadIdx.x + blockDim.x * blockIdx.x;

	// Validation: Do not continue if currentFFT exceeds the total length of fft samples.
	if (currentFFT > fftlen)
	{
		return;
	}

	// Get data samples.
	register float data_real = data[currentFFT].x;
	register float data_imag = data[currentFFT].y;

//	register int iterator = 0;
//	register int currentNumzs = 0;
	register int segmentedNumzs = blockIdx.y * KERNEL_MULTIPLIER;
	// Current numzs without the iterator (z)
	register int currentNumzs = 0;
	register int maxNumzs = (segmentedNumzs + KERNEL_MULTIPLIER);
	if(maxNumzs > kernelTotalNumzs)
	{
		maxNumzs = kernelTotalNumzs;
	}

	for(currentNumzs = segmentedNumzs; currentNumzs < maxNumzs; currentNumzs++)
	{
		int idx = currentFFT + (currentNumzs * fftlen); //currentFFT + (currentNumzs * fftlen);

		// Get kernel data for the current template.
		register float kern_real = kernels[idx].x;
		register float kern_imag = kernels[idx].y;

		// Perform convolution.
		outputdata[idx].x = (data_real * kern_real + data_imag * kern_imag);
		outputdata[idx].y = (data_imag * kern_real - data_real * kern_imag);

//			if (currentNumzs <5 & currentFFT < 5)
//			{
//				printf("Convolved signal #%i %i [%i] X: %f Y: %f\n", currentNumzs, currentFFT, idx, outputdata[idx].x, outputdata[idx].y);
//			}
	}
}


__global__ void generatePowers(int kernelTotalNumzs, int numrs, int fftlen, cufftComplex* data, float scale, float* outputdata)
{
	register int currentNumrs = threadIdx.x + blockDim.x * blockIdx.x;

	// Validation: Do not continue if currentNumzs exceeds total good bins (numrs).
	if (currentNumrs >= numrs)
	{
		return;
	}

	// Current numzs without the iterator (z)
	register int segmentedNumzs = blockIdx.y * KERNEL_MULTIPLIER;
	// Current numzs without the iterator (z)
	register int currentNumzs = 0;
	register int maxNumzs = (segmentedNumzs + KERNEL_MULTIPLIER);
	if(maxNumzs > kernelTotalNumzs)
	{
		maxNumzs = kernelTotalNumzs;
	}

	for(currentNumzs = segmentedNumzs; currentNumzs < maxNumzs; currentNumzs++)
	{
		//int currentNumzs = segmentedNumzs + iterator; //blockIdx.y * z + iterator;
		int output_idx = currentNumrs + (currentNumzs * numrs); //output_idx = currentNumrs + (currentNumzs * numrs);
		int data_idx = currentNumrs + (currentNumzs * fftlen); //currentNumrs + (currentNumzs * fftlen) + offset;

		// Normalize.
		register float data_real = data[data_idx].x;
		register float data_imag = data[data_idx].y;
		outputdata[output_idx] = (data_real * data_real + data_imag * data_imag) * scale;
	}

}


__global__ void addPowers(int fundamental_numzs, int fundamental_numrs, int subharm_fftlen, int fundamental_zlo, int subharm_zlo, int subharm_rlo, double harmonicFraction, cufftComplex* data, float scale, float* output_powers, int fullrlo)
{
	register int currentNumrs = threadIdx.x + blockDim.x * blockIdx.x;

	// Validation: Do not continue if currentNumzs exceeds total good bins (numrs).
	if (currentNumrs >= fundamental_numrs)
	{
		return;
	}

	// Current numzs without the iterator (z)
	register int segmentedNumzs = blockIdx.y * KERNEL_MULTIPLIER;
	// Current numzs without the iterator (z)
	register int currentNumzs = 0;
	register int maxNumzs = (segmentedNumzs + KERNEL_MULTIPLIER);
	if(maxNumzs > fundamental_numzs)
	{
		maxNumzs = fundamental_numzs;
	}

	double rr = fullrlo + currentNumrs * ACCEL_DR;
	double subr = (int) (ACCEL_RDR * rr * harmonicFraction + 0.5) * ACCEL_DR;
	register unsigned short subharmonic_idx_rind = (unsigned short) ((subr - subharm_rlo) * ACCEL_RDR);
	//register int subharmonic_idx_rinds = rinds[currentNumrs];

	for(currentNumzs = segmentedNumzs; currentNumzs < maxNumzs; currentNumzs++)
	{
		// Calculate ZINDs.
		int zz = fundamental_zlo + currentNumzs * ACCEL_DZ;
		//double z_temp = (ACCEL_RDZ * zz * harmonicFraction);
		//int nearestInt = lrint(z_temp + 0.1); // To Nearest int.
		//int subz = nearestInt * ACCEL_DZ;
		//int subz = z_temp *ACCEL_DZ;
		int subz = zz * harmonicFraction;
		int zind = (int) ((subz - subharm_zlo) * ACCEL_RDZ);


		int powers_idx = currentNumrs + (currentNumzs * fundamental_numrs);
		int subharmonic_idx = subharmonic_idx_rind + (subharm_fftlen * zind); //rinds[currentNumrs] + (subharm_fftlen * zind) + offset;

		// Normalize.
		register float data_real = data[subharmonic_idx].x;
		register float data_imag = data[subharmonic_idx].y;
		output_powers[powers_idx] += (data_real * data_real + data_imag * data_imag) * scale;

//		if (currentNumzs < 8 && currentNumrs < 8)
//		{
			//printf("Numzs: %i Numrs: %i DataIdx: %i Val: %f \n",currentNumzs, currentNumrs, data_idx, ((data_real * data_real + data_imag * data_imag) * scale));
//			printf("POW: %i ZIND: %i RIND: %i data: %f \n", powers_idx, zind, rinds[currentNumrs + rindsOffset], (data_real * data_real + data_imag * data_imag) * scale);
//		}
	}
}
