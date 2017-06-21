extern "C"{
	#include "../headers/bpd_math.h"
	#include <omp.h>
}

#include "../headers/bpd_search.cuh"
#include "../headers/bpd_gpu_kernels.cuh"
#include "../headers/bpd_fft.cuh"

cufftHandle *plans_C2CFFT, *plans_IC2CFFT;
cudaStream_t *streams;
texture<unsigned short> tex_device_rinds;
dim3 threadsPerBlock(THREADS);


cufftComplex* get_fourier_amplitudes(int loBin, int numbins, cufftComplex* fft,
		long dataNumbins, long dataLoBin)
{

	cufftComplex *outputData;
	outputData = (cufftComplex *) malloc(sizeof(cufftComplex) * numbins);

	// Zero-pad if we try to read before the beginning of the FFT.
	long offset = 0;
	long binsDifference = loBin - dataLoBin;
	if (binsDifference < 0)
	{
		offset = llabs(loBin - dataLoBin);
		memset(outputData, 0, sizeof(cufftComplex) * offset);
	}
	long firstbin = binsDifference + offset;
	long newnumbins = numbins - offset;

	// Zero-pad if we try to read beyond the end of the FFT.

	if ((firstbin + newnumbins) > dataNumbins)
	{
		long numpad = firstbin + newnumbins - dataNumbins;
		newnumbins = newnumbins - numpad;
		memset(outputData + (numbins - numpad), 0, sizeof(cufftComplex) * numpad);
	}

	// Copy data.
	memcpy(outputData + offset, fft + firstbin, sizeof(cufftComplex) * newnumbins);

	return outputData;
}

/// Spread the input array for correlation.
/// Parameters:
///		'inputData' is the FFT data to be prepared.
///		'inputElements' is the number of elements in the 'inputData'.
///		'interval' is the number of bins in between.
/// 	'startIdx' is the starting location of the output array.
/// 	'outputData' is an array with the length of 'inputElements' * 'interval'  of the spreaded data.
void spreadData(cufftComplex * inputData, int inputElements, int interval, long startIdx, cufftComplex *outputData)
{
	int outputElements = inputElements * interval;

	memset(outputData + startIdx, 0, sizeof(cufftComplex) * outputElements);

	int inputCounter, outputCounter;
	for (inputCounter = 0, outputCounter = startIdx; inputCounter < inputElements; inputCounter++, outputCounter += interval)
	{
		outputData[outputCounter] = inputData[inputCounter];
	}
}

/// Generate Fourier Frequency Derivative information.
/// Paramters:
/// 	'output_binOffsetArray' = An array of bin offset, equivalent to the kernelHalfWidth for each harmonic. This is needed for the next step.
void generate_harmonics_ffdot(unsigned short totalHarmonics, unsigned short harmonicStages, SubharmonicInfo *subharmonicsInfo, double fullrlo, double fullrhi, double zlo, ffdotInfo *output_ffdot, unsigned short *output_binOffsetArray)
{
	unsigned short counter = 0;
	short int currentStage = 0;

	// For each stage...
	#pragma omp parallel for
	for(currentStage = 0; currentStage < harmonicStages; currentStage++)
	{
		// Process subharmonics.
		short int currentHarmonic = 1;
		short int maxHarmonic = (1 << currentStage); // binary shift left to get power of 2.

		// Cater for fundamental harmonic (this is performed only once).
		if (currentStage < 1)
		{
			maxHarmonic = 1;
		}

		// For each harmonic in stage...
		for (; currentHarmonic <= maxHarmonic; currentHarmonic += 2)
		{
			SubharmonicInfo *shi;
			shi = &subharmonicsInfo[counter];

			// Get the required information.
			double harmonicFraction = (double) currentHarmonic / (double) maxHarmonic;
			double drlo = calc_required_r(harmonicFraction, fullrlo);
			double drhi = calc_required_r(harmonicFraction, fullrhi);
			output_ffdot[counter].currentHarmonic = currentHarmonic;
			output_ffdot[counter].maxHarmonic = maxHarmonic;
			output_ffdot[counter].rlo = (int) floor(drlo);
			output_ffdot[counter].rhi = drhi;
			output_ffdot[counter].zlo = calc_required_z(harmonicFraction, zlo);

			if (currentHarmonic == 1 && maxHarmonic == 1) // Fundamental harmonic.
			{
				output_ffdot[counter].numrs = ACCEL_USELEN;
			}
			else
			{
				output_ffdot[counter].numrs = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
				if (output_ffdot[counter].numrs % ACCEL_RDR)
				{
					output_ffdot[counter].numrs = (output_ffdot[counter].numrs / ACCEL_RDR + 1) * ACCEL_RDR;
				}
			}

			output_ffdot[counter].numzs = shi->totalKernelsTemplates;
			output_ffdot[counter].fftlen = shi->kernelFFTLength;
			output_binOffsetArray[counter] = shi->kernelHalfWidth[0];

			counter++;
		}
	}
}


void generate_harmonic_ffdot_data(unsigned short totalHarmonics, long totalDataLength, ffdotInfo *ffdot, unsigned short *binoffsetArray, long *dataStartIdx, cufftComplex *fft, long dataNumbins, long dataLoBin, cufftComplex *output_data)
{
	// Determine data size and set the starting location of each struct.
	unsigned short iterator = 0;

	// Initialize Data.
	#pragma omp parallel for
	for	(iterator = 0; iterator < totalHarmonics; iterator++)
	{
		// Init bin information.
		int binoffset = binoffsetArray[iterator];
		int lobin = ffdot[iterator].rlo - binoffset;
		int hibin = (int) ceil(ffdot[iterator].rhi) + binoffset;
		int numdata = hibin - lobin + 1;
		int nice_numdata = get2PowerNGreaterThan(numdata);     // for FFTs

		if (nice_numdata != (ffdot[iterator].fftlen / ACCEL_NUMBETWEEN))
		{
			logMessage("ATTN: nice_numdata != equal to fftlen/2.");
		}

		// Create Local Fourier Amplitudes.
		cufftComplex *fourierAmplitudes;
		fourierAmplitudes = get_fourier_amplitudes(lobin, nice_numdata, fft, dataNumbins, dataLoBin);

		// Normalize the Fourier Amplitudes.
		{
			float *powers;
			powers = (float *) malloc((size_t) (sizeof(float) * numdata));
			int iCounter;
			for (iCounter = 0; iCounter < numdata; iCounter++)
			{
				// Get complex power.
				// power = r^2 + i^2
				powers[iCounter] = (fourierAmplitudes[iCounter].x * fourierAmplitudes[iCounter].x) + (fourierAmplitudes[iCounter].y * fourierAmplitudes[iCounter].y);
			}

			// Perform Median normalization.
			// Normalization Constant is calculated as 1.0 / sqrt(median/log(2)))
			double normalizeConst = inverse_square_root(fast_median(powers, numdata) / log(2.0));

			for (iCounter = 0; iCounter < numdata; iCounter++)
			{
				fourierAmplitudes[iCounter].x *= normalizeConst;
				fourierAmplitudes[iCounter].y *= normalizeConst;
			}

			free(powers);
		}

		// Prepare and spread the data with no padding, and FFT the data
		spreadData(fourierAmplitudes, ffdot[iterator].fftlen / ACCEL_NUMBETWEEN, ACCEL_NUMBETWEEN, dataStartIdx[iterator], output_data);
		free(fourierAmplitudes);
	}
}

void initialize_cuda_plans(unsigned short totalHarmonics, SubharmonicInfo *subharmonicInfo)
{
	plans_C2CFFT = (cufftHandle*) malloc(sizeof(cufftHandle) * totalHarmonics);
	plans_IC2CFFT = (cufftHandle*) malloc(sizeof(cufftHandle) * totalHarmonics);
	streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * totalHarmonics);

	// Perform C2C FFT on each harmonic signal.
	unsigned short counter = 0;
	for (; counter < totalHarmonics; counter++)
	{
		// Create Streams & Plan.
		//cudaStreamCreate(&streams[counter]);
		cufftPlan1d(&plans_C2CFFT[counter], subharmonicInfo[counter].kernelFFTLength, CUFFT_C2C, 1);
		cufftPlan1d(&plans_IC2CFFT[counter], subharmonicInfo[counter].kernelFFTLength, CUFFT_C2C, subharmonicInfo[counter].totalKernelsTemplates);

		cudaStreamCreate(&streams[counter]);
		// Set Stream to plan.
		//cufftSetStream(plans[counter], streams[counter]);
	}
}

void free_cuda_plans(unsigned short totalHarmonics)
{
	// Cuda Memory Clean-up;
	unsigned short counter = 0;
	for (; counter < totalHarmonics; counter++)
	{
		cufftDestroy(plans_C2CFFT[counter]);
		cufftDestroy(plans_IC2CFFT[counter]);
		cudaStreamDestroy(streams[counter]);
	}

	free(plans_C2CFFT);
	free(plans_IC2CFFT);
	free(streams);
}

/// Perform C2C FFT on each harmonic and return it on 'output_device_fftdata'
/// Parameters:
///			'device_data'	 : an array that contains all the data for each harmonic. This must be allocated on the GPU.
/// 		'output_device_fftdata' : one output array that is large enough to cater for the fft signals of each 'harmonicInfo'.
///							   This must already be initialized on the GPU.
void perform_fft_harmonics(unsigned short totalHarmonics, cufftComplex *device_data, long *dataStartIdx, cufftComplex *output_device_fftdata)
{
	// Perform C2C FFT on each harmonic signal.
	unsigned short counter = 0;
	for (counter = 0; counter < totalHarmonics; counter++)
	{
		// Perform C2C FFT.
		cufftExecC2C(plans_C2CFFT[counter], device_data + dataStartIdx[counter], output_device_fftdata + dataStartIdx[counter], CUFFT_FORWARD);
		checkCudaErrors("Performing FFT");
	}
	//cudaDeviceSynchronize();

}


void perform_convolution(unsigned short totalHarmonics, cufftComplex *device_fft_data,  ffdotInfo *harmonicInfo, SubharmonicInfo *shi, long *dataStartIdx, cufftComplex *output_deviceConvolvedSignal, long *convolutionHarmonicIdx)
{
	unsigned short counter = 0;
	for (counter = 0; counter < totalHarmonics; counter++)
	{
		// Execute Kernel to perform convolution.
		dim3 conv_numBlocks(ceil((float) harmonicInfo[counter].fftlen / (float) THREADS), ceil(harmonicInfo[counter].numzs / (float) KERNEL_MULTIPLIER));
		//logMessage("Convolution: Number of blocks X: %i Y: %i", conv_numBlocks.x, conv_numBlocks.y);

//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);

//		cudaEventRecord(start);
		convolveSignal<<<conv_numBlocks, threadsPerBlock,0, streams[counter]>>>(harmonicInfo[counter].numzs, harmonicInfo[counter].fftlen, device_fft_data + dataStartIdx[counter], shi[counter].kernels_data, harmonicInfo[counter].fftlen, output_deviceConvolvedSignal + convolutionHarmonicIdx[counter]);
//		cudaDeviceSynchronize();
//		cudaEventRecord(stop);
//
//		cudaEventSynchronize(stop);
//		float milliseconds = 0;
//		cudaEventElapsedTime(&milliseconds, start, stop);
//		printf("Performed kernel in: %f \n",milliseconds);
//		printf("Numzs %d FFTLEN: %d: \n",harmonicInfo[counter].numzs, harmonicInfo[counter].fftlen);
//		exit(-1);
	}
	//cudaDeviceSynchronize();
}

/// Perform C2C IFFT on each harmonic and numzs, and return it on 'output_device_ifftdata'
/// Parameters:
///			'device_convolvedSignal'	 : an array that contains all the data that has been previously convolved for each harmonic and numzs. This must be allocated on the GPU.
/// 		'output_device_ifftdata' : one output array that is large enough to cater for the fft signals for each numzs of each 'harmonicInfo'.
///							   This must already be initialized on the GPU.
void perform_inversefft(unsigned short totalHarmonics, cufftComplex *device_convolvedSignal, long *convolutionHarmonicIdx, cufftComplex *output_device_ifftdata)
{
	// Perform C2C IFFT on each convolved signal.
	unsigned short counter = 0;
	for (counter = 0; counter < totalHarmonics; counter++)
	{
		// Perform C2C IFFT.
		cufftExecC2C(plans_IC2CFFT[counter], device_convolvedSignal + convolutionHarmonicIdx[counter], output_device_ifftdata + convolutionHarmonicIdx[counter], CUFFT_INVERSE);
		checkCudaErrors("Performing IFFT");
	}
	//cudaDeviceSynchronize();
}


/// Generate noramlized powers using the inverse fft data.
/// Paramters:
///			'output_devicePowers': an array that contains all powers of all harmonics. This must already be initialized on the GPU.
void generate_fundamental_powers(unsigned short totalHarmonics, cufftComplex *device_ifft_data, long *ifftHarmonicIdx, ffdotInfo *harmonicInfo, SubharmonicInfo *shi, float *output_devicePowers)
{
	unsigned short counter = 0;

	// Execute Kernel to perform normalization.
	dim3 norm_numBlocks(ceil((float) harmonicInfo[counter].numrs / (float) THREADS), ceil(harmonicInfo[counter].numzs / (float) KERNEL_MULTIPLIER));
	//dim3 norm_numBlocks(ceil((float) harmonicInfo[counter].numrs / (float) THREADS), ceil(harmonicInfo[counter].numzs / (float) KERNEL_MULTIPLIER));

	int offset = shi[counter].kernelHalfWidth[0] * ACCEL_NUMBETWEEN;
	float normalizationConst = 1.0 / (harmonicInfo[counter].fftlen * harmonicInfo[counter].fftlen);

	generatePowers<<<norm_numBlocks, threadsPerBlock, 0, streams[counter]>>>(harmonicInfo[counter].numzs, harmonicInfo[counter].numrs, harmonicInfo[counter].fftlen, device_ifft_data + ifftHarmonicIdx[counter] + offset, normalizationConst, output_devicePowers);

//	cudaDeviceSynchronize();
}

/// Add Subharmonic Fourier Frequency derivative (f-dot) powers to fundamental.
void perform_harmonic_summing(unsigned short totalHarmonics, cufftComplex *device_ifft_data, long *ifftHarmonicIdx, ffdotInfo *harmonicInfo, SubharmonicInfo *shi, float *output_devicePowers, int fullrlo, unsigned short *host_rinds, unsigned short *device_rinds, int totalRindsBins)
{
	// Calculate RINDs.
//	unsigned short counter = 0;
//	unsigned short iCounter = 0;
//	for (counter = 1; counter < totalHarmonics; counter++)
//	{
//		ffdotInfo subharmonic = harmonicInfo[counter];
//		double harmonicFraction = (double) subharmonic.currentHarmonic / (double) subharmonic.maxHarmonic;
//		int offset = (counter - 1) * ACCEL_USELEN;
//
//		// Initialize the lookup indices.
//		for (iCounter = 0; iCounter < ACCEL_USELEN; iCounter++)
//		{
//			double rr = fullrlo + iCounter * ACCEL_DR;
//			//double subr = calc_required_r(harmonicFraction, rr);
//			double subr = (int) (ACCEL_RDR * rr * harmonicFraction + 0.5) * ACCEL_DR;
//			unsigned short rind = (unsigned short) ((subr - subharmonic.rlo) * ACCEL_RDR);
//			host_rinds[iCounter + offset] = rind;
//		}
//	}
//	cudaMemcpy(device_rinds, host_rinds, totalRindsBins * sizeof(unsigned short), cudaMemcpyHostToDevice);

	// Perform Search on the first harmonic. Use first streams[0]
	// performSearch();
	cudaDeviceSynchronize();

	// Add subharmonics
	unsigned short counter = 0;
	for(counter = 1; counter < totalHarmonics; counter++)
	{
		double harmonicFraction = (double) harmonicInfo[counter].currentHarmonic / (double) harmonicInfo[counter].maxHarmonic;

		// Execute Kernel to perform normalization.
		dim3 pow_numBlocks(ceil((float) harmonicInfo[0].numrs / (float) THREADS), ceil(harmonicInfo[0].numzs / (float) KERNEL_MULTIPLIER ));
		int offset = shi[counter].kernelHalfWidth[0] * ACCEL_NUMBETWEEN;
		float normalizationConst = 1.0 / (harmonicInfo[counter].fftlen * harmonicInfo[counter].fftlen);

		addPowers<<<pow_numBlocks, threadsPerBlock>>>(harmonicInfo[0].numzs, harmonicInfo[0].numrs, harmonicInfo[counter].fftlen, harmonicInfo[0].zlo,  harmonicInfo[counter].zlo, harmonicInfo[counter].rlo, harmonicFraction, device_ifft_data + ifftHarmonicIdx[counter] + offset, normalizationConst, output_devicePowers, fullrlo);

		if (((counter + 1) & (counter)) == 0)
		{
			// Perform Search on each stage of composite sub-harmonic.
			// Perform everything in the main stream, i.e. do not use streams.
			// All streams are synchronised in the previous barrier i.e. cudaDeviceSynchornise() call.
			// performSearch();
		}
	}

	//cudaDeviceSynchronize();
}
