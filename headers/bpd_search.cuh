/*
 * bpd_search.h
 *
 *  Created on: September 21, 2016
 *  Author: Keith Azzopardi
 */
#ifndef BPD_SEARCH_CUH_
#define BPD_SEARCH_CUH_

#include "bpd_cudacommon.cuh"
#include "bpd_subharmonics.cuh"



typedef struct ffdotInfo
{
	unsigned short currentHarmonic;
	unsigned short maxHarmonic;

	int numrs;        					// Number of Fourier Frequency present.
	int numzs; 							// Number of Fourier f-dots present.
	int rlo; 							// Lowest Fourier Frequency present.
	int zlo; 							// Lowest Fourier Frequency derivative (f-dot) present.
	double rhi;							// Highest Fourier Frequency bin.


	// Data.
//	cufftComplex *pdata;				// Being cleared after processing.
//	float *powers; 						// Powers array of length numrs for numzs times.
//	unsigned short *rinds; 				// Table of indices for Fourier Frequencies.
	int fftlen;							// Length of Fourier Transform.
} ffdotInfo;

cufftComplex* get_fourier_amplitudes(int loBin, int numbins, cufftComplex* fft, long dataNumbins, long dataLoBin);
void spreadData(cufftComplex * inputData, int inputElements, int interval, unsigned int startIdx, cufftComplex *outputData);

void generate_harmonics_ffdot(unsigned short totalHarmonics, unsigned short harmonicStages, SubharmonicInfo *subharmonicsInfo, double fullrlo, double fullrhi, double zlo, ffdotInfo *output_ffdot, unsigned short *output_binOffsetArray);
void generate_harmonic_ffdot_data(unsigned short totalHarmonics,  long totalDataLength, ffdotInfo *ffdot, unsigned short *binoffsetArray, long *dataStartIdx, cufftComplex *fft, long dataNumbins, long dataLoBin, cufftComplex *output_data);
void perform_fft_harmonics(unsigned short totalHarmonics, cufftComplex *device_data, long *dataStartIdx, cufftComplex *output_device_fftdata);
void perform_convolution(unsigned short totalHarmonics, cufftComplex *device_fft_data,  ffdotInfo *harmonicInfo, SubharmonicInfo *shi, long *dataStartIdx, cufftComplex *output_deviceConvolvedSignal, long *convolutionHarmonicIdx);
void perform_inversefft(unsigned short totalHarmonics, cufftComplex *device_convolvedSignal, long *convolutionHarmonicIdx, cufftComplex *output_device_ifftdata);
void generate_fundamental_powers(unsigned short totalHarmonics, cufftComplex *device_ifft_data, long *ifftHarmonicIdx,  ffdotInfo *harmonicInfo, SubharmonicInfo *shi, float *output_devicePowers);
void perform_harmonic_summing(unsigned short totalHarmonics, cufftComplex *device_ifft_data, long *ifftHarmonicIdx, ffdotInfo *harmonicInfo, SubharmonicInfo *shi, float *output_devicePowers, int fullrlo, unsigned short *host_rinds, unsigned short *device_rinds, int totalRindsBins);

void initialize_cuda_plans(unsigned short totalHarmonics, SubharmonicInfo *subharmonicInfo);
void free_cuda_plans(unsigned short totalHarmonics);
#endif /* BPD_SEARCH_CUH_ */
