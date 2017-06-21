/*
 * bpd_subharmonics.h
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */

#ifndef BPD_SUBHARMONICS_CUH_
#define BPD_SUBHARMONICS_CUH_

#include "bpd_cudacommon.cuh"

typedef struct SubharmonicInfo
{
	unsigned short maxHarmonic; 		  // The number of sub-harmonics
	unsigned short currentHarmonic; 	  // The sub-harmonic number (fundamental = numharm)
	int zmax; 							  // The maximum Fourier f-dot for this harmonic

	// Kernel Templates
	unsigned short totalKernelsTemplates; // Number of kernels in the vector
	int kernelFFTLength;			      // Length of each kernel
	int *z;								  // The fourier f-dot of each kernel
	unsigned short *kernelHalfWidth;      // Half width (bins) of each raw kernel;
	cufftComplex *kernels_data;			  // Kernel FFT data
} SubharmonicInfo;

SubharmonicInfo *createSubharmonicInfo(int numHarmonicStages, int zhi);

int calc_required_z(double harmonicFraction, double zfull);
double calc_required_r(double harmonicFraction, double rfull);
int index_from_r(double r, double rlo);
int index_from_z(double z, double zlo);
int getTotalHarmonics(int harmonicStages);

#endif /* BPD_SUBHARMONICS_CUH_ */
