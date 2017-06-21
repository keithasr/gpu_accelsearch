/*
 * bpd_convolution.cuh
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */
#ifndef BPD_GPU_KERNELS_CUH_
#define BPD_GPU_KERNELS_CUH_

#include "bpd_cudacommon.cuh"
#include "bpd_subharmonics.cuh"


/// Perform a Fourier Domain convolution of each segment with all templates.
__global__ void convolveSignal(int kernelTotalNumzs, int fftlen, cufftComplex* data, cufftComplex *kernels, int kernelTemplateLength, cufftComplex* outputdata);
__global__ void generatePowers(int kernelTotalNumzs, int numrs, int fftlen, cufftComplex* data, float scale, float* outputdata);
__global__ void addPowers(int fundamental_numzs, int fundamental_numrs, int subharm_fftlen, int fundamental_zlo, int subharm_zlo, int subharm_rlo, double harmonicFraction, cufftComplex* data, float scale, float* output_powers, int fullrlo);
#endif /* BPD_GPU_KERNELS_CUH_ */

