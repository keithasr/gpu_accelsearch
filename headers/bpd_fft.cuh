
/*
 * bpd_fft.cuh
 *
 *  Created on: Jan 2, 2016
 *      Author: keithasr
 */
#ifndef BPD_FFT_CUH_
#define BPD_FFT_CUH_

#include "bpd_common.h"
#include "bpd_cudacommon.cuh"

unsigned realToComplexFFT(float* idata, long inputElements, cufftComplex **odata, int batch);
unsigned complexToComplexFFT(cufftComplex* device_idata, unsigned int inputElements, cufftComplex **device_odata, int batch);
unsigned inverseComplexToComplexFFT(cufftComplex* device_idata, unsigned int inputElements, cufftComplex **device_odata, int batch);

#endif /* BPD_FFT_CUH_ */
