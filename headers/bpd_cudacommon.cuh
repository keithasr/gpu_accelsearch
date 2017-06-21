/*
 * bpd_cudacommon.cuh
 *
 *  Created on: September 21, 2016
 *  Author: Keith Azzopardi
 */
#ifndef BPD_CUDACOMMON_CUH_
#define BPD_CUDACOMMON_CUH_

extern "C"{
	#include "bpd_common.h"
}
#include <cuda_runtime.h>
#include <cufft.h>


// This will output the proper error string when calling cudaGetLastError
#define checkCudaErrors(msg)      __checkCudaErrors (msg, __FILE__, __LINE__)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define KERNEL_MULTIPLIER 30
#define THREADS 128

void __checkCudaErrors(const char *errorMessage, const char *file, const int line);
void gpuAssert(cudaError_t code, const char *file, int line);
void checkGPUMemory();

#endif /* BPD_CUDACOMMON_CUH_ */
