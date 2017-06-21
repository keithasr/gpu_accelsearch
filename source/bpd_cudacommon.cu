#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void __checkCudaErrors(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i): getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(EXIT_FAILURE);
   }
}

void checkGPUMemory()
{
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status;
    cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){

        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

        exit(1);

    }
    else{
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    }
}
