#include "../headers/bpd_cudacommon.cuh"
#include "../headers/bpd_fft.cuh"

unsigned realToComplexFFT(float* idata, long inputElements, cufftComplex **odata, int batch)
{
	// ---------------------- Initialise CUDA stuff ---------------------------------
	cudaSetDevice(0);
	cudaEvent_t event_start, event_stop;
	float timestamp;

	//  Events
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

	// FFT size.
	unsigned inputSize = inputElements * sizeof(cufftReal);
	unsigned outputElements = (floor(inputElements/2) + 1);
	unsigned outputSize = outputElements * sizeof(cufftComplex);

	// Init variables.
	cufftHandle plan;
	cufftReal *device_idata, *host_idata;
	cufftComplex *device_odata, *host_odata;
	host_idata = (cufftReal*) malloc(inputSize);
	host_odata = (cufftComplex *) malloc(outputSize);

	// ---------------------- Copy data to GPU ----------------------------------
	cudaMalloc((void **) &device_idata, inputSize);
	cudaMalloc((void **) &device_odata, outputSize);
	checkCudaErrors("Allocating memory on device");


	host_idata = (cufftReal*) idata;

	cudaEventRecord(event_start, 0);
	cudaMemcpy(device_idata, host_idata, inputSize, cudaMemcpyHostToDevice);
	checkCudaErrors("Copying data to device");
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	checkCudaErrors("Copying data to device timer");
	//printf("Copied to GPU in: %lf\n", timestamp);

	// ---------------------- FFT all the channels in place ----------------------
	cufftPlan1d(&plan, inputElements, CUFFT_R2C, batch);

	cudaEventRecord(event_start, 0);
	cufftExecR2C(plan, device_idata, device_odata);
	cudaThreadSynchronize();
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	checkCudaErrors("Performing FFT");
	//printf("Performed forward FFT: %lf\n", timestamp);

	cudaMemcpy(host_odata, device_odata, outputSize, cudaMemcpyDeviceToHost);

	*odata = host_odata;

//_________________________________________________________________________________________________________
// Perform IFFT.
//cufftHandle testPlan;
//cufftPlan1d(&testPlan, inputSamples, CUFFT_C2R, 1);
//cufftExecC2R(testPlan, device_odata, device_idata);
//checkCudaErrors("Performing IFFT");

//cudaMemcpy(host_idata, device_idata, outputSize, cudaMemcpyDeviceToHost);

//_________________________________________________________________________________________________________


	// ---------------------- Cuda Memory Cleanup ----------------------
	cufftDestroy(plan);
	cudaFree(device_idata);
	cudaFree(device_odata);

	return outputElements;
}

unsigned complexToComplexFFT(cufftComplex* device_idata, unsigned int inputElements, cufftComplex **device_odata, int batch)
{
	// ---------------------- Initialise CUDA stuff ---------------------------------
	cudaSetDevice(0);
	cudaEvent_t event_start, event_stop;
	float timestamp;

	//  Events
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

	// FFT size.
	unsigned int outputElements = inputElements;
	unsigned int outputSize = outputElements * sizeof(cufftComplex) * batch;

	// Init variables.
	cufftHandle plan;
	cufftComplex *odata;
	cudaMalloc((void **) &odata, outputSize);
	checkCudaErrors("Allocating memory on device");


	// ---------------------- FFT all the channels in place ----------------------
	cufftPlan1d(&plan, inputElements, CUFFT_C2C, batch);

	cudaEventRecord(event_start, 0);
	cufftExecC2C(plan, device_idata, odata,CUFFT_FORWARD);
	cudaThreadSynchronize();
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	checkCudaErrors("Performing FFT");
	//printf("Performed forward FFT: %lf\n", timestamp);

	*device_odata = odata;

//_________________________________________________________________________________________________________
// Perform IFFT.
//cufftHandle testPlan;
//cufftPlan1d(&testPlan, inputSamples, CUFFT_C2R, 1);
//cufftExecC2R(testPlan, device_odata, device_idata);
//checkCudaErrors("Performing IFFT");

//cudaMemcpy(host_idata, device_idata, outputSize, cudaMemcpyDeviceToHost);

//_________________________________________________________________________________________________________


	// ---------------------- Cuda Memory Cleanup ----------------------
	cufftDestroy(plan);

	return outputElements;
}

unsigned inverseComplexToComplexFFT(cufftComplex* device_idata, unsigned int inputElements, cufftComplex **device_odata, int batch)
{
	// ---------------------- Initialise CUDA stuff ---------------------------------
	cudaSetDevice(0);
	cudaEvent_t event_start, event_stop;
	float timestamp;

	//  Events
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

	// FFT size.
	unsigned int outputElements = inputElements;
	unsigned int outputSize = outputElements * sizeof(cufftComplex) * batch;

	// Init variables.
	cufftHandle plan;
	cufftComplex *odata;
	cudaMalloc((void **) &odata, outputSize);
	checkCudaErrors("Allocating memory on device");

	// ---------------------- FFT all the channels in place ----------------------
	cufftPlan1d(&plan, inputElements, CUFFT_C2C, batch);

	cudaEventRecord(event_start, 0);
	cufftExecC2C(plan, device_idata, odata,CUFFT_INVERSE);
	cudaThreadSynchronize();
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	checkCudaErrors("Performing FFT");
	//printf("Performed forward FFT: %lf\n", timestamp);


	*device_odata = odata;

//_________________________________________________________________________________________________________
// TESTING: Perform IFFT.
//cufftHandle testPlan;
//cufftPlan1d(&testPlan, inputSamples, CUFFT_C2R, 1);
//cufftExecC2R(testPlan, device_odata, device_idata);
//checkCudaErrors("Performing IFFT");

//cudaMemcpy(host_idata, device_idata, outputSize, cudaMemcpyDeviceToHost);

//_________________________________________________________________________________________________________


	// ---------------------- Cuda Memory Cleanup ----------------------
	cufftDestroy(plan);

	return outputElements;
}
