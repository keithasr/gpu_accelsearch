extern "C" {
	#include "headers/bpd_common.h"
	#include "headers/bpd_math.h"
	#include "headers/bpd_noise.h"
	#include "headers/bpd_data.h"
	#include <sys/time.h>
}
#include "headers/bpd_cudacommon.cuh"
#include "headers/bpd_fft.cuh"
#include "headers/bpd_subharmonics.cuh"
#include "headers/bpd_gpu_kernels.cuh"
#include "headers/bpd_search.cuh"
#include "headers/bpd_candidates.cuh"
#include "headers/test_powers.cuh"

typedef struct SearchParameters
{
	unsigned short numHarmonics;  		// The number of Harmonics to search.
	unsigned short numHarmonicStages; 	// Number of stages of harmonic summing.
	int zlo; 						// The min Fourier Frequency derivative (f-dot) to search.
	int zhi;      					// The max Fourier Frequency derivative (f-dot) to search.

	unsigned rlo; 					// Minumum Fourier Frequency to search.
	unsigned rhi;					// Maximum Fourier Frequency to search.

} SearchParameters;

typedef struct DataInfo
{
	long binsInTimeSeries;
	float timeSeriesWidth;					// Time Series width in seconds.

	unsigned totalNumberBins;				// Total number of bins in the data.
	unsigned lowestBin;						// Lowest spectral bin present in the data.
	unsigned highestBin;					// Highest spectral bin present in the data.

	float *powersCutoff;       				// Cutoff powers to choose a cand (per harmsummed).
	long long *independentSpectraCount;		// Count of independent spectra per summed harmonic.

} DataInfo;


/* =======================================================================
 * Acceleration Search Main Loop
 * =======================================================================
 */

int main()
{
	int iCounter = 0;

	// Set search parameters.
	int zmax = 300;          //  int value between 0 and 1200.
	SearchParameters searchParam;
	searchParam.numHarmonics = 8;
	searchParam.zlo = -zmax;
	searchParam.zhi = zmax;
	searchParam.numHarmonicStages = fast_log2(searchParam.numHarmonics) + 1;
	searchParam.rlo = 0;     // Set to zero.
	searchParam.rhi = 0;     // Set to zero.

	/* =======================================================================
	 * Kernels
	 * =======================================================================
	 */

	// Generating correlation kernels
	SubharmonicInfo *subharmonicInfo;
	subharmonicInfo = createSubharmonicInfo(searchParam.numHarmonicStages, searchParam.zhi);

	/* =======================================================================
	 * Time Series input signal.
	 * =======================================================================
	 */

	// Initialize data.
	logMessage("Reading Data...");
	float *data;
	long dataLength = readData(
			"/users/keithazzopardi/BinaryPulsarDetection/Data/Lband_topo_DM0.00.dat", &data);

	// FFT time series signal.
	logMessage("FFTing signal...");
	cufftComplex *fft;
	unsigned fftLength = realToComplexFFT(data, dataLength, &fft, 1);

	// Set Data info.
	DataInfo dataInfo;
	dataInfo.binsInTimeSeries = dataLength;
	dataInfo.timeSeriesWidth = 0.000072;
	dataInfo.totalNumberBins = fftLength - 1; // NOTE: Subtraction of 1 because of R2C FFT Length differs by one
	dataInfo.lowestBin = 0;
	dataInfo.highestBin = fftLength - 1;     // NOTE: Subtraction of 1 because of R2C FFT Length differs by one

	dataInfo.powersCutoff = (float *) malloc(searchParam.numHarmonicStages * sizeof(float));
	dataInfo.independentSpectraCount = (long long *) malloc(searchParam.numHarmonicStages * sizeof(long long));
	// TODO: Set powersCutOff and independentSpectraCount


	// Set search info.
	searchParam.rlo = floor(1.0 * (dataInfo.binsInTimeSeries * dataInfo.timeSeriesWidth));
	searchParam.rhi = fftLength - 1;     	// TODO: subract 1

	logMessage("Reddening signal...");
	deredden(fft, dataInfo.totalNumberBins);

//	logMessage("Min Frequency to Search %i", searchParam.rlo);
//	logMessage("Max Frequency to Search %i", searchParam.rhi);
//	logMessage("First Fourier Frequency  %i", dataInfo.lowestBin);
//	logMessage("Highest Bin %i", dataInfo.highestBin);


	/* =======================================================================
	 * Acceleration Search Main Loop
	 * =======================================================================
	 */
	{
		// NOTE: Fourier Frequency,r, = f*T_obs = T_obs/P
		// where T_obs is the observation time and P_obs is the orbital period.

		logMessage("ACCEL_USELEN %i", ACCEL_USELEN);
		logMessage("ACCEL_DR %f", ACCEL_DR);

		// Get the total number of harmonics in ALL STAGES.
		// NOTE: Each stage has multiple harmonics.
		unsigned short totalHarmonics = getTotalHarmonics(searchParam.numHarmonicStages);

		// Init search variables.
		ffdotInfo *harmonicInfo;
		harmonicInfo = (ffdotInfo *) malloc(totalHarmonics * sizeof(ffdotInfo));		// NOTE: 1D array is created instead of jagged array.
		unsigned short *binOffsetArray;
		binOffsetArray = (unsigned short *) malloc (totalHarmonics * sizeof(unsigned short));
	    unsigned int counter = 0;

	    // Determine sizes.
	    long totalFFTBins = 0; 						// Number of FFT in all harmonics.
	    long totalConvolutionBins = 0; 				// Number of bins required for all convolution, i.e fftlen * numzs
	    long *dataStartIdx;			   				// The index of starting location of the data for this structure.
	    long *convolutionHarmonicIdx;				// The index of starting location of the convolution array.
	    dataStartIdx = (long *) malloc (totalHarmonics * sizeof(long));// The starting idx of each harmonic data signal.
	    convolutionHarmonicIdx = (long *) malloc (totalHarmonics * sizeof(long));// The starting idx the convolution each harmonic.
	    for (counter = 0; counter < totalHarmonics; counter++)
	    {
	    	dataStartIdx[counter] = totalFFTBins;

	    	// Determine FFT variables.
	    	totalFFTBins += subharmonicInfo[counter].kernelFFTLength;

	    	// Determine convolution variables
	    	convolutionHarmonicIdx[counter] = totalConvolutionBins;
	    	// NOTE: Size is FFTLEN but convolution is perform for each NUMZS.
	    	//logMessage("FFTLen %d Numzs %d", subharmonicInfo[counter].kernelFFTLength, subharmonicInfo[counter].totalKernelsTemplates);
	    	totalConvolutionBins += (subharmonicInfo[counter].kernelFFTLength * subharmonicInfo[counter].totalKernelsTemplates);// NOTE: harmonicInfo[counter].fftlen * harmonicInfo[counter].numzs;
	    }
	    //Init Data variables.
		cufftComplex *host_data;
		//host_data = (cufftComplex *) malloc((size_t) (sizeof(cufftComplex) * totalFFTBins));
		cudaMallocHost((void **) &host_data, sizeof(cufftComplex) * totalFFTBins);
		//cudaHostAlloc((void **) &host_data, sizeof(cufftComplex) * totalFFTBins,cudaHostAllocMapped);

		checkGPUMemory();

	    checkCudaErrors("GPU MEMORY");

	    //Init FFT Variables.
	    cufftComplex *device_data, *device_fftdata;
	    cudaMalloc((void **) &device_data, totalFFTBins * sizeof(cufftComplex));
	    cudaMalloc((void **) &device_fftdata, totalFFTBins * sizeof(cufftComplex));
		checkCudaErrors("Allocating memory on device.");

		//Init Convolution arrays.
		cufftComplex *device_convolvedSignal, * device_ifftdata;
		cudaMalloc((void**) &device_convolvedSignal, sizeof(cufftComplex) * totalConvolutionBins);
		cudaMalloc((void**) &device_ifftdata, sizeof(cufftComplex) * totalConvolutionBins);
	    checkCudaErrors("Allocating Convolution Array on device.");

	    logMessage("Total Convolution Bins: %ld", totalConvolutionBins);

	    // Init Powers arrays.
	    unsigned short *host_rinds, *device_rinds;
		unsigned short totalRindsBins = ACCEL_USELEN * (totalHarmonics - 1);
		//host_rinds = (unsigned short *) malloc( totalRindsBins * sizeof(unsigned short));
		cudaMallocHost((void **) &host_rinds, totalRindsBins * sizeof(unsigned short));
		cudaMalloc((void**) &device_rinds, totalRindsBins * sizeof(unsigned short));

		initialize_cuda_plans(totalHarmonics,subharmonicInfo);

		// Set GPU Device.
		cudaSetDevice(0);

		unsigned short USELEN_MUL_DR = ACCEL_USELEN * ACCEL_DR;


		struct timeval  t1, t2;
//	    clock_t t3, t4;
	    double total_t;
	    double time[10];


	    float *device_powers;
    	cudaMalloc((void**) &device_powers, sizeof(float) * ACCEL_USELEN * subharmonicInfo[0].totalKernelsTemplates);

    	checkGPUMemory();

//	    double total_t;
//	    double time[10];
//	    t1 = clock();
	    gettimeofday(&t1,0);


	    int _iter = 0;
		int iteration = 0;

		double startr, lastr, nextr;
	    while (_iter < 10)
	    {
			startr = searchParam.rlo;
			lastr = 0.0;
			nextr = 0.0;


		while ((startr + (USELEN_MUL_DR)) < dataInfo.highestBin)
		{
			nextr = startr + USELEN_MUL_DR;
			lastr = nextr - ACCEL_DR;

			logMessage("Performed iteration %d",iteration);

			// Generate harmonics information struct.
//			t3 = clock();
			generate_harmonics_ffdot(totalHarmonics, searchParam.numHarmonicStages,subharmonicInfo, startr, lastr, searchParam.zlo, harmonicInfo, binOffsetArray);

		    // Generate Data.
		    generate_harmonic_ffdot_data(totalHarmonics, totalFFTBins, harmonicInfo, binOffsetArray, dataStartIdx, fft, dataInfo.totalNumberBins, dataInfo.lowestBin, host_data);
//			t4 = clock();
//			total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
//			time[0] += total_t;

//		    checkGPUMemory();

			// Copy segmented and spreaded FFT data on device.
		    // NOTE: For each harmonic, each size is (fftlen / ACCEL_NUMBETWEEN) * ACCEL_NUMBETWEEN that is equivalent to FFTLEN.
		    cudaMemcpy(device_data, host_data, totalFFTBins * sizeof(cufftComplex), cudaMemcpyHostToDevice);
		    //checkCudaErrors("Error copying data to device.");

		    // Perform C2C FFT on each harmonic and save output to 'device_fftdata'.
//		    t3 = clock();
    	    perform_fft_harmonics(totalHarmonics, device_data, dataStartIdx, device_fftdata);
//    	    cudaDeviceSynchronize();
//			t4 = clock();
//			total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
//			time[1] += total_t;
//			checkCudaErrors("Error FFTing Harmonics.");

		    // Perform Convolution of each harmonic.
//			t3 = clock();
		    perform_convolution(totalHarmonics, device_fftdata, harmonicInfo, subharmonicInfo, dataStartIdx, device_convolvedSignal, convolutionHarmonicIdx);
//			cudaDeviceSynchronize();
//		    t4 = clock();
//			total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
//			time[2] += total_t;
//		    checkCudaErrors("Error performing Convolution.");

		    // Perform Inverse FFT.
//		    t3 = clock();
		    perform_inversefft(totalHarmonics, device_convolvedSignal, convolutionHarmonicIdx, device_ifftdata);
//			cudaDeviceSynchronize();
//		    t4 = clock();
//			total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
//			time[3] += total_t;

//		    cufftComplex *host_result;
//		    host_result = (cufftComplex *) malloc(sizeof(cufftComplex) * 1024);
//		    cudaMemcpy(host_result, device_ifftdata, 1024 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
//		    int testCounter = 0;
//		    for(testCounter = 0; testCounter < 1024; testCounter++)
//		    {
//		     printf("#%d X: %f Y: %f \n", testCounter, host_result[testCounter].x, host_result[testCounter].y);
//		    }

//		    exit(-1);

			//t4 = clock();
			//total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
			//time[3] += total_t;
//		    checkCudaErrors("Error performing inverse fft.");
//
//		    t3 = clock();
		    generate_fundamental_powers(totalHarmonics, device_ifftdata, convolutionHarmonicIdx, harmonicInfo, subharmonicInfo, device_powers);
//		    checkCudaErrors("Error generating powers.");

		    perform_harmonic_summing(totalHarmonics, device_ifftdata, convolutionHarmonicIdx, harmonicInfo, subharmonicInfo, device_powers, startr, host_rinds, device_rinds, totalRindsBins);
//		    cudaDeviceSynchronize();
//		    t4 = clock();
//		    total_t = (double) (t4 - t3) / (double) CLOCKS_PER_SEC;
//		    time[4] += total_t;

//		    cudaDeviceSynchronize();
//		    float *host_powers;
//			host_powers = (float *) malloc(sizeof(float) * ACCEL_USELEN * subharmonicInfo[0].totalKernelsTemplates);
//			cudaMemcpy(host_powers, device_powers, ACCEL_USELEN * subharmonicInfo[0].totalKernelsTemplates * sizeof(float), cudaMemcpyDeviceToHost);
//
//			// Write to file.
//			FILE *fp;
//			fp=fopen("fundamental_powers_added_r0_cuda_new.csv","w+");
//			int _idx = 0;
//			int _start = 0;
//			int _end = harmonicInfo[1].numrs;
//			for(_idx = _start ;_idx < _end; _idx++)
//			{
//			  fprintf(fp,"%i",_idx);
//			  if (_idx < _end - 1) fprintf(fp,",");
//			}
//			fprintf(fp,"\n");
//
//			for(_idx = _start ;_idx < _end; _idx++)
//			{
//			  fprintf(fp,"%f",host_powers[_idx]);
//			  if (_idx < _end - 1) fprintf(fp,",");
//			}
//			exit(-1);

			startr = nextr;
			iteration++;
		  }
		_iter++;
	    }

	    gettimeofday(&t2, 0);
	    long elapsed = (t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec;

//	t2 = clock();
//	total_t = (double) (t2 - t1) / (double) CLOCKS_PER_SEC;
    logMessage("Performed search in \t %f", elapsed/(float) 1000000);
	int a = 0;
	for(a = 0; a< 10; a++)
	{
		logMessage("Performed %d in \t %f", a, time[a]  );
	}


	// Memory Clean-up.
    free_cuda_plans(totalHarmonics);
    free(dataStartIdx);
    free(convolutionHarmonicIdx);
	free(harmonicInfo);
	cudaFreeHost(host_data);
	cudaFreeHost(host_rinds);
	free(fft);
	cudaFree(device_data);
	cudaFree(device_fftdata);
	cudaFree(device_convolvedSignal);

    for (counter = 0; counter < totalHarmonics; counter++)
    {
    	free(subharmonicInfo[counter].kernelHalfWidth);
        cudaFree(subharmonicInfo[counter].kernels_data);
    }
	free(subharmonicInfo);

    cudaDeviceReset();

	}


	/* =======================================================================
	 * Optimize Candidates
	 * =======================================================================
	 */
	{

		int numCandidates = 0;
		// TODO: Set number of candidates.
		Candidates *cands;
		cands = (Candidates *) malloc((size_t) (sizeof(Candidates) * 1));


		if (numCandidates > 0)
		{
			// Sort candidates according to the optimized sigmas.
			sortCandidates();

			// Eliminate (most of) the harmonically related candidates.
			if ((searchParam.numHarmonics > 1))
			{
				eliminateHarmonics();
			}

			// Optimize each candidate and its harmonics.
			for (iCounter = 0; iCounter < numCandidates; iCounter++)
			{
				optimizeCandidates();
			}

			// Calculate the properties of the fundamentals.
			for (iCounter = 0; iCounter < numCandidates; iCounter++)
			{
				calculateProperties();
			}
		}
	}

	return 0;
}
