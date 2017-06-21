extern "C"
{
	#include "../headers/bpd_math.h"
	#include "../headers/bpd_fresnl.h"
}
#include "../headers/bpd_subharmonics.cuh"
#include "../headers/bpd_fft.cuh"

/// Calculate the required Fourier Dots (z) for subharmonic.
int calc_required_z(double harmonicFraction, double zfull)
{
	return NEAREST_INT(ACCEL_RDZ * zfull * harmonicFraction) * ACCEL_DZ;
}

/// Calculate the required Fourier Frequencies (r) for subharmonic.
double calc_required_r(double harmonicFraction, double rfull)
{
	return (int) (ACCEL_RDR * rfull * harmonicFraction + 0.5) * ACCEL_DR;
}

/// Return the approximate kernel half-width in FFT bins required to achieve
/// a reasonable high accuracy correlation based correction or interpolation for a Fourier signal
/// with constant f-dot. (i.e  a constant frequency derivative)
/// Parameters:
///   'z' is the Fourier Frequency derivative (# of bins the signal
///      smears over during the observation).
///   'accuracy' is either LOWACC or HIGHACC.
/// Result:
///    Since, this is based on the kernel half-width, the result must be multiplied by 2*'numbetween'
///    to get the actual length of the array required to hold such a kernel.
int z_response_halfwidth(double z)
{
	z = fabs(z);

	int m = (int) (z * (0.00089 * z + 0.3131) + NUMFINTBINS);
	m = (m < NUMFINTBINS) ? NUMFINTBINS : m;

	/* Prevent the equation from blowing up in large z cases */
	if (z > 100 && m > 0.6 * z)
	{
		m = 0.6 * z;
	}

	return m;
}

/// Return an index for a Fourier Frequency given an array with stepsize ACCEL_DR.
int index_from_r(double r, double rlo)
{
	return (int) ((r - rlo) * ACCEL_RDR + DBLCORRECT);
}

/// Return an index of Fourier Frequency Dot (z) given an array with stepsize of ACCEl_DZ.
int index_from_z(double z, double zlo)
{
	return (int) ((z - zlo) * ACCEL_RDZ + DBLCORRECT);
}

/// Calculate the length of the fft required to process subharmonics.
int calculateFFTLength(unsigned short numharm, unsigned short harmnum, int max_zfull)
{
	double harmonicFraction = (double) harmnum / (double) numharm;

	int bins_needed = ((ACCEL_USELEN * harmnum) / numharm) + 2;
	int end_effects = 2 * ACCEL_NUMBETWEEN * z_response_halfwidth(calc_required_z(harmonicFraction, max_zfull));
	return get2PowerNGreaterThan(bins_needed + end_effects);
}



/// Generate a complex response function for Fourier Interpolation (r).
///  Paramters:
///    'roffset' is the offset in Fourier bins for the full response
///       (i.e. At this point, the response would equal 1.0)
///    'numbetween' is the number of points to interpolate between
///       each standard FFT bin.  (i.e. 'numbetween' = 2 = interbins)
///    'numkern' is the number of complex points that the kernel will       contain.
cufftComplex *generate_r_response(double roffset, int numbetween, int numkern)
{
	cufftComplex *response;

	// Validation: Check that parameters are valid.
	if (roffset < 0.0 || roffset >= 1.0)
	{
		throwError("roffset = %f is out of bounds in generate_r_response().", roffset);
	}
	if (numbetween < 1 || numbetween >= 20000)
	{
		throwError("numbetween = %d out of bounds in generate_r_response().", numbetween);
	}
	if (numkern < numbetween)
	{
		throwError("numkern = %d out of bounds in generate_r_response().", numkern);
	}
	if ((numkern % (2 * numbetween)) != 0)
	{
		throwError("numkern %% (2 * numbetween) != 0 in generate_r_response().");
	}

	// Initialize variables.
	response = (cufftComplex *) malloc(
			(size_t) (sizeof(cufftComplex) * numkern));
	double startr = PI * (numkern / (double) (2 * numbetween) + roffset);
	double delta = -PI / numbetween;
	double tmp = sin(0.5 * delta);
	double alpha = -2.0 * tmp * tmp;
	double beta = sin(delta);
	double c = cos(startr);
	double s = sin(startr);

	// Generate points.
	int iCounter = 0;
	double r = startr;
	double sinc;
	for (iCounter = 0; iCounter < numkern; iCounter++, r += delta)
	{
		if (r == 0.0)
			sinc = 1.0;
		else
			sinc = s / r;
		response[iCounter].x = c * sinc;
		response[iCounter].y = s * sinc;
		c = alpha * (tmp = c) - beta * s + c;
		s = alpha * s + beta * tmp + s;
	}

	// Correct for divide by zero when the roffset is close to zero.
	if (roffset < 1E-3)
	{
		response[numkern / 2].x = 1
				- 6.579736267392905746 * (tmp = roffset * roffset);
		response[numkern / 2].y = roffset * (PI - 10.335425560099940058 * tmp);
	}
	return response;
}

/// Generate the response function for Fourier f-dot interpolation (z).
/// Parameters:
///   'roffset' is the offset in Fourier bins for the full response
///      (i.e. At this point, the response would equal 1.0)
///   'numbetween' is the number of points to interpolate between
///       each standard FFT bin.  (i.e. 'numbetween' = 2 = interbins)
///    'z' is the Fourier Frequency derivative (# of bins the signal
///       smears over during the observation).
///    'numkern' is the number of complex points that the kernel will contain.
cufftComplex *generate_z_response(double roffset, int numbetween, double z, int numkern)
{
	cufftComplex *response;

	// Validation: Check parameters boundaries.
	if (roffset < 0.0 || roffset >= 1.0)
	{
		throwError("roffset = %f out of bounds in generate_z_response().", roffset);
	}
	if (numbetween < 1 || numbetween >= 20000)
	{
		throwError("numbetween = %d out of bounds in generate_z_response().", numbetween);
	}
	if (numkern < numbetween)
	{
		throwError("numkern = %d out of bounds in generate_z_response().", numkern);
	}
	if ((numkern % (2 * numbetween)) != 0)
	{
		throwError("numkern %% (2 * numbetween) != 0 in generate_z_response().");
	}

	// Validation: Use the normal Fourier interpolation kernel when z~=0 .
	double absz = fabs(z);
	if (absz < 1E-4)
	{
		response = generate_r_response(roffset, numbetween, numkern);
		return response;
	}

	response = (cufftComplex *) malloc(
			(size_t) (sizeof(cufftComplex) * numkern));

	// Perform calculations.
	double startr = roffset - (0.5 * z);
	double tmprl;
	double startroffset = (startr < 0) ? 1.0 + modf(startr, &tmprl) : modf(startr, &tmprl);
	int signz = (z < 0.0) ? -1 : 1;
	double zd = signz * SQRT2 / sqrt(absz);
	double cons = zd / 2.0;
	double pibyz = PI / z;
	startr += numkern / (double) (2 * numbetween);
	double delta = -1.0 / numbetween;

	int iCounter;
	double s, c;
	double tmp, r, xx, yy, zz;
	double fressy, frescy, fressz, frescz, tmpim; // fresnl variables.
	for (iCounter = 0, r = startr; iCounter < numkern; iCounter++, r += delta)
	{
		yy = r * zd;
		zz = yy + z * zd;
		xx = pibyz * r * r;
		c = cos(xx);
		s = sin(xx);
		fresnl(yy, &fressy, &frescy);
		fresnl(zz, &fressz, &frescz);
		tmprl = signz * (frescz - frescy);
		tmpim = fressy - fressz;
		response[iCounter].x = ((tmp = tmprl) * c - tmpim * s) * cons;
		response[iCounter].y = -(tmp * s + tmpim * c) * cons;
	}

	// Correct for divide by zero when the roffset and z is close to zero.
	int numkernby2;
	if (startroffset < 1E-3 && absz < 1E-3)
	{
		zz = z * z;
		xx = startroffset * startroffset;
		numkernby2 = numkern / 2;
		response[numkernby2].x = 1.0 - 0.16449340668482264365 * zz;
		response[numkernby2].y = -0.5235987755982988731 * z;
		response[numkernby2].x += startroffset * 1.6449340668482264365 * z;
		response[numkernby2].y += startroffset
				* (PI - 0.5167712780049970029 * zz);
		response[numkernby2].x += xx
				* (-6.579736267392905746 + 0.9277056288952613070 * zz);
		response[numkernby2].y += xx * (3.1006276680299820175 * z);
	}
	return response;
}

int getTotalHarmonics(int harmonicStages)
{
	return (1 << (harmonicStages - 1));
}

void initializeSubharmonicInfo(unsigned short maxHarmonic, unsigned short currentSubharmonic, int zhi, SubharmonicInfo *shi)
{
	double harmonicFraction = (double) currentSubharmonic / (double) maxHarmonic;
	shi->maxHarmonic = maxHarmonic;
	shi->currentHarmonic = currentSubharmonic;
	shi->zmax = calc_required_z(harmonicFraction, zhi);
	shi->totalKernelsTemplates = (shi->zmax / ACCEL_DZ) * 2 + 1;

	// Initialize kernels.
	shi->kernelFFTLength = calculateFFTLength(maxHarmonic, currentSubharmonic, zhi);
	shi->z = (int *) malloc(shi->totalKernelsTemplates * sizeof(int));
	shi->kernelHalfWidth = (unsigned short *) malloc (shi->totalKernelsTemplates * sizeof(unsigned short));
	int inputSize = shi->totalKernelsTemplates * sizeof(cufftComplex) * shi->kernelFFTLength;
	shi->kernels_data = (cufftComplex *) malloc(inputSize);

	// Initialize array to 0.
	memset(shi->kernels_data, 0, sizeof(cufftComplex) * shi->kernelFFTLength * shi->totalKernelsTemplates);

	// Create Kernels
	int iCounter = 0;
	for (iCounter = 0; iCounter < shi->totalKernelsTemplates; iCounter++)
	{
		shi->z[iCounter] = -shi->zmax + iCounter * ACCEL_DZ;
		shi->kernelHalfWidth[iCounter] = z_response_halfwidth((double) shi->z[iCounter]);

		// Place kernel data into a zero filled array.
		int num_kernel = 2 * ACCEL_NUMBETWEEN * shi->kernelHalfWidth[iCounter];
		int halfwidth = num_kernel / 2;
		cufftComplex *temp_kernel;
		temp_kernel = generate_z_response(0.0, ACCEL_NUMBETWEEN, shi->z[iCounter], num_kernel);
		memcpy(shi->kernels_data + (iCounter * shi->kernelFFTLength), temp_kernel + halfwidth, sizeof(cufftComplex) * halfwidth);
		memcpy(shi->kernels_data + (iCounter * shi->kernelFFTLength) + shi->kernelFFTLength - halfwidth, temp_kernel,
				sizeof(cufftComplex) * halfwidth);
		free(temp_kernel);
	}

	// FFT signal.
	cufftComplex *device_idata, *device_fft;
	cudaMalloc((void **) &device_idata, inputSize);
	cudaMemcpy(device_idata, shi->kernels_data, inputSize, cudaMemcpyHostToDevice);
	complexToComplexFFT(device_idata, shi->kernelFFTLength, &device_fft, shi->totalKernelsTemplates);
	free(shi->kernels_data);

	// Set kernel in device.
	shi->kernels_data = device_fft;

}

SubharmonicInfo **createSubharmonicInfo2(int numHarmonicStages, int zhi)
{
	short int totalHarmonics = getTotalHarmonics(numHarmonicStages);

	SubharmonicInfo **subharmInfo;
	subharmInfo = (SubharmonicInfo **) malloc(numHarmonicStages * sizeof(SubharmonicInfo *));

	// Create subharmonics info.
	short int currentStage;
	short int counter = 0;
	for (currentStage = 0; currentStage < numHarmonicStages; currentStage++)
	{
		short int currentHarmonic = 1;
		short int maxHarmonic = (1 << currentStage);

		// Cater for fundamental harmonic (this is performed only once).
		if (currentStage < 1)
		{
			maxHarmonic = 1;
		}

		subharmInfo[currentStage] = (SubharmonicInfo *) malloc(maxHarmonic * sizeof(SubharmonicInfo));
		for (currentHarmonic = 1; currentHarmonic <= maxHarmonic; currentHarmonic += 2)
		{
			initializeSubharmonicInfo(maxHarmonic, currentHarmonic, zhi, &subharmInfo[currentStage][currentHarmonic - 1]);

			printf("Harmonic %i/%i has %i kernel(s) from z = %i to %i,  FFT length = %i\n", currentHarmonic,
					maxHarmonic, subharmInfo[currentStage][currentHarmonic - 1].totalKernelsTemplates,
					-subharmInfo[currentStage][currentHarmonic - 1].zmax, subharmInfo[currentStage][currentHarmonic - 1].zmax,
					subharmInfo[currentStage][currentHarmonic - 1].kernelFFTLength);

			counter++;
		}
	}

	return subharmInfo;
}

SubharmonicInfo *createSubharmonicInfo(int numHarmonicStages, int zhi)
{
	short int totalHarmonics = getTotalHarmonics(numHarmonicStages);

	SubharmonicInfo *subharmInfo;
	subharmInfo = (SubharmonicInfo *) malloc(totalHarmonics * sizeof(SubharmonicInfo));

	// Create subharmonics info.
	short int currentStage;
	short int counter = 0;
	for (currentStage = 0; currentStage < numHarmonicStages; currentStage++)
	{
		short int currentHarmonic = 1;
		short int maxHarmonic = (1 << currentStage);

		// Cater for fundamental harmonic (this is performed only once).
		if (currentStage < 1)
		{
			maxHarmonic = 1;
		}

		for (currentHarmonic = 1; currentHarmonic <= maxHarmonic; currentHarmonic += 2)
		{
			initializeSubharmonicInfo(maxHarmonic, currentHarmonic, zhi, &subharmInfo[counter]);

			printf("Harmonic %i/%i has %i kernel(s) from z = %i to %i,  FFT length = %i\n", currentHarmonic,
					maxHarmonic, subharmInfo[counter].totalKernelsTemplates,
					-subharmInfo[counter].zmax, subharmInfo[counter].zmax,
					subharmInfo[counter].kernelFFTLength);

			counter++;
		}
	}

	return subharmInfo;
}


