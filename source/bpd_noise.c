#include <stdlib.h>
#include "../headers/bpd_noise.h"
#include "../headers/bpd_math.h"

/// Attempt to remove red-noise from the time series by using
/// a median-filter of logarithmically increasing width.
/// subtracting a running median and normalising the local
/// root mean square will result in the witened spectrum having
/// a zero mean and unit root mean square.
/// With this normalisation scheme, the S/N ratio of any specteral
/// feature is simply its amplitude.
void deredden(cufftComplex *fft, unsigned numOfBins)
{
	int ind, initialbuflen = 6, lastbuflen, maxbuflen = 200;
	long lCounter = 0;
	int binnum = 1, numwrote = 1;
	float *powbuf,  dslope = 1.0;
	float powargr, powargi;

	/* Takes care of the DC term */
	fft[0].x = 1.0;
	fft[0].y = 0.0;

	// TODO: GPUIZE Power Init
	/* Step through the input FFT and create powers */
	powbuf = (float*) malloc (numOfBins * sizeof(float));
	for (lCounter = 0; lCounter < numOfBins; lCounter++)
	{
		powbuf[lCounter] = fft[lCounter].x * fft[lCounter].x + fft[lCounter].y* fft[lCounter].y;
	}

	/* Calculate initial values */
	int buflen = initialbuflen;
	float mean_old = fast_median(powbuf + binnum, buflen) / log(2.0);

	// Write the first half of the normalized block
	// Note that this does *not* include a slope, but since it
	// is only a few bins, that is probably OK.
	float norm = inverse_square_root(mean_old);
	for (ind = numwrote; ind < binnum + buflen / 2; ind++)
	{
		fft[ind].x *= norm;
		fft[ind].y *= norm;
	}
	numwrote += buflen / 2;
	binnum += buflen;
	lastbuflen = buflen;
	buflen = initialbuflen * log(binnum);
	if (buflen > maxbuflen)
	{
		buflen = maxbuflen;
	}

	while ((binnum + buflen) < numOfBins)
	{
		// Calculate the next mean
		double mean_new = fast_median(powbuf + binnum, buflen) / log(2.0);
		// The slope between the last block median and the current median
		dslope = (mean_new - mean_old) / (0.5 * (lastbuflen + buflen));

		// Correct the last-half of the old block...
		for (lCounter = 0, ind = numwrote; ind < binnum + buflen / 2; lCounter++, ind++)
		{
			norm = inverse_square_root(mean_old + dslope * lCounter);
			fft[ind].x *= norm;
			fft[ind].y *= norm;
		}
		numwrote += lCounter;

		/* Update our values */
		binnum += buflen;
		lastbuflen = buflen;
		mean_old = mean_new;
		buflen = initialbuflen * log(binnum);
		if (buflen > maxbuflen)
		{
			buflen = maxbuflen;
		}
	}

	// TODO: GPUIZE
	// Deal with the last chunk (assume same slope as before)
	for (lCounter = 0, ind = numwrote; ind < numOfBins; lCounter++, ind++)
	{
		norm = inverse_square_root(mean_old + dslope * lCounter);
		fft[ind].x *= norm;
		fft[ind].y *= norm;
	}

	/* Free the powers */
	free(powbuf);
}
