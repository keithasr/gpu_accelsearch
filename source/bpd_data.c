#include "../headers/bpd_common.h"
#include "../headers/bpd_data.h"

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>


long readData(char* path, float **output_data)
{
	fflush(NULL);

	long lCounter = 0;

	FILE *file;
	file = fopen(path, "rb");
	if (file == NULL)
	{
		throwError("Cannot open file");
	}

	// Validation: Check time series length.
	struct stat buf;
	int fileNum = fileno(file);
	int statRet = fstat(fileNum, &buf);
	if (statRet < 0)
	{
		throwError("Cannot retrieve file descriptor");
	}
	unsigned long fileLength = (buf.st_size / sizeof(float));
	if (fileLength <= 0)
	{
		throwError("File descriptor was not read.");
	}
	else if (fileLength > 67108864)
	{
		throwError("The input time series is too large");
	}

	// Initialize data.
	long dataPoints = fileLength + (2 * ACCEL_PADDING);
	float *data;
	data = (float *) malloc(sizeof(float) * dataPoints);
	for (lCounter = 0; lCounter < dataPoints; lCounter++)
	{
		data[lCounter] = 0.0;
	}

	// NOTE: Data padding.
	// The padding allows us to search very short time series
	// using correlations without having to worry about
	// accessing data before or after the valid FFT freqs.

	// Position and read the data.
	// Initial padding is set to zero.
	// Last padding will be overlapped.
	int rt = fseeko(file, 0, SEEK_SET);
	if (rt < 0)
	{
		throwError("Error in fseek");
	}
	long readBins = fread(data + ACCEL_PADDING, sizeof(float),
			(dataPoints - ACCEL_PADDING), file);
	if ((readBins != fileLength) && ferror(file))
	{
		throwError("Reading input file");
	}

	fclose(file);

	// Set output data array.
	*output_data = data + ACCEL_PADDING;

	return fileLength;
}
