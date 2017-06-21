extern "C"{
 #include "unistd.h"
}
#include "../headers/test_convolution.cuh"

/// Unit test for convolved signal.
bool testConvolution(cufftComplex* input_data, size_t totalElements, size_t elementSize)
{
	char* test_name = "Unit Test Convolution";
	char* data_path = "../Data/output_convolution.dat";

	// Save data to file if does not exists.
	if(!(access(data_path, F_OK ) != -1 )) // if file does not exists.
	{
		// Write to file.
		FILE *fp;
		if((fp=fopen(data_path, "wb")) == NULL)
		{
		   throwError("Cannot open file.");
		}
		if(fwrite(input_data, elementSize, totalElements, fp) != totalElements)
		{
			 throwError("Number of elements saved is not equal to the expected number of elements.");
		}
		fclose(fp);
		logMessage("%s: Data saved to file.", test_name);
	}

	// Read data.
	cufftComplex *output_data;
	FILE *fp;
    fp = fopen(data_path, "rb");
    output_data = (cufftComplex *) malloc((size_t) (elementSize * totalElements));
    if (fp != 0)
    {
        if (fread(output_data, elementSize, totalElements, fp) != totalElements)
        {
            throwError("Failed to read from data from file.");
        }
        fclose(fp);
    }
	logMessage("%s: Data read from file.", test_name);


	int ii;
	for(ii = 0; ii < totalElements; ii++)
	{
	   if (output_data[ii].x != input_data[ii].x || output_data[ii].y != input_data[ii].y )
	   {
	    	throwError("%s: Failed Data mismatch at element %i",test_name, ii);
	    	return false;
	   }
	 }
	logMessage("%s: Passed.", test_name);
	free(output_data);
	return true;
}
