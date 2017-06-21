extern "C"{
 #include "unistd.h"
}
#include "../headers/test_powers.cuh"


/// Unit test for convolved signal.
int testPowers(float* input_data, size_t totalElements, size_t elementSize)
{
	char* test_name = "Unit Test Powers";
	char* data_file = "../Data/output_powers.dat";

	// Save input data to file if does not exists.
	if(!(access(data_file, F_OK ) != -1 )) // if file does not exists.
	{
		// Write file.
		FILE *fp;
		if((fp=fopen(data_file, "wb")) == NULL)
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

	// Read input_data.
	float *output_data;
	FILE *fp;
	fp = fopen(data_file, "rb");
	output_data = (float *) malloc((size_t) (elementSize * totalElements));
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
	   if (output_data[ii] != input_data[ii])
	   {
		  // logMessage("Total number of elements are: %i", totalElements);
		   //logMessage("%s: Before Values %i %f %f", test_name, ii -1,  output_data[ii - 1], input_data[ii - 1]);
		   //logMessage("%s: After Values %i %f %f", test_name, ii + 1,  output_data[ii + 1], input_data[ii + 1]);
		   logMessage("%s: Data mismatch at element %i original: %f new: %f",test_name, ii, output_data[ii], input_data[ii]);
	   }
	 }
	free(output_data);
	logMessage("%s: Passed.", test_name);
	return 0;
}
