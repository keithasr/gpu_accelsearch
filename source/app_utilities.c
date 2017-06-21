#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "../headers/app_utilities.h"

void throwError(const char *errorMessage, ...)
{
	va_list args;
    va_start(args, errorMessage);
    vprintf(errorMessage, args);
    printf("\n");
    va_end(args);

    exit(EXIT_FAILURE);
}

void logMessage(const char *msg, ...)
{
	//vprintf("%s \n",message);
	va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    printf("\n");
    va_end(args);
}
